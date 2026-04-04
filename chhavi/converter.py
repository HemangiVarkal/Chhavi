#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

──────────────────────────────────────────────────────────────────────────────
Description
──────────────────────────────────────────────────────────────────────────────
This script converts RAMSES simulation outputs into the VTKHDF format that
ParaView and other VTK-based tools can read.

──────────────────────────────────────────────────────────────────────────────
WHY THIS EXISTS
──────────────────────────────────────────────────────────────────────────────
- RAMSES uses an Adaptive Mesh Refinement (AMR) grid that’s great for HPC, but
  not directly plug-and-play for most visualization pipelines.
- VTKHDF’s **OverlappingAMR** stores data by refinement level, with cell-centered
  fields and AMR indexing metadata. That’s perfect for ParaView.

──────────────────────────────────────────────────────────────────────────────
IT SUPPORTS:
──────────────────────────────────────────────────────────────────────────────
 - Vectorized spatial filtering with normalized ranges (--x-range/--y-range/--z-range)
 - Per-snapshot parallel processing (ProcessPoolExecutor)
 - Field discovery (--list-fields) and explicit field selection (--fields)
 - Dry-run mode (--dry-run) to print the plan without writing files
 - Number of cores (--nproc) for number of cores to be used for parallel processing (default: 1)
 - Embedding run metadata (CLI command, timestamp, code version) in each output file

"""


from __future__ import annotations

import logging
import shlex
import sys
import time
import argparse
from typing import Dict, Iterable, List, Optional, Tuple
import os

import numpy as np
import h5py as h5
import osyris


def setup_logging(verbose: bool) -> None:
    """
    Configure global logging.

    Args:
        verbose: If True, set DEBUG level, otherwise INFO.
    """

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Reduce noise from libraries (adjustable)
    logging.getLogger("h5py").setLevel(logging.WARNING)


logger = logging.getLogger("chhavi")


def parse_output_numbers(arg: str) -> List[int]:
    """
    Parse output numbers strings like '5', '1,3,5', or '2-7' into a list of ints.

    Args:
        arg: user-provided string

    Returns:
        List of ints representing snapshot/output numbers.

    Raises:
        argparse.ArgumentTypeError on invalid format.
    """

    if "-" in arg and "," in arg:
        raise argparse.ArgumentTypeError("Do not mix ranges and lists; use either 'a-b' or 'a,b,c'.")

    if "-" in arg:
        try:
            start, end = map(int, arg.split("-", 1))
        except ValueError:
            raise argparse.ArgumentTypeError("Invalid range; use 'start-end'.")
        if end < start:
            raise argparse.ArgumentTypeError("Range end must be >= start.")
        return list(range(start, end + 1))

    if "," in arg:
        nums = []
        for x in arg.split(","):
            x = x.strip()
            if x == "":
                continue
            try:
                nums.append(int(x))
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid integer in list: '{x}'")
        return nums

    try:
        return [int(arg)]
    except ValueError:
        raise argparse.ArgumentTypeError("Output number must be an integer.")


def parse_norm_range(arg: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse normalized axis range spec: 'min:max', ':max', 'min:', or ':'.

    Returns (min_norm, max_norm) where each entry is a float in [0,1], or None when not provided.

    Args:
        arg: string or None
    """

    if arg is None:
        return (None, None)

    s = arg.strip()

    if s == "":
        return (None, None)

    if ":" not in s:
        raise argparse.ArgumentTypeError("Axis range must be 'min:max' (e.g., 0.2:0.8, :0.6, 0.1:, :).")

    left, right = s.split(":", 1)
    minv = float(left) if left.strip() != "" else 0.0
    maxv = float(right) if right.strip() != "" else 1.0

    if not (0.0 <= minv <= 1.0 and 0.0 <= maxv <= 1.0):
        raise argparse.ArgumentTypeError("Axis normalized bounds must be within [0, 1].")

    if minv > maxv:
        raise argparse.ArgumentTypeError("Axis min cannot be greater than axis max.")

    return (minv, maxv)


def parse_fields_arg(arg: Optional[str]) -> Optional[List[str]]:
    """
    Parse the --fields argument which is a comma-separated list of field names.

    Returns None if user didn't pass anything (means use defaults or auto-detect).
    """

    if arg is None:
        return None

    fields = [f.strip() for f in arg.split(",") if f.strip() != ""]

    return fields if fields else None


class ChhaviConverter:
    """
    Convert RAMSES (osyris) outputs into VTKHDF OverlappingAMR files.

    This class is intentionally small and unit-test friendly: separate methods perform
    small responsibilities (boxlength inference, mask building, vector extraction, HDF5 writes).

    Args:
        input_folder: Directory with RAMSES outputs.
        output_prefix: Prefix for output files.
        fields: List of fields to include (None for defaults).
        level_start: Minimum AMR level.
        level_end: Maximum AMR level.
        x/y/z_range_norm: Normalized spatial ranges.
        dry_run: If True, no files written.
        output_directory: Optional directory for output files; defaults to input_folder.
    """

    def __init__(
        self,
        input_folder: str,
        output_prefix: str = "overlapping_amr",
        fields: Optional[List[str]] = None,
        level_start: Optional[int] = None,
        level_end: Optional[int] = None,
        x_range_norm: Tuple[Optional[float], Optional[float]] = (None, None),
        y_range_norm: Tuple[Optional[float], Optional[float]] = (None, None),
        z_range_norm: Tuple[Optional[float], Optional[float]] = (None, None),
        dry_run: bool = False,
        output_directory: Optional[str] = None,
    ):
        # Inputs & configuration
        self.input_folder = input_folder
        self.output_prefix = output_prefix

        # User-requested fields (None = auto-detect later)
        self.requested_fields = fields

        # Level bounds (None => no filter)
        self.level_start = level_start
        self.level_end = level_end

        # normalized ranges (None/None => no filter)
        self.x_range_norm = x_range_norm
        self.y_range_norm = y_range_norm
        self.z_range_norm = z_range_norm

        self.dry_run = dry_run

        # Directory to save output files
        self.output_directory = output_directory or input_folder

        # dtype aliases for HDF5 datasets
        self.float_dtype = "f"  # float32
        self.int_dtype = "i8"

        # physical bounds converted after boxlength known (dict with 'x','y','z' -> (lo,hi) or None)
        self._phys_bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None

    
    def _infer_boxlength_from_data(self, data) -> Optional[float]:
        """
        Try to infer simulation box length from osyris dataset `data`.
    
        This version multiplies 'boxlen' by 'unit_l' from the dataset's metadata to
        ensure spatial filtering units match the physical coordinates.
    
        Returns:
            box length as float, or None if not found.
        """
        try:
            if (
                hasattr(data, "meta")
                and isinstance(data.meta, dict)
                and "boxlen" in data.meta
                and "unit_l" in data.meta
            ):
                return float(data.meta["boxlen"] * data.meta["unit_l"])
        except Exception:
            logger.debug("Error while reading data.meta for boxlen or unit_l")
            pass
        return None


    def _compute_physical_bounds(self, data) -> None:
        """
        Convert normalized filters to physical coordinates using inferred box length.

        If any axis filter is requested but boxlength cannot be inferred, a ValueError is raised.
        """
        any_filter = any(
            [
                self.x_range_norm != (None, None),
                self.y_range_norm != (None, None),
                self.z_range_norm != (None, None),
            ]
        )

        if not any_filter:
            self._phys_bounds = None
            return

        boxlen = self._infer_boxlength_from_data(data)

        if boxlen is None:
            logger.warning(
                "Spatial filtering was requested but boxlength could not be inferred from dataset. "
                "Filtering will fail. Ensure your RAMSES/osyris data includes box length metadata."
            )
            raise ValueError(
                "Spatial filtering requested but boxlength could not be inferred from dataset. "
                "Ensure your RAMSES/osyris data includes box length metadata."
            )

        def conv(r: Tuple[Optional[float], Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
            if r == (None, None):
                return (None, None)

            lo, hi = r

            return (lo * boxlen, hi * boxlen)

        xb = conv(self.x_range_norm)
        yb = conv(self.y_range_norm)
        zb = conv(self.z_range_norm)

        self._phys_bounds = {"x": xb, "y": yb, "z": zb}

        logger.info("Using boxlength = %s", boxlen)

        if xb != (None, None):
            logger.info("x filter (physical): [%.6g, %.6g]", xb[0], xb[1])
        if yb != (None, None):
            logger.info("y filter (physical): [%.6g, %.6g]", yb[0], yb[1])
        if zb != (None, None):
            logger.info("z filter (physical): [%.6g, %.6g]", zb[0], zb[1])

    def read_data(self, output_num: int):
        """
        Load a RAMSES snapshot using osyris.RamsesDataset and return the loaded dataset.
        On failure, logs the error and returns None (so caller can handle).
        """
        try:
            ds = osyris.RamsesDataset(output_num, path=self.input_folder).load()
            return ds
        except Exception as e:
            logger.error("Failed to load output %s from '%s': %s", output_num, self.input_folder, e)
            logger.debug("Exception details:", exc_info=True)
            return None

    def _filter_levels(self, levels: Iterable[int]) -> List[int]:
        """
        Apply level_start/level_end filters to the list/iterable of levels.
        """
        levels = list(levels)
        if self.level_start is not None:
            levels = [lvl for lvl in levels if lvl >= self.level_start]
        if self.level_end is not None:
            levels = [lvl for lvl in levels if lvl <= self.level_end]
        return levels

    def _build_mask(self, px: np.ndarray, py: np.ndarray, pz: np.ndarray) -> np.ndarray:
        """
        Build boolean mask to apply spatial filtering.

        If no physical bounds are set (self._phys_bounds is None) returns all True mask.
        """
        if self._phys_bounds is None:
            return np.ones_like(px, dtype=bool)

        mask = np.ones_like(px, dtype=bool)

        (xmin, xmax) = self._phys_bounds["x"]
        (ymin, ymax) = self._phys_bounds["y"]
        (zmin, zmax) = self._phys_bounds["z"]

        if xmin is not None and xmax is not None:
            mask &= (px >= xmin) & (px <= xmax)
        if ymin is not None and ymax is not None:
            mask &= (py >= ymin) & (py <= ymax)
        if zmin is not None and zmax is not None:
            mask &= (pz >= zmin) & (pz <= zmax)

        return mask

    def _extract_vector(self, vec_field) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract vector components x,y,z from a vector-like osyris field.

        Tries an efficient fast path (vec_field.x.values / vec_field['x'].values) and falls back
        to safe iteration if necessary.

        Returns:
            tuple of 1D numpy arrays (vx, vy, vz)
        """
        try:
            if hasattr(vec_field, "__getitem__"):
                try:
                    vx = np.asarray(vec_field["x"].values, dtype=float)
                    vy = np.asarray(vec_field["y"].values, dtype=float)
                    vz = np.asarray(vec_field["z"].values, dtype=float)
                    return vx, vy, vz
                except Exception:
                    pass
            try:
                vx = np.asarray(vec_field.x.values, dtype=float)
                vy = np.asarray(vec_field.y.values, dtype=float)
                vz = np.asarray(vec_field.z.values, dtype=float)
                return vx, vy, vz
            except Exception:
                pass
        except Exception:
            logger.debug("Fast path vector extraction failed; using safe fallback.", exc_info=True)

        vx_list = []
        vy_list = []
        vz_list = []

        for v in vec_field:
            try:
                vx_list.append(float(v.x.values))
                vy_list.append(float(v.y.values))
                vz_list.append(float(v.z.values))
            except Exception:
                try:
                    vx_list.append(float(v.x))
                    vy_list.append(float(v.y))
                    vz_list.append(float(v.z))
                except Exception as ee:
                    logger.debug("Vector element extraction failed for element %r: %s", v, ee)
                    raise RuntimeError("Unable to extract vector components from osyris field; incompatible format.")

        return np.asarray(vx_list, dtype=float), np.asarray(vy_list, dtype=float), np.asarray(vz_list, dtype=float)

    def _write_level_to_hdf5(
        self,
        level_group,
        spacing: float,
        corner_coords: np.ndarray,
        scalars_dict: Dict[str, np.ndarray],
        vectors_dict: Dict[str, np.ndarray],
    ) -> None:
        """
        Write one AMR level group to HDF5 following VTKHDF OverlappingAMR expectations.

        Args:
            level_group: h5py Group for level (already created)
            spacing: cell spacing (dx) for this level
            corner_coords: Nx3 array of cell corner coordinates (min corner)
            scalars_dict: mapping name -> 1D array (length N) of scalar values
            vectors_dict: mapping name -> Nx3 array of vector components
        """
        level_group.attrs.create("Spacing", [spacing, spacing, spacing], dtype=self.float_dtype)
        level_group.attrs["NumberOfBlocks"] = len(corner_coords)

        inv = 1.0 / spacing
        corner_coords = np.asarray(corner_coords)
        i_min = (corner_coords[:, 0] * inv).astype(np.int32)
        j_min = (corner_coords[:, 1] * inv).astype(np.int32)
        k_min = (corner_coords[:, 2] * inv).astype(np.int32)
        amr_values = np.column_stack([i_min, i_min, j_min, j_min, k_min, k_min]).astype(np.int32)
        level_group.create_dataset("AMRBox", data=amr_values)

        celldata = level_group.create_group("CellData")

        for name, arr in scalars_dict.items():
            celldata.create_dataset(name, data=arr.reshape(-1, 1).astype(self.float_dtype))

        for name, arr in vectors_dict.items():
            celldata.create_dataset(name, data=arr.astype(self.float_dtype))

        level_group.create_group("PointData")
        level_group.create_group("FieldData")

    def _collect_fields_from_mesh(self, mesh) -> List[str]:
        """
        Try to collect available field names from the mesh object.
        This attempts multiple access patterns to be robust across osyris versions.
        """
        try:
            keys = list(mesh.keys())
            return keys
        except Exception as e:
            logger.error("Cannot extract fields from mesh: %s", e)
            return []

    def convert_one(self, output_num: int, data) -> None:
        """
        Convert one loaded dataset to VTKHDF and write to disk (unless dry-run).

        Args:
            output_num: snapshot number
            data: object returned by osyris.RamsesDataset(...).load()
        """
        if data is None:
            logger.warning("No data for output %s; skipping.", output_num)
            return

        if "mesh" not in data:
            logger.error("Dataset %s does not contain a 'mesh' key; skipping.", output_num)
            return

        mesh = data["mesh"]

        try:
            self._compute_physical_bounds(data)
        except ValueError as e:
            logger.error("Skipping output %s: %s", output_num, e)
            return

        missing_fields = set()

        if self.requested_fields is None:
            self.requested_fields = ["density", "pressure", "velocity"]

        try:
            levels = sorted(np.unique(mesh["level"].values))
        except Exception:
            try:
                levels = sorted(np.unique(np.asarray([row["level"] for row in mesh])))
            except Exception as e:
                logger.error("Unable to determine AMR levels for output %s: %s", output_num, e)
                logger.debug("mesh sample: %r", mesh)
                return

        levels = self._filter_levels(levels)
        logger.info("Levels to write for output %s: %s", output_num, levels)

        if self.dry_run:
            logger.info(
                "[dry-run] Would write file '%s_%05d.vtkhdf' with fields: %s (levels: %s), if available.",
                self.output_prefix,
                output_num,
                self.requested_fields,
                levels,
            )
            return

        t0 = time.time()
        output_filename = os.path.join(
            self.output_directory,
            f"{self.output_prefix}_{output_num:05d}.vtkhdf"
        )

        level_data = []

        try:
            for new_level, actual_level in enumerate(levels):
                level_sel = mesh[mesh["level"].values == actual_level]

                dx_arr = level_sel["dx"].values

                if len(dx_arr) == 0:
                    logger.debug("Level %s has zero cells; skipping.", actual_level)
                    continue

                spacing = float(dx_arr[0])

                pos = level_sel["position"]
                px = np.asarray([float(p.x.values) for p in pos])
                py = np.asarray([float(p.y.values) for p in pos])
                pz = np.asarray([float(p.z.values) for p in pos])

                logger.debug("Level %s: %d cells before filtering", actual_level, len(px))

                mask = self._build_mask(px, py, pz)
                kept = int(np.count_nonzero(mask))
                logger.debug("Level %s: %d cells after spatial filtering", actual_level, kept)
                if kept == 0:
                    continue

                l = level_sel["dx"].values.astype(float)

                l = l[mask]
                cx = px[mask] - l / 2.0
                cy = py[mask] - l / 2.0
                cz = pz[mask] - l / 2.0
                corner_coords = np.column_stack([cx, cy, cz])

                scalars: Dict[str, np.ndarray] = {}
                vectors: Dict[str, np.ndarray] = {}

                def get_scalar(name: str) -> Optional[np.ndarray]:
                    try:
                        if name in level_sel:
                            arr = np.asarray(level_sel[name].values, dtype=float)[mask]
                            return arr
                    except Exception:
                        logger.debug("Scalar field '%s' reading failed on level %s", name, actual_level, exc_info=True)
                    return None

                def get_vector(name: str) -> Optional[np.ndarray]:
                    try:
                        if name in level_sel:
                            vf = level_sel[name]
                            vx, vy, vz = self._extract_vector(vf)
                            vec = np.column_stack([vx[mask], vy[mask], vz[mask]])
                            return vec
                    except Exception:
                        logger.debug("Vector field '%s' reading failed on level %s", name, actual_level, exc_info=True)
                    return None

                for fieldname in self.requested_fields:
                    if fieldname in {"density", "pressure", "grav_potential"}:
                        arr = get_scalar(fieldname)
                        if arr is not None:
                            scalars[fieldname.capitalize()] = arr
                        else:
                            missing_fields.add(fieldname)

                    elif fieldname in {"velocity", "grav_acceleration", "magnetic_field"}:
                        vec = get_vector(fieldname)
                        if vec is not None and vec.size:
                            vectors[fieldname.replace("_", " ").title().replace(" ", "_")] = vec
                        else:
                            missing_fields.add(fieldname)

                    else:
                        arr = get_scalar(fieldname)
                        if arr is not None:
                            scalars[fieldname] = arr
                        else:
                            vec = get_vector(fieldname)
                            if vec is not None:
                                vectors[fieldname] = vec
                            else:
                                missing_fields.add(fieldname)

                if scalars or vectors:
                    level_data.append((new_level, spacing, corner_coords, scalars, vectors))

            if not level_data:
                for mf in sorted(missing_fields):
                    logger.warning(
                        "Requested field '%s' not found in output %s; skipping.",
                        mf,
                        output_num,
                    )
                logger.warning("Skipping output %s: no file created (all cells filtered out or requested fields missing).", output_num)
                return

            with h5.File(output_filename, "w") as f:
                root = f.create_group("VTKHDF", track_order=True)
                root.attrs["Version"] = (2, 2)
                root.attrs.create(
                    "Type", b"OverlappingAMR", dtype=h5.string_dtype("ascii", len(b"OverlappingAMR"))
                )
                root.attrs.create(
                    "GridDescription", b"XYZ", dtype=h5.string_dtype("ascii", len(b"XYZ"))
                )
                root.attrs.create(
                    "Origin", [0.0, 0.0, 0.0], dtype=self.float_dtype
                )
                root.attrs["NumberOfLevels"] = len(level_data)

                root.attrs["generator_command"] = shlex.join(sys.argv)
                root.attrs["generator_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                root.attrs["generator_version"] = __version__

                for new_level, spacing, corner_coords, scalars, vectors in level_data:
                    level_group = root.create_group(f"Level{new_level}")
                    self._write_level_to_hdf5(
                        level_group=level_group,
                        spacing=spacing,
                        corner_coords=corner_coords,
                        scalars_dict=scalars,
                        vectors_dict=vectors,
                    )

            logger.info("DONE: Saved '%s_%05d.vtkhdf' in %.2fs", self.output_prefix, output_num, time.time() - t0)
        except Exception as e:
            logger.exception("Failed to convert/write output %s: %s", output_num, e)

    def process_output(self, output_num: int) -> None:
        """
        Read and convert a single output (load + convert_one).
        This wrapper isolates exceptions so callers (parallel runner) can continue on failure.
        """
        data = self.read_data(output_num)
        try:
            self.convert_one(output_num, data)
        except Exception as e:
            logger.exception("Unexpected failure converting output %s: %s", output_num, e)


def list_fields_for_snapshot(input_folder: str, output_num: int) -> List[str]:
    """
    Load one snapshot and return a best-effort list of available fields found in mesh.

    This function is tolerant: when structures differ across osyris versions it will attempt
    a few strategies and return the union of discovered names.
    """
    ds = None
    try:
        ds = osyris.RamsesDataset(output_num, path=input_folder).load()
    except Exception as e:
        logger.error("Failed to load output %s for field listing: %s", output_num, e)
        return []

    if ds is None or "mesh" not in ds:
        logger.warning("No 'mesh' found in snapshot %s; cannot list fields.", output_num)
        return []

    mesh = ds["mesh"]

    conv = ChhaviConverter(input_folder=input_folder)

    fields = conv._collect_fields_from_mesh(mesh)

    unique_fields = []

    for f in fields:
        if f not in unique_fields:
            unique_fields.append(f)
    return unique_fields
