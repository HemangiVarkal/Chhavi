#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAMSES Radial Density Profiles using Osyris

Compute radial density profiles for RAMSES datasets loaded via OSYRIS.

Features:
- Auto-detects OR manually specifies output snapshots to process
- Cross-platform: Windows (serial), Linux/macOS (parallel via multiprocessing)
- Universal OSYRIS loader (tries modern `load()` then legacy `RamsesDataset`)
- Computes mean/std/min/max density per unique radial bin from mesh center
- Saves clean CSV files to ./profile_outputs/: osyris_profile_XXXXX.csv

CLI Usage:
    python3 compute_osyris_profile.py --base-dir ../ramses_outputs --folder-name sedov_3d
    python3 compute_osyris_profile.py --base-dir ../ramses_outputs --folder-name sedov_3d -n 1,5
"""

import os
import re
import argparse
import logging
from pathlib import Path
import numpy as np
from typing import List
from multiprocessing import Pool

try:
    import osyris
except ImportError:
    raise ImportError("Osyris not installed. Install with: pip install osyris")

logger = logging.getLogger("osyris_profiles")


def parse_output_numbers(value: str) -> List[int]:
    """
    Parse output numbers: "1", "1,3,5", "10-15".
    """
    numbers = set()
    parts = value.split(",")
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        if "-" in part:
            try:
                start, end = map(int, re.findall(r"\d+", part)) 
                numbers.update(range(start, end + 1))
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid range '{part}'")
        else:
            try:
                numbers.add(int(part))
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid number '{part}'")
    
    if not numbers:
        raise argparse.ArgumentTypeError(f"No valid numbers in: {value}")
    
    return sorted(list(numbers))


def detect_available_outputs(base_dir: str, folder_name: str) -> List[int]:
    """
    Scan folder for output_XXXXX directories.
    """
    folder_path = Path(base_dir) / folder_name
    
    logger.info("Scanning: %s", folder_path)
    
    if not folder_path.exists():
        logger.warning("Folder not found: %s", folder_path)
        return []

    pattern = re.compile(r"output_(\d{5})") 
    outputs = []
    try:
        for item in folder_path.iterdir():
            if item.is_dir():  
                match = pattern.fullmatch(item.name)
                if match:
                    outputs.append(int(match.group(1)))
        outputs.sort()
        if outputs:
            logger.info("Found %d outputs: %s", 
                       len(outputs), 
                       outputs[:5] + ["..."] if len(outputs)>5 else outputs)
        else:
            logger.warning("No output_XXXXX directories in: %s", folder_path)
    except PermissionError:
        logger.error("Permission denied: %s", folder_path)
        return []
    
    return outputs


def match_short_numbers(numbers: List[int], available: List[int]) -> List[int]:
    """
    Match short numbers to available 5-digit outputs.
    """
    matched = []
    for n in numbers:
        n_str = str(n)
        if len(n_str) >= 3:
            if n in available:
                matched.append(n)
        else:
            for pad_len in [5, 4, 3]:
                padded = int(f"{n:0{pad_len}d}")
                if padded in available:
                    matched.append(padded)
                    break
    return sorted(set(matched))


def universal_osyris_loader(output_number: int, base_dir: str, folder_name: str):
    """
    Load dataset using Osyris
    """
    output_path = Path(base_dir) / folder_name / f"output_{output_number:05d}"
    
    if not output_path.exists():
        raise RuntimeError(f"Output directory missing: {output_path}")

    try:
        return osyris.load(output_path)
    except:
        try:
            dataset = osyris.RamsesDataset(output_number, path=Path(base_dir) / folder_name)
            return dataset.load()
        except Exception as e:
            raise RuntimeError(f"Failed to load {output_path}: {e}")


def compute_radial_profile(data, output_number: int, output_csv: str):
    """
    Compute radial density statistics and save CSV.
    """
    logger.info("Processing output_%05d", output_number)

    mesh = data["mesh"]
    meta = data.meta
    center = meta["boxlen"] * meta["unit_l"] * np.array([0.5, 0.5, 0.5])

    positions = mesh["position"].values
    radial_dist = np.linalg.norm(positions - center, axis=1)
    rval_unique, rinv_idx = np.unique(radial_dist, return_inverse=True)

    stats = []
    for i in range(len(rval_unique)):
        idx = np.where(rinv_idx == i)[0]
        densities = mesh["density"].values[idx]
        stats.append([
            rval_unique[i],
            np.mean(densities),
            np.std(densities),
            np.min(densities),
            np.max(densities)
        ])

    profile_array = np.array(stats)
    header = "Radius,Density_mean,Density_std,Density_min,Density_max"
    np.savetxt(output_csv, profile_array, delimiter=",", header=header)
    logger.info("Saved: %s (%d bins)", output_csv, len(stats))


def process_single_snapshot(args_tuple):
    """
    Multiprocessing worker.
    """
    args, output_number = args_tuple
    data = universal_osyris_loader(output_number, args.base_dir, args.folder_name)
    # HARDCODED: Always saves to ./profile_outputs/
    csv_path = Path("profile_outputs") / f"osyris_profile_{output_number:05d}.csv"
    os.makedirs(csv_path.parent, exist_ok=True)
    compute_radial_profile(data, output_number, str(csv_path))


def run_serial(args, numbers: List[int]):
    """
    Windows-safe serial execution.
    """
    logger.info("Running SERIAL mode (Windows)")
    for num in numbers:
        try:
            process_single_snapshot((args, num))
        except Exception as e:
            logger.error("Failed snapshot %d: %s", num, e)


def run_parallel(args, numbers: List[int]):
    """
    Unix parallel execution.
    """
    nproc = args.nproc or os.cpu_count() or 1
    logger.info("Running PARALLEL mode (%d cores)", nproc)
    with Pool(processes=nproc) as pool:
        pool.map(process_single_snapshot, [(args, num) for num in numbers])


def main() -> None:
    """
    Main CLI entrypoint.
    """
    parser = argparse.ArgumentParser(
        description="RAMSES Radial Density Profiles via Osyris",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--base-dir", type=str, required=True, 
                       help="RAMSES root directory (e.g., '../ramses_outputs')")
    parser.add_argument("--folder-name", type=str, required=True, 
                       help="Simulation output folder (e.g., 'sedov_3d')")
    parser.add_argument("-n", "--numbers", type=parse_output_numbers, default=None, 
                       help="Output numbers to process: '1', '1,3,5', '10-15' (omit=ALL)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--nproc", type=int, default=None, help="CPU cores (Unix only)")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s"
    )

    # Normalize input path
    args.base_dir = os.path.abspath(args.base_dir)
    input_folder = os.path.join(args.base_dir, args.folder_name)
    logger.info("Input folder: %s", input_folder)

    # Detect available snapshots
    all_outputs = detect_available_outputs(args.base_dir, args.folder_name)

    # Select snapshots to process
    if args.numbers is not None:
        numbers = match_short_numbers(args.numbers, all_outputs)
        if not numbers:
            logger.warning("No matches for %s. Using all %d available.", args.numbers, len(all_outputs))
            numbers = all_outputs
    else:
        numbers = all_outputs

    if not numbers:
        logger.info("No snapshots found to process.")
        return

    # HARDCODED OUTPUT: Always ./profile_outputs/ (relative to profiles/)
    output_dir = "profile_outputs"
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Output directory: %s", os.path.abspath(output_dir))

    # Platform-specific execution
    try:
        if os.name == "nt":  # Windows
            run_serial(args, numbers)
        else:  # Unix
            run_parallel(args, numbers)
        logger.info("Completed: generated %d OSYRIS profiles in %s", len(numbers), output_dir)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception("Pipeline failed")


if __name__ == "__main__":
    main()
