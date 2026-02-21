#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

─────────────────────────────────────────────────────────────
Example Usage of Chhavi
─────────────────────────────────────────────────────────────

This script demonstrates how to use the `ChhaviConverter`
class to explore and convert RAMSES simulation outputs into
VTKHDF OverlappingAMR files.

Features demonstrated:
1. Listing available fields in snapshots
2. Inspecting basic dataset info
3. Performing a dry-run conversion (no files written)
4. Showing physical fields that will be added to the output

Note: The output directory is created automatically by Chhavi
if it does not already exist.

─────────────────────────────────────────────────────────────

"""

import os
from typing import List
from chhavi.converter import ChhaviConverter, list_fields_for_snapshot

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

RAMSES_OUTPUT_ROOT = "ramses_outputs/sedov_3d"

SNAPSHOT_FOLDERS = ["output_00001", "output_00002"]

FIELDS_TO_INSPECT = ["density", "velocity", "pressure"]

DRY_RUN = True

OUTPUT_DIR = None  # Set to e.g. "vtk_outputs" to specify output location, or None to use input folder


# ──────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────


def print_snapshot_info(snapshot_num: int, snapshot_name: str, fields: List[str]):

    print(f"\n🔹 Processing snapshot {snapshot_num} ({snapshot_name})...")
    if fields:
        print(f"Detected fields: {', '.join(fields)}")
    else:
        print("Detected fields: None")


def print_physical_fields(fields: List[str]):

    scalar_fields = [f for f in fields if f.lower() in {"density", "pressure", "grav_potential"}]
    vector_fields = [f for f in fields if f.lower() in {"velocity", "grav_acceleration", "magnetic_field"}]

    print("Fields that will be added to VTKHDF:")
    if scalar_fields:
        print("  Scalars:", ", ".join(scalar_fields))
    if vector_fields:
        print("  Vectors:", ", ".join(vector_fields))
    if not scalar_fields and not vector_fields:
        print("  None (no standard fields detected)")


# ──────────────────────────────────────────────────────────────
# Main Example Workflow
# ──────────────────────────────────────────────────────────────

def main():

    print("=== Chhavi Example Usage ===")
    print("This example inspects snapshots and performs a dry-run conversion.\n")

    for idx, snapshot_name in enumerate(SNAPSHOT_FOLDERS, start=1):
        try:
            available_fields = list_fields_for_snapshot(RAMSES_OUTPUT_ROOT, idx)
        except Exception as e:
            print(f"⚠️ Failed to list fields for snapshot {idx}: {e}")
            available_fields = []

        print_snapshot_info(idx, snapshot_name, available_fields)

        converter = ChhaviConverter(
            input_folder=RAMSES_OUTPUT_ROOT,
            output_prefix=f"example_{snapshot_name}",
            fields=FIELDS_TO_INSPECT,
            dry_run=DRY_RUN,
            output_directory=OUTPUT_DIR,
        )

        print_physical_fields(converter.requested_fields)

        try:
            converter.process_output(idx)
            print(f"✅ Snapshot {idx} dry-run conversion completed successfully!")
        except Exception as e:
            print(f"❌ Failed to process snapshot {idx}: {e}")

    print("\nExample usage finished!")
    print("Set `dry_run=False` to write actual VTKHDF files.")


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
