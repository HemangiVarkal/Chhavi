# -*- coding: utf-8 -*-

"""

Chhavi: RAMSES → VTKHDF (Overlapping AMR) Converter
===================================================

──────────────────────────────────────────────────────────────────────────────
Description
──────────────────────────────────────────────────────────────────────────────
Chhavi converts RAMSES simulation outputs into the VTKHDF format that
ParaView and other VTK-based visualization tools can read.

──────────────────────────────────────────────────────────────────────────────
WHY THIS EXISTS
──────────────────────────────────────────────────────────────────────────────
- RAMSES uses an Adaptive Mesh Refinement (AMR) grid suitable for HPC, but not
  directly compatible with most visualization pipelines.
- VTKHDF’s OverlappingAMR format stores data by refinement level, with
  cell-centered fields and AMR indexing metadata, ideal for ParaView.

"""

from .converter import (
    ChhaviConverter,
    parse_output_numbers,
    parse_norm_range,
    parse_fields_arg,
    list_fields_for_snapshot,
)

from .parallel import (
    process_single_output,
    run_parallel_conversion,
)

__version__ = "1.0.1"
