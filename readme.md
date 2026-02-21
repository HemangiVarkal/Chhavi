# Chhavi: A Python tool for converting RAMSES outputs to VTKHDF

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![Python Version](https://img.shields.io/badge/python-3.9%2B-brightgreen)]()  
[![Build Status](https://img.shields.io/badge/tests-passing-brightgreen)]()


## Overview

**Chhavi** is a Python package that converts **[RAMSES](https://ramses-organisation.readthedocs.io/en/latest/)** simulation outputs into **[VTKHDF](https://vtk.org/documentation/)** OverlappingAMR format.

It provides both a **command-line interface (CLI)** and a **Python API**, making it easy to visualize and analyze data in [ParaView](https://docs.paraview.org/en/latest/) and other compatible tools.

The package also includes test coverage, example scripts, profile analysis suite for **Osyris** vs **VTKHDF** validation and clear documentation, ensuring it is reproducible and accessible for scientific use.

---

## Features

- Convert RAMSES AMR outputs into VTKHDF OverlappingAMR files  
- Uses [Osyris](https://osyris.readthedocs.io/en/stable/) for data analysis and extraction  
- Support for both **scalar fields** (density, pressure, grav_potential) and  
  **vector fields** (velocity, magnetic_field, grav_acceleration)  
- Dry-run mode to preview what would be written without creating files  
- CLI and Python API for flexible use  
- Parallel conversion support with configurable number of workers (`--nproc`)  
- Customizable output directory through `--output-dir` option
- Profile Analysis - **Osyris** vs **VTKHDF** validation (CCC > 0.99)
- Fully tested with `pytest`  

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/HemangiVarkal/Chhavi.git
cd Chhavi
pip install -r requirements.txt
```

---

## Usage

### Command-Line Interface (CLI)

Example:

```bash
python -m chhavi.cli --base-dir ramses_outputs/ 
       --folder-name sedov_3d/ -n 1 
       --output-prefix sedov_test 
       --fields density,velocity,pressure 
       --dry-run
       --output-dir ./vtk_outputs
       --nproc 1
```

Key options:  
- `--base-dir` → Parent folder  
- `--folder-name` → Folder containing RAMSES outputs  
- `-n` → Snapshot number(s)  
- `--output-prefix` → Prefix for generated `.vtkhdf` files  
- `--fields` → Comma-separated list of fields (scalars/vectors)  
- `--dry-run` → Run without writing files
- `--output-dir` → Directory to store `.vtkhdf` output files. Defaults to `--base-dir/--folder-name`.
- `--nproc` → Number of CPU cores to use for parallel conversion (default: 1). Falls back to serial if unavailable.

---

### Python API

```python
from chhavi.converter import ChhaviConverter

converter = ChhaviConverter(
input_folder="ramses_outputs/sedov_3d",
output_prefix="sedov_test",
fields=["density", "velocity"],
dry_run=True,
output_directory="./vtk_outputs"
)

converter.process_output(1)
```

### Profile Analysis Workflow

Profile analysis tools are provided in `profiles/` directory:
```python
cd profiles/
python .\compute_osyris_profile.py --base-dir ..\ramses_outputs --folder-name sedov_3d --numbers 4
python .\compute_vtk_profile.py --base-dir ..\vtk_outputs --folder-name sedov_3d --numbers 4
python .\analyzing_profiles.py -n 4

```
This suite:

1. Generates radial density profiles from OSYRIS (RAMSES native) and VTKHDF outputs, saved as CSV files in profile_outputs/ folder:
    osyris_profile_00002.csv [radius, mean, std, min, max]
    vtk_profile_00002.csv [radius, mean, std, min, max]
2. Computes CCC validation metric between CSV profiles (CCC > 0.99 confirms equivalence)
3. Creates publication-quality comparison plots profile_comparison_00002.png with error bands
4. test_profile_analysis.py in tests/ validates snapshot output_00004 with 7 automated tests

---

### Example Script

An example is provided in `examples/example_usage.py`:

```bash
python -m examples.example_usage
```

This script:  
1. Lists available fields in snapshots  
2. Prints dataset info  
3. Performs a dry-run conversion  
4. Displays scalar/vector fields to be written
   
You can specify an output directory in the example usage by setting `OUTPUT_DIR='your/path'`.

---

## Output File Structure

```lua
<output-prefix>_00001.vtkhdf
<output-prefix>_00002.vtkhdf
...
```

---

## Tests

Run the test suite with:

```bash
pytest tests/
```

---

## Repository Structure

```
Chhavi/
├── chhavi/          # Core Python package
│ ├── __init__.py
│ ├── cli.py
│ ├── converter.py
│ └── parallel.py
│
├── tests/ # Unit tests
│ ├── __init__.py
│ ├── test_cli.py
│ ├── test_converter.py
│ ├── test_import.py
│ ├── test_parallel.py
│ ├── test_parser.py
│ └── test_profile_analysis.py
│
├── examples/         # Example usage scripts
│ └── example_usage.py
│
├── profiles/        #  Profile analysis suite
│ ├── compute_osyris_profile.py
│ ├── compute_vtk_profile.py
│ ├── analyzing_profiles.py
│
├── papers/           # JOSS submission papers
│ ├── paper.md
│ └── paper.bib
│
├── ramses_outputs/   # Sample real RAMSES outputs
│ └── sedov_3d/
│ ├── output_00001/
│ ├── output_00002/
│ ├── output_00003/
│ ├── output_00004/
│ └── output_00005/
│
├── LICENSE
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Notes & Best Practices 

- Default fields if `--fields` is not specified: `density`, `pressure`, `velocity`
- Profile validation confirms **Osyris** and **VTKHDF** produce equivalent results
- Verbose mode `--verbose` provides step-by-step information, including the number of cells retained per level. 
- Parallel execution automatically uses specified CPU cores (`--nproc`); falls back to serial execution if needed  
- Output directory is auto-created if it does not exist — no manual setup required  
- If no cells survive filtering or fields are missing, the output file is skipped, with warnings logged  

---

## License

This project is licensed under the terms of the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

## Authors

- **[Hemangi C. Varkal](https://github.com/HemangiVarkal)** — Developer
- **Shubhankar R. Gharote** — Space Applications Centre (SAC), ISRO  
- **Dr. Munn Vinayak Shukla** — Space Applications Centre (SAC), ISRO
- **Dr. Mehul Pandya** - Space Applications Centre (SAC), ISRO

---

## Acknowledgements

- This work was carried out at the **Space Applications Centre (SAC), Indian Space Research Organisation (ISRO), Ahmedabad, India**.  
- The author expresses sincere appreciation to **Dr. Rashmi Sharma (DD, EPSA)** for continuous encouragement and institutional support.  
- Special thanks are due to **Dr. Mehul Pandya (Group Director, SESG/EPSA)**, **Dr. Munn Vinayak Shukla (Head, SSD/SESG/EPSA)** and **Shubhankar R. Gharote (Scientist, SSD/SESG/EPSA)** for their invaluable guidance, technical insights, and collaboration throughout the development of this work.  
- Computations were performed using the **SAGAR High Performance Computing (HPC) Facility** of SAC.  
- Implementation follows **ParaView VTKHDF OverlappingAMR** conventions.  

---

## Citation

If you use **Chhavi** in your work, please cite it as:

> Varkal, H. (2026). *Chhavi: A Python tool for converting RAMSES outputs to VTKHDF*.  
> GitHub repository: [https://github.com/HemangiVarkal/Chhavi](https://github.com/HemangiVarkal/Chhavi)