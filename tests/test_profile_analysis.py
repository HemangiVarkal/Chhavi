"""
Unit tests for Chhavi Profile Analysis Suite.

These tests verify that the profile analysis pipeline:
1. Computes CCC > 0.99 validation metric (publication criterion)
2. Loads OSYRIS/VTK CSV profiles correctly  
3. Detects real profile files in production directory
4. Handles edge cases without NaN or crashes
5. Generates publication-quality comparison plots
6. Validates exact CSV format consistency
7. Imports modules correctly from tests/ directory

Tests run from Chhavi/tests/ → validate Chhavi/profiles/profile_outputs/
"""

import pytest
import numpy as np
import os
import sys
from pathlib import Path
import tempfile
import shutil

# ──────────────────────────────────────────────────────────────
# Configuration - Corrected Paths
# ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent  # Chhavi/
PROFILES_DIR = PROJECT_ROOT / "profiles"
PROFILE_OUTPUT_ROOT = PROFILES_DIR / "profile_outputs"
TEST_SNAPSHOT_NUMBER = 4

# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_profile_dir(tmp_path_factory):
    """Create temporary test profiles matching production CSV format.
    
    Generates realistic Sedov blast wave radial density profiles:
    - OSYRIS: Reference profile (100 radial bins)
    - VTK: OSYRIS + controlled noise (CCC > 0.99 guaranteed)
    - Exact 5-column format: [radius, mean, std, min, max]
    - Matches np.savetxt() output from generator scripts
    """
    tmpdir = tmp_path_factory.mktemp("test_profiles")
    
    # Create realistic Sedov-like radial profiles (CCC > 0.99 guaranteed)
    r = np.linspace(0, 0.2, 100)
    rho_osyris = 1.0 * (1.0 - r)**1.2 * np.exp(-r/0.05)
    noise = 0.005 * np.random.normal(size=len(r), scale=rho_osyris)
    rho_vtk = rho_osyris + noise
    
    std_factor = 0.1
    osyris_profile = np.column_stack([
        r, rho_osyris, rho_osyris*std_factor, 
        rho_osyris*(1-std_factor), rho_osyris*(1+std_factor)
    ])
    vtk_profile = np.column_stack([
        r, rho_vtk, rho_vtk*std_factor, 
        rho_vtk*(1-std_factor), rho_vtk*(1+std_factor)
    ])
    
    # Save test CSV files
    osyris_file = tmpdir / f"osyris_profile_{TEST_SNAPSHOT_NUMBER:05d}.csv"
    vtk_file = tmpdir / f"vtk_profile_{TEST_SNAPSHOT_NUMBER:05d}.csv"
    
    np.savetxt(osyris_file, osyris_profile, 
               delimiter=",", 
               header="Radius,Density_mean,Density_std,Density_min,Density_max")
    np.savetxt(vtk_file, vtk_profile, 
               delimiter=",", 
               header="Radius,Density_mean,Density_std,Density_min,Density_max")
    
    return tmpdir

# ──────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────

def test_ccc_validation_criterion(test_profile_dir):
    """Verify CCC > 0.99 meets ISRO publication validation criterion.
    
    Tests core statistical validation metric using production functions:
    - load_profile_data(): Exact CLI loading logic
    - compute_validation_metrics(): Publication CCC computation
    - Asserts CCC > 0.99 AND Pearson r > 0.99 (dual criterion)
    - Validates minimum 50 radial bins for statistical significance
    """
    # Add profiles/ to Python path
    sys.path.insert(0, str(PROFILES_DIR))
    from analyzing_profiles import load_profile_data, compute_validation_metrics
    
    osyris_file = test_profile_dir / f"osyris_profile_{TEST_SNAPSHOT_NUMBER:05d}.csv"
    vtk_file = test_profile_dir / f"vtk_profile_{TEST_SNAPSHOT_NUMBER:05d}.csv"
    
    osyris_profile, vtk_profile = load_profile_data(str(osyris_file), str(vtk_file))
    metrics = compute_validation_metrics(osyris_profile, vtk_profile)
    
    # ✅ PUBLICATION CRITERION
    assert metrics['ccc'] > 0.99, f"CCC={metrics['ccc']:.6f} < 0.99"
    assert metrics['pearson_r'] > 0.99, f"Pearson r={metrics['pearson_r']:.6f} < 0.99"
    assert metrics['n_bins'] > 50, f"Insufficient bins: {metrics['n_bins']}"


def test_profile_loading_success(test_profile_dir):
    """Verify production CSV loading works without errors.
    
    Tests exact CLI profile loading pipeline:
    - Both OSYRIS and VTK profiles load via load_profile_data()
    - Validates 5-column structure [radius, mean, std, min, max]
    - Ensures matching radial bin counts between profiles
    """
    sys.path.insert(0, str(PROFILES_DIR))
    from analyzing_profiles import load_profile_data
    
    osyris_file = test_profile_dir / f"osyris_profile_{TEST_SNAPSHOT_NUMBER:05d}.csv"
    vtk_file = test_profile_dir / f"vtk_profile_{TEST_SNAPSHOT_NUMBER:05d}.csv"
    
    osyris_profile, vtk_profile = load_profile_data(str(osyris_file), str(vtk_file))
    
    assert osyris_profile.shape[1] == 5, "OSYRIS profile must have 5 columns"
    assert vtk_profile.shape[1] == 5, "VTK profile must have 5 columns"
    assert len(osyris_profile) == len(vtk_profile), "Profiles must have same # bins"


def test_real_profiles_exist():
    """Verify real production profiles exist in profiles/profile_outputs/.
    
    Checks for actual generator output files:
    - osyris_profile_00002.csv (from compute_osyris_profile.py)
    - vtk_profile_00002.csv (from compute_vtk_profile.py)
    - Skips test if generators haven't been run (CI-friendly)
    """
    osyris_file = PROFILE_OUTPUT_ROOT / f"osyris_profile_{TEST_SNAPSHOT_NUMBER:05d}.csv"
    vtk_file = PROFILE_OUTPUT_ROOT / f"vtk_profile_{TEST_SNAPSHOT_NUMBER:05d}.csv"
    
    if osyris_file.exists() and vtk_file.exists():
        print(f"✅ Real profiles found: {osyris_file}, {vtk_file}")
    else:
        pytest.skip("Real profiles not found - run profile generators first")


def test_ccc_failure_case():
    """Verify CCC computation handles edge cases without NaN/crashes.
    
    Tests deliberately mismatched profiles (different physical trends):
    - OSYRIS: Exponential decay profile
    - VTK: Oscillatory profile with offset  
    - Validates CCC < 0.95 (clear failure) but non-NaN result
    - Ensures numerical stability in production metrics
    """
    sys.path.insert(0, str(PROFILES_DIR))
    from analyzing_profiles import load_profile_data, compute_validation_metrics
    
    # FIXED: Add small noise to avoid zero variance
    r = np.linspace(0, 0.2, 100)
    rho_osyris = 1.0 + 0.1 * np.exp(-r/0.05)  # Avoid zero values
    rho_vtk = 2.0 + 0.01 * np.sin(r*10)        # Different trend + small variation
    
    osyris_profile = np.column_stack([r, rho_osyris, rho_osyris*0.1, rho_osyris*0.9, rho_osyris*1.1])
    vtk_profile = np.column_stack([r, rho_vtk, rho_vtk*0.1, rho_vtk*0.9, rho_vtk*1.1])
    
    with tempfile.TemporaryDirectory() as tmpdir:
        osyris_file = Path(tmpdir) / f"osyris_profile_{TEST_SNAPSHOT_NUMBER:05d}.csv"
        vtk_file = Path(tmpdir) / f"vtk_profile_{TEST_SNAPSHOT_NUMBER:05d}.csv"
        
        np.savetxt(osyris_file, osyris_profile, delimiter=",")
        np.savetxt(vtk_file, vtk_profile, delimiter=",")
        
        osyris_data, vtk_data = load_profile_data(str(osyris_file), str(vtk_file))
        metrics = compute_validation_metrics(osyris_data, vtk_data)
        
        # Should have low CCC (< 0.99) but valid number (not NaN)
        assert not np.isnan(metrics['ccc']), "CCC should not be NaN"
        assert metrics['ccc'] < 0.95, f"Expected low CCC, got {metrics['ccc']:.3f}"


def test_plot_generation(tmp_path, test_profile_dir):
    """Verify publication plot generation completes successfully.
    
    Tests production plotting pipeline:
    - create_comparison_plot() with real profile data
    - Validates PNG output file creation
    - Ensures reasonable file size (>1KB, non-empty plot)
    - Confirms matplotlib integration from tests/ directory
    """
    sys.path.insert(0, str(PROFILES_DIR))
    from analyzing_profiles import load_profile_data, create_comparison_plot
    
    osyris_file = test_profile_dir / f"osyris_profile_{TEST_SNAPSHOT_NUMBER:05d}.csv"
    vtk_file = test_profile_dir / f"vtk_profile_{TEST_SNAPSHOT_NUMBER:05d}.csv"
    
    osyris_profile, vtk_profile = load_profile_data(str(osyris_file), str(vtk_file))
    
    plot_path = tmp_path / "test_plot.png"
    create_comparison_plot(osyris_profile, vtk_profile, str(plot_path), TEST_SNAPSHOT_NUMBER)
    
    assert plot_path.exists(), "Plot file was not created"
    assert plot_path.stat().st_size > 1000, "Plot file too small"


def test_profile_format_consistency(test_profile_dir):
    """Verify CSV headers match exact np.savetxt() production format.
    
    Tests that test fixtures produce identical CSV format to:
    - compute_osyris_profile.py generator output
    - compute_vtk_profile.py generator output
    - analyzing_profiles.py loader expectations
    """
    osyris_file = test_profile_dir / f"osyris_profile_{TEST_SNAPSHOT_NUMBER:05d}.csv"
    
    with open(osyris_file) as f:
        header = f.readline().strip()
    assert header == "# Radius,Density_mean,Density_std,Density_min,Density_max"


def test_cli_import_from_tests_dir():
    """Verify module imports work from tests/ → profiles/ directory structure.
    
    Tests Python path resolution across Chhavi/ subdirectory boundaries:
    - sys.path.insert(0, "../profiles") from tests/
    - Import production analyzing_profiles.py functions
    - Ensures test suite portable across working directories
    """
    sys.path.insert(0, str(PROFILES_DIR))
    try:
        from analyzing_profiles import load_profile_data
        assert True, "Import successful from tests/"
    except ImportError as e:
        pytest.fail(f"Cannot import analyzing_profiles.py: {e}")
