# cold-disk

---

Numerical simulation of cold accretion disks


<div align="center">

[![License](https://img.shields.io/github/license/MilkMilkMilky/cold-disk)](https://github.com/MilkMilkMilky/cold-disk/blob/main/LICENSE)
[![Commit activity](https://img.shields.io/github/commit-activity/m/MilkMilkMilky/cold-disk)](https://github.com/MilkMilkMilky/cold-disk/commits/main)
[![python versions](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://github.com/MilkMilkMilky/cold-disk)
</div>


<div align="center">
    <a href="https://github.com/MilkMilkMilky/cold-disk">Github</a>
    ·
    <a href="https://MilkMilkMilky.github.io/cold-disk">Documentation</a>
</div>


## Overview

**Cold Disk** is a comprehensive Python package for numerical simulation of cold accretion disks around black holes. The project provides state-of-the-art computational tools for modeling both standard (Shakura-Sunyaev) and slim accretion disk models, enabling researchers to study the physics of matter accretion in astrophysical systems.

### Background

Accretion disks are fundamental structures in astrophysics, powering some of the most energetic phenomena in the universe, including active galactic nuclei (AGN), X-ray binaries, and gamma-ray bursts. Understanding the structure and evolution of these disks requires sophisticated numerical models that can capture the complex interplay between gravity, hydrodynamics, radiation, and magnetic fields.

### Abstract

This package implements two complementary disk models:

- **Standard Disk Model**: Based on the classical Shakura-Sunyaev prescription, assuming thin, optically thick disks with Keplerian angular velocity and including both gas and radiation pressure contributions.

- **Slim Disk Model**: An advanced generalization that accounts for advective cooling, radial pressure gradients, and transonic flow near the inner boundary, providing more accurate descriptions for high accretion rate systems.

### Goals

The primary goals of this project are to:

1. **Provide robust numerical solvers** for both standard and slim accretion disk models with high accuracy and stability
2. **Enable parameter space exploration** through efficient batch computation workflows and HDF5-based data management
3. **Support research applications** in black hole physics, AGN studies, and high-energy astrophysics
4. **Offer a clean, well-documented API** that facilitates both educational use and advanced research applications
5. **Ensure computational efficiency** through optimized algorithms and proper error handling for large-scale parameter studies

## Features

### Core Disk Models

- **Standard Disk Solver**: Implements the classical Shakura-Sunyaev model with Keplerian angular velocity, gas and radiation pressure, and Rosseland mean opacity
- **Slim Disk Solver**: Advanced model with advective cooling, radial pressure gradients, transonic flow, and shooting method for global solutions
- **Dual Model Support**: Seamless switching between standard and slim disk models for different physical regimes

### Numerical Methods

- **Robust ODE Integration**: Uses scipy's adaptive integrators with fixed tolerances for numerical stability
- **Shooting Method**: Sophisticated root-finding algorithms for transonic solutions in slim disk models
- **Error Handling**: Comprehensive error detection and logging for failed computations
- **Convergence Control**: Automatic tolerance adjustment and convergence monitoring

### Parameter Space Exploration

- **Batch Computation**: Efficient processing of large parameter spaces with HDF5-based storage
- **Multiple Dispatch Modes**: Support for 'parasweep', 'pairscan', and 'fullfactorial' parameter generation
- **Resumable Computations**: Task completion tracking enables restarting interrupted batch jobs


### Data Management

- **HDF5 Storage**: Efficient, compressed storage of parameter spaces and computation results
- **Structured Data**: Organized datasets with metadata, task states, and result groups
- **Date-Based Organization**: Automatic file naming with timestamps for result tracking
- **Atomic Operations**: Safe file operations with proper error handling and rollback capabilities

### Physical Accuracy

- **CGS Units**: All computations in consistent CGS units for astrophysical applications
- **Physical Constants**: Comprehensive set of fundamental and astrophysical constants

### API Design

- **Static Methods**: Stateless, thread-safe design with no instance requirements
- **Type Hints**: Full type annotation support for better IDE integration and code reliability
- **Dataclass Parameters**: Clean parameter structures with validation and documentation
- **Modular Architecture**: Well-separated concerns with clear interfaces between components

### Research Tools

- **Spectral Energy Distribution**: Built-in SED computation for radiation output analysis of slim disk
- **Luminosity Calculations**: Total and bolometric luminosity computation with proper integration of slim disk
- **Disk Structure Analysis**: Complete radial profiles of density, temperature, pressure, velocity, etc.
- **Wind Effects**: Configurable power-law slim disk wind prescriptions for outflow modeling

## Installation

To install the project, run the following command:

```bash
python -m pip install git+https://github.com/MilkMilkMilky/cold-disk.git
```

Or install from local:

```bash
git clone https://github.com/MilkMilkMilky/cold-disk.git
cd cold-disk
python -m pip install .
```

## Usage

### Basic Single Disk Computation

The simplest way to use Cold Disk is to compute a single disk solution for a specific set of parameters:

```python
from cold_disk import DiskParams, StandardDisk, SlimDisk

# Define disk parameters
par = DiskParams(
    alpha_viscosity=0.1,        # Viscosity parameter
    dimless_accrate=1.0,        # Accretion rate (Eddington units)
    dimless_bhmass=1e8,         # Black hole mass (solar masses)
    gas_index=3.0,              # Polytropic gas index
    wind_index=0.0,             # Wind index (0 = no wind)
    dimless_radius_in=3.0,      # Inner radius (Schwarzschild units)
    dimless_radius_out=10000.0, # Outer radius (Schwarzschild units)
)

# Compute standard disk solution at a specific radius
std_result = StandardDisk.get_standard_solve_result(
    par=par, 
    dimless_radius=10.0
)
print(f"Temperature: {std_result['temperature']:.2e} K")
print(f"Density: {std_result['density']:.2e} g/cm³")

# Compute slim disk solution (global profile)
slim_result, info = SlimDisk.slim_disk_odeint_solver(par=par)
print(f"Slim disk converged: {info['shoot_succcess']}")
print(f"Radial profile length: {len(slim_result['radius'])} points")
```

### Parameter Space Exploration

For research applications requiring systematic parameter studies:

```python
from cold_disk import ParaspaceGeneratorTools, ResultGeneratorTools
from pathlib import Path

# Load default parameter ranges
adjparams = ParaspaceGeneratorTools.load_adjparams_default()

# Create HDF5 file for results
h5path = ParaspaceGeneratorTools.load_disk_datafiles(
    data_date="20250101", 
    disktype="slim"
)

# Initialize parameter space (full factorial design)
ParaspaceGeneratorTools.paramspace_init(
    hdf5_file_path=h5path,
    adjparams_obj=adjparams,
    dispatch_mode="fullfactorial"
)

# Compute all disk solutions
ResultGeneratorTools.slimdisk_normalresult_generator(hdf5_file_path=h5path)

# Compute radiation outputs
ResultGeneratorTools.slimdisk_radiationresult_generator(hdf5_file_path=h5path)
```

### Custom Parameter Ranges

Define your own parameter space for targeted studies:

```python
import numpy as np
from cold_disk.disk_driver.paraspace_generator import AdjustableParams

# Custom parameter ranges
custom_params = AdjustableParams(
    alpha_viscosity=np.array([0.01, 0.1, 0.3]),
    dimless_accrate=np.array([0.1, 1.0, 10.0]),
    dimless_bhmass=np.array([1e6, 1e7, 1e8]),
    gas_index=np.array([1.4, 3.0]),
    wind_index=np.array([0.0, 0.5]),
    dimless_radius_in=np.array([3.0]),
    dimless_radius_out=np.array([1000.0])
)

# Generate parameter space
param_space = ParaspaceGeneratorTools.adjparams_dispatcher(
    adjparams_obj=custom_params,
    dispatch_mode="fullfactorial"
)
print(f"Total parameter combinations: {len(param_space)}")
```

### Accessing Physical Constants

The package provides comprehensive physical constants for astrophysical calculations:

```python
from cold_disk import cgs_consts, consts

# CGS constants for disk calculations
print(f"Speed of light: {cgs_consts.cgs_c:.2e} cm/s")
print(f"Solar mass: {cgs_consts.cgs_msun:.2e} g")
print(f"Stefan-Boltzmann: {cgs_consts.cgs_sb:.2e} erg/s/cm²/K⁴")

# SI constants
print(f"Gravitational constant: {consts.gravitational_constant:.2e} m³/kg/s²")
print(f"Planck constant: {consts.planck_constant:.2e} J·s")
```

### Analyzing Results

Extract and analyze computation results from HDF5 files:

```python
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load results from HDF5 file
with h5py.File("slimdiskdata_20250101.h5", "r") as f:
    # Get parameter space
    paramspace = f["adjparamspace"][:]
    
    # Get results for first task
    task_group = f["task_0"]
    result = task_group["resultnormal"][:]
    
    # Extract radial profiles
    radius = result["radius"]
    temperature = result["temperature"]
    density = result["density"]

# Plot temperature profile
plt.figure(figsize=(8, 6))
plt.loglog(radius, temperature)
plt.xlabel("Radius (cm)")
plt.ylabel("Temperature (K)")
plt.title("Disk Temperature Profile")
plt.grid(True)
plt.show()
```

### Advanced Usage: Custom Disk Models

For advanced users, the package allows direct access to individual disk structure calculations:

```python
from cold_disk import DiskParams, StandardDisk, SlimDisk

par = DiskParams(
    alpha_viscosity=0.1,
    dimless_accrate=1.0,
    dimless_bhmass=1e8,
    gas_index=3.0,
    wind_index=0.0,
    dimless_radius_in=3.0,
    dimless_radius_out=10000.0,
)

# Individual physical quantities
radius = 10.0
ang_vel = StandardDisk.get_standard_angvel(par=par, dimless_radius=radius)
pressure = StandardDisk.get_standard_pressure(par=par, dimless_radius=radius)
opacity = StandardDisk.get_standard_averopacity(par=par, dimless_radius=radius)

print(f"Angular velocity: {ang_vel:.2e} rad/s")
print(f"Pressure: {pressure:.2e} dyn/cm²")
print(f"Opacity: {opacity:.2e} cm²/g")
```

### Error Handling and Logging

The package provides comprehensive error handling for robust computations:

```python
from cold_disk import ResultGeneratorTools
from pathlib import Path

# Check computation status
h5path = Path("slimdiskdata_20250101.h5")
try:
    ResultGeneratorTools.slimdisk_normalresult_generator(hdf5_file_path=h5path)
    print("All computations completed successfully")
except Exception as e:
    print(f"Computation failed: {e}")
    
# Check log files for detailed error information
log_file = h5path.parent / f"{h5path.stem}_solver.log"
if log_file.exists():
    with open(log_file, "r") as f:
        print("Solver log:", f.read())
```

## License

This project is licensed under the GNU LGPLv3.
Check the [LICENSE](LICENSE) file for more details.

## Contributing

Please follow the [Contributing Guide](https://github.com/MilkMilkMilky/cold-disk/blob/main/CONTRIBUTING.md) to contribute to this project.

## Contact

For support or inquiries, please contact:

- Email: milkcowmilky@gmail.com
