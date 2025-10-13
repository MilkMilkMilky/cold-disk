"""Cold Disk: Numerical Simulation of Accretion Disks.

A comprehensive Python package for numerical simulation of cold accretion disks,
providing tools for both standard (Shakura-Sunyaev) and slim disk models with
batch computation capabilities and parameter space exploration.

The package is organized into three main subpackages:

- `parameters`: Physical constants and default model parameters
- `disk_solver`: Numerical methods for solving disk structure equations
- `disk_driver`: Parameter space generation and batch computation workflows

Public API
----------
AdjustableParams : dataclass
    Dataclass containing arrays of adjustable model parameters used to build
    parameter spaces for batch computations.
DiskParams : dataclass
    Dataclass representing a single set of adjustable parameters for disk
    simulations. Stores scalar values directly used by disk solvers.
DiskTools : class
    Collection of static utility methods for accretion disk calculations,
    including unit conversions, black hole mass calculations, and general
    disk-related coefficients.
ParaspaceGeneratorTools : class
    Collection of static methods for parameter space generation and HDF5 file
    management. Handles creation of parameter combinations, file initialization,
    and parameter space dispatching for batch computations.
ResultGeneratorTools : class
    Collection of static methods for computing and storing accretion disk
    solutions in HDF5 files. Manages batch computation workflows, result
    storage, and completion tracking for parameter space exploration.
SlimDisk : class
    Collection of static methods for slim accretion disk calculations.
    Extends the standard model with advective cooling, radial pressure
    gradients, and transonic flow near the inner boundary.
StandardDisk : class
    Collection of static methods for standard (Shakura-Sunyaev) accretion
    disk calculations. Assumes thin, optically thick disk with Keplerian
    angular velocity and includes both gas and radiation pressure.
cgs_consts : CGSConsts
    Read-only singleton containing CGS-unit physical constants (cm, g, s, erg, etc.)
    used throughout the disk computation framework.
consts : _ConstsWrapper
    Read-only singleton combining fundamental constants and application constants.
    Provides all relevant physical constants with SI units.
model_params : ModelParams
    Read-only singleton providing default arrays of adjustable parameters
    for accretion disk modeling, including viscosity, accretion rate,
    black hole mass, gas index, wind index, and inner/outer radii.

Notes
-----
- All solver classes (`DiskTools`, `StandardDisk`, `SlimDisk`) contain only
  static methods and are stateless; no instances are required.
- `DiskParams` instances store scalar parameter values for single model
  evaluations within the parameter space.
- The slim disk solver uses a shooting method to find transonic solutions
  and requires fixed numerical tolerances for stability.
- HDF5 files are used for storing parameter spaces and computation results
  with proper error handling and resumable computation capabilities.
- All physical quantities are computed in CGS units for consistency.

Examples
--------
>>> from cold_disk import DiskParams, StandardDisk, SlimDisk, ParaspaceGeneratorTools
>>> # Create disk parameters
>>> par = DiskParams(
...     alpha_viscosity=0.1,
...     dimless_accrate=1.0,
...     dimless_bhmass=1e8,
...     gas_index=3,
...     wind_index=0.0,
...     dimless_radius_in=3.0,
...     dimless_radius_out=10000.0,
... )
>>> # Standard disk solution
>>> std_result = StandardDisk.get_standard_solve_result(par=par, dimless_radius=10.0)
>>> # Slim disk solution
>>> slim_result, info = SlimDisk.slim_disk_odeint_solver(par=par)
>>> # Batch computation workflow
>>> adjparams = ParaspaceGeneratorTools.load_adjparams_default()
>>> h5path = ParaspaceGeneratorTools.load_disk_datafiles(disktype="slim")
>>> ParaspaceGeneratorTools.paramspace_init(
...     hdf5_file_path=h5path, adjparams_obj=adjparams, dispatch_mode="fullfactorial"
... )

"""
from cold_disk.disk_driver import AdjustableParams, ParaspaceGeneratorTools, ResultGeneratorTools
from cold_disk.disk_solver import DiskParams, DiskTools, SlimDisk, StandardDisk, cgs_consts
from cold_disk.parameters import consts, model_params

__all__ = [
    "AdjustableParams",
    "DiskParams",
    "DiskTools",
    "ParaspaceGeneratorTools",
    "ResultGeneratorTools",
    "SlimDisk",
    "StandardDisk",
    "cgs_consts",
    "consts",
    "model_params",
]
