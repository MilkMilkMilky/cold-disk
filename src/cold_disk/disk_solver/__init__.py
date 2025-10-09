"""Cold Disk Solver Subpackage.

This subpackage provides numerical methods and utilities for solving accretion
disk models in the cold disk modeling framework. It includes both standard
(Shakura-Sunyaev) and slim disk solvers, along with parameter structures and
computational tools.

Public API
----------
DiskParams : dataclass
    Dataclass representing a single set of adjustable parameters for disk
    simulations. Stores scalar values directly used by disk solvers.
cgs_consts : CGSConsts
    Read-only singleton containing CGS-unit physical constants (cm, g, s, erg, etc.)
    used throughout the disk computation framework.
DiskTools : class
    Collection of static utility methods for accretion disk calculations,
    including unit conversions, black hole mass calculations, and general
    disk-related coefficients.
StandardDisk : class
    Collection of static methods for standard (Shakura-Sunyaev) accretion
    disk calculations. Assumes thin, optically thick disk with Keplerian
    angular velocity and includes both gas and radiation pressure.
SlimDisk : class
    Collection of static methods for slim accretion disk calculations.
    Extends the standard model with advective cooling, radial pressure
    gradients, and transonic flow near the inner boundary.

Notes
-----
- All solver classes (`DiskTools`, `StandardDisk`, `SlimDisk`) contain only
  static methods and are stateless; no instances are required.
- `DiskParams` instances store scalar parameter values for single model
  evaluations within the parameter space.
- `cgs_consts` provides all physical constants in CGS units for consistency
  across the computation framework.
- The slim disk solver uses a shooting method to find transonic solutions
  and requires fixed numerical tolerances for stability.
- All methods expect `DiskParams` objects as input for disk parameter data.

Examples
--------
>>> from cold_disk import DiskParams, StandardDisk, SlimDisk, cgs_consts
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
>>> # Physical constants
>>> cgs_consts.cgs_c
29979245800.0

"""
from cold_disk.disk_solver.parameter_init import DiskParams, cgs_consts
from cold_disk.disk_solver.solve_slim import SlimDisk
from cold_disk.disk_solver.solve_standard import StandardDisk
from cold_disk.disk_solver.solve_tools import DiskTools

__all__ = ["DiskParams", "DiskTools", "SlimDisk", "StandardDisk", "cgs_consts"]
