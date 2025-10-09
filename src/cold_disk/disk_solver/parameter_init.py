"""Module `cold_disk.disk_solver.parameter_init`.

Provides the core parameter structures and CGS constants for accretion disk modeling.

Public API:
------------
- `DiskParams`: Dataclass representing a single set of adjustable parameters
  for disk simulations.
- `cgs_consts`: Module-level singleton containing CGS-unit physical constants.

Notes:
-----
- `DiskParams` instances store **scalar values** directly used by disk solvers.
- `cgs_consts` contains standard constants in CGS units (cm, g, s, erg, etc.).
- Internal arrays and temporary constants used for construction are not part of
  the public API.
- Typical usage is for higher-level disk computation routines, not direct
  manual construction.

Example:
-------
>>> from cold_disk import DiskParams, cgs_consts
>>> dp = DiskParams(
...     alpha_viscosity=0.1,
...     dimless_accrate=1.0,
...     dimless_bhmass=1e8,
...     gas_index=3,
...     wind_index=0.0,
...     dimless_radius_in=3.0,
...     dimless_radius_out=10000.0,
... )
>>> dp.dimless_bhmass
1e8
>>> cgs_consts.cgs_c
2.99792458e10

"""
from dataclasses import dataclass

from cold_disk.parameters import consts

__all__ = ["DiskParams", "cgs_consts"]

cgs_c = consts.vacuum_light_speed * 100
cgs_h = consts.planck_constant * 1e7
cgs_kb = consts.boltzmann_constant * 1e7
cgs_gra = consts.gravitational_constant * 1e3
cgs_rg = consts.avogadro_constant * cgs_kb
cgs_mp = consts.proton_mass * 1e3
cgs_mh = consts.atomic_mass_constant * 1.008 * 1e3
cgs_sb = consts.stefan_boltzmann_constant * 1e3
cgs_amm = 0.617
cgs_msun = consts.solar_mass * 1e3
cgs_a = 4 * cgs_sb / cgs_c
cgs_kes = 0.34
cgs_kra = 6.4e22


@dataclass(frozen=True)
class CGSConsts:
    """Container for physical constants in CGS units.

    Provides values for fundamental constants (speed of light, Planck constant,
    gravitational constant, etc.) and standard astrophysical constants (solar mass,
    Stefan-Boltzmann constant, etc.) in CGS units (cm, g, s, erg, etc.).

    """

    cgs_c: float
    """Vacuum light speed (CGS, cm)."""

    cgs_h: float
    """Planck constant (CGS, erg.s)."""

    cgs_kb: float
    """Boltzmann constant (CGS, erg/K)."""

    cgs_gra: float
    """Gravitational constant (CGS, cm^3·g^-1·s^-2)."""

    cgs_rg: float
    """Molar gas constant (CGS, erg/(mol·K))."""

    cgs_mp: float
    """Proton mass (CGS, g)."""

    cgs_mh: float
    """Hydrogen atomic mass (CGS, g)."""

    cgs_sb: float
    """Stefan-Boltzmann constant (CGS, erg·s^-1·cm^-2·K^-4)."""

    cgs_amm: float
    """Average molar mass (CGS, g·mol^-1)."""

    cgs_msun: float
    """Solar mass (CGS, g)."""

    cgs_a: float
    """Radiation constant.  (CGS, erg·cm^-3·K^-4)."""

    cgs_kes: float
    """Electron scattering opacity. (CGS, cm^2·g^-1)."""

    cgs_kra: float
    """Kramers' opacity coefficient. (CGS, cm^5.5·g^-2·K^3.5)."""


cgs_consts = CGSConsts(
    cgs_c=cgs_c,
    cgs_h=cgs_h,
    cgs_kb=cgs_kb,
    cgs_gra=cgs_gra,
    cgs_rg=cgs_rg,
    cgs_mp=cgs_mp,
    cgs_mh=cgs_mh,
    cgs_sb=cgs_sb,
    cgs_amm=cgs_amm,
    cgs_msun=cgs_msun,
    cgs_a=cgs_a,
    cgs_kes=cgs_kes,
    cgs_kra=cgs_kra,
)


@dataclass
class DiskParams:
    """Container for a single physical parameter set used in disk-model computations.

    This dataclass defines one specific combination of adjustable parameters
    describing a single instance of a accretion-disk model.
    Each field stores a **scalar value** that will be used directly in numerical
    solvers (e.g. the ODE integrator or the global slim-disk solver).

    It serves as the per-task parameter structure corresponding to one row of
    the parameter space generated from :class:`_AdjustableParams`.

    Attributes
    ----------
    alpha_viscosity : float
        Dimensionless viscosity parameter.
    dimless_accrate : float
        Dimensionless accretion rate, typically normalized by the Eddington rate.
    dimless_bhmass : float
        Dimensionless black-hole mass (e.g., in solar-mass units).
    gas_index : float
        Polytropic gas index.
    wind_index : float
        Disk-wind index parameter describing the outflow strength or scaling.
    dimless_radius_in : float
        Inner boundary radius of the computational domain (dimensionless).
    dimless_radius_out : float
        Outer boundary radius of the computational domain (dimensionless).

    Notes
    -----
    - This class represents a *single model evaluation point* within the
      multi-dimensional parameter space defined by :class:`_AdjustableParams`.
    - Instances of this class are typically constructed automatically by higher-level
      routines (e.g. disk_driver) rather than manually.
    - All quantities are assumed to be expressed in **dimensionless** or **CGS-based**
      units consistent with the solver conventions.

    Examples
    --------
    >>> from cold_disk.disk_solver.parameter_init import DiskParams
    >>> dp = DiskParams(
    ...     alpha_viscosity=0.1,
    ...     dimless_accrate=1.0,
    ...     dimless_bhmass=1e8,
    ...     gas_index=1.4,
    ...     wind_index=0.0,
    ...     dimless_radius_in=3.0,
    ...     dimless_radius_out=100.0,
    ... )
    >>> dp.alpha_viscosity
    0.1

    """

    alpha_viscosity: float
    dimless_accrate: float
    dimless_bhmass: float
    gas_index: float
    wind_index: float
    dimless_radius_in: float
    dimless_radius_out: float
