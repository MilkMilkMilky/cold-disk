"""Cold Disk Parameters Subpackage.

This subpackage provides access to physical constants and
default accretion disk parameters used in the cold disk modeling
framework. It exposes module-level singletons for convenient
read-only access.

Public API
----------
consts : _ConstsWrapper
    Read-only singleton combining fundamental constants (`consts_fund`)
    and application constants (`consts_app`). Provides all relevant
    physical constants with SI units.
model_params : ModelParams
    Read-only singleton providing default arrays of adjustable parameters
    for accretion disk modeling, including viscosity, accretion rate,
    black hole mass, gas index, wind index, and inner/outer radii.

Notes
-----
- All constants in `consts` are read-only; attempts to modify them
  will raise `AttributeError`.
- `consts` merges both fundamental physical constants (from
  `phy_consts_fund`) and derived/application constants
  (from `phy_consts_app`) for convenient access.
- `_ConstsWrapper` is an internal class and should not be used directly.
- `model_params` contains typical example values and can be used
  as a template for generating parameter grids for simulations.

Examples
--------
>>> from cold_disk import consts, model_params
>>> consts.vacuum_light_speed
299792458
>>> model_params.dimless_accrate
array([0.1, 1.0, 10.0])

"""
from typing import TYPE_CHECKING, Any, NoReturn, cast

from cold_disk.parameters import phy_consts_app, phy_consts_fund
from cold_disk.parameters.model_parameters import model_params

__all__ = ["consts", "model_params"]


class _ConstsWrapper:
    """Read-only wrapper combining fundamental and application constants."""

    __slots__ = ("_app", "_fund")

    def __init__(self, fund, app) -> None:
        object.__setattr__(self, "_fund", fund)
        object.__setattr__(self, "_app", app)

    def __getattr__(self, name) -> Any:
        if hasattr(self._fund, name):
            return getattr(self._fund, name)
        if hasattr(self._app, name):
            return getattr(self._app, name)
        raise AttributeError(f"No such constant: {name}")

    def __setattr__(self, key, value) -> NoReturn:
        raise AttributeError("Constants are read-only")

    def __delattr__(self, key) -> NoReturn:
        raise AttributeError("Constants are read-only")


consts = _ConstsWrapper(phy_consts_fund.consts_fund, phy_consts_app.consts_app)

if TYPE_CHECKING:

    class ConstsType:
        # Fundamental constants
        caesium_frequency: int
        vacuum_light_speed: int
        planck_constant: float
        elementary_charge: float
        boltzmann_constant: float
        avogadro_constant: float
        luminous_efficacy_kcd: int
        gravitational_constant: float
        vacuum_electric_permittivity: float
        vacuum_magnetic_permeability: float
        atomic_mass_constant: float
        proton_mass: float
        neutron_mass: float
        electron_mass: float
        # Application constants
        reduced_planck_constant: float
        fine_structure_constant: float
        molar_gas_constant: float
        rydberg_constant: float
        stefan_boltzmann_constant: float
        classical_electron_radius: float
        bohr_radius: float
        thomson_cross_section: float
        electron_compton_wavelength: float
        solar_mass: float
        earth_mass: float
        standard_gravitational_accel: float
        standard_atmosphere: float
        astronomical_unit: float
        julian_year_seconds: float
        light_year: float
        parsec: float

    consts = cast("ConstsType", consts)
