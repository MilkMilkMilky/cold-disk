from typing import TYPE_CHECKING, Any, NoReturn, cast

from parameters import phy_consts_app, phy_consts_fund
from parameters.model_parameters import model_params

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
