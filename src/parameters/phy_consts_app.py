import math
from dataclasses import dataclass

from parameters.phy_consts_fund import consts_fund

# ---------------------
# Derived constants
# ---------------------
reduced_planck_constant = consts_fund.planck_constant / 2 / math.pi
fine_structure_constant = consts_fund.elementary_charge**2 / (
    4 * math.pi * consts_fund.vacuum_electric_permittivity * reduced_planck_constant * consts_fund.vacuum_light_speed
)
molar_gas_constant = consts_fund.avogadro_constant * consts_fund.boltzmann_constant
rydberg_constant = 10973731.568157
stefan_boltzmann_constant = 5.670374419e-8
classical_electron_radius = 2.8179403205e-15
bohr_radius = 5.29177210544e-11
thomson_cross_section = 6.6524587051e-29
electron_compton_wavelength = consts_fund.planck_constant / consts_fund.electron_mass / consts_fund.vacuum_light_speed
# ---------------------
# Application constants
# ---------------------
solar_mass = 1.98847e30
earth_mass = 5.972e24
standard_gravitational_accel = 9.80665
standard_atmosphere = 1.01325e5
astronomical_unit = 1.495978707e11
julian_year_seconds = 365.25 * 24 * 3600
light_year = consts_fund.vacuum_light_speed * julian_year_seconds
parsec = (648000 / math.pi) * astronomical_unit


@dataclass(frozen=True)
class PhyConstsApp:
    reduced_planck_constant: float
    """Reduced Planck constant (SI, J·s)."""

    fine_structure_constant: float
    """Fine-structure constant (dimensionless)."""

    molar_gas_constant: float
    """Molar gas constant (SI, J/(mol·K))."""

    rydberg_constant: float
    """Rydberg constant (SI, m^-1)."""

    stefan_boltzmann_constant: float
    """Stefan-Boltzmann constant (SI, W·m^-2·K^-4)."""

    classical_electron_radius: float
    """Classical_electron_radius (SI, m)."""

    bohr_radius: float
    """Bohr radius (SI, m)."""

    thomson_cross_section: float
    """Thomson cross section (SI, m^2)."""

    electron_compton_wavelength: float
    """Electron Compton wavelength (SI, m)."""

    solar_mass: float
    """Solar mass (SI, kg)."""

    standard_gravitational_accel: float
    """Standard gravitational acceleration (SI, m·s^-2)."""

    standard_atmosphere: float
    """Standard atmosphere (SI, Pa)."""

    earth_mass: float
    """Earth mass (SI, kg)."""

    astronomical_unit: float
    """Astronomical Unit (SI, m)."""

    julian_year_seconds: float
    """Julian year seconds (SI, s)."""

    light_year: float
    """Light year (SI, m)."""

    parsec: float
    """Parsec (SI, m)."""


consts_app = PhyConstsApp(
    reduced_planck_constant=reduced_planck_constant,
    fine_structure_constant=fine_structure_constant,
    molar_gas_constant=molar_gas_constant,
    rydberg_constant=rydberg_constant,
    stefan_boltzmann_constant=stefan_boltzmann_constant,
    classical_electron_radius=classical_electron_radius,
    bohr_radius=bohr_radius,
    thomson_cross_section=thomson_cross_section,
    electron_compton_wavelength=electron_compton_wavelength,
    solar_mass=solar_mass,
    standard_gravitational_accel=standard_gravitational_accel,
    standard_atmosphere=standard_atmosphere,
    earth_mass=earth_mass,
    astronomical_unit=astronomical_unit,
    julian_year_seconds=julian_year_seconds,
    light_year=light_year,
    parsec=parsec,
)

__all__ = ["consts_app"]
