# Stubs for parameters.phy_consts C extension
# This file provides type hints for the exposed `consts` object.

class PhyConstsFund:
    caesium_frequency: int
    """Caesium frequency (SI, Hz)."""

    vacuum_light_speed: int
    """Speed of light in vacuum (SI, m/s)."""

    planck_constant: float
    """Planck constant (SI, J·s)."""

    elementary_charge: float
    """Elementary charge (SI, C)."""

    boltzmann_constant: float
    """Boltzmann constant (SI, J/K)."""

    avogadro_constant: float
    """Avogadro constant (SI, mol^-1)."""

    luminous_efficacy_kcd: int
    """Luminous efficacy Kcd (SI, lm/W)."""

    gravitational_constant: float
    """Gravitational constant (SI, m^3·kg^-1·s^-2)."""

    vacuum_electric_permittivity: float
    """Vacuum electric permittivity (SI, F/m)."""

    vacuum_magnetic_permeability: float
    """Vacuum magnetic permeability (SI, N/A^2)."""

    atomic_mass_constant: float
    """Atomic mass constant (SI, kg)."""

    proton_mass: float
    """Proton mass (SI, kg)."""

    neutron_mass: float
    """Neutron mass (SI, kg)."""

    electron_mass: float
    """Electron mass (SI, kg)."""

consts_fund: PhyConstsFund

__all__ = ["consts_fund"]
