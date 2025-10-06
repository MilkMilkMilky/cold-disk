import math
from dataclasses import dataclass

from parameters import consts

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
    alpha_viscosity: float
    dimless_accrate: float
    dimless_bhmass: float
    gas_index: float
    wind_index: float
    dimless_radius_in: float
    dimless_radius_out: float
