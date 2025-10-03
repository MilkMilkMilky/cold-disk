import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from disk_solver import DiskParams, DiskTools, SlimDisk, StandardDisk, cgs_consts
from parameters import model_params


def print_cgs_constants():
    """Print all CGS constants for verification"""
    print("=== CGS Constants Verification ===")
    print()

    # Print each constant with its description and value
    print(f"cgs_c (Vacuum light speed): {cgs_consts.cgs_c:.6e} cm/s")
    print(f"cgs_h (Planck constant): {cgs_consts.cgs_h:.6e} erg·s")
    print(f"cgs_kb (Boltzmann constant): {cgs_consts.cgs_kb:.6e} erg/K")
    print(f"cgs_gra (Gravitational constant): {cgs_consts.cgs_gra:.6e} cm³·g⁻¹·s⁻²")
    print(f"cgs_rg (Molar gas constant): {cgs_consts.cgs_rg:.6e} erg/(mol·K)")
    print(f"cgs_mp (Proton mass): {cgs_consts.cgs_mp:.6e} g")
    print(f"cgs_mh (Hydrogen atomic mass): {cgs_consts.cgs_mh:.6e} g")
    print(f"cgs_sb (Stefan-Boltzmann constant): {cgs_consts.cgs_sb:.6e} erg·s⁻¹·cm⁻²·K⁻⁴")
    print(f"cgs_amm (Average molar mass): {cgs_consts.cgs_amm:.6e} g·mol⁻¹")
    print(f"cgs_msun (Solar mass): {cgs_consts.cgs_msun:.6e} g")
    print(f"cgs_a (Radiation constant): {cgs_consts.cgs_a:.6e} erg·cm⁻³·K⁻⁴")
    print(f"cgs_kes (Electron scattering opacity): {cgs_consts.cgs_kes:.6e} cm²·g⁻¹")
    print(f"cgs_kra (Kramers' opacity coefficient): {cgs_consts.cgs_kra:.6e} cm⁵·⁵·g⁻²·K³·⁵")

    print()
    print("=== Expected Values for Verification ===")
    print("cgs_c should be ~2.997925e10 cm/s")
    print("cgs_h should be ~6.626070e-27 erg·s")
    print("cgs_kb should be ~1.380649e-16 erg/K")
    print("cgs_gra should be ~6.674301e-8 cm³·g⁻¹·s⁻²")
    print("cgs_sb should be ~5.670374e-5 erg·s⁻¹·cm⁻²·K⁻⁴")
    print("cgs_msun should be ~1.988470e33 g")
    print("cgs_a should be ~7.565733e-15 erg·cm⁻³·K⁻⁴")
    print("cgs_kes should be 0.34 cm²·g⁻¹")
    print("cgs_kra should be 6.4e22 cm⁵·⁵·g⁻²·K³·⁵")

    print()
    print("=== Cross-checks ===")
    # Verify cgs_a = 4 * cgs_sb / cgs_c
    expected_cgs_a = 4 * cgs_consts.cgs_sb / cgs_consts.cgs_c
    print(f"cgs_a verification (4*cgs_sb/cgs_c): {expected_cgs_a:.6e}")
    print(f"cgs_a actual value: {cgs_consts.cgs_a:.6e}")
    print(f"Match: {abs(cgs_consts.cgs_a - expected_cgs_a) < 1e-10}")

    # Verify cgs_rg = cgs_avogadro * cgs_kb
    print(f"cgs_rg verification: {cgs_consts.cgs_rg:.6e} erg/(mol·K)")

    print()
    print("=== All constants printed successfully ===")


if __name__ == "__main__":
    print_cgs_constants()
