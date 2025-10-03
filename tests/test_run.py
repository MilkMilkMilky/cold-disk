import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from disk_solver import DiskParams, DiskTools, SlimDisk, StandardDisk, cgs_consts
from parameters import model_params

if __name__ == "__main__":
    para = DiskParams(
        alpha_viscosity=model_params.alpha_viscosity[0],
        dimless_accrate=model_params.dimless_accrate[0],
        dimless_bhmass=model_params.dimless_bhmass[0],
        gas_index=model_params.gas_index[0],
        wind_index=model_params.wind_index[0],
        dimless_radius_in=model_params.dimless_radius_in[0],
        dimless_radius_out=model_params.dimless_radius_out[0],
    )
    lin = SlimDisk.get_slim_angmomin(par=para, dimless_angmomin=1.5)
    solve, solveinfo = SlimDisk.slim_disk_solver(par=para)
    # dimless_radius_solve_array = solveinfo["tcur"]
    # slimintresult = solve[: dimless_radius_solve_array.shape[0]]
    dimless_radius_solve_array = solve[0]
    angmom_solve_array, coffeta_solve_array = solve[1], solve[2]
    rveltosvel_solve_array = SlimDisk.get_slim_rveltosvel_fromfirst(
        par=para,
        dimless_radius=dimless_radius_solve_array,
        angmom=angmom_solve_array,
        coff_eta=coffeta_solve_array,
        angmomin=lin,
    )
    rveltosvel_solve_array = np.asarray(rveltosvel_solve_array)
    print(solveinfo)
    print(max(rveltosvel_solve_array))
    print(max(coffeta_solve_array))
    plt.figure(figsize=(15, 5))

    # angmom vs radius
    plt.subplot(1, 3, 1)
    plt.plot(dimless_radius_solve_array, angmom_solve_array, marker="o", linestyle="-", color="b")
    plt.xlabel("Dimless Radius")
    plt.ylabel("Angular Momentum (angmom)")
    plt.title("Angular Momentum vs Radius")
    plt.grid(True)

    # coff_eta vs radius
    plt.subplot(1, 3, 2)
    plt.plot(dimless_radius_solve_array, coffeta_solve_array, marker="o", linestyle="-", color="r")
    plt.xlabel("Dimless Radius")
    plt.ylabel("Coff_eta")
    plt.title("Coff_eta vs Radius")
    plt.grid(True)

    # rvel / sound speed vs radius
    plt.subplot(1, 3, 3)
    plt.plot(dimless_radius_solve_array, rveltosvel_solve_array, marker="o", linestyle="-", color="g")
    plt.xlabel("Dimless Radius")
    plt.ylabel("Radial Velocity / Sound Speed")
    plt.title("rvel/sound_speed vs Radius")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
