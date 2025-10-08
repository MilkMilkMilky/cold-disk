from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

from cold_disk import DiskParams, ParaspaceGeneratorTools, ResultGeneratorTools, SlimDisk


def slimtest_save_csv(par: DiskParams):
    solveresult, solveinfo = SlimDisk.slim_disk_odeint_solver(par=par)
    print(solveinfo)
    print(np.max(solveresult["rveltosvel"]))
    results_df = pd.DataFrame(solveresult)
    results_df = results_df.set_index("dimless_radius")
    save_path = Path("./slim_results.csv")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(save_path, index=True)


if __name__ == "__main__":
    filepath = ParaspaceGeneratorTools.load_disk_datafiles(data_date="20251008", disktype="slim")
    adjparams = ParaspaceGeneratorTools.load_adjparams_default()
    ParaspaceGeneratorTools.paramspace_init(
        hdf5_file_path=filepath,
        adjparams_obj=adjparams,
        dispatch_mode="fullfactorial",
    )
    ResultGeneratorTools.slimdisk_normalresult_generator(hdf5_file_path=filepath)
    # ResultGeneratorTools.slimdisk_radiationresult_generator(hdf5_file_path=filepath)
