from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

from cold_disk import DiskParams, ParaspaceGeneratorTools, ResultGeneratorTools, SlimDisk

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
