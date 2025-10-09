"""Cold Disk Driver Subpackage.

This subpackage provides tools for managing parameter spaces and computing
accretion disk solutions in batch workflows. It handles parameter space
generation, HDF5 file management, and batch computation orchestration
for the cold disk modeling framework.

Public API
----------
ParaspaceGeneratorTools : class
    Collection of static methods for parameter space generation and HDF5 file
    management. Handles creation of parameter combinations, file initialization,
    and parameter space dispatching for batch computations.
ResultGeneratorTools : class
    Collection of static methods for computing and storing accretion disk
    solutions in HDF5 files. Manages batch computation workflows, result
    storage, and completion tracking for parameter space exploration.

Notes
-----
- Both classes contain only static methods and are stateless; no instances are required.
- `ParaspaceGeneratorTools` works with parameter arrays to generate parameter spaces
  in three modes: 'parasweep', 'pairscan', and 'fullfactorial'.
- `ResultGeneratorTools` operates on HDF5 files with pre-initialized parameter spaces
  and provides robust error handling and logging for batch computations.
- HDF5 files are organized with dated subdirectories and contain parameter space
  datasets, task state tracking, and result storage groups.
- All file operations are atomic and include proper error handling for reliability.

Examples
--------
>>> from cold_disk import ParaspaceGeneratorTools, ResultGeneratorTools
>>> from pathlib import Path
>>> # Generate parameter space and initialize HDF5 file
>>> adjparams = ParaspaceGeneratorTools.load_adjparams_default()
>>> h5path = ParaspaceGeneratorTools.load_disk_datafiles(disktype="slim")
>>> ParaspaceGeneratorTools.paramspace_init(
...     hdf5_file_path=h5path, adjparams_obj=adjparams, dispatch_mode="fullfactorial"
... )
>>> # Compute disk solutions
>>> ResultGeneratorTools.slimdisk_normalresult_generator(hdf5_file_path=h5path)
>>> ResultGeneratorTools.slimdisk_radiationresult_generator(hdf5_file_path=h5path)

"""
from cold_disk.disk_driver.paraspace_generator import ParaspaceGeneratorTools
from cold_disk.disk_driver.result_generator import ResultGeneratorTools

__all__ = ["ParaspaceGeneratorTools", "ResultGeneratorTools"]
