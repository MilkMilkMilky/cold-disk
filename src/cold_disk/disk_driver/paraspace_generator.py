"""Module `cold_disk.disk_driver.paraspace_generator`.

Provides tools for generating and managing parameter spaces for accretion disk
modeling. This module handles the creation of parameter combinations, HDF5 file
initialization, and parameter space dispatching for batch computations.

The primary class is `ParaspaceGeneratorTools`, which offers static methods for
parameter space generation, file management, and data organization.

Notes:
-----
- Parameter spaces are generated from arrays of adjustable parameters.
- HDF5 files are used for storing parameter combinations and computation results.
- Three dispatch modes are supported: 'parasweep', 'pairscan', and 'fullfactorial'.
- All methods are static and stateless; no instance of `ParaspaceGeneratorTools` is required.

Example:
-------
>>> from cold_disk import ParaspaceGeneratorTools
>>> # Load default parameter ranges
>>> adjparams = ParaspaceGeneratorTools.load_adjparams_default()
>>> # Generate parameter space
>>> space = ParaspaceGeneratorTools.adjparams_dispatcher(adjparams_obj=adjparams, dispatch_mode="fullfactorial")
>>> # Initialize HDF5 file
>>> h5path = ParaspaceGeneratorTools.load_disk_datafiles(disktype="slim")
>>> ParaspaceGeneratorTools.paramspace_init(
...     hdf5_file_path=h5path, adjparams_obj=adjparams, dispatch_mode="fullfactorial"
... )

"""
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from cold_disk.parameters import model_params

__all__ = ["AdjustableParams", "ParaspaceGeneratorTools"]

@dataclass
class AdjustableParams:
    """Container for the adjustable parameter ranges used in parameter-space generation.

    This dataclass defines all tunable physical and model parameters that may vary
    across the parameter space for disk-model computations.
    Each attribute is an array specifying the *allowed values* (not a single value)
    for the corresponding quantity.

    The collection of these arrays serves as the foundation for constructing the
    full parameter space matrix via
    :func:`ParaspaceGeneratorTools.adjparams_dispatcher`.

    Attributes
    ----------
    alpha_viscosity : np.ndarray
        Dimensionless viscosity parameter values.
    dimless_accrate : np.ndarray
        Dimensionless accretion rate, typically normalized to the Eddington rate.
    dimless_bhmass : np.ndarray
        Dimensionless black hole mass values, e.g., normalized by solar masses.
    gas_index : np.ndarray
        Polytropic gas index values.
    wind_index : np.ndarray
        Disk-wind index values controlling the outflow prescription.
    dimless_radius_in : np.ndarray
        Inner radius of the disk (dimensionless), e.g., in units of gravitational radii.
    dimless_radius_out : np.ndarray
        Outer radius of the disk (dimensionless), defining the computational domain extent.

    Notes
    -----
    - Each field stores **an array of candidate values**, not a single scalar.
      This enables automatic expansion into a parameter grid for batch computations.

    Examples
    --------
    >>> from cold_disk.disk_driver.paraspace_generator import AdjustableParams
    >>> import numpy as np
    >>> adj = AdjustableParams(
    ...     alpha_viscosity=np.array([0.01, 0.1]),
    ...     dimless_accrate=np.array([0.1, 1.0, 10.0]),
    ...     dimless_bhmass=np.array([1e7]),
    ...     gas_index=np.array([1.4]),
    ...     wind_index=np.array([0.0]),
    ...     dimless_radius_in=np.array([3.0]),
    ...     dimless_radius_out=np.array([100.0]),
    ... )
    >>> adj.dimless_accrate
    array([ 0.1,  1. , 10. ])

    """

    alpha_viscosity: np.ndarray
    dimless_accrate: np.ndarray
    dimless_bhmass: np.ndarray
    gas_index: np.ndarray
    wind_index: np.ndarray
    dimless_radius_in: np.ndarray
    dimless_radius_out: np.ndarray


class ParaspaceGeneratorTools:
    """Collection of static methods for parameter space generation and HDF5 file management.

    This class provides tools for creating parameter spaces from adjustable parameter
    arrays, managing HDF5 data files, and organizing computation workflows for
    accretion disk modeling.

    Methods include:

    - `get_current_utcdate`: Get current UTC date string for file timestamping.
    - `load_disk_datafiles`: Prepare and load HDF5 data files for disk computations.
    - `load_adjparams_default`: Load default adjustable parameter ranges from model_params.
    - `adjparams_dispatcher`: Generate parameter space matrix using specified dispatch mode.
    - `paramspace_init`: Initialize parameter space structure in HDF5 files.

    All methods are static and handle the workflow from parameter space generation
    to HDF5 file initialization for batch disk computations.

    Notes
    -----
    - Parameter spaces can be generated in three modes: 'parasweep' (single parameter
      variation), 'pairscan' (pairwise parameter scanning), and 'fullfactorial'
      (Cartesian product of all parameters).
    - HDF5 files are organized with dated subdirectories and contain parameter
      space datasets, task state tracking, and result storage groups.
    - The class works with `AdjustableParams` instances containing parameter arrays.
    - All file operations are atomic and include proper error handling.

    """

    @staticmethod
    def get_current_utcdate() -> str:
        """Get the current UTC date string in compact numeric format.

        This utility returns the current date and time in UTC timezone,
        formatted as an 8-digit string ``'YYYYMMDD'``. The resulting string
        is typically used to timestamp HDF5 files.

        Returns
        -------
        str
            Current UTC date represented as an 8-digit string (``'YYYYMMDD'``).

        Examples
        --------
        >>> ParaspaceGeneratorTools.get_current_utcdate()
        '20251001'

        """
        utcdate_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
        return utcdate_str

    @staticmethod
    def load_disk_datafiles(
        *,
        data_date: str | None = None,
        disktype: str,
        base_path: str | Path | None = None,
    ) -> Path:
        """Prepare and load the target HDF5 data file for storing disk-model data.

        Constructs (and if necessary creates) a dated subdirectory for either
        slim- or standard-disk data and returns the path to the HDF5 file to be
        used as the main storage file for model outputs and related datasets.

        The directory / file layout created is::

            <data_root>/ <disk_folder> / <disk_folder>_YYYYMMDD / <disk_folder>_YYYYMMDD.h5

        where `<disk_folder>` is selected from ``{'slimdiskdata', 'standarddiskdata'}``
        according to ``disktype``. If ``data_date`` is omitted, the current UTC
        date (see :func:`get_current_utcdate`) is used.

        Parameters
        ----------
        data_date : str or None, optional
            Date string in the format ``'YYYYMMDD'`` specifying which day's
            data directory and file to load. If ``None`` (default), the current
            UTC date is used.
        disktype : str
            Disk type identifier, must be either ``'slim'`` or ``'standard'``.
            Controls which top-level data folder is used.
        base_path : str or pathlib.Path or None, optional
            Optional base path for the project workspace. If provided, the
            function will use ``base_path / 'data'`` as the data root.
            The provided ``base_path`` is expanded (``~`` handling) and resolved
            to an absolute path. If ``None`` (default), the project root is
            inferred as three levels above this file's directory and
            ``<project_root>/data`` is used.

        Returns
        -------
        pathlib.Path
            Absolute path to the HDF5 file corresponding to the requested date
            and disk type. If the target directory or file do not exist, they are
            created (an empty HDF5 file is initialized when missing).

        Raises
        ------
        ValueError
            - If ``disktype`` is not ``'slim'`` or ``'standard'``.
            - If ``data_date`` is not a valid 8-digit string in the format ``'YYYYMMDD'``.
            - If ``base_path`` is provided but does not exist or is not a directory.

        Notes
        -----
        - The returned HDF5 file is intended to serve as the main data container
          for disk-model computations (parameter-space datasets, solver outputs,
          diagnostic data, etc.).
        - When ``base_path`` is provided it is processed with ``Path(base_path).expanduser().resolve()``
          before validation; this allows users to pass ``'~'``-style paths.
        - The function guarantees the directory structure exists and initializes
          an empty HDF5 file when necessary.

        Examples
        --------
        >>> ParaspaceGeneratorTools.load_disk_datafiles(disktype="slim")
        PosixPath('/.../data/slimdiskdata/slimdiskdata_20251001/slimdiskdata_20251001.h5')

        >>> ParaspaceGeneratorTools.load_disk_datafiles(
        ...     data_date="19491001", disktype="standard", base_path="~/projects/cold_disk"
        ... )
        PosixPath('/home/user/projects/cold_disk/data/standarddiskdata/standarddiskdata_19491001/standarddiskdata_19491001.h5')

        """
        if disktype not in ("slim", "standard"):
            raise ValueError(f"Invalid disktype '{disktype}', must be 'standard' or 'slim'")

        if data_date is None:
            data_date = ParaspaceGeneratorTools.get_current_utcdate()

        if not isinstance(data_date, str) or len(data_date) != 8 or not data_date.isdigit():
            raise ValueError(f"data_date must be 'YYYYMMDD', got: {data_date}")

        if base_path is None:
            current_file_dir = Path(__file__).resolve().parent
            project_root = (current_file_dir.parent.parent.parent).resolve()
            data_root = project_root / "data"
        else:
            base_path = Path(base_path).expanduser().resolve()
            if not base_path.exists():
                raise ValueError(f"Provided base_path does not exist: {base_path}")
            if not base_path.is_dir():
                raise ValueError(f"Provided base_path is not a directory: {base_path}")
            data_root = base_path / "data"

        disk_map = {"slim": "slimdiskdata", "standard": "standarddiskdata"}
        diskdata_dir = data_root / disk_map[disktype]
        target_dir_name = f"{disk_map[disktype]}_{data_date}"
        target_dir = diskdata_dir / target_dir_name
        hdf5_file_path = target_dir / f"{target_dir_name}.h5"
        target_dir.mkdir(parents=True, exist_ok=True)

        if not hdf5_file_path.exists():
            h5py.File(hdf5_file_path, "w").close()

        return hdf5_file_path

    @staticmethod
    def load_adjparams_default() -> AdjustableParams:
        """Load the default adjustable-parameter ranges for the parameter-space generator.

        This function constructs and returns an instance of
        :class:`cold_disk.disk_driver.paraspace_generator.AdjustableParams`,
        whose fields are populated from the global singleton ``model_params``.
        Each field in the dataclass corresponds to one adjustable model parameter
        and is represented as a NumPy array specifying its possible values
        (i.e., the range or grid over which the parameter will be scanned).

        The resulting object defines the base parameter-space configuration
        used by :func:`ParaspaceGeneratorTools.adjparams_dispatcher` to construct
        concrete parameter combinations for subsequent disk-model computations.

        Returns
        -------
        AdjustableParams
            Instance of :class:`cold_disk.disk_driver.paraspace_generator.AdjustableParams`
            containing NumPy arrays of adjustable parameter ranges:

            - ``alpha_viscosity`` : alpha-viscosity parameter
            - ``dimless_accrate`` : dimensionless accretion rate
            - ``dimless_bhmass`` : dimensionless black-hole mass
            - ``gas_index`` : gas-pressure dominance index
            - ``wind_index`` : wind-driving index
            - ``dimless_radius_in`` : inner disk radius (dimensionless)
            - ``dimless_radius_out`` : outer disk radius (dimensionless)

        Notes
        -----
        - All parameter arrays are obtained directly from the module-level singleton
          ``cold_disk.parameters.model_params``.
        - This dataclass instance is a container for parameter *ranges*, not single values.
          It serves as input for parameter-space generation modes such as
          ``'parasweep'``, ``'pairscan'``, and ``'fullfactorial'``.

        Examples
        --------
        >>> adj = ParaspaceGeneratorTools.load_adjparams_default()
        >>> adj.alpha_viscosity
        array([0.01, 0.1, 0.3])
        >>> type(adj)
        <class 'cold_disk.disk_driver.paraspace_generator.AdjustableParams'>

        """
        adjparams = AdjustableParams(
            alpha_viscosity=model_params.alpha_viscosity,
            dimless_accrate=model_params.dimless_accrate,
            dimless_bhmass=model_params.dimless_bhmass,
            gas_index=model_params.gas_index,
            wind_index=model_params.wind_index,
            dimless_radius_in=model_params.dimless_radius_in,
            dimless_radius_out=model_params.dimless_radius_out,
        )
        return adjparams

    @staticmethod
    def adjparams_dispatcher(*, adjparams_obj: AdjustableParams, dispatch_mode: str) -> np.ndarray:
        """Generate the full parameter-space matrix according to the specified dispatching mode.

        This function expands the adjustable-parameter ranges stored in an
        :class:`cold_disk.disk_driver.paraspace_generator.AdjustableParams` instance
        into a structured NumPy array of concrete parameter combinations (the *parameter space*).
        Each row of the output array corresponds to one specific combination of parameters
        that can be passed to a disk-model solver.

        The generation behavior depends on the selected ``dispatch_mode``:

        - ``'parasweep'`` : Sweep only one parameter while keeping all others fixed.
          Exactly one parameter array must have length > 1.
        - ``'pairscan'`` : Scan multiple parameters pairwise; all arrays must have the same length.
        - ``'fullfactorial'`` : Compute the Cartesian product of all parameter arrays
          to enumerate all possible parameter combinations.

        Parameters
        ----------
        adjparams_obj : AdjustableParams
            Instance of :class:`cold_disk.disk_driver.paraspace_generator.AdjustableParams`
            containing NumPy arrays for each adjustable parameter.
        dispatch_mode : {'parasweep', 'pairscan', 'fullfactorial'}
            Mode controlling how parameter combinations are generated:

            - ``'parasweep'`` : vary one parameter, hold others constant.
            - ``'pairscan'`` : scan parameters pairwise (same array lengths required).
            - ``'fullfactorial'`` : perform full Cartesian expansion across all arrays.

        Returns
        -------
        np.ndarray
            A structured NumPy array representing the generated parameter space.
            The dtype fields are:

            - ``id`` : (int) unique identifier for each parameter combination.
            - All adjustable parameters defined in ``adjparams_obj`` (float).

            Example dtype:
            ``[('id','i4'), ('alpha_viscosity','f8'), ('dimless_accrate','f8'), ...]``

        Raises
        ------
        ValueError
            If ``dispatch_mode`` is invalid, or the array-length conditions for the
            chosen mode are not satisfied.

        Notes
        -----
        - The resulting array serves as the canonical parameter-space table used
          for HDF5 initialization in :func:`ParaspaceGeneratorTools.paramspace_init`.
        - For reproducibility, the field order in the structured array strictly
          follows the dataclass field order of ``AdjustableParams``.
        - The ``id`` field is automatically assigned as a zero-based index.

        Examples
        --------
        >>> adj = ParaspaceGeneratorTools.load_adjparams_default()
        >>> space = ParaspaceGeneratorTools.adjparams_dispatcher(adjparams_obj=adj, dispatch_mode="fullfactorial")
        >>> space.dtype.names
        ('id', 'alpha_viscosity', 'dimless_accrate', 'dimless_bhmass', ...)
        >>> space.shape
        (144,)

        """
        allowed_modes = ("parasweep", "pairscan", "fullfactorial")
        if dispatch_mode not in allowed_modes:
            raise ValueError(f"dispatch_mode must be one of {allowed_modes}, got '{dispatch_mode}'")

        adjparams_dict = {field.name: getattr(adjparams_obj, field.name) for field in fields(adjparams_obj)}
        adjparams_names = list(adjparams_dict.keys())

        if dispatch_mode == "parasweep":
            lengths = [len(value) for value in adjparams_dict.values()]
            num_multi_value_params = sum(length_value > 1 for length_value in lengths)
            if num_multi_value_params != 1:
                raise ValueError(
                    "In 'parasweep' mode, exactly one adjustable parameter must have length > 1; "
                    "all other parameters must have length 1.",
                )
            max_length = max(lengths)
            adjparams_arrays = [
                np.full(max_length, value[0]) if len(value) == 1 else value for value in adjparams_dict.values()
            ]
            adjparams_matrix = np.column_stack(adjparams_arrays)

        elif dispatch_mode == "pairscan":
            lengths = [len(value) for value in adjparams_dict.values()]
            if len(set(lengths)) != 1:
                raise ValueError("In 'pairscan' mode, all adjustable parameters must have the same length.")
            adjparams_matrix = np.column_stack(list(adjparams_dict.values()))

        elif dispatch_mode == "fullfactorial":
            grids = np.meshgrid(*adjparams_dict.values(), indexing="ij")
            adjparams_matrix = np.column_stack([grid.ravel() for grid in grids])

        dtype = [("id", "i4")] + [(name, "f8") for name in adjparams_names]
        num_tasks = adjparams_matrix.shape[0]
        adjparams_space = np.zeros(num_tasks, dtype=dtype)
        adjparams_space["id"] = np.arange(num_tasks)
        for index, name in enumerate(adjparams_names):
            adjparams_space[name] = adjparams_matrix[:, index]

        return adjparams_space

    @staticmethod
    def paramspace_init(
        *,
        hdf5_file_path: Path,
        adjparams_obj: AdjustableParams,
        dispatch_mode: str,
        clearfile: bool = False,
    ) -> None:
        """Initialize the parameter-space structure inside the main HDF5 data file.

        This function prepares the *parameter-space* and *task-indexing* datasets
        within the main HDF5 storage file that holds all disk model computation data.
        It ensures that the file contains a consistent base structure before any
        solver modules (e.g., standard-disk or slim-disk) begin writing their results.

        The created datasets serve as the registry and bookkeeping layer for all
        subsequent numerical tasks, associating each adjustable-parameter combination
        with a dedicated task group that will later store its computation outputs.

        Parameters
        ----------
        hdf5_file_path : Path
            Path to the main HDF5 data file that stores **all simulation content**,
            including the parameter space, solver results, and metadata.
        adjparams_obj : AdjustableParams
            Instance of :class:`cold_disk.disk_driver.paraspace_generator.AdjustableParams`,
            containing the adjustable-parameter arrays from which the parameter space
            will be constructed.
        dispatch_mode : {'parasweep', 'pairscan', 'fullfactorial'}
            Parameter-space generation mode used by
            :func:`ParaspaceGeneratorTools.adjparams_dispatcher` to expand the parameter grid.
        clearfile : bool, optional
            If ``True``, clear and recreate the entire file (open mode ``'w'``).
            If ``False`` (default), append or update the existing structure (open mode ``'a'``).

        Returns
        -------
        None
            The HDF5 file is updated in place; the function does not return any value.

        Raises
        ------
        TypeError
            If an existing ``/adjparamspace`` object is not a valid HDF5 dataset.
        ValueError
            If ``dispatch_mode`` is invalid for
            :func:`ParaspaceGeneratorTools.adjparams_dispatcher`.

        Notes
        -----
        - The HDF5 file after initialization will contain (at minimum):

          - ``/adjparamspace`` : structured dataset of all adjustable-parameter combinations.
          - ``/taskstate`` : boolean 1D array marking each task's completion status.
          - ``/task_{id}`` : empty groups reserved for solver outputs.

        - This routine does **not** perform any computation or solver dispatch;
          it only prepares the file structure for downstream modules to use.

        - Setting ``clearfile=True`` will permanently delete existing results and metadata.

        Examples
        --------
        >>> h5path = Path("./slimdisk_data_main.h5")
        >>> adj = ParaspaceGeneratorTools.load_adjparams_default()
        >>> ParaspaceGeneratorTools.paramspace_init(
        ...     hdf5_file_path=h5path,
        ...     adjparams_obj=adj,
        ...     dispatch_mode="fullfactorial",
        ...     clearfile=True,
        ... )
        >>> with h5py.File(h5path, "r") as f:
        ...     print(list(f.keys()))
        ['adjparamspace', 'taskstate', 'task_0', 'task_1', ...]

        """
        file_mode = "w" if clearfile else "a"

        with h5py.File(hdf5_file_path, file_mode) as h5file:
            if "adjparamspace" in h5file:
                adjparamspace_ds = h5file["adjparamspace"]
                if not isinstance(adjparamspace_ds, h5py.Dataset):
                    raise TypeError("'adjparamspace' must be an h5py.Dataset")
                task_ids = adjparamspace_ds.fields("id")[:]
                num_tasks = len(task_ids)
            else:
                adjparams_space = ParaspaceGeneratorTools.adjparams_dispatcher(
                    adjparams_obj=adjparams_obj,
                    dispatch_mode=dispatch_mode,
                )
                h5file.create_dataset("adjparamspace", data=adjparams_space, compression="gzip", compression_opts=9)
                num_tasks = len(adjparams_space)
                task_ids = np.arange(num_tasks)

            if "taskstate" not in h5file:
                taskstate_ds = h5file.create_dataset(
                    "taskstate",
                    shape=(num_tasks,),
                    dtype="bool",
                    compression="gzip",
                    compression_opts=9,
                )
                taskstate_ds[:] = False

            for task_id in task_ids:
                group_name = f"task_{task_id}"
                if group_name not in h5file:
                    h5file.create_group(group_name)
        return None
