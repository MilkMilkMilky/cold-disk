import pathlib

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from cold_disk import ParaspaceGeneratorTools, cgs_consts

if __name__ == "__main__":
    a = pathlib.Path(r"""D:\CowOwl""") / "output.txt"
    with pathlib.Path.open(a, "w"):
        a.write_text("hello world", encoding="utf-8")
