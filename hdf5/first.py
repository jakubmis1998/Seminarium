import h5py
import numpy as np

# sizes = [2**4, 2**8, 2**12, 2**16, 2**20, 2**24, 2**28]
# l = [1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 0, 0, 1, 2]
# l4 = (sizes[0] / 16) * l
# l8 = (sizes[1] / 16) * l
# l12 = (sizes[2] / 16) * l
# l16 = (sizes[3] / 16) * l
# l20 = (sizes[4] / 16) * l
# l24 = (sizes[5] / 16) * l
# l28 = (sizes[6] / 16) * l

# with h5py.File("plik.hdf5", "a") as hdf_file:
#     hdf_file.create_dataset(name="tablica4", data=l4)
#     hdf_file.create_dataset(name="tablica8", data=l8)
#     hdf_file.create_dataset(name="tablica12", data=l12)
#     hdf_file.create_dataset(name="tablica16", data=l16)
#     hdf_file.create_dataset(name="tablica20", data=l20)
#     hdf_file.create_dataset(name="tablica24", data=l24)
#     hdf_file.create_dataset(name="tablica28", data=l28)

with h5py.File("plik.hdf5", "r") as hdf_file:
    xd = hdf_file.keys()
    for cos in hdf_file.values():
        print(cos)