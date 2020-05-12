import numpy as np
import time

"""
axes=(0,0) - 1 i 3 wiersz taki sam, ale ogolnie wyniki zle
wyniki ogolnie sie roznia
"""

if __name__ == "__main__":
    l = [1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 0, 0, 1, 2]
    l2d = [[1, 2, 3, 3, 2, 1, 0, 0],
           [1, 2, 3, 3, 0, 0, 1, 2]]
    l4d = [[1, 2, 3, 3], 
           [2, 1, 0, 0],
           [1, 2, 3, 3],
           [0, 0, 1, 2]]

    arr = np.asarray(l, np.float32)
    arr2d = np.asarray(l2d, np.float32)
    arr4d = np.asarray(l4d, np.float32)

    arrf = np.fft.fft(arr)
    arrf2d = np.fft.fft2(arr2d)
    arrf4d = np.fft.fft2(arr4d)

    print("1D\n{}".format(arrf))
    print("2D\n{}".format(arrf2d))
    print("4D\n{}".format(arrf4d))
