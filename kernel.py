#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""public static void intMaskMultiThread(int[][] m, int[][] result, int[][] mask, int X, int Y, int R) {
    int[][] pola = new int[X][Y];
    Ring halfRing = new Ring(R);
    IntStream.range(0, X).parallel().forEach(x0 -> {
        int lxbound = Math.max(0, x0 - R);
        int rxbound = Math.min(X, x0 + R + 1);
        IntStream.range(0, Y).forEach(y0 -> {
            IntStream.range(lxbound, rxbound).forEach(x -> {
                final int dx = x0 - x;
                final int dRx = dx + R;
                final int ry = halfRing.getHalfRing(dx);
                IntStream.range(Math.max(0, y0 - ry + 1), Math.min(Y, y0 + ry)).forEach(y -> {
                    pola[x0][y0]++;
                    final int dry = y + R - y0;
                    result[x0][y0] += ((m[x0][y0] - m[x][y]) >> 31) * mask[dRx][dry];
                });
            });
        });
    });
}"""

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import time

def int_mask_multi_thread(m, result, mask, X, Y, R):
    """
    m - macierz wejsciowa X x Y
    result - wyjscie
    mask - macierz wag 2R+1 x 2R+1
    R - promien otoczenia
    """
    pola = np.empty([X, Y], dtype=np.int)
    half_ring = None  # tu bedzie jakis obiekt
    
    for x0 in range(X):
        lxbound = max(0, x0 - R)
        rxbound = min(X, x0 + R + 1)
        for y0 in range(Y):
            for x in range(lxbound, rxbound):
                dx = x0 - x
                drx = dx + R
                ry = 0  # tu bedzie jakas liczba
                # Zamieniam sobie miejscami max i min, bo tu zawsze lewa > prawa i nie wykonuje sie nigdy petla
                for y in range(min(Y, y0 + ry), max(0, y0 - ry + 1)):
                    dry = y + R - y0
                    pola[x0][y0] += mask[drx][dry]
                    # 0 jeśli m[x0][y0] > m[x][y], else -1
                    result[x0][y0] += ((m[x0][y0] - m[x][y]) >> 31) * mask[drx][dry]


if __name__ == "__main__":
    X, Y = 3, 3
    R = 3

    # Dane CPU
    m = np.random.randint(10, size=(X, Y))
    mask = np.random.randint(10, size=(2*R + 1, 2*R + 1))
    result = np.zeros([X, Y], dtype=np.int)

    # Zmienne do pomiaru czasu na GPU
    start = cuda.Event()
    end = cuda.Event()

    """
    Każdy wątek oblicza jeden element m. Czyli wątków jest X * Y.
    Każdy wątek oblicza sobie dwie ostatnie pętle.
    """

    # Kernel
    mod = SourceModule("""
    #define MAX(a,b) (a)<(b)?(b):(a)
    #define MIN(a,b) (a)>(b)?(b):(a)

    __global__ void gpu_int_mask_multi_thread(const int X, const int Y, const int R, const int** mask, int* m, int* result)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y; // numer wiersza macierzy m i result
        int col = blockIdx.x * blockDim.x + threadIdx.x; // numer kolumny macierzy m i result

        if ((row < X) && (col < Y)) {
            int x, y, dx, drx, ry, dry;
            int lxbound = MAX(0,row - R);
            int rxbound = MIN(X,row + R + 1);

            for(x=lxbound; x<rxbound; x++) {
                dx = row - x;
                drx = dx - R;
                ry = 0;
                for(y=MIN(Y,col+ry); y<MAX(0,col-ry+1); y++) {
                    dry = y + R - col;
                    result[row * X + col] += ((m[row * X + col] - m[x * X + y]) >> 31) * mask[drx][dry];
                }
            }
        }
    }
    """)
    gpu_int_mask_multi_thread = mod.get_function("gpu_int_mask_multi_thread")

    # Dane GPU
    start.record()
    m_gpu = gpuarray.to_gpu(m)
    mask_gpu = gpuarray.to_gpu(mask)
    result_gpu = gpuarray.empty((X, Y), np.int)

    # Wywołanie kernel'a
    gpu_int_mask_multi_thread(
        np.int32(X),
        np.int32(Y),
        np.int32(R),
        mask_gpu,
        m_gpu,
        result_gpu,
        block=(32, 32, 1),
        grid=((X+31)//32, (X+31)//32, 1)
    )

    # Wynik GPU
    result_gpu_kernel = result_gpu.get()
    end.record()
    end.synchronize()
    secs = start.time_till(end)*1e-3
    print("GPU: %.7f s" % secs)
    
    # Wynik CPU
    s = time.time()
    int_mask_multi_thread(m, result, mask, X, Y, R)
    print("CPU: %.7f s" % (time.time() - s))

