#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Przykładowa funkcja
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
    # pola = np.empty([X, Y], dtype=np.int)
    # half_ring = None  # tu bedzie jakis obiekt
    
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
                    # pola[x0][y0] += mask[drx][dry]
                    # 0 jeśli m[x0][y0] > m[x][y], else -1
                    result[x0][y0] += ((m[x0][y0] - m[x][y]) >> 31) * mask[drx][dry]


if __name__ == "__main__":

    # Rozmiary
    X, Y = 8, 8
    R = 2
    print("Rozmiar: {}x{}".format(X, Y))

    # Dane CPU
    m = np.random.randint(10, size=(X, Y), dtype=np.int32)
    mask = np.random.randint(10, size=(2*R + 1, 2*R + 1), dtype=np.int32)
    result = np.zeros([X, Y], dtype=np.int32)

    # Zmienne do pomiaru czasu na GPU
    start = cuda.Event()
    end = cuda.Event()

    # Kernel
    # Każdy wątek oblicza jeden element result z m - czyli wątków jest X * Y.
    # Każdy wątek oblicza sobie dwie ostatnie pętle.
    mod = SourceModule("""
    #define MAX(a, b) (a)<(b)?(b):(a)
    #define MIN(a, b) (a)>(b)?(b):(a)

    __global__ void gpu_int_mask_multi_thread(const int X, const int Y, const int R, const int *mask, const int *m, int *result)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y; // numer wiersza macierzy m i result
        int col = blockIdx.x * blockDim.x + threadIdx.x; // numer kolumny macierzy m i result

        if ((row < X) && (col < Y)) {
            int x, y, dx, drx, ry, dry, from, to, lxbound, rxbound;
            lxbound = MAX(0, row - R);
            rxbound = MIN(X, row + R + 1);

            for(x = lxbound; x < rxbound; x++) {
                dx = row - x;
                drx = dx + R;
                ry = 0;
                from = MIN(Y, col + ry);
                to = MAX(0, col - ry + 1);
                for(y = from; y < to; y++) {
                    dry = y + R - col;
                    result[row * Y + col] += (((m[row * Y + col] - m[x * Y + y]) >> 31) * mask[drx * (2*R+1) + dry]);
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
    result_gpu = gpuarray.empty((X, Y), np.int32)

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
    e = time.time() - s
    print("CPU: %.7f s" % e)
    
    # print("Computation error:\n {}".format(abs(np.subtract(result_gpu_kernel, result))))

"""
Rozmiar: 3x3
GPU: 0.0029615 s
CPU: 0.0005479 s
Przyspieszenie: x 0.18

Rozmiar: 8x8
GPU: 0.0030379 s
CPU: 0.0043349 s
Przyspieszenie: x 1.43

Rozmiar: 32x32
GPU: 0.0025339 s
CPU: 0.0515258 s
Przyspieszenie: x 25.50

Rozmiar: 128x128
GPU: 0.0046817 s
CPU: 0.5255759 s
Przyspieszenie: x 131.38

Rozmiar: 512x512
GPU: 0.0070015 s
CPU: 7.8603940 s
Przyspieszenie: x 1122.86

Rozmiar: 1024x1024
GPU: 0.0084219 s
CPU: 36.8409948 s
Przyspieszenie: x 4605.00

Rozmiar: 4096x4096
GPU: 0.0544873 s
CPU: 580.9967351 s
Przyspieszenie: x 11619.80
"""
