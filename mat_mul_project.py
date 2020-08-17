#!/usr/bin/env python
# -*- coding: utf-8 -*-

###################################################### IMPORTS ###########################################

# Matrices operations
import numpy as np

# PyCuda
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

# Sci-kit Cuda modules
import skcuda.linalg as linalg
import skcuda.cublas as cublas

# Reikna
import reikna.cluda as cluda
from reikna.linalg import MatrixMul

# Time measuring
import time

# PyCuda initialization
import pycuda.autoinit

# Linalg initialization
linalg.init()

# Reikna initialization
api = cluda.ocl_api()  # It will be enough to make the code use CUDA
thr = api.Thread.create()  # Prepare and compilation stage


##################################### KERNELS #########################################


# Naive kernel for matrices multiplications
mod = SourceModule("""
    __global__ void naive_martices_multiplication_kernel(int N, float *A, float *B, float* out)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y; // numer wiersza
        int col = blockIdx.x * blockDim.x + threadIdx.x; // numer kolumny

        float out_ij = 0.0;
        if ((row < N) && (col < N)){
            // Jeden watek uzywa jednego wiersza macierzy A i jedna kolumne macierzy B
            // do obliczenia jednego elementu macierzy wynikowej
            for (int k = 0; k < N; k++) {
                out_ij += A[col * N + k] * B[k * N + row];
            }
        }

        out[col * N + row] = out_ij;
    }
""")
naive_martices_multiplication_kernel = mod.get_function("naive_martices_multiplication_kernel")


# Kernel for matrices multiplication with shared memory and tiles
# N musi byc wielokrotnoscia BlockDim.x (N = k * BLOCKSIZE.X)
# W przypadku mniejszego rozmiaru mozna dopelnic macierz zerami do N x N
# Ilosc kafelkow = ilosc blokow - jeden blok oblicza jeden kafelek
# Ilosc watkow = elementow kafelka - jeden watek oblicza jeden element kafelka
mod = SourceModule("""
    __global__ void shared_matrices_multiplication_kernel(int N, float *A, float *B, float* out)
    {
        const int TILE_DIM = 32;
        __shared__ float A_shared[TILE_DIM][TILE_DIM];
        __shared__ float B_shared[TILE_DIM][TILE_DIM];

        int tx = threadIdx.x; // Id x w danym bloku
        int ty = threadIdx.y; // Id y w danym bloku
        int i = blockIdx.x * blockDim.x + threadIdx.x; // Ogolny id x - w odpowiednim kafelku
        int j = blockIdx.y * blockDim.y + threadIdx.y; // Ogolny id y - w odpowiednim kafelku

        float out_ij = 0.0;
        // Jeden watek wypelnia macierz wspoldzielona m-tego kafelka
        // Z pamieci globalnej do pamieci dzielonej
        for (int m = 0; m < (N / TILE_DIM); m++) { // m-ty kafelek
            A_shared[tx][ty] = A[i * N + (m * TILE_DIM + ty)];
            B_shared[tx][ty] = B[(m * TILE_DIM + tx) * N + j];

            __syncthreads(); // zapewnia zaladowanie macierzy

            for (int k = 0; k < TILE_DIM; k++) { // Suma czesciowa - element m-tego kafelka
                out_ij += A_shared[tx][k] * B_shared[k][ty];
            }

            __syncthreads(); // zapewnia obliczenia w pelni wykonane
        }

        out[i * N + j] = out_ij; // pamiec globalna - suma pelna - jeden element m-tego kafelka
    }
""")
shared_matrices_multiplication_kernel = mod.get_function("shared_matrices_multiplication_kernel")


# Kernel for matrices multiplication with tiles, shared memory and any size
"""
Zoptymalizowane mnozenie macierzy z pamiecia dzielona wykorzystujac kafelkowanie
Dowolne N - sprawdzanie w kernelu
"""
mod = SourceModule("""
    __global__ void tiles_matrices_multiplication_kernel(float* A, float* B, float* C, int N)
    {
        const int TILE_DIM = 32;
        __shared__ float A_shared[TILE_DIM][TILE_DIM];
        __shared__ float B_shared[TILE_DIM][TILE_DIM];

        int tx = threadIdx.x; // Id x w danym bloku
        int ty = threadIdx.y; // Id y w danym bloku
        int i = blockIdx.x * blockDim.x + threadIdx.x; // Ogolny id x - w odpowiednim kafelku
        int j = blockIdx.y * blockDim.y + threadIdx.y; // Ogolny id y - w odpowiednim kafelku

        float out_ij = 0.0;
        for (int m = 0; m < (TILE_DIM + N - 1) / TILE_DIM; m++) {

            if ((i < N) && (m * TILE_DIM + ty < N)) {
                A_shared[tx][ty] = A[i * N + (m * TILE_DIM + ty)];
            } else {
                A_shared[tx][ty] = 0.0;
            }

            if ((j < N) && (m * TILE_DIM + tx < N)) {
                B_shared[tx][ty] = B[(m * TILE_DIM + tx) * N + j];
            } else {
                B_shared[tx][ty] = 0.0;
            }

            __syncthreads();

            for (int k = 0; k < TILE_DIM; k++) {
                out_ij += A_shared[tx][k] * B_shared[k][ty];
            }

            __syncthreads();
        }

        if (j < N && i < N)
            C[i * N + j] = out_ij;
    }
""")
tiles_matrices_multiplication_kernel = mod.get_function("tiles_matrices_multiplication_kernel")


#################################### RESULTS METHODS ##############################################


def print_results(description, size, time):
    """Prints information of matrices multiplication results"""
    print("\n" + description)
    print("Size: \t{}x{}".format(size, size))
    print("Time: \t%.7f s" % time)
    return None


def print_computation_error(A_result, B_result):
    """Prints computation error of matrices multiplication"""
    print("Computation error:\n {}".format(abs(np.subtract(A_result, B_result))))
    return None


#################################### MATRICES MULTIPLICATION METHODS #####################################


def mat_mul_cpu(A, B, N):
    """Naive matrices multiplication on CPU"""

    # Result matrix
    result = np.empty([N, N], dtype=np.float32)

    # Start time measuring
    start = time.time()

    # Computing matrices multiplication
    for i in range(N):
        for j in range(N):
            tmp = 0.0
            for k in range(N):
                tmp += A[i][k] * B[k][j]
            result[i][j] = tmp

    # Return time and result matrix
    return {
        "time": time.time() - start,
        "result": result
    }
    

"""
Numpy array is a collection of similar data-types that are densely packed in memory.
A Python list can have different data-types, which puts lots of extra constraints while doing computation on it.
Numpy is able to divide a task into multiple subtasks and process them parallelly.
Numpy functions are implemented in C. Which again makes it faster compared to Python Lists.
"""
def numpy_mat_mul_cpu(A, B, N):
    """Matrices multiplications on CPU using NumPy"""

    # Start time measuring
    start = time.time()

    # Comupting with NumPy
    result = np.dot(A, B)

    # Return time and result matrix
    return {
        "time": time.time() - start,
        "result": result
    }


def naive_mat_mul_kernel(A, B, N):
    """
    Matrices multiplications on GPU using naive kernel

    CPU
    1 2 3
    4 5 6
    7 8 9
    GPU
    1 2 3 4 5 6 7 8 9

    row - ktory wiersz przetwarza dany watek
    col - ktora kolumne przetwarza dany watek

    row * N + col ---> wielokrotnosc wiersza + kolumna. 5 = N(3) * 1 + 2 - wynik
    row * N + k   ---> wielkrotnosc wiersza + 0, 1, 2, ..., N - caly wiersz(kolejno N-ki)
    k * N + col   ---> wielokrotnosc kolumny + N, 2N, 3N - cala kolumna(co N)
    """

    # For time measuring on GPU
    start = cuda.Event()
    end = cuda.Event()

    # Result matrix
    result = gpuarray.empty((N, N), np.float32)

    # Start time measuring
    start.record()

    # Copy in - data
    a_gpu = gpuarray.to_gpu(A)
    b_gpu = gpuarray.to_gpu(B)

    # Computing matrices multiplication
    naive_martices_multiplication_kernel(
        np.int32(N),
        a_gpu, b_gpu,
        result,
        block=(32, 32, 1),
        grid=((N+31)//32, (N+31)//32, 1)
    )

    # Copy out - data
    result_gpu = result.get()

    # End time measuring
    end.record()
    end.synchronize()

    # Return time and result matrix
    return {
        "time": start.time_till(end)*1e-3,
        "result": result_gpu
    }


def shared_mat_mul_kernel(A, B, N):
    """Matrices multiplication on GPU using shared memory"""

    # For time measuring on GPU
    start = cuda.Event()
    end = cuda.Event()

    # Result matrix
    result = gpuarray.empty((N, N), np.float32)

    # Start time measuring
    start.record()

    # Copy in - data
    a_gpu = gpuarray.to_gpu(A)
    b_gpu = gpuarray.to_gpu(B)

    # Computing matrices multiplication
    shared_matrices_multiplication_kernel(
        np.int32(N),
        a_gpu,
        b_gpu,
        result,
        block=(32, 32, 1),
        grid=((N+31)//32, (N+31)//32, 1)
    )

    # Copy out - data
    result_gpu = result.get()

    # End time measuring
    end.record()
    end.synchronize()

    # Return time and result matrix
    return {
        "time": start.time_till(end)*1e-3,
        "result": result_gpu
    }


def tiles_mat_mul_kernel(A, B, N):
    """Matrices multiplication on GPU using tiles"""

    # For time measuring on GPU
    start = cuda.Event()
    end = cuda.Event()

    # Result matrix
    result = gpuarray.empty((N, N), np.float32)

    # Start time measuring
    start.record()

    # Copy in - data
    a_gpu = gpuarray.to_gpu(A)
    b_gpu = gpuarray.to_gpu(B)

    # Computing matrices multiplication
    tiles_matrices_multiplication_kernel(
        a_gpu,
        b_gpu,
        result,
        np.int32(N),
        block=(32, 32, 1),
        grid=((N+31)//32, (N+31)//32, 1)
    )

    # Copy out - data
    result_gpu = result.get()

    # End time measuring
    end.record()
    end.synchronize()

    # Return time and result matrix
    return {
        "time": start.time_till(end)*1e-3,
        "result": result_gpu
    }


def skcuda_linalg_mat_mul(A, B, N):
    """Matrices multiplications on GPU using skcuda linalg"""

    # Start time measuring on GPU
    start = cuda.Event()
    end = cuda.Event()
    start.record()

    # Copy in - data
    a_gpu = gpuarray.to_gpu(A)
    b_gpu = gpuarray.to_gpu(B)

    # Computing with skcuda linalg
    result = linalg.dot(a_gpu, b_gpu)

    # Copy out - data
    result_gpu = result.get()

    # End time measuring
    end.record()
    end.synchronize()

    # Return time and result matrix
    return {
        "time": start.time_till(end)*1e-3,
        "result": result_gpu
    }


def reikna_mat_mul(A, B, N):
    """Matrices multiplication on GPU using REIKNA"""

    # For time measuring on GPU
    start = cuda.Event()
    end = cuda.Event()

    # Result matrix
    result_gpu = thr.array((N, N), dtype=np.float32)

    # Start time measuring
    start.record()

    # Copy in - data
    a_gpu = thr.to_device(A)
    b_gpu = thr.to_device(B)

    # Computing with reikna
    dot = MatrixMul(a_gpu, b_gpu, out_arr=result_gpu)  # preparation
    dotc = dot.compile(thr)  # compilation
    dotc(result_gpu, a_gpu, b_gpu)  # call

    # Copy out - data
    result = result_gpu.get()

    # End time measuring
    end.record()
    end.synchronize()

    # Return time and result matrix
    return {
        "time": start.time_till(end)*1e-3,
        "result": result
    }


def skcuda_cublas_mat_mul(A, B, N):
    """Matrices multiplications on GPU using skcuda cublas"""

    # Start time measuring on GPU
    start = cuda.Event()
    end = cuda.Event()
    start.record()

    # Prepare matrices
    A = np.asarray(A, order = 'F')
    B = np.asarray(B, order = 'F')

    # Result matrix
    result = gpuarray.empty((N, N), np.float32)
    alpha = np.float32(1.0)
    beta  = np.float32(0.0)

    # Copy in - data
    a_gpu = gpuarray.to_gpu(A)
    b_gpu = gpuarray.to_gpu(B)

    # Computing with skcuda cublas
    cublas_handle = cublas.cublasCreate()
    cublas.cublasSgemm(
        # C = α op ( A ) op ( B ) + β C
        # op ( A ) = A if  transa == CUBLAS_OP_N
        # op ( B ) = B if  transb == CUBLAS_OP_N
        cublas_handle, # handle to the cuBLAS library context.
        'n', # operation op(A) that is non- or (conj.) transpose.
        'n', # operation op(B) that is non- or (conj.) transpose.
        N, # m
        N, # n
        N, # k
        alpha, # scalar used for multiplication
        a_gpu.gpudata, # array of dimensions lda x k with lda>=max(1,m) if transa == CUBLAS_OP_N.
        N, # leading dimension of two-dimensional array used to store the matrix A.
        b_gpu.gpudata, # array of dimension ldb x n with ldb>=max(1,k) if transb == CUBLAS_OP_N and ldb x k with ldb>=max(1,n) otherwise.
        N, # leading dimension of two-dimensional array used to store matrix B.
        beta, # scalar used for multiplication. If beta==0, C does not have to be a valid input.
        result.gpudata, # array of dimensions ldc x n with ldc>=max(1,m).
        N # leading dimension of a two-dimensional array used to store the matrix C.
    )
    cublas.cublasDestroy(cublas_handle)

    # Copy out - data
    result_gpu = result.reshape(result.shape, order = 'F').get()

    # End time measuring
    end.record()
    end.synchronize()

    # Return time and result matrix
    return {
        "time": start.time_till(end)*1e-3,
        "result": result_gpu
    }


if __name__ == "__main__":

    # Matrices and size
    N = 32
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)

    #################################### CPU NAIVE MATRICES MULTIPLICATION #######################################

    print_results(
        description="CPU NAIVE", 
        size=N,
        time=mat_mul_cpu(A, B, N).get("time")
    )

    #################################### CPU WITH NUMPY MATRICES MULTIPLICATION #######################################

    print_results(
        description="CPU using NUMPY", 
        size=N,
        time=numpy_mat_mul_cpu(A, B, N).get("time")
    )

    #################################### GPU NAIVE KERNEL MATRICES MULTIPLICATION #######################################

    print_results(
        description="GPU using NAIVE KERNEL", 
        size=N,
        time=naive_mat_mul_kernel(A, B, N).get("time")
    )

    #################################### GPU KERNEL MATRICES MULTIPLICATION WITH SHARED MEMORY AND TILES ##########################

    print_results(
        description="GPU KERNEL using TILES and SHARED MEMORY", 
        size=N,
        time=shared_mat_mul_kernel(A, B, N).get("time")
    )


    #################################### GPU KERNEL MATRICES MULTIPLICATION WITH SHARED MEMORY, TILES AND ANY SIZE ###############

    print_results(
        description="GPU KERNEL using TILES, SHARED MEMORY and ANY SIZE", 
        size=N,
        time=tiles_mat_mul_kernel(A, B, N).get("time")
    )

    #################################### GPU SKCUDA LINALG MATRICES MULTIPLICATION ##########################

    print_results(
        description="GPU SKCUDA LINALG", 
        size=N,
        time=skcuda_linalg_mat_mul(A, B, N).get("time")
    )

    #################################### GPU REIKNA MATRICES MULTIPLICATION ##########################

    print_results(
        description="GPU REIKNA", 
        size=N,
        time=reikna_mat_mul(A, B, N).get("time")
    )

    #################################### GPU SKCUDA CUBLAS MATRICES MULTIPLICATION ##########################

    print_results(
        description="GPU SKCUDA CUBLAS", 
        size=N,
        time=skcuda_cublas_mat_mul(A, B, N).get("time")
    )

    # Computation error between two matrices
    print_computation_error(skcuda_cublas_mat_mul(A, B, N).get("result"), skcuda_linalg_mat_mul(A, B, N).get("result"))
