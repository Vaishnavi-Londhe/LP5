// 1. CUDA C++ Code for Addition of Two Large Vectors
#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    int n;

    cout << "Enter size of vectors: ";
    cin >> n;

    int *a = new int[n];
    int *b = new int[n];
    int *c = new int[n];
    int *cpu = new int[n];

    cout << "Enter elements of first vector:" << endl;
    for (int i = 0; i < n; i++) {
        cin >> a[i];
    }

    cout << "Enter elements of second vector:" << endl;
    for (int i = 0; i < n; i++) {
        cin >> b[i];
    }

    for (int i = 0; i < n; i++) {
        cpu[i] = a[i] + b[i];
    }

    int *d_a, *d_b, *d_c;
    int size = n * sizeof(int);

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

    vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float time = 0;
    cudaEventElapsedTime(&time, start, end);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    int error = 0;
    for (int i = 0; i < n; i++) {
        error += cpu[i] - c[i];
    }

    cout << endl;
    cout << "Result of Vector Addition:" << endl;
    for (int i = 0; i < n; i++) {
        cout << c[i] << " ";
    }

    cout << endl;
    cout << "Error: " << error << endl;
    cout << "Time Elapsed: " << time << " ms" << endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    delete[] a;
    delete[] b;
    delete[] c;
    delete[] cpu;

    return 0;
}
// Enter size of vectors: 5
// Enter elements of first vector:
// 1 2 3 4 5
// Enter elements of second vector:
// 10 20 30 40 50



// How to Compile and Run CUDA Programs
// Compile Vector Addition
// nvcc vector_add.cu -o vector_add 
// vector_add.exe