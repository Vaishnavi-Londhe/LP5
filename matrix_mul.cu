#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void matrixMultiplication(int *a, int *b, int *c, int n) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < n && col < n) {
        int sum = 0;

        for (int k = 0; k < n; k++) {
            sum += a[row * n + k] * b[k * n + col];
        }

        c[row * n + col] = sum;
    }
}

int main() {
    int n;

    cout << "Enter matrix size n: ";
    cin >> n;

    int total = n * n;
    int size = total * sizeof(int);

    int *a = new int[total];
    int *b = new int[total];
    int *c = new int[total];
    int *cpu = new int[total];

    cout << "Enter elements of first matrix:" << endl;
    for (int i = 0; i < total; i++) {
        cin >> a[i];
    }

    cout << "Enter elements of second matrix:" << endl;
    for (int i = 0; i < total; i++) {
        cin >> b[i];
    }

    int *d_a, *d_b, *d_c;

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

    matrixMultiplication<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float time = 0;
    cudaEventElapsedTime(&time, start, end);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            int sum = 0;

            for (int k = 0; k < n; k++) {
                sum += a[row * n + k] * b[k * n + col];
            }

            cpu[row * n + col] = sum;
        }
    }

    int error = 0;
    for (int i = 0; i < total; i++) {
        error += cpu[i] - c[i];
    }

    cout << endl;
    cout << "Result Matrix:" << endl;

    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            cout << c[row * n + col] << " ";
        }
        cout << endl;
    }

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
// Enter matrix size n: 3
// Enter elements of first matrix:
// 1 2 3
// 4 5 6
// 7 8 9
// Enter elements of second matrix:
// 1 0 0
// 0 1 0
// 0 0 1



//o/p
// Result Matrix:
// 1 2 3
// 4 5 6
// 7 8 9
// Error: 0
// Time Elapsed: 0.035 ms