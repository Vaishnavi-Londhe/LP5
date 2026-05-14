!nvidia-smi


%%writefile cuda_program.cu

#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// VECTOR ADDITION KERNEL
__global__ void vectorAdd(int *a, int *b, int *c, int n) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

// MATRIX MULTIPLICATION KERNEL
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

    int choice;

    cout << "===== CUDA PROGRAM =====" << endl;
    cout << "1. Vector Addition" << endl;
    cout << "2. Matrix Multiplication" << endl;
    cout << "Enter your choice: ";
    cin >> choice;

    // VECTOR ADDITION
    if (choice == 1) {

        int n;

        cout << endl;
        cout << "Enter size of vectors: ";
        cin >> n;

        int *a = new int[n];
        int *b = new int[n];
        int *c = new int[n];
        int *cpu = new int[n];

        cout << endl;
        cout << "Enter elements of first vector:" << endl;

        for (int i = 0; i < n; i++) {
            cin >> a[i];
        }

        cout << "Enter elements of second vector:" << endl;

        for (int i = 0; i < n; i++) {
            cin >> b[i];
        }

        // CPU Verification
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
    }

    // MATRIX MULTIPLICATION
    else if (choice == 2) {

        int n;

        cout << endl;
        cout << "Enter matrix size n: ";
        cin >> n;

        int total = n * n;
        int size = total * sizeof(int);

        int *a = new int[total];
        int *b = new int[total];
        int *c = new int[total];
        int *cpu = new int[total];

        cout << endl;
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

        dim3 blocksPerGrid(
            (n + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (n + threadsPerBlock.y - 1) / threadsPerBlock.y
        );

        cudaEvent_t start, end;

        cudaEventCreate(&start);
        cudaEventCreate(&end);

        cudaEventRecord(start);

        matrixMultiplication<<<blocksPerGrid, threadsPerBlock>>>(
            d_a, d_b, d_c, n
        );

        cudaEventRecord(end);
        cudaEventSynchronize(end);

        float time = 0;

        cudaEventElapsedTime(&time, start, end);

        cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

        // CPU Verification
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
    }

    else {

        cout << "Invalid Choice!" << endl;
    }

    return 0;
}



!nvcc cuda_program.cu -o cuda_program

!printf "1\n5\n1 2 3 4 5\n10 20 30 40 50\n" | ./cuda_program

!printf "2\n3\n1 2 3\n4 5 6\n7 8 9\n1 0 0\n0 1 0\n0 0 1\n" | ./cuda_program


  
