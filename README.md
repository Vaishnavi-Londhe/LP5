# LP5








!nvidia-smi

%%writefile cuda_program.cu

#include 
#include 

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

        vectorAdd<<>>(d_a, d_b, d_c, n);

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

        matrixMultiplication<<>>(
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






## HPC 3
#include <iostream>
#include <vector>
#include <climits>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

class ParallelReducer {
private:
    vector<int> arr;

public:
    ParallelReducer(const vector<int>& input) : arr(input) {}

    // ---------------- PARALLEL MIN ----------------
    int parallelMin() {
        int min_val = INT_MAX;

        #pragma omp parallel for reduction(min:min_val)
        for (int i = 0; i < arr.size(); i++) {
            if (arr[i] < min_val)
                min_val = arr[i];
        }

        return min_val;
    }

    // ---------------- PARALLEL MAX ----------------
    int parallelMax() {
        int max_val = INT_MIN;

        #pragma omp parallel for reduction(max:max_val)
        for (int i = 0; i < arr.size(); i++) {
            if (arr[i] > max_val)
                max_val = arr[i];
        }

        return max_val;
    }

    // ---------------- PARALLEL SUM ----------------
    int parallelSum() {
        int sum = 0;

        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < arr.size(); i++) {
            sum += arr[i];
        }

        return sum;
    }

    // ---------------- PARALLEL AVERAGE ----------------
    double parallelAverage() {
        double total = parallelSum();
        return total / arr.size();
    }
};

int main() {
    int size;

    cout << "Enter size of the array: ";
    cin >> size;

    vector<int> data(size);

    cout << "Enter " << size << " elements:\n";

    for (int i = 0; i < size; i++)
        cin >> data[i];

    ParallelReducer reducer(data);

    // ---------------- MINIMUM ----------------
    auto startMin = high_resolution_clock::now();

    int minResult = reducer.parallelMin();

    auto endMin = high_resolution_clock::now();

    auto durationMin =
        duration_cast<microseconds>(endMin - startMin);

    cout << "\nParallel Minimum: "
         << minResult << endl;

    cout << "Execution Time (Min): "
         << durationMin.count()
         << " microseconds\n";

    // ---------------- MAXIMUM ----------------
    auto startMax = high_resolution_clock::now();

    int maxResult = reducer.parallelMax();

    auto endMax = high_resolution_clock::now();

    auto durationMax =
        duration_cast<microseconds>(endMax - startMax);

    cout << "\nParallel Maximum: "
         << maxResult << endl;

    cout << "Execution Time (Max): "
         << durationMax.count()
         << " microseconds\n";

    // ---------------- SUM ----------------
    auto startSum = high_resolution_clock::now();

    int sumResult = reducer.parallelSum();

    auto endSum = high_resolution_clock::now();

    auto durationSum =
        duration_cast<microseconds>(endSum - startSum);

    cout << "\nParallel Sum: "
         << sumResult << endl;

    cout << "Execution Time (Sum): "
         << durationSum.count()
         << " microseconds\n";

    // ---------------- AVERAGE ----------------
    auto startAvg = high_resolution_clock::now();

    double avgResult = reducer.parallelAverage();

    auto endAvg = high_resolution_clock::now();

    auto durationAvg =
        duration_cast<microseconds>(endAvg - startAvg);

    cout << "\nParallel Average: "
         << avgResult << endl;

    cout << "Execution Time (Average): "
         << durationAvg.count()
         << " microseconds\n";

    return 0;
}
