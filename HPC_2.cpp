#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

void sequentialBubbleSort(vector<int>& arr) {
    int n = arr.size();

    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

void parallelBubbleSort(vector<int>& arr) {
    int n = arr.size();

    for (int i = 0; i < n; i++) {
        int phase = i % 2;

        #pragma omp parallel for
        for (int j = phase; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

void merge(vector<int>& arr, int low, int mid, int high) {
    vector<int> temp;
    int i = low;
    int j = mid + 1;

    while (i <= mid && j <= high) {
        if (arr[i] <= arr[j]) {
            temp.push_back(arr[i]);
            i++;
        } else {
            temp.push_back(arr[j]);
            j++;
        }
    }

    while (i <= mid) {
        temp.push_back(arr[i]);
        i++;
    }

    while (j <= high) {
        temp.push_back(arr[j]);
        j++;
    }

    for (int k = 0; k < temp.size(); k++) {
        arr[low + k] = temp[k];
    }
}

void sequentialMergeSort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int mid = (low + high) / 2;

        sequentialMergeSort(arr, low, mid);
        sequentialMergeSort(arr, mid + 1, high);

        merge(arr, low, mid, high);
    }
}

void parallelMergeSort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int mid = (low + high) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                parallelMergeSort(arr, low, mid);
            }

            #pragma omp section
            {
                parallelMergeSort(arr, mid + 1, high);
            }
        }

        merge(arr, low, mid, high);
    }
}

void printArray(vector<int> arr) {
    for (int x : arr) {
        cout << x << " ";
    }
    cout << endl;
}

int main() {
    int n;

    omp_set_num_threads(4);

    cout << "Enter number of elements: ";
    cin >> n;

    vector<int> original(n);

    cout << "Enter elements:" << endl;
    for (int i = 0; i < n; i++) {
        cin >> original[i];
    }

    vector<int> arr1 = original;
    vector<int> arr2 = original;
    vector<int> arr3 = original;
    vector<int> arr4 = original;

    double start, end;

    start = omp_get_wtime();
    sequentialBubbleSort(arr1);
    end = omp_get_wtime();

    cout << endl;
    cout << "Sequential Bubble Sort Result: ";
    printArray(arr1);
    cout << "Sequential Bubble Sort Time: " << end - start << " seconds" << endl;

    start = omp_get_wtime();
    parallelBubbleSort(arr2);
    end = omp_get_wtime();

    cout << endl;
    cout << "Parallel Bubble Sort Result: ";
    printArray(arr2);
    cout << "Parallel Bubble Sort Time: " << end - start << " seconds" << endl;

    start = omp_get_wtime();
    sequentialMergeSort(arr3, 0, n - 1);
    end = omp_get_wtime();

    cout << endl;
    cout << "Sequential Merge Sort Result: ";
    printArray(arr3);
    cout << "Sequential Merge Sort Time: " << end - start << " seconds" << endl;

    start = omp_get_wtime();
    parallelMergeSort(arr4, 0, n - 1);
    end = omp_get_wtime();

    cout << endl;
    cout << "Parallel Merge Sort Result: ";
    printArray(arr4);
    cout << "Parallel Merge Sort Time: " << end - start << " seconds" << endl;

    return 0;
}
// Enter number of elements: 5
// Enter elements:
// 5 4 3 2 1

// // o/p
// Sequential Bubble Sort Result: 1 2 3 4 5
// Sequential Bubble Sort Time: 0.000001 seconds

// Parallel Bubble Sort Result: 1 2 3 4 5
// Parallel Bubble Sort Time: 0.000214 seconds

// Sequential Merge Sort Result: 1 2 3 4 5
// Sequential Merge Sort Time: 0.000004 seconds

// Parallel Merge Sort Result: 1 2 3 4 5
// Parallel Merge Sort Time: 0.000173 seconds


//run 
// g++ parallel_sort.cpp -fopenmp -o parallel_sort.exe
// parallel_sort.exe
