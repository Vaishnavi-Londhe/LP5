// Combined Program: Sequential and Parallel Bubble Sort + Merge Sort
// Measures execution time using chrono and OpenMP

#include <omp.h>
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;
using namespace std::chrono;

// ---------------- DISPLAY ARRAY ----------------
void displayArray(string message, vector<int>& nums)
{
    cout << "\t" << message << ": [";

    for (int i = 0; i < nums.size(); i++)
    {
        cout << nums[i];

        if (i != nums.size() - 1)
            cout << ", ";
    }

    cout << "]" << endl;
}

// ==========================================================
//                SEQUENTIAL BUBBLE SORT
// ==========================================================
void sequentialBubbleSort(vector<int>& nums)
{
    int n = nums.size();

    for (int i = 0; i < n - 1; i++)
    {
        for (int j = 0; j < n - i - 1; j++)
        {
            if (nums[j] > nums[j + 1])
            {
                swap(nums[j], nums[j + 1]);
            }
        }
    }
}

// ==========================================================
//                PARALLEL BUBBLE SORT
// ==========================================================
void parallelBubbleSort(vector<int>& nums, int n)
{
    for (int i = 0; i < n; i++)
    {
        int start = i % 2;

#pragma omp parallel for
        for (int j = start; j < n - 1; j += 2)
        {
            if (nums[j] > nums[j + 1])
            {
                swap(nums[j], nums[j + 1]);
            }
        }
    }
}

// ==========================================================
//                    MERGE FUNCTION
// ==========================================================
void merge(vector<int>& nums, int start, int mid, int end)
{
    int n1 = mid - start + 1;
    int n2 = end - mid;

    vector<int> left(n1), right(n2);

    for (int i = 0; i < n1; i++)
        left[i] = nums[start + i];

    for (int j = 0; j < n2; j++)
        right[j] = nums[mid + 1 + j];

    int i = 0, j = 0, k = start;

    while (i < n1 && j < n2)
    {
        if (left[i] <= right[j])
        {
            nums[k++] = left[i++];
        }
        else
        {
            nums[k++] = right[j++];
        }
    }

    while (i < n1)
    {
        nums[k++] = left[i++];
    }

    while (j < n2)
    {
        nums[k++] = right[j++];
    }
}

// ==========================================================
//                SEQUENTIAL MERGE SORT
// ==========================================================
void sequentialMergeSort(vector<int>& nums, int start, int end)
{
    if (start < end)
    {
        int mid = (start + end) / 2;

        sequentialMergeSort(nums, start, mid);
        sequentialMergeSort(nums, mid + 1, end);

        merge(nums, start, mid, end);
    }
}

// ==========================================================
//                PARALLEL MERGE SORT
// ==========================================================
void parallelMergeSort(vector<int>& nums, int start, int end)
{
    if (start < end)
    {
        int mid = (start + end) / 2;

#pragma omp parallel sections
        {
#pragma omp section
            parallelMergeSort(nums, start, mid);

#pragma omp section
            parallelMergeSort(nums, mid + 1, end);
        }

        merge(nums, start, mid, end);
    }
}

// ==========================================================
//                         MAIN
// ==========================================================
int main()
{
    int n;

    cout << "Enter number of elements: ";
    cin >> n;

    vector<int> original(n);

    cout << "Enter " << n << " elements:\n";

    for (int i = 0; i < n; i++)
    {
        cin >> original[i];
    }

    // Create copies for each algorithm
    vector<int> seqBubble = original;
    vector<int> parBubble = original;
    vector<int> seqMerge = original;
    vector<int> parMerge = original;

    // ======================================================
    //              SEQUENTIAL BUBBLE SORT
    // ======================================================
    cout << "\n===== Sequential Bubble Sort =====\n";

    displayArray("Before", seqBubble);

    auto start1 = high_resolution_clock::now();

    sequentialBubbleSort(seqBubble);

    auto end1 = high_resolution_clock::now();

    displayArray("After", seqBubble);

    auto duration1 =
        duration_cast<microseconds>(end1 - start1);

    cout << "Execution Time: "
         << duration1.count()
         << " microseconds\n";

    // ======================================================
    //              PARALLEL BUBBLE SORT
    // ======================================================
    cout << "\n===== Parallel Bubble Sort =====\n";

    displayArray("Before", parBubble);

    auto start2 = high_resolution_clock::now();

    parallelBubbleSort(parBubble, n);

    auto end2 = high_resolution_clock::now();

    displayArray("After", parBubble);

    auto duration2 =
        duration_cast<microseconds>(end2 - start2);

    cout << "Execution Time: "
         << duration2.count()
         << " microseconds\n";

    // ======================================================
    //              SEQUENTIAL MERGE SORT
    // ======================================================
    cout << "\n===== Sequential Merge Sort =====\n";

    displayArray("Before", seqMerge);

    auto start3 = high_resolution_clock::now();

    sequentialMergeSort(seqMerge, 0, n - 1);

    auto end3 = high_resolution_clock::now();

    displayArray("After", seqMerge);

    auto duration3 =
        duration_cast<microseconds>(end3 - start3);

    cout << "Execution Time: "
         << duration3.count()
         << " microseconds\n";

    // ======================================================
    //              PARALLEL MERGE SORT
    // ======================================================
    cout << "\n===== Parallel Merge Sort =====\n";

    displayArray("Before", parMerge);

    auto start4 = high_resolution_clock::now();

    parallelMergeSort(parMerge, 0, n - 1);

    auto end4 = high_resolution_clock::now();

    displayArray("After", parMerge);

    auto duration4 =
        duration_cast<microseconds>(end4 - start4);

    cout << "Execution Time: "
         << duration4.count()
         << " microseconds\n";

    return 0;
}
