// Write a program to implement Sequential Bubble Sort and Merge Sort
// and measure their performance using execution time.

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

// ---------------- SEQUENTIAL BUBBLE SORT ----------------
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

// ---------------- MERGE FUNCTION ----------------
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
            nums[k] = left[i];
            i++;
        }
        else
        {
            nums[k] = right[j];
            j++;
        }

        k++;
    }

    while (i < n1)
    {
        nums[k] = left[i];
        i++;
        k++;
    }

    while (j < n2)
    {
        nums[k] = right[j];
        j++;
        k++;
    }
}

// ---------------- SEQUENTIAL MERGE SORT ----------------
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

// ---------------- MAIN ----------------
int main()
{
    int n;

    cout << "Enter number of elements: ";
    cin >> n;

    vector<int> arr1(n), arr2(n);

    cout << "Enter " << n << " elements:\n";

    for (int i = 0; i < n; i++)
    {
        cin >> arr1[i];
        arr2[i] = arr1[i]; // Copy for merge sort
    }

    // ---------------- BUBBLE SORT ----------------
    cout << "\nSequential Bubble Sort:" << endl;

    displayArray("Before", arr1);

    auto start1 = high_resolution_clock::now();

    sequentialBubbleSort(arr1);

    auto end1 = high_resolution_clock::now();

    displayArray("After", arr1);

    auto duration1 = duration_cast<microseconds>(end1 - start1);

    cout << "Execution Time: "
         << duration1.count()
         << " microseconds\n";

    // ---------------- MERGE SORT ----------------
    cout << "\nSequential Merge Sort:" << endl;

    displayArray("Before", arr2);

    auto start2 = high_resolution_clock::now();

    sequentialMergeSort(arr2, 0, n - 1);

    auto end2 = high_resolution_clock::now();

    displayArray("After", arr2);

    auto duration2 = duration_cast<microseconds>(end2 - start2);

    cout << "Execution Time: "
         << duration2.count()
         << " microseconds\n";

    return 0;
}
