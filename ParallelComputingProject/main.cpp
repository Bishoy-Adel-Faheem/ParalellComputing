#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_set> 
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <climits>
#include <cerrno>


using namespace std;


#include <direct.h>  
#define MKDIR(dir) _mkdir(dir)


//=============================================================================
// IO Handler
//=============================================================================

class IOHandler {
public:
    // Read data from a file specified by full path
    static vector<int> readInputFile(const string& fullPath) {
        vector<int> data;
        ifstream file(fullPath);

        if (!file) {
            cerr << "Error opening file: " << fullPath << endl;
            return data;
        }

        string line;
        while (getline(file, line)) {
            istringstream iss(line);
            int value;
            while (iss >> value) {
                data.push_back(value);
            }
        }

        file.close();
        cout << "Read " << data.size() << " values from file: " << fullPath << endl;
        return data;
    }

    // Write data to a file specified by full path
    static void writeOutputFile(const string& fullPath, const vector<int>& data) {
        // Extract directory from path
        size_t lastSlash = fullPath.find_last_of("/\\");
        string directoryPath;


        ofstream file(fullPath);

        if (!file) {
            cerr << "Error opening file for writing: " << fullPath << endl;
            return;
        }

        for (const auto& value : data) {
            file << value << " ";
        }

        file.close();
        cout << "Wrote " << data.size() << " values to file: " << fullPath << endl;
    }

    // Write to a file in a directory with a specific filename
    static void writeOutputFile(const string& directory, const string& filename,
        const vector<int>& data) {
        string fullPath = combinePath(directory, filename);
        writeOutputFile(fullPath, data);
    }

    // Combine directory path and filename
    static string combinePath(const string& directory, const string& filename) {
        if (directory.empty()) {
            return filename;
        }

        char lastChar = directory[directory.length() - 1];
        if (lastChar == '/' || lastChar == '\\') {
            return directory + filename;
        }
        else {
            // Use platform-appropriate path separators

            return directory + "\\" + filename;
        }
    }

};

//=============================================================================
// Timer Utility
//=============================================================================

class Timer {
public:
    void start() {
        start_time = chrono::high_resolution_clock::now();
    }

    double stop() {
        auto end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end_time - start_time;
        return elapsed.count();
    }

private:
    chrono::high_resolution_clock::time_point start_time;
};


//=============================================================================
// Algorithm 1: Quick Search
//=============================================================================


class QuickSearch {
public:
    static vector<int> parallelSearch(const vector<int>& data, int search_value, int world_size) {
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        int data_size = data.size();

        // Logging
        if (world_rank == 0) {
            cout << endl << "[Master] Broadcasting data size...\n";
            cout << endl << "[Master] Average chunk size per process: ~" << (data_size / world_size) << " elements\n";
            cout << "[Master] Distributing data across " << world_size << " processes...\n";
        }

        // Broadcast the data size to all processes
        MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Calculate the base chunk size and remainder
        int chunk_size = data_size / world_size;
        int remainder = data_size % world_size;

        // Determine the local chunk size for this process
        int local_size = (world_rank < remainder) ? chunk_size + 1 : chunk_size;

        // Calculate the starting index in the global array for this process
        int local_offset = world_rank * chunk_size + min(world_rank, remainder);

        // Allocate memory for the local chunk
        vector<int> local_data(local_size);

        // Master process distributes the data
        if (world_rank == 0) {
            // Copy master's portion
            copy(data.begin(), data.begin() + local_size, local_data.begin());

            // Send remaining chunks to other processes
            for (int i = 1; i < world_size; ++i) {
                int recipient_size = (i < remainder) ? chunk_size + 1 : chunk_size;
                int recipient_start = i * chunk_size + min(i, remainder);

                MPI_Send(&data[recipient_start], recipient_size, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        }
        else {
            // Receive data chunk from master
            MPI_Status status;
            MPI_Recv(local_data.data(), local_size, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        }

        // Logging chunk assignment
        cout << endl << "[Process " << world_rank << "] Received " << local_size << " elements." << endl;

        // Each process finds all matching indices locally
        vector<int> local_indices = searchAllInChunk(local_data, search_value);

        // Convert local indices to global indices
        for (int& idx : local_indices) {
            idx += local_offset;
        }

        // Determine the max number of matches any process found (needed for gather)
        int local_count = local_indices.size();
        int max_count;
        MPI_Allreduce(&local_count, &max_count, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        // Pad local indices to match max_count for uniform gathering
        local_indices.resize(max_count, -1);

        // Allocate buffer on root to collect all results
        vector<int> all_indices;
        if (world_rank == 0) {
            all_indices.resize(world_size * max_count);
        }

        // Gather all results at root
        MPI_Gather(local_indices.data(), max_count, MPI_INT,
            all_indices.data(), max_count, MPI_INT, 0, MPI_COMM_WORLD);

        // Root prints final results
        if (world_rank == 0) {
            for (int rank = 0; rank < world_size; ++rank) {
                for (int j = 0; j < max_count; ++j) {
                    int value = all_indices[rank * max_count + j];
                    if (value != -1) {
                        cout << endl << "[Process " << rank << "] Found value " << search_value << " at index " << value << endl;
                    }
                }
            }

            // Collect and return all valid results
            vector<int> final_results;
            for (int value : all_indices) {
                if (value != -1) {
                    final_results.push_back(value);
                }
            }
            return final_results;
        }

        // Other processes return an empty vector
        return {};
    }

private:
    // Search for all occurrences of the search value in a chunk
    static vector<int> searchAllInChunk(const vector<int>& chunk, int search_value) {
        vector<int> indices;
        for (int i = 0; i < chunk.size(); ++i) {
            if (chunk[i] == search_value) {
                indices.push_back(i);
            }
        }
        return indices;
    }
};


//=============================================================================
// Algorithm 2: Prime Number Finding
//=============================================================================

class ParallelPrimeFinder {
private:
    // Rank of the current process and total number of processes
    int world_rank, world_size;

    // Global range to search for prime numbers
    int global_start = 1, global_end = 100000;

    // Local and global vectors to store found prime numbers
    std::vector<int> local_primes;
    std::vector<int> global_primes;

    // Check if a number is prime
    bool isPrime(int num) {
        if (num < 2) return false;
        for (int i = 2; i <= std::sqrt(num); ++i)
            if (num % i == 0) return false;
        return true;
    }

    // Calculate the range each process will be responsible for
    void calculateLocalRange(int& local_start, int& local_end) {
        int total_range = global_end - global_start + 1;

        // Base range for each process and remaining numbers
        int base = total_range / world_size;
        int remainder = total_range % world_size;

        // Calculate local start and end for each process
        local_start = global_start + world_rank * base + std::min(world_rank, remainder);
        local_end = local_start + base - 1;
        if (world_rank < remainder) local_end += 1;
    }

public:
    // Set the range of numbers to search for primes
    void setRange(int start, int end) {
        global_start = start;
        global_end = end;
    }

    // Initialize MPI environment
    void initialize() {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    }

    // Entry point for finding primes in parallel
    std::vector<int> run() {
        initialize();          // Get MPI rank and size
        findLocalPrimes();     // Each process finds primes in its range
        gatherResults();       // Gather all primes to rank 0

        // Only rank 0 processes the final result
        if (world_rank == 0) {
            // Remove duplicates and sort the final list
            std::unordered_set<int> unique(global_primes.begin(), global_primes.end());
            std::vector<int> final_primes(unique.begin(), unique.end());
            std::sort(final_primes.begin(), final_primes.end());
            return final_primes;
        }

        // Other processes return an empty vector
        return std::vector<int>();
    }

private:
    // Find prime numbers in the local range assigned to the process
    void findLocalPrimes() {
        int local_start, local_end;
        calculateLocalRange(local_start, local_end);

        // Check each number in the local range
        for (int num = local_start; num <= local_end; ++num) {
            if (isPrime(num)) {
                local_primes.push_back(num);
            }
        }
    }

    // Gather all local primes from every process to the root (rank 0)
    void gatherResults() {
        int local_size = local_primes.size();

        // Each process sends the size of its local_primes
        std::vector<int> recvcounts(world_size);
        MPI_Gather(&local_size, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Calculate displacements and total size on root process
        std::vector<int> displs(world_size);
        int total_size = 0;
        if (world_rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < world_size; ++i) {
                displs[i] = displs[i - 1] + recvcounts[i - 1];
            }
            total_size = displs[world_size - 1] + recvcounts[world_size - 1];
            global_primes.resize(total_size);
        }

        // Gather all local prime vectors into global_primes on rank 0
        MPI_Gatherv(
            local_primes.data(), local_size, MPI_INT,
            global_primes.data(), recvcounts.data(), displs.data(), MPI_INT,
            0, MPI_COMM_WORLD
        );
    }
};


//=============================================================================
// Algorithm 3: Bitonic Sort
//=============================================================================

class BitonicSort {
private:
    // Helper function to find the next power of two ≥ n
    static int nextPowerOfTwo(int n) {
        int power = 1;
        while (power < n) {
            power *= 2;
        }
        return power;
    }

    // Swap two elements if out of ascending order
    static void ascendingSwap(int index1, int index2, vector<int>& data) {
        if (data[index2] < data[index1]) {
            swap(data[index1], data[index2]);
        }
    }

    // Swap two elements if out of descending order
    static void descendingSwap(int index1, int index2, vector<int>& data) {
        if (data[index1] < data[index2]) {
            swap(data[index1], data[index2]);
        }
    }

    // Core part of bitonic sort: performs local ordering within a bitonic sequence
    static void bitonicSortFromBitonicSequence(int startIndex, int lastIndex, int dir, vector<int>& data) {
        if (startIndex >= lastIndex) return;

        int counter = 0;
        int noOfElements = lastIndex - startIndex + 1;

        // Iteratively compare and swap in pairs
        for (int j = noOfElements / 2; j > 0; j /= 2) {
            counter = 0;
            for (int i = startIndex; i + j <= lastIndex; i++) {
                if (counter < j) {
                    if (i >= 0 && i < data.size() && i + j >= 0 && i + j < data.size()) {
                        if (dir == 1)
                            ascendingSwap(i, i + j, data);
                        else
                            descendingSwap(i, i + j, data);
                    }
                    counter++;
                }
                else {
                    counter = 0;
                    i = i + j - 1; // skip next j items to avoid duplicate comparisons
                }
            }
        }
    }

    // Builds a bitonic sequence from a portion of the data
    static void bitonicSequenceGenerator(int startIndex, int lastIndex, vector<int>& data) {
        int noOfElements = lastIndex - startIndex + 1;

        // Create bitonic sequences by merging pairs of sorted subsequences
        for (int j = 2; j <= noOfElements; j *= 2) {
#pragma omp parallel for // Parallelize each merge step with OpenMP
            for (int i = startIndex; i < startIndex + noOfElements; i += j) {
                int end = min(i + j - 1, startIndex + noOfElements - 1);
                if (((i / j) % 2) == 0) {
                    bitonicSortFromBitonicSequence(i, end, 1, data);  // ascending
                }
                else {
                    bitonicSortFromBitonicSequence(i, end, 0, data);  // descending
                }
            }
        }
    }

public:
    // Main function to perform parallel bitonic sort using MPI
    static vector<int> parallelSort(const vector<int>& data, int world_size) {
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        vector<int> result;

        if (world_rank == 0) {
            cout << "[Master] Bitonic sort initiated with " << world_size << " processes." << endl;

            // Ensure number of processes is a power of 2 (Bitonic sort works best this way)
            int power_of_two = 1;
            while (power_of_two < world_size) {
                power_of_two *= 2;
            }

            if (power_of_two != world_size) {
                cout << "[Master] Warning: Bitonic sort is optimal with power-of-two process counts." << endl;
            }

            vector<int> input_data = data;
            int original_size = input_data.size();
            int padded_size = nextPowerOfTwo(original_size);  // pad to power of 2
            input_data.resize(padded_size, INT_MAX);          // pad with INT_MAX as sentinel

            // Broadcast size information to all processes
            MPI_Bcast(&padded_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&original_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

            // Calculate how many elements each process will handle
            int elements_per_proc = padded_size / world_size;
            if (elements_per_proc < 1) elements_per_proc = 1;

            // Distribute chunks of data to each worker process
            for (int i = 1; i < world_size; i++) {
                int start_idx = i * elements_per_proc;
                int chunk_size = (start_idx < padded_size) ? min(elements_per_proc, padded_size - start_idx) : 0;
                MPI_Send(&chunk_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                if (chunk_size > 0) {
                    MPI_Send(&input_data[start_idx], chunk_size, MPI_INT, i, 0, MPI_COMM_WORLD);
                }
            }

            // Master process sorts its own chunk
            int root_chunk_size = min(elements_per_proc, padded_size);
            vector<int> local_data(root_chunk_size);
            copy(input_data.begin(), input_data.begin() + root_chunk_size, local_data.begin());

            if (root_chunk_size > 1) {
                bitonicSequenceGenerator(0, root_chunk_size - 1, local_data);
            }

            // Gather sorted chunks from all processes
            vector<int> gathered_data(padded_size);
            copy(local_data.begin(), local_data.end(), gathered_data.begin());

            for (int i = 1; i < world_size; i++) {
                int start_idx = i * elements_per_proc;
                if (start_idx < padded_size) {
                    int chunk_size = min(elements_per_proc, padded_size - start_idx);
                    MPI_Recv(&gathered_data[start_idx], chunk_size, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }

            // Perform full bitonic merge on gathered data
            for (int k = 2; k <= padded_size; k *= 2) {
                for (int j = k / 2; j > 0; j /= 2) {
                    for (int i = 0; i < padded_size; i++) {
                        int l = i ^ j;
                        if (l > i) {
                            if ((i & k) == 0) {
                                if (gathered_data[i] > gathered_data[l]) {
                                    swap(gathered_data[i], gathered_data[l]);
                                }
                            }
                            else {
                                if (gathered_data[i] < gathered_data[l]) {
                                    swap(gathered_data[i], gathered_data[l]);
                                }
                            }
                        }
                    }
                }
            }

            // Remove padding and return result
            result.assign(gathered_data.begin(), gathered_data.begin() + original_size);
            cout << "[Master] Bitonic sort completed. Final result size: " << result.size() << "." << endl;
        }
        else {
            // Worker processes

            // Receive size information from master
            int padded_size, original_size;
            MPI_Bcast(&padded_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&original_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

            int chunk_size;
            MPI_Recv(&chunk_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (chunk_size > 0) {
                vector<int> local_data(chunk_size);
                MPI_Recv(local_data.data(), chunk_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (chunk_size > 1) {
                    bitonicSequenceGenerator(0, chunk_size - 1, local_data);
                }

                // Send sorted chunk back to master
                MPI_Send(local_data.data(), chunk_size, MPI_INT, 0, 1, MPI_COMM_WORLD);
            }
        }

        return result;  // only master returns the sorted vector
    }
};


//=============================================================================
// Algorithm 4: Radix Sort
//=============================================================================

class RadixSort {
public:
    static vector<int> parallelSort(const vector<int>& data, int world_size) {
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        vector<int> result;

        if (world_rank == 0) {
         
            cout << endl << "[Master] Finding maximum value to determine digit count...\n";

            // Find maximum number to know number of digits
            int max_num = getMax(data);

            cout << endl << "[Master] Distributing data to worker processes...\n";

            // Calculate chunk size
            int chunk_size = data.size() / world_size;
            int remainder = data.size() % world_size;

            // Distribute data
            vector<int> local_chunk;
            vector<vector<int>> chunks(world_size);

            // Prepare chunks
            for (int i = 0; i < world_size; i++) {
                int start = i * chunk_size + min(i, remainder);
                int end = (i + 1) * chunk_size + min(i + 1, remainder);
                chunks[i].assign(data.begin() + start, data.begin() + end);
            }

            // Keep first chunk for master
            local_chunk = chunks[0];

            // Send chunks to workers
            for (int i = 1; i < world_size; i++) {
                int size = chunks[i].size();
                MPI_Send(&size, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(chunks[i].data(), size, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&max_num, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            }

            // Process radix sort locally
            for (int exp = 1; max_num / exp > 0; exp *= 10) {
                countSort(local_chunk, exp);

                // Tell workers to do the same
                for (int i = 1; i < world_size; i++) {
                    MPI_Send(&exp, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
                }
            }

            // Signal end of processing
            int end_signal = -1;
            for (int i = 1; i < world_size; i++) {
                MPI_Send(&end_signal, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            }

            cout << "[Master] Gathering sorted chunks from worker processes...\n";

            // Gather all sorted chunks
            vector<int> sorted_chunks_sizes(world_size);
            sorted_chunks_sizes[0] = local_chunk.size();

            int total_size = local_chunk.size();
            for (int i = 1; i < world_size; i++) {
                int size;
                MPI_Recv(&size, 1, MPI_INT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                sorted_chunks_sizes[i] = size;
                total_size += size;
            }

            // Resize result vector and copy master's chunk
            result.resize(total_size);
            copy(local_chunk.begin(), local_chunk.end(), result.begin());

            // Receive and add workers' chunks
            int offset = local_chunk.size();
            for (int i = 1; i < world_size; i++) {
                vector<int> worker_chunk(sorted_chunks_sizes[i]);
                MPI_Recv(worker_chunk.data(), sorted_chunks_sizes[i], MPI_INT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                copy(worker_chunk.begin(), worker_chunk.end(), result.begin() + offset);
                offset += sorted_chunks_sizes[i];
            }

            cout << "[Master] Final merge sort on gathered chunks...\n";

            result = mergeSort(result);

        }
        else {

            // Worker process
            // Receive chunk
            int chunk_size;
            MPI_Recv(&chunk_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            vector<int> local_chunk(chunk_size);
            MPI_Recv(local_chunk.data(), chunk_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Receive max number
            int max_num;
            MPI_Recv(&max_num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Process radix sort
            int exp;
            while (true) {
                MPI_Recv(&exp, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (exp == -1) break;

                // Count sort on this digit
                countSort(local_chunk, exp);
            }

            // Send sorted chunk back
            int size = local_chunk.size();
            MPI_Send(&size, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
            MPI_Send(local_chunk.data(), size, MPI_INT, 0, 2, MPI_COMM_WORLD);
        }

        return result;
    }

private:
    static int getMax(const vector<int>& data) {
        if (data.empty()) return 0;
        return *max_element(data.begin(), data.end());
    }

    static void countSort(vector<int>& arr, int exp) {
        int n = arr.size();
        vector<int> output(n);
        vector<int> count(10, 0);

        for (int i = 0; i < n; i++) {
            count[(arr[i] / exp) % 10]++;
        }

        // Change count[i] so that count[i] contains
        // actual position of this digit in output[]
        for (int i = 1; i < 10; i++) {
            count[i] += count[i - 1];
        }

        // Build the output array
        for (int i = n - 1; i >= 0; i--) {
            output[count[(arr[i] / exp) % 10] - 1] = arr[i];
            count[(arr[i] / exp) % 10]--;
        }

        // Copy the output array to arr[]
        for (int i = 0; i < n; i++) {
            arr[i] = output[i];
        }
    }

    // Merge two sorted subarrays
    static void merge(vector<int>& arr, int l, int m, int r) {
        int n1 = m - l + 1;
        int n2 = r - m;

        vector<int> L(n1), R(n2);

        for (int i = 0; i < n1; i++)
            L[i] = arr[l + i];
        for (int j = 0; j < n2; j++)
            R[j] = arr[m + 1 + j];

        int i = 0, j = 0, k = l;
        while (i < n1 && j < n2) {
            if (L[i] <= R[j]) {
                arr[k++] = L[i++];
            }
            else {
                arr[k++] = R[j++];
            }
        }

        while (i < n1) {
            arr[k++] = L[i++];
        }

        while (j < n2) {
            arr[k++] = R[j++];
        }
    }

    // Recursive merge sort
    static void mergeSortRecursive(vector<int>& arr, int l, int r) {
        if (l < r) {
            int m = l + (r - l) / 2;
            mergeSortRecursive(arr, l, m);
            mergeSortRecursive(arr, m + 1, r);
            merge(arr, l, m, r);
        }
    }

    // Wrapper to call mergeSortRecursive
    static vector<int> mergeSort(vector<int> arr) {
        mergeSortRecursive(arr, 0, arr.size() - 1);
        return arr;
    }
};


//=============================================================================
// Algorithm 5: Sample Sort
//=============================================================================


class SampleSort {
public:
    static vector<int> parallelSort(const vector<int>& data, int world_size) {
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        vector<int> result;
        int data_size = data.size();




        MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Calculate local chunk size
        int chunk_size = data_size / world_size;
        int remainder = data_size % world_size;
        int local_size = (world_rank < remainder) ? chunk_size + 1 : chunk_size;



        // Allocate memory for local chunk
        vector<int> local_chunk(local_size);

        // Distribute data
        if (world_rank == 0) {
            cout << endl << "[Master] Broadcasting data size...\n";
            cout << endl << "[Master] Average chunk size per process: ~" << (data_size / world_size) << " elements\n";
            cout << "[Master] Distributing data across " << world_size << " processes...\n";

            // Copy local chunk for master
            copy(data.begin(), data.begin() + local_size, local_chunk.begin());

            // Send chunks to workers
            for (int i = 1; i < world_size; i++) {
                int recv_size = (i < remainder) ? chunk_size + 1 : chunk_size;
                int recv_start = i * chunk_size + min(i, remainder);

                MPI_Send(&data[recv_start], recv_size, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
            cout << endl << "[Master] Initial data distribution complete.\n";
        }
        else {
            // Receive local chunk
            MPI_Recv(local_chunk.data(), local_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Sort local chunk
        sort(local_chunk.begin(), local_chunk.end());

        if (world_rank == 0) {
            cout << "[Master] Each process has sorted its local chunk.\n";
            cout << "[Master] Selecting sample elements for pivot determination...\n";
        }

        // ---- Optimized sampling ----
        // Each process selects regular samples (world_size-1 samples)
        vector<int> local_samples;

        // Make sure we have enough elements to sample
        if (local_size >= world_size) {
            local_samples.resize(world_size - 1);
            double step = static_cast<double>(local_size) / world_size;

            for (int i = 1; i < world_size; i++) {
                int idx = static_cast<int>(i * step);
                if (idx < local_size) {
                    local_samples[i - 1] = local_chunk[idx];
                }
                else {
                    // Use the last element if index is out of bounds
                    local_samples[i - 1] = local_chunk.back();
                }
            }
        }
        else {
            // If we don't have enough elements, just use what we have
            for (int i = 0; i < local_size; i++) {
                local_samples.push_back(local_chunk[i]);
            }
        }

        // Gather all samples at rank 0
        int local_sample_count = local_samples.size();
        vector<int> all_sample_counts(world_size);

        MPI_Gather(&local_sample_count, 1, MPI_INT,
            all_sample_counts.data(), 1, MPI_INT,
            0, MPI_COMM_WORLD);

        vector<int> all_samples;
        vector<int> displacements;

        if (world_rank == 0) {
            // Calculate displacements and total samples
            displacements.resize(world_size);
            int total_samples = 0;

            for (int i = 0; i < world_size; i++) {
                displacements[i] = total_samples;
                total_samples += all_sample_counts[i];
            }

            // Resize to hold all samples
            all_samples.resize(total_samples);

            cout << "[Master] Gathering " << total_samples << " samples for pivot selection...\n";
        }

        // Gather all samples
        MPI_Gatherv(local_samples.data(), local_sample_count, MPI_INT,
            all_samples.data(), all_sample_counts.data(), displacements.data(),
            MPI_INT, 0, MPI_COMM_WORLD);

        // Select and broadcast pivots
        vector<int> pivots;

        if (world_rank == 0) {
            cout << "[Master] Selecting pivots from gathered samples...\n";

            // Sort the gathered samples
            sort(all_samples.begin(), all_samples.end());

            // Select world_size-1 evenly spaced pivots
            pivots.resize(world_size - 1);
            double step = static_cast<double>(all_samples.size()) / world_size;

            for (int i = 1; i < world_size; i++) {
                int idx = static_cast<int>(i * step);
                if (idx < all_samples.size()) {
                    pivots[i - 1] = all_samples[idx];
                }
                else {
                    pivots[i - 1] = all_samples.back();
                }
            }

            cout << "[Master] Broadcasting " << pivots.size() << " pivots to all processes...\n";
        }

        // Broadcast the number of pivots
        int pivot_count = (world_rank == 0) ? pivots.size() : 0;
        MPI_Bcast(&pivot_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Resize pivots on non-root processes and broadcast
        if (world_rank != 0) {
            pivots.resize(pivot_count);
        }

        MPI_Bcast(pivots.data(), pivot_count, MPI_INT, 0, MPI_COMM_WORLD);

        // ---- Optimized partitioning ----
        if (world_rank == 0) {
            cout << "[Master] Partitioning data based on pivots...\n";
        }

        // Each process partitions its local data
        vector<vector<int>> buckets(world_size);

        for (int value : local_chunk) {
            int bucket = 0;
            while (bucket < pivots.size() && value > pivots[bucket]) {
                bucket++;
            }
            buckets[bucket].push_back(value);
        }

        // Optimized all-to-all communication
        if (world_rank == 0) {
            cout << "[Master] Exchanging partitioned data between processes...\n";
        }

        // First, gather the sizes of all buckets from all processes
        vector<int> local_bucket_sizes(world_size);
        for (int i = 0; i < world_size; i++) {
            local_bucket_sizes[i] = buckets[i].size();
        }

        vector<int> all_bucket_sizes(world_size * world_size);

        MPI_Allgather(local_bucket_sizes.data(), world_size, MPI_INT,
            all_bucket_sizes.data(), world_size, MPI_INT,
            MPI_COMM_WORLD);

        // Each process will receive buckets from other processes
        vector<int> recv_counts(world_size);
        for (int i = 0; i < world_size; i++) {
            recv_counts[i] = all_bucket_sizes[i * world_size + world_rank];
        }

        // Calculate the total size of the data each process will receive
        int total_recv_size = 0;
        for (int count : recv_counts) {
            total_recv_size += count;
        }

        // Prepare send/receive displacements
        vector<int> sdispls(world_size), rdispls(world_size);
        int send_displ = 0, recv_displ = 0;

        for (int i = 0; i < world_size; i++) {
            sdispls[i] = send_displ;
            send_displ += local_bucket_sizes[i];

            rdispls[i] = recv_displ;
            recv_displ += recv_counts[i];
        }

        // Flatten buckets into a single send buffer
        vector<int> send_buffer;
        for (const auto& bucket : buckets) {
            send_buffer.insert(send_buffer.end(), bucket.begin(), bucket.end());
        }

        // Receive buffer for all-to-all data exchange
        vector<int> recv_buffer(total_recv_size);

        // Perform all-to-all data exchange
        MPI_Alltoallv(send_buffer.data(), local_bucket_sizes.data(), sdispls.data(), MPI_INT,
            recv_buffer.data(), recv_counts.data(), rdispls.data(), MPI_INT,
            MPI_COMM_WORLD);

        if (world_rank == 0) {
            cout << "[Master] Each process sorting its final bucket...\n";
        }

        // Sort received data
        sort(recv_buffer.begin(), recv_buffer.end());

        // Gather the sizes of sorted buckets at root
        int local_sorted_size = recv_buffer.size();
        vector<int> sorted_sizes(world_size);

        MPI_Gather(&local_sorted_size, 1, MPI_INT,
            sorted_sizes.data(), 1, MPI_INT,
            0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            // Calculate displacements for gathering final results
            displacements.resize(world_size);
            int total_size = 0;

            for (int i = 0; i < world_size; i++) {
                displacements[i] = total_size;
                total_size += sorted_sizes[i];
            }

            // Resize result vector to hold all sorted data
            result.resize(total_size);

            cout << "[Master] Gathering all sorted data to create final result...\n";
        }

        // Gather all sorted data
        MPI_Gatherv(recv_buffer.data(), local_sorted_size, MPI_INT,
            result.data(), sorted_sizes.data(), displacements.data(),
            MPI_INT, 0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            cout << "[Master] Sample Sort complete!\n";

            //// Visualize results
            //IOHandler::visualizeData(result, "Sample Sort");
        }

        return result;
    }
};



//=============================================================================
// Main Function
//=============================================================================

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);


    // Master process (rank 0) handles user interaction
    if (world_rank == 0) {
        cout << "===============================================" << endl;
        cout << "Welcome to Parallel Algorithm Simulation with MPI" << endl;
        cout << "===============================================" << endl;
        cout << "Please choose an algorithm to execute: " << endl;
        cout << "01 - Quick Search" << endl;
        cout << "02 - Prime Number Finding" << endl;
        cout << "03 - Bitonic Sort" << endl;
        cout << "04 - Radix Sort" << endl;
        cout << "05 - Sample Sort" << endl;
        cout << "Enter the number of the algorithm to run: ";

        int algorithm_choice;
        cin >> algorithm_choice;


        // Broadcast algorithm choice to all processes
        MPI_Bcast(&algorithm_choice, 1, MPI_INT, 0, MPI_COMM_WORLD);

        double elapsed = 0.0;
        Timer timer;

        // Execute the selected algorithm
        switch (algorithm_choice) {
        case 1: // Quick Search
        {
            cout << endl << "-----------------------------" << endl;
            cout << " Quick Search Selected " << endl;
            cout << "-----------------------------" << endl;

            // Ask for input file path
            cout << "Enter the full path to the input file: ";
            string filepath;
            cin >> filepath;

            // Read input data from file - BEFORE TIMING STARTS
            vector<int> data = IOHandler::readInputFile(filepath);

            if (data.empty()) {
                cout << "Error: Input data is empty or file could not be read." << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
                MPI_Finalize();
                return 1;
            }

            int search_value;
            cout << endl << "Enter value to search: ";
            cin >> search_value;

            // Broadcast search value before timing starts
            MPI_Bcast(&search_value, 1, MPI_INT, 0, MPI_COMM_WORLD);

            // Start timing only for algorithm execution
            timer.start();
            vector<int> indeces = QuickSearch::parallelSearch(data, search_value, world_size);
            elapsed = timer.stop();

            if (indeces.empty()) {
                cout << endl << "Result: Value " << search_value << " not found" << endl;
            }
            else
            {
				cout << endl << "Result: Value " << search_value << " found at indices: ";
				for (int index : indeces) {
					cout << index << " ";
				}
				cout << endl;
            }
       
        }
        break;


        case 2: // Prime Number Finding
        {
            cout << endl << "-----------------------------" << endl;
            cout << " Prime Number in Range Selected " << endl;
            cout << "-----------------------------" << endl;
            int start_range, end_range;
            if (world_rank == 0) {
                cout << "Enter start of range: ";
                cin >> start_range;
                cout << "Enter end of range: ";
                cin >> end_range;
                if (start_range > end_range) {
                    swap(start_range, end_range);
                }
            }
            MPI_Bcast(&start_range, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&end_range, 1, MPI_INT, 0, MPI_COMM_WORLD);

            timer.start();
            ParallelPrimeFinder finder;
            finder.setRange(start_range, end_range);
            vector<int> primes = finder.run();
            elapsed = timer.stop();

            if (world_rank == 0) {
                string output_path;
                cout << endl << "Enter path to save results: ";
                cin >> output_path;
                IOHandler::writeOutputFile(output_path, primes);

                cout << "\nFound " << primes.size() << " prime numbers in range ["
                    << start_range << ", " << end_range << "]" << endl;
            }
        
        }
        break;

        case 3: // Bitonic Sort
        {
            cout << endl << "-----------------------------" << endl;
            cout << " Bitonic Sort Selected " << endl;
            cout << "-----------------------------" << endl;

            // Ask for input file path
            cout << "Enter the full path to the input file: ";
            string filepath;
            cin >> filepath;

            // Read input data from file - BEFORE TIMING STARTS
            vector<int> data = IOHandler::readInputFile(filepath);

            if (data.empty()) {
                cout << "Error: Input data is empty or file could not be read." << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
                MPI_Finalize();
                return 1;
            }

            // Start timer just before algorithm execution
            timer.start();
            vector<int> sorted_data = BitonicSort::parallelSort(data, world_size);
            elapsed = timer.stop();

            cout << endl << "Sorted " << sorted_data.size() << " elements using Bitonic Sort" << endl;

            // Save results to file - AFTER TIMING ENDS
            string output_path;
            cout << endl << "Enter path to save results: ";
            cin >> output_path;
            IOHandler::writeOutputFile(output_path, sorted_data);
        }
        break;

        case 4: // Radix Sort
        {
            cout << endl << "-----------------------------" << endl;
            cout << " Radix Sort Selected " << endl;
            cout << "-----------------------------" << endl;


            // Ask for input file path
            cout << "Enter the full path to the input file: ";
            string filepath;
            cin >> filepath;

            // Read input data from file - BEFORE TIMING STARTS
            vector<int> data = IOHandler::readInputFile(filepath);

            if (data.empty()) {
                cout << "Error: Input data is empty or file could not be read." << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
                MPI_Finalize();
                return 1;
            }

            // Start timer just before algorithm execution
            timer.start();
            vector<int> sorted_data = RadixSort::parallelSort(data, world_size);
            elapsed = timer.stop();

            cout << endl << "Sorted " << sorted_data.size() << " elements using Radix Sort" << endl;

            // Save results to file - AFTER TIMING ENDS
            string output_path;
            cout << endl << "Enter path to save results: ";
            cin >> output_path;
            IOHandler::writeOutputFile(output_path, sorted_data);
        }
        break;

        case 5: // Sample Sort
        {
            cout << endl << "-----------------------------" << endl;
            cout << " Sample Sort Selected " << endl;
            cout << "-----------------------------" << endl;
            
            // Ask for input file path
            cout << "Enter the full path to the input file: ";
            string filepath;
            cin >> filepath;

            // Read input data from file - BEFORE TIMING STARTS
            vector<int> data = IOHandler::readInputFile(filepath);

            if (data.empty()) {
                cout << "Error: Input data is empty or file could not be read." << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
                MPI_Finalize();
                return 1;
            }
            
            // Start timer just before algorithm execution
            timer.start();
            vector<int> sorted_data = SampleSort::parallelSort(data, world_size);
            elapsed = timer.stop();

            cout << endl << "Sorted " << sorted_data.size() << " elements using Sample Sort" << endl;

            // Save results to file - AFTER TIMING ENDS
            string output_path;
            cout << endl << "Enter path to save results: ";
            cin >> output_path;
            IOHandler::writeOutputFile(output_path, sorted_data);
        }
        break;

        default:
            cout << endl << "Invalid algorithm choice!" << endl;
            break;
        }

        // Display execution time of only the algorithm
        cout << endl << "Algorithm execution time: " << elapsed << " seconds" << endl;
    }
    else {
        // Worker processes

        // Receive algorithm choice from master
        int algorithm_choice;
        MPI_Bcast(&algorithm_choice, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Execute the selected algorithm
        switch (algorithm_choice) {
        case 1: // Quick Search
        {
            int search_value;
            MPI_Bcast(&search_value, 1, MPI_INT, 0, MPI_COMM_WORLD);
            QuickSearch::parallelSearch(vector<int>(), search_value, world_size);
        }
        break;

        case 2: // Prime Number Finding
        {
            int start_range, end_range;
            MPI_Bcast(&start_range, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&end_range, 1, MPI_INT, 0, MPI_COMM_WORLD);
            ParallelPrimeFinder finder;
            finder.setRange(start_range, end_range);
            finder.run();
        }
        break;

        case 3: // Bitonic Sort
            BitonicSort::parallelSort(vector<int>(), world_size);
            break;

        case 4: // Radix Sort
            RadixSort::parallelSort(vector<int>(), world_size);
            break;

        case 5: // Sample Sort
            SampleSort::parallelSort(vector<int>(), world_size);
            break;

        default:
            // Do nothing for invalid choices
            break;
        }
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
