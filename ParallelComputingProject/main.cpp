#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <climits>
#include <cerrno>

using namespace std;

// Platform-specific includes for directory operations

#include <direct.h>  // For _mkdir on Windows
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

        if (lastSlash != string::npos) {
            directoryPath = fullPath.substr(0, lastSlash);
            ensureDirectoryExists(directoryPath);
        }

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
#ifdef _WIN32
            return directory + "\\" + filename;
#else
            return directory + "/" + filename;
#endif
        }
    }

    // Ensure directory exists, create if needed
    static bool ensureDirectoryExists(const string& directory) {
        // Check if directory already exists
        struct stat info;
        if (stat(directory.c_str(), &info) == 0) {
            return (info.st_mode & S_IFDIR) != 0;  // Return true if it's a directory
        }

        // Create the directory path recursively
        size_t position = 0;
        string currentPath;
        int result = 0;

        // Skip device part (e.g., C:)
        if (directory.length() >= 2 && directory[1] == ':') {
            currentPath = directory.substr(0, 2);
            position = 2;
        }

        // Create each directory in the path
        while (position < directory.length()) {
            position = directory.find_first_of("/\\", position);

            if (position == string::npos) {
                // Last part of the path
                currentPath = directory;
            }
            else {
                currentPath = directory.substr(0, position++);
            }

            if (!currentPath.empty()) {
                // Skip if it's just a separator like C:\
                if (currentPath.length() <= 3 && currentPath.back() == '\\') {
                continue;
            }

            result = MKDIR(currentPath.c_str());

            if (result != 0 && errno != EEXIST) {
                cerr << "Error creating directory: " << currentPath << endl;
                return false;
            }
            if (position == string::npos) {
                break;
            }
        }


        return true;
    }

    


// Visualization function
static void visualizeData(const vector<int>& data, const string& title) {
    // Simple console visualization
    cout << "Data Visualization: " << title << endl;
    cout << "------------------------" << endl;

    if (data.empty()) {
        cout << "No data to visualize." << endl;
        cout << "------------------------" << endl;
        return;
    }

    // Find min and max for scaling
    int min_val = *min_element(data.begin(), data.end());
    int max_val = *max_element(data.begin(), data.end());
    int range = max_val - min_val;

    // Display up to 20 elements with simple bar chart
    int display_limit = min(20, static_cast<int>(data.size()));
    for (int i = 0; i < display_limit; ++i) {
        cout << data[i] << ": ";

        // Normalize to 0-50 range for display
        int bar_length = 1;
        if (range > 0) {
            bar_length = 1 + (data[i] - min_val) * 49 / range;
        }

        for (int j = 0; j < bar_length; ++j) {
            cout << "#";
        }
        cout << endl;
    }

    if (data.size() > display_limit) {
        cout << "... and " << (data.size() - display_limit) << " more values" << endl;
    }

    cout << "------------------------" << endl;
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
    static int parallelSearch(const vector<int>& data, int search_value, int world_size) {
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        // Logging
        if (world_rank == 0) {
            cout << endl << "Distributing data across processes..." << endl;
        }

        int data_size = data.size();

        // Broadcast the data size
        MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Calculate chunk size and offset
        int chunk_size = data_size / world_size;
        int remainder = data_size % world_size;

        // Each process will have at least chunk_size elements
        // The first 'remainder' processes get one extra element
        int local_size = (world_rank < remainder) ? chunk_size + 1 : chunk_size;

        // Calculate local offset for this process
        int local_offset = world_rank * chunk_size + min(world_rank, remainder);

        // Allocate memory for local chunk
        vector<int> local_data(local_size);

        // Distribute data to all processes
        if (world_rank == 0) {
            // Copy master's portion
            copy(data.begin(), data.begin() + local_size, local_data.begin());

            // Master sends data to all other processes
            for (int i = 1; i < world_size; ++i) {
                int recipient_size = (i < remainder) ? chunk_size + 1 : chunk_size;
                int recipient_start = i * chunk_size + min(i, remainder);

                MPI_Send(&data[recipient_start], recipient_size, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        }
        else {
            // Receive data from master
            MPI_Status status;
            MPI_Recv(local_data.data(), local_size, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        }

        // Each process searches its chunk and returns local index if found
        int local_result = searchInChunk(local_data, search_value);

        // Convert local index to global index if found
        int global_result = -1;
        if (local_result != -1) {
            global_result = local_offset + local_result;
            // Print which process found the value and at what index
            cout << "[Process " << world_rank << "] Found value " << search_value << " at index " << global_result << endl;

        }

        // Gather results
        int final_result = -1;
        MPI_Allreduce(&global_result, &final_result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        return final_result;
    }

private:
    static int searchInChunk(const vector<int>& chunk, int search_value) {
        auto it = find(chunk.begin(), chunk.end(), search_value);
        if (it != chunk.end()) {
            return distance(chunk.begin(), it);
        }
        return -1;  // Not found
    }
};


//=============================================================================
// Algorithm 2: Prime Number Finding
//=============================================================================

class PrimeFinder {
public:
    static vector<int> findPrimes(const vector<int>& data, int world_size) {
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        // Logging header
        if (world_rank == 0) {

            cout << endl << "Reading data and distributing to processes..." << endl;
        }
        // First broadcast the data size
        int data_size = data.size();
        MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Calculate chunk size for data distribution
        int chunk_size = data_size / world_size;
        int remainder = data_size % world_size;

        // Each process will have at least chunk_size elements
        // The first 'remainder' processes get one extra element
        int local_size = (world_rank < remainder) ? chunk_size + 1 : chunk_size;

        // Allocate memory for local chunk
        vector<int> local_data(local_size);

        // Distribute data to all processes
        if (world_rank == 0) {
            // Copy master's portion
            copy(data.begin(), data.begin() + local_size, local_data.begin());

            // Master sends data to all other processes
            for (int i = 1; i < world_size; ++i) {
                int recipient_size = (i < remainder) ? chunk_size + 1 : chunk_size;
                int recipient_start = i * chunk_size + min(i, remainder);

                MPI_Send(&data[recipient_start], recipient_size, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        }
        else {
            // Receive data from master
            MPI_Status status;
            MPI_Recv(local_data.data(), local_size, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        }


        // Logging chunk assignment
        cout << endl << "[Process " << world_rank << "] Received " << local_size << " elements." << endl;


        // Find primes in local data
        vector<int> local_primes;
        for (int num : local_data) {
            if (isPrime(num)) {
                local_primes.push_back(num);
            }
        }

        // Logging local prime results
        cout << endl << "[Process " << world_rank << "] Found " << local_primes.size() << " prime numbers." << endl;

        // Gather results
        int local_count = local_primes.size();
        vector<int> all_counts(world_size);

        MPI_Gather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        vector<int> global_primes;
        if (world_rank == 0) {
            // Calculate displacements for gathering
            vector<int> displacements(world_size);
            int total_count = 0;
            for (int i = 0; i < world_size; ++i) {
                displacements[i] = total_count;
                total_count += all_counts[i];
            }

            // Resize global primes vector
            global_primes.resize(total_count);

            // Gather all primes
            MPI_Gatherv(local_primes.data(), local_count, MPI_INT, global_primes.data(),
                all_counts.data(), displacements.data(), MPI_INT, 0, MPI_COMM_WORLD);

            // Remove duplicates and sort
            sort(global_primes.begin(), global_primes.end());
            auto last = unique(global_primes.begin(), global_primes.end());
            global_primes.erase(last, global_primes.end());

            // Final logging
            cout << "[Master] Total unique primes found: " << global_primes.size() << endl;

            // Visualize results
            IOHandler::visualizeData(global_primes, "Prime Numbers in Input");
        }
        else {
            // Non-root processes just send their data
            MPI_Gatherv(local_primes.data(), local_count, MPI_INT, nullptr,
                nullptr, nullptr, MPI_INT, 0, MPI_COMM_WORLD);
        }

        return global_primes;
    }

private:
    static bool isPrime(int number) {
        if (number <= 1) return false;
        if (number <= 3) return true;
        if (number % 2 == 0 || number % 3 == 0) return false;

        for (int i = 5; i * i <= number; i += 6) {
            if (number % i == 0 || number % (i + 2) == 0)
                return false;
        }

        return true;
    }
};

//=============================================================================
// Algorithm 3: Bitonic Sort
//=============================================================================


class BitonicSort {
private:
    // Function to find the next power of two greater than or equal to n
    static int nextPowerOfTwo(int n) {
        int power = 1;
        while (power < n) {
            power *= 2;
        }
        return power;
    }

    // Swap elements in ascending order
    static void ascendingSwap(int index1, int index2, vector<int>& data) {
        if (data[index2] < data[index1]) {
            swap(data[index1], data[index2]);
        }
    }

    // Swap elements in descending order
    static void descendingSwap(int index1, int index2, vector<int>& data) {
        if (data[index1] < data[index2]) {
            swap(data[index1], data[index2]);
        }
    }

    // Perform Bitonic Sort on a bitonic sequence
    static void bitonicSortFromBitonicSequence(int startIndex, int lastIndex, int dir, vector<int>& data) {
        if (startIndex >= lastIndex) return; // Handle edge case

        int counter = 0;
        int noOfElements = lastIndex - startIndex + 1;
        for (int j = noOfElements / 2; j > 0; j = j / 2) {
            counter = 0;
            for (int i = startIndex; i + j <= lastIndex; i++) {
                if (counter < j) {
                        // Ensure we don't access out of bounds
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
                    i = i + j - 1;
                }
            }
        }
    }

    // Generate bitonic sequence
    static void bitonicSequenceGenerator(int startIndex, int lastIndex, vector<int>& data) {
        int noOfElements = lastIndex - startIndex + 1;
        for (int j = 2; j <= noOfElements; j = j * 2) {
#pragma omp parallel for
            for (int i = startIndex; i < startIndex + noOfElements; i = i + j) {
                int end = min(i + j - 1, startIndex + noOfElements - 1);
                if (((i / j) % 2) == 0) {
                    bitonicSortFromBitonicSequence(i, end, 1, data); // Ascending
                }
                else {
                    bitonicSortFromBitonicSequence(i, end, 0, data); // Descending
                }
            }
        }
    }

public:
    // Parallel Bitonic Sort using MPI
    static vector<int> parallelSort(const vector<int>& data, int world_size) {
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        vector<int> result;

        if (world_rank == 0) {
            cout << "[Master] Bitonic sort initiated with " << world_size << " processes." << endl;

            // Ensure world size is a power of two
            int power_of_two = 1;
            while (power_of_two < world_size) {
                power_of_two *= 2;
            }

            if (power_of_two != world_size) {
                cout << "[Master] Warning: Bitonic sort is optimal with power-of-two process counts." << endl;
            }

            vector<int> input_data = data;
            int original_size = input_data.size();
            int padded_size = nextPowerOfTwo(original_size); // Pad to power of 2
            input_data.resize(padded_size, INT_MAX); // Pad with max int values

            // Broadcast padded and original size
            MPI_Bcast(&padded_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&original_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

            // Distribute elements equally
            int elements_per_proc = padded_size / world_size;
            if (elements_per_proc < 1) {
                elements_per_proc = 1;
                cout << "[Master] Warning: Data size is smaller than number of processes." << endl;
            }

            for (int i = 1; i < world_size; i++) {
                int start_idx = i * elements_per_proc;
                int chunk_size = (start_idx < padded_size) ? min(elements_per_proc, padded_size - start_idx) : 0;
                MPI_Send(&chunk_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                if (chunk_size > 0) {
                    MPI_Send(&input_data[start_idx], chunk_size, MPI_INT, i, 0, MPI_COMM_WORLD);
                    cout << "[Master] Sent chunk of size " << chunk_size << " to process " << i << "." << endl;
                }
            }

            // Master process sorts its own chunk
            int root_chunk_size = min(elements_per_proc, padded_size);
            vector<int> local_data(root_chunk_size);
            copy(input_data.begin(), input_data.begin() + root_chunk_size, local_data.begin());
            cout << "[Master] Sorting local chunk of size " << root_chunk_size << "." << endl;

            if (root_chunk_size > 1) {
                bitonicSequenceGenerator(0, root_chunk_size - 1, local_data);
            }

            vector<int> gathered_data(padded_size);
            copy(local_data.begin(), local_data.end(), gathered_data.begin());

            // Receive sorted chunks from other processes
            for (int i = 1; i < world_size; i++) {
                int start_idx = i * elements_per_proc;
                if (start_idx < padded_size) {
                    int chunk_size = min(elements_per_proc, padded_size - start_idx);
                    MPI_Recv(&gathered_data[start_idx], chunk_size, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    cout << "[Master] Received sorted chunk of size " << chunk_size << " from process " << i << "." << endl;
                }
            }

            // Final bitonic merging
            vector<int> merged_data = gathered_data;
            for (int j = world_size; j <= padded_size; j *= 2) {
                for (int i = 0; i < padded_size; i += j) {
                    int end = min(i + j - 1, padded_size - 1);
                    int dir = ((i / j) % 2 == 0) ? 1 : 0; // Ascending or Descending
                    if (end >= i) {
                        bitonicSortFromBitonicSequence(i, end, dir, merged_data);
                    }
                }
            }

            // Remove padded INT_MAX values
            result.clear();
            for (int i = 0; i < padded_size && result.size() < original_size; i++) {
                if (merged_data[i] != INT_MAX) {
                    result.push_back(merged_data[i]);
                }
            }

            // Edge case: fallback if trimming fails
            if (result.size() < original_size) {
                result.resize(original_size);
                copy(merged_data.begin(), merged_data.begin() + original_size, result.begin());
            }

            cout << "[Master] Bitonic sort completed. Final result size: " << result.size() << "." << endl;
            IOHandler::visualizeData(result, "Bitonic Sort");
        }
        else {
            // Other processes receive metadata
            int padded_size, original_size;
            MPI_Bcast(&padded_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&original_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

            // Receive chunk
            int chunk_size;
            MPI_Recv(&chunk_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cout << "[Process " << world_rank << "] Received chunk size: " << chunk_size << "." << endl;

            if (chunk_size > 0) {
                vector<int> local_data(chunk_size);
                MPI_Recv(local_data.data(), chunk_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Local sorting
                cout << "[Process " << world_rank << "] Performing local sort on received chunk." << endl;
                if (chunk_size > 1) {
                    bitonicSequenceGenerator(0, chunk_size - 1, local_data);
                }

                // Send back sorted data
                MPI_Send(local_data.data(), chunk_size, MPI_INT, 0, 1, MPI_COMM_WORLD);
                cout << "[Process " << world_rank << "] Sent sorted chunk back to master." << endl;
            }
        }

        return result;
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
                cout << "[Master] Performing count sort for digit place " << exp << "...\n";
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

            // Visualize results
            IOHandler::visualizeData(result, "Radix Sort");
        }
        else {
            cout << "[Worker " << world_rank << "] Waiting to receive data chunk...\n";

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

            // Visualize results
            IOHandler::visualizeData(result, "Sample Sort");
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

    // Check if we have at least 2 processes
    if (world_size < 2) {
        cerr << "This program requires at least 2 processes" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

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

            int search_value;
            cout << endl << "Enter value to search: ";
            cin >> search_value;

            // Broadcast search value before timing starts
            MPI_Bcast(&search_value, 1, MPI_INT, 0, MPI_COMM_WORLD);

            // Start timing only for algorithm execution
            timer.start();
            int index = QuickSearch::parallelSearch(data, search_value, world_size);
            elapsed = timer.stop();

            if (index == -1) {
                cout << endl << "Result: Value " << search_value << " not found" << endl;
            }
       
        }
        break;


        case 2: // Prime Number Finding
        {
            cout << endl << "-----------------------------" << endl;
            cout << " Prime Number Selected " << endl;
            cout << "-----------------------------" << endl;
            // Start timer just before algorithm execution
            timer.start();
            vector<int> primes = PrimeFinder::findPrimes(data, world_size);
            elapsed = timer.stop();

            cout << endl << "Found " << primes.size() << " prime numbers in the input data" << endl;

            // Save results to file - AFTER TIMING ENDS
            string output_path;
            cout << endl << "Enter path to save results: ";
            cin >> output_path;
            IOHandler::writeOutputFile(output_path, primes);
        }
        break;

        case 3: // Bitonic Sort
        {
            cout << endl << "-----------------------------" << endl;
            cout << " Bitonic Sort Selected " << endl;
            cout << "-----------------------------" << endl;

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
            PrimeFinder::findPrimes(vector<int>(), world_size);
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
