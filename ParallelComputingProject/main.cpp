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

// Platform-specific includes for directory operations
#ifdef _WIN32
#include <direct.h>  // For _mkdir on Windows
#define MKDIR(dir) _mkdir(dir)
#else
#include <sys/stat.h>  // For mkdir on Unix/Linux
#define MKDIR(dir) mkdir(dir, 0777)
#endif

//=============================================================================
// IO Handler
//=============================================================================

class IOHandler {
public:
    // Read data from a file specified by full path
    static std::vector<int> readInputFile(const std::string& fullPath) {
        std::vector<int> data;
        std::ifstream file(fullPath);

        if (!file) {
            std::cerr << "Error opening file: " << fullPath << std::endl;
            return data;
        }

        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            int value;
            while (iss >> value) {
                data.push_back(value);
            }
        }

        file.close();
        std::cout << "Read " << data.size() << " values from file: " << fullPath << std::endl;
        return data;
    }

    // Write data to a file specified by full path
    static void writeOutputFile(const std::string& fullPath, const std::vector<int>& data) {
        // Extract directory from path
        size_t lastSlash = fullPath.find_last_of("/\\");
        std::string directoryPath;

        if (lastSlash != std::string::npos) {
            directoryPath = fullPath.substr(0, lastSlash);
            ensureDirectoryExists(directoryPath);
        }

        std::ofstream file(fullPath);

        if (!file) {
            std::cerr << "Error opening file for writing: " << fullPath << std::endl;
            return;
        }

        for (const auto& value : data) {
            file << value << " ";
        }

        file.close();
        std::cout << "Wrote " << data.size() << " values to file: " << fullPath << std::endl;
    }

    // Write to a file in a directory with a specific filename
    static void writeOutputFile(const std::string& directory, const std::string& filename,
        const std::vector<int>& data) {
        std::string fullPath = combinePath(directory, filename);
        writeOutputFile(fullPath, data);
    }

    // Combine directory path and filename
    static std::string combinePath(const std::string& directory, const std::string& filename) {
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
    static bool ensureDirectoryExists(const std::string& directory) {
        // Check if directory already exists
        struct stat info;
        if (stat(directory.c_str(), &info) == 0) {
            return (info.st_mode & S_IFDIR) != 0;  // Return true if it's a directory
        }

        // Create the directory path recursively
        size_t position = 0;
        std::string currentPath;
        int result = 0;

        // Skip device part (e.g., C:)
        if (directory.length() >= 2 && directory[1] == ':') {
            currentPath = directory.substr(0, 2);
            position = 2;
        }

        // Create each directory in the path
        while (position < directory.length()) {
            position = directory.find_first_of("/\\", position);

            if (position == std::string::npos) {
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
                std::cerr << "Error creating directory: " << currentPath << std::endl;
                return false;
            }
            if (position == std::string::npos) {
                break;
            }
        }


        return true;
    }

    


// Visualization function
static void visualizeData(const std::vector<int>& data, const std::string& title) {
    // Simple console visualization
    std::cout << "Data Visualization: " << title << std::endl;
    std::cout << "------------------------" << std::endl;

    if (data.empty()) {
        std::cout << "No data to visualize." << std::endl;
        std::cout << "------------------------" << std::endl;
        return;
    }

    // Find min and max for scaling
    int min_val = *std::min_element(data.begin(), data.end());
    int max_val = *std::max_element(data.begin(), data.end());
    int range = max_val - min_val;

    // Display up to 20 elements with simple bar chart
    int display_limit = std::min(20, static_cast<int>(data.size()));
    for (int i = 0; i < display_limit; ++i) {
        std::cout << data[i] << ": ";

        // Normalize to 0-50 range for display
        int bar_length = 1;
        if (range > 0) {
            bar_length = 1 + (data[i] - min_val) * 49 / range;
        }

        for (int j = 0; j < bar_length; ++j) {
            std::cout << "#";
        }
        std::cout << std::endl;
    }

    if (data.size() > display_limit) {
        std::cout << "... and " << (data.size() - display_limit) << " more values" << std::endl;
    }

    std::cout << "------------------------" << std::endl;
}
};

//=============================================================================
// Timer Utility
//=============================================================================

class Timer {
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        return elapsed.count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;
};

//=============================================================================
// Algorithm 1: Quick Search
//=============================================================================

class QuickSearch {
public:
    static bool parallelSearch(const std::vector<int>& data, int search_value, int world_size) {
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        int data_size = data.size();

        // Broadcast the data size
        MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Calculate chunk size and offset
        int chunk_size = data_size / world_size;
        int remainder = data_size % world_size;

        // Each process will have at least chunk_size elements
        // The first 'remainder' processes get one extra element
        int local_size = (world_rank < remainder) ? chunk_size + 1 : chunk_size;

        // Allocate memory for local chunk
        std::vector<int> local_data(local_size);

        // Distribute data to all processes
        if (world_rank == 0) {
            // Copy master's portion
            std::copy(data.begin(), data.begin() + local_size, local_data.begin());

            // Master sends data to all other processes
            for (int i = 1; i < world_size; ++i) {
                int recipient_size = (i < remainder) ? chunk_size + 1 : chunk_size;
                int recipient_start = i * chunk_size + std::min(i, remainder);

                MPI_Send(&data[recipient_start], recipient_size, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        }
        else {
            // Receive data from master
            MPI_Status status;
            MPI_Recv(local_data.data(), local_size, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        }

        // Each process searches its chunk
        bool local_found = searchInChunk(local_data, search_value);

        // Gather results
        int global_found = 0;
        MPI_Reduce(&local_found, &global_found, 1, MPI_INT, MPI_LOR, 0, MPI_COMM_WORLD);

        return global_found != 0;
    }

private:
    static bool searchInChunk(const std::vector<int>& chunk, int search_value) {
        return std::find(chunk.begin(), chunk.end(), search_value) != chunk.end();
    }
};

//=============================================================================
// Algorithm 2: Prime Number Finding
//=============================================================================

class PrimeFinder {
public:
    static std::vector<int> findPrimes(const std::vector<int>& data, int world_size) {
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

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
        std::vector<int> local_data(local_size);

        // Distribute data to all processes
        if (world_rank == 0) {
            // Copy master's portion
            std::copy(data.begin(), data.begin() + local_size, local_data.begin());

            // Master sends data to all other processes
            for (int i = 1; i < world_size; ++i) {
                int recipient_size = (i < remainder) ? chunk_size + 1 : chunk_size;
                int recipient_start = i * chunk_size + std::min(i, remainder);

                MPI_Send(&data[recipient_start], recipient_size, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        }
        else {
            // Receive data from master
            MPI_Status status;
            MPI_Recv(local_data.data(), local_size, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        }

        // Find primes in local data
        std::vector<int> local_primes;
        for (int num : local_data) {
            if (isPrime(num)) {
                local_primes.push_back(num);
            }
        }

        // Gather results
        int local_count = local_primes.size();
        std::vector<int> all_counts(world_size);

        MPI_Gather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<int> global_primes;
        if (world_rank == 0) {
            // Calculate displacements for gathering
            std::vector<int> displacements(world_size);
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
            std::sort(global_primes.begin(), global_primes.end());
            auto last = std::unique(global_primes.begin(), global_primes.end());
            global_primes.erase(last, global_primes.end());

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
    // Find the next power of 2 that is greater than or equal to n
    static int nextPowerOfTwo(int n) {
        int power = 1;
        while (power < n) {
            power *= 2;
        }
        return power;
    }

    // Bitonic merge operation for sorting
    static void bitonicMerge(std::vector<int>& data, int low, int count, int dir, int stride) {
        if (count > 1) {
            for (int i = low; i < low + count - stride; i += stride * 2) {
                for (int j = 0; j < stride; j++) {
                    if (dir == (data[i + j] > data[i + j + stride])) {
                        std::swap(data[i + j], data[i + j + stride]);
                    }
                }
            }

            if (stride > 1) {
                bitonicMerge(data, low, count / 2, dir, stride / 2);
                bitonicMerge(data, low + count / 2, count / 2, dir, stride / 2);
            }
        }
    }

public:
    static std::vector<int> parallelSort(const std::vector<int>& data, int world_size) {
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        std::vector<int> result;

        if (world_rank == 0) {
            // Make a copy of the input data
            std::vector<int> input_data = data;
            int original_size = input_data.size();

            // Pad data to a power of 2
            int padded_size = nextPowerOfTwo(original_size);
            input_data.resize(padded_size, INT_MAX);

            // Broadcast padded size and original size to all processes
            MPI_Bcast(&padded_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&original_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

            // Calculate elements per process
            int elements_per_proc = padded_size / world_size;

            // Send chunks to worker processes
            for (int i = 1; i < world_size; i++) {
                int start_idx = i * elements_per_proc;
                MPI_Send(&input_data[start_idx], elements_per_proc, MPI_INT, i, 0, MPI_COMM_WORLD);
            }

            // Process root's own chunk
            std::vector<int> local_data(elements_per_proc);
            std::copy(input_data.begin(), input_data.begin() + elements_per_proc, local_data.begin());

            // Perform bitonic sort phases
            for (int k = 2; k <= padded_size; k *= 2) {
                // k is the size of the bitonic sequence
                // For ascending sort, dir=1
                for (int j = k / 2; j > 0; j /= 2) {
                    // j is the size of the merge

                    // Perform local bitonic merge with dir=1 (ascending)
                    bitonicMerge(local_data, 0, elements_per_proc, 1, j);

                    // Tell workers to perform their merge
                    int phase[2] = { k, j };
                    for (int i = 1; i < world_size; i++) {
                        MPI_Send(phase, 2, MPI_INT, i, 1, MPI_COMM_WORLD);
                    }

                    // Perform global exchanges if needed
                    if (j >= elements_per_proc) {
                        // Global exchanges are needed
                        std::vector<int> recv_data(elements_per_proc);

                        for (int i = 0; i < world_size; i++) {
                            int partner = i ^ (j / elements_per_proc);
                            if (i < partner && partner < world_size) {
                                if (i == 0) {
                                    // Master exchanges with partner
                                    MPI_Recv(recv_data.data(), elements_per_proc, MPI_INT, partner, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                                    // Compare and exchange based on bitonic sort rules
                                    // Calculate direction for this exchange
                                    bool ascending = ((i / (k / (2 * elements_per_proc))) % 2) == 0;
                                    for (int idx = 0; idx < elements_per_proc; idx++) {
                                        if ((local_data[idx] > recv_data[idx]) == ascending) {
                                            std::swap(local_data[idx], recv_data[idx]);
                                        }
                                    }

                                    // Send back exchanged data
                                    MPI_Send(recv_data.data(), elements_per_proc, MPI_INT, partner, 3, MPI_COMM_WORLD);
                                }
                                else {
                                    // Master coordinates exchange between worker processes
                                    std::vector<int> data_i(elements_per_proc), data_p(elements_per_proc);

                                    MPI_Recv(data_i.data(), elements_per_proc, MPI_INT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                                    MPI_Recv(data_p.data(), elements_per_proc, MPI_INT, partner, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                                    // Calculate direction for this exchange
                                    bool ascending = ((i / (k / (2 * elements_per_proc))) % 2) == 0;
                                    for (int idx = 0; idx < elements_per_proc; idx++) {
                                        if ((data_i[idx] > data_p[idx]) == ascending) {
                                            std::swap(data_i[idx], data_p[idx]);
                                        }
                                    }

                                    // Send exchanged data back
                                    MPI_Send(data_i.data(), elements_per_proc, MPI_INT, i, 3, MPI_COMM_WORLD);
                                    MPI_Send(data_p.data(), elements_per_proc, MPI_INT, partner, 3, MPI_COMM_WORLD);
                                }
                            }
                        }
                    }
                }
            }

            // Signal end of sorting phases
            int final_signal[2] = { -1, -1 };
            for (int i = 1; i < world_size; i++) {
                MPI_Send(final_signal, 2, MPI_INT, i, 1, MPI_COMM_WORLD);
            }

            // Gather all sorted chunks
            std::vector<int> gathered_data(padded_size);
            std::copy(local_data.begin(), local_data.end(), gathered_data.begin());

            for (int i = 1; i < world_size; i++) {
                int start_idx = i * elements_per_proc;
                MPI_Recv(&gathered_data[start_idx], elements_per_proc, MPI_INT, i, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            // Create final sorted result, removing INT_MAX padding values
            result.clear();
            for (int i = 0; i < padded_size && result.size() < original_size; i++) {
                if (gathered_data[i] != INT_MAX) {
                    result.push_back(gathered_data[i]);
                }
            }

            // If we don't have enough values (might happen if input has INT_MAX values)
            // Just take the first 'original_size' elements
            if (result.size() < original_size) {
                result.resize(original_size);
                std::copy(gathered_data.begin(), gathered_data.begin() + original_size, result.begin());
            }

            // Visualize results
            IOHandler::visualizeData(result, "Bitonic Sort");
        }
        else {
            // Worker process
            // Receive broadcast of sizes
            int padded_size, original_size;
            MPI_Bcast(&padded_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&original_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

            // Calculate chunk size
            int elements_per_proc = padded_size / world_size;

            // Receive local data chunk
            std::vector<int> local_data(elements_per_proc);
            MPI_Recv(local_data.data(), elements_per_proc, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Process bitonic sort phases
            while (true) {
                int phase[2];
                MPI_Status status;

                // Receive phase information or termination signal
                MPI_Recv(phase, 2, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);

                if (phase[0] == -1 && phase[1] == -1) {
                    // End of sorting, send local data back to master
                    MPI_Send(local_data.data(), elements_per_proc, MPI_INT, 0, 4, MPI_COMM_WORLD);
                    break;
                }

                int k = phase[0];
                int j = phase[1];

                // Perform local bitonic merge with dir=1 (ascending)
                bitonicMerge(local_data, 0, elements_per_proc, 1, j);

                // If global exchange is needed
                if (j >= elements_per_proc) {
                    // Send data to master for exchange
                    MPI_Send(local_data.data(), elements_per_proc, MPI_INT, 0, 2, MPI_COMM_WORLD);

                    // Receive exchanged data
                    MPI_Recv(local_data.data(), elements_per_proc, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
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
    static std::vector<int> parallelSort(const std::vector<int>& data, int world_size) {
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        std::vector<int> result;

        if (world_rank == 0) {
            // Find maximum number to know number of digits
            int max_num = getMax(data);

            // Calculate chunk size
            int chunk_size = data.size() / world_size;
            int remainder = data.size() % world_size;

            // Distribute data
            std::vector<int> local_chunk;
            std::vector<std::vector<int>> chunks(world_size);

            // Prepare chunks
            for (int i = 0; i < world_size; i++) {
                int start = i * chunk_size + std::min(i, remainder);
                int end = (i + 1) * chunk_size + std::min(i + 1, remainder);
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
                // Count sort on each digit
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

            // Gather all sorted chunks
            std::vector<int> sorted_chunks_sizes(world_size);
            sorted_chunks_sizes[0] = local_chunk.size();

            // Calculate total size and displacements
            int total_size = local_chunk.size();
            for (int i = 1; i < world_size; i++) {
                int size;
                MPI_Recv(&size, 1, MPI_INT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                sorted_chunks_sizes[i] = size;
                total_size += size;
            }

            // Resize result vector and copy master's chunk
            result.resize(total_size);
            std::copy(local_chunk.begin(), local_chunk.end(), result.begin());

            // Receive and add workers' chunks
            int offset = local_chunk.size();
            for (int i = 1; i < world_size; i++) {
                std::vector<int> worker_chunk(sorted_chunks_sizes[i]);
                MPI_Recv(worker_chunk.data(), sorted_chunks_sizes[i], MPI_INT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::copy(worker_chunk.begin(), worker_chunk.end(), result.begin() + offset);
                offset += sorted_chunks_sizes[i];
            }

            // Final merge sort to combine all sorted chunks
            result = mergeSort(result);

            // Visualize results
            IOHandler::visualizeData(result, "Radix Sort");
        }
        else {
            // Worker process
            // Receive chunk
            int chunk_size;
            MPI_Recv(&chunk_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            std::vector<int> local_chunk(chunk_size);
            MPI_Recv(local_chunk.data(), chunk_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Receive max number
            int max_num;
            MPI_Recv(&max_num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Process radix sort
            int exp;
            while (true) {
                MPI_Recv(&exp, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (exp == -1) break; // End signal

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
    static int getMax(const std::vector<int>& data) {
        if (data.empty()) return 0;
        return *std::max_element(data.begin(), data.end());
    }

    static void countSort(std::vector<int>& arr, int exp) {
        int n = arr.size();
        std::vector<int> output(n);
        std::vector<int> count(10, 0);

        // Store count of occurrences in count[]
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
    static void merge(std::vector<int>& arr, int l, int m, int r) {
        int n1 = m - l + 1;
        int n2 = r - m;

        std::vector<int> L(n1), R(n2);

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
    static void mergeSortRecursive(std::vector<int>& arr, int l, int r) {
        if (l < r) {
            int m = l + (r - l) / 2;
            mergeSortRecursive(arr, l, m);
            mergeSortRecursive(arr, m + 1, r);
            merge(arr, l, m, r);
        }
    }

    // Wrapper to call mergeSortRecursive
    static std::vector<int> mergeSort(std::vector<int> arr) {
        mergeSortRecursive(arr, 0, arr.size() - 1);
        return arr;
    }
};

//=============================================================================
// Algorithm 5: Sample Sort
//=============================================================================

class SampleSort {
public:
    static std::vector<int> parallelSort(const std::vector<int>& data, int world_size) {
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        std::vector<int> result;
        int data_size = data.size();

        // Broadcast data size
        MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Calculate local chunk size
        int chunk_size = data_size / world_size;
        int remainder = data_size % world_size;
        int local_size = (world_rank < remainder) ? chunk_size + 1 : chunk_size;

        // Allocate memory for local chunk
        std::vector<int> local_chunk(local_size);

        // Distribute data
        if (world_rank == 0) {
            // Copy local chunk for master
            std::copy(data.begin(), data.begin() + local_size, local_chunk.begin());

            // Send chunks to workers
            for (int i = 1; i < world_size; i++) {
                int recv_size = (i < remainder) ? chunk_size + 1 : chunk_size;
                int recv_start = i * chunk_size + std::min(i, remainder);

                MPI_Send(&data[recv_start], recv_size, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        }
        else {
            // Receive local chunk
            MPI_Recv(local_chunk.data(), local_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Sort local chunk
        std::sort(local_chunk.begin(), local_chunk.end());

        // Choose regular samples from each process
        int samples_per_proc = world_size - 1;
        std::vector<int> local_samples;

        // Ensure we have enough elements to sample
        if (local_size > 1) {
            local_samples.resize(std::min(samples_per_proc, local_size - 1));
            double step = static_cast<double>(local_size - 1) / (local_samples.size() + 1);
            for (int i = 0; i < local_samples.size(); i++) {
                int idx = static_cast<int>((i + 1) * step + 0.5);
                local_samples[i] = local_chunk[idx];
            }
        }
        else if (local_size == 1) {
            local_samples = { local_chunk[0] };
        }

        // Gather all samples at rank 0
        std::vector<int> all_samples;

        if (world_rank == 0) {
            // First, gather the sizes of samples
            std::vector<int> sample_sizes(world_size);
            int local_sample_size = local_samples.size();

            MPI_Gather(&local_sample_size, 1, MPI_INT, sample_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

            // Calculate displacements and total size
            std::vector<int> displacements(world_size);
            int total_samples = 0;
            for (int i = 0; i < world_size; i++) {
                displacements[i] = total_samples;
                total_samples += sample_sizes[i];
            }

            // Resize all_samples to hold all samples
            all_samples.resize(total_samples);

            // Gather all samples
            MPI_Gatherv(local_samples.data(), local_samples.size(), MPI_INT,
                all_samples.data(), sample_sizes.data(), displacements.data(),
                MPI_INT, 0, MPI_COMM_WORLD);

            // Sort all samples
            std::sort(all_samples.begin(), all_samples.end());

            // Select pivots (world_size-1 pivots)
            std::vector<int> pivots;
            if (all_samples.size() >= world_size - 1) {
                pivots.resize(world_size - 1);
                double step = static_cast<double>(all_samples.size()) / world_size;
                for (int i = 0; i < world_size - 1; i++) {
                    int idx = static_cast<int>((i + 1) * step + 0.5);
                    pivots[i] = all_samples[idx];
                }
            }
            else {
                // Handle edge case: not enough samples
                pivots = all_samples;
            }

            // Broadcast pivot size and pivots to all processes
            int pivot_size = pivots.size();
            MPI_Bcast(&pivot_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (pivot_size > 0) {
                MPI_Bcast(pivots.data(), pivot_size, MPI_INT, 0, MPI_COMM_WORLD);
            }

            // Partition local data based on pivots
            std::vector<std::vector<int>> partitions(world_size);

            for (int value : local_chunk) {
                int partition_idx = 0;
                while (partition_idx < pivot_size && value > pivots[partition_idx]) {
                    partition_idx++;
                }
                partitions[partition_idx].push_back(value);
            }

            // Exchange partitions among processes
            // Send sizes and data to each process
            for (int i = 1; i < world_size; i++) {
                for (int p = 0; p < world_size; p++) {
                    int size = partitions[p].size();
                    MPI_Send(&size, 1, MPI_INT, i, 2, MPI_COMM_WORLD);

                    if (size > 0) {
                        MPI_Send(partitions[p].data(), size, MPI_INT, i, 2, MPI_COMM_WORLD);
                    }
                }
            }

            // Receive partitions from other processes for bucket 0
            for (int i = 1; i < world_size; i++) {
                int size;
                MPI_Recv(&size, 1, MPI_INT, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (size > 0) {
                    std::vector<int> received(size);
                    MPI_Recv(received.data(), size, MPI_INT, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    partitions[0].insert(partitions[0].end(), received.begin(), received.end());
                }
            }

            // Sort the local bucket
            std::sort(partitions[0].begin(), partitions[0].end());

            // Gather all sorted buckets
            std::vector<int> bucket_sizes(world_size);
            bucket_sizes[0] = partitions[0].size();

            for (int i = 1; i < world_size; i++) {
                MPI_Recv(&bucket_sizes[i], 1, MPI_INT, i, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            // Calculate total size and prepare result vector
            int total_size = 0;
            for (int size : bucket_sizes) {
                total_size += size;
            }

            result.resize(total_size);

            // Copy master's bucket
            std::copy(partitions[0].begin(), partitions[0].end(), result.begin());

            // Receive sorted buckets from workers
            int offset = bucket_sizes[0];
            for (int i = 1; i < world_size; i++) {
                if (bucket_sizes[i] > 0) {
                    MPI_Recv(&result[offset], bucket_sizes[i], MPI_INT, i, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    offset += bucket_sizes[i];
                }
            }

            // Visualize results
            IOHandler::visualizeData(result, "Sample Sort");
        }
        else {
            // Worker processes

            // First, gather the sizes of samples
            int local_sample_size = local_samples.size();
            MPI_Gather(&local_sample_size, 1, MPI_INT, nullptr, 0, MPI_INT, 0, MPI_COMM_WORLD);

            // Gather all samples
            MPI_Gatherv(local_samples.data(), local_samples.size(), MPI_INT,
                nullptr, nullptr, nullptr, MPI_INT, 0, MPI_COMM_WORLD);

            // Receive pivots from master
            int pivot_size;
            MPI_Bcast(&pivot_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

            std::vector<int> pivots(pivot_size);
            if (pivot_size > 0) {
                MPI_Bcast(pivots.data(), pivot_size, MPI_INT, 0, MPI_COMM_WORLD);
            }

            // Partition local data based on pivots
            std::vector<std::vector<int>> partitions(world_size);

            for (int value : local_chunk) {
                int partition_idx = 0;
                while (partition_idx < pivot_size && value > pivots[partition_idx]) {
                    partition_idx++;
                }
                partitions[partition_idx].push_back(value);
            }

            // Receive partitions from master and other processes
            std::vector<std::vector<int>> received_partitions(world_size);

            for (int p = 0; p < world_size; p++) {
                int size;
                MPI_Recv(&size, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (size > 0) {
                    received_partitions[p].resize(size);
                    MPI_Recv(received_partitions[p].data(), size, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }

            // Send local partitions to the corresponding processes
            for (int p = 0; p < world_size; p++) {
                if (p != world_rank) {
                    int size = partitions[p].size();
                    MPI_Send(&size, 1, MPI_INT, p, 3, MPI_COMM_WORLD);

                    if (size > 0) {
                        MPI_Send(partitions[p].data(), size, MPI_INT, p, 3, MPI_COMM_WORLD);
                    }
                }
            }

            // Add received elements for my bucket
            for (int p = 0; p < world_size; p++) {
                if (p != world_rank && !received_partitions[world_rank].empty()) {
                    partitions[world_rank].insert(
                        partitions[world_rank].end(),
                        received_partitions[world_rank].begin(),
                        received_partitions[world_rank].end()
                    );
                }
            }

            // Sort my bucket
            std::sort(partitions[world_rank].begin(), partitions[world_rank].end());

            // Send bucket size to master
            int bucket_size = partitions[world_rank].size();
            MPI_Send(&bucket_size, 1, MPI_INT, 0, 4, MPI_COMM_WORLD);

            // Send sorted bucket to master
            if (bucket_size > 0) {
                MPI_Send(partitions[world_rank].data(), bucket_size, MPI_INT, 0, 4, MPI_COMM_WORLD);
            }
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
        std::cerr << "This program requires at least 2 processes" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    // Master process (rank 0) handles user interaction
    if (world_rank == 0) {
        std::cout << "===============================================" << std::endl;
        std::cout << "Welcome to Parallel Algorithm Simulation with MPI" << std::endl;
        std::cout << "===============================================" << std::endl;
        std::cout << "Please choose an algorithm to execute: " << std::endl;
        std::cout << "01 - Quick Search" << std::endl;
        std::cout << "02 - Prime Number Finding" << std::endl;
        std::cout << "03 - Bitonic Sort" << std::endl;
        std::cout << "04 - Radix Sort" << std::endl;
        std::cout << "05 - Sample Sort" << std::endl;
        std::cout << "Enter the number of the algorithm to run: ";

        int algorithm_choice;
        std::cin >> algorithm_choice;

        // Ask for input file path
        std::cout << "Enter the full path to the input file: ";
        std::string filepath;
        std::cin >> filepath;

        // Read input data from file
        std::vector<int> data = IOHandler::readInputFile(filepath);

        if (data.empty()) {
            std::cout << "Error: Input data is empty or file could not be read." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            MPI_Finalize();
            return 1;
        }

        // Broadcast algorithm choice to all processes
        MPI_Bcast(&algorithm_choice, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Start timer
        Timer timer;
        timer.start();

        // Execute the selected algorithm
        switch (algorithm_choice) {
        case 1:
            // Quick Search
        {
            int search_value;
            std::cout << "Enter value to search: ";
            std::cin >> search_value;
            MPI_Bcast(&search_value, 1, MPI_INT, 0, MPI_COMM_WORLD);

            bool found = QuickSearch::parallelSearch(data, search_value, world_size);
            std::cout << "Value " << search_value << (found ? " found" : " not found") << std::endl;
        }
        break;

        case 2:
            // Prime Number Finding
        {
            // Modified to pass the entire data array rather than just finding upper bound
            std::vector<int> primes = PrimeFinder::findPrimes(data, world_size);
            std::cout << "Found " << primes.size() << " prime numbers in the input data" << std::endl;

            // Save results to file
            std::string output_path;
            std::cout << "Enter path to save results: ";
            std::cin >> output_path;
            IOHandler::writeOutputFile(output_path, primes);
        }
        break;


        case 3:
            // Bitonic Sort
        {
            std::vector<int> sorted_data = BitonicSort::parallelSort(data, world_size);
            std::cout << "Sorted " << sorted_data.size() << " elements using Bitonic Sort" << std::endl;

            // Save results to file
            std::string output_path;
            std::cout << "Enter path to save results: ";
            std::cin >> output_path;
            IOHandler::writeOutputFile(output_path, sorted_data);
        }
        break;

        case 4:
            // Radix Sort
        {
            std::vector<int> sorted_data = RadixSort::parallelSort(data, world_size);
            std::cout << "Sorted " << sorted_data.size() << " elements using Radix Sort" << std::endl;

            // Save results to file
            std::string output_path;
            std::cout << "Enter path to save results: ";
            std::cin >> output_path;
            IOHandler::writeOutputFile(output_path, sorted_data);
        }
        break;

        case 5:
            // Sample Sort
        {
            std::vector<int> sorted_data = SampleSort::parallelSort(data, world_size);
            std::cout << "Sorted " << sorted_data.size() << " elements using Sample Sort" << std::endl;

            // Save results to file
            std::string output_path;
            std::cout << "Enter path to save results: ";
            std::cin >> output_path;
            IOHandler::writeOutputFile(output_path, sorted_data);
        }
        break;

        default:
            std::cout << "Invalid algorithm choice!" << std::endl;
            break;
        }

        // Stop timer and display elapsed time
        double elapsed = timer.stop();
        std::cout << "Execution time: " << elapsed << " seconds" << std::endl;

    }
    else {
        // Worker processes

        // Receive algorithm choice from master
        int algorithm_choice;
        MPI_Bcast(&algorithm_choice, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Execute the selected algorithm
        switch (algorithm_choice) {
        case 1:
            // Quick Search
        {
            int search_value;
            MPI_Bcast(&search_value, 1, MPI_INT, 0, MPI_COMM_WORLD);
            QuickSearch::parallelSearch(std::vector<int>(), search_value, world_size);
        }
        break;

        case 2:
            // Prime Number Finding
        {
            PrimeFinder::findPrimes(std::vector<int>(), world_size);
        }
        break;

        break;

        case 3:
            // Bitonic Sort
            BitonicSort::parallelSort(std::vector<int>(), world_size);
            break;

        case 4:
            // Radix Sort
            RadixSort::parallelSort(std::vector<int>(), world_size);
            break;

        case 5:
            // Sample Sort
            SampleSort::parallelSort(std::vector<int>(), world_size);
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
