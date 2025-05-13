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

    static void writeOutputFile(const string& fullPath, const vector<int>& data) {


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

    static void writeOutputFile(const string& directory, const string& filename,
        const vector<int>& data) {
        string fullPath = combinePath(directory, filename);
        writeOutputFile(fullPath, data);
    }

    static string combinePath(const string& directory, const string& filename) {
        if (directory.empty()) {
            return filename;
        }

        char lastChar = directory[directory.length() - 1];
        if (lastChar == '/' || lastChar == '\\') {
            return directory + filename;
        }
        else {
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
    static int parallelSearch(const vector<int>& data, int search_value, int world_size) {
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        if (world_rank == 0) {
            cout << endl << "Distributing data across processes..." << endl;
        }

        int data_size = data.size();

        MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

        int chunk_size = data_size / world_size;
        int remainder = data_size % world_size;

        int local_size = (world_rank < remainder) ? chunk_size + 1 : chunk_size;

        int local_offset = world_rank * chunk_size + min(world_rank, remainder);

        vector<int> local_data(local_size);

        if (world_rank == 0) {
            copy(data.begin(), data.begin() + local_size, local_data.begin());

            for (int i = 1; i < world_size; ++i) {
                int recipient_size = (i < remainder) ? chunk_size + 1 : chunk_size;
                int recipient_start = i * chunk_size + min(i, remainder);

                MPI_Send(&data[recipient_start], recipient_size, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        }
        else {
            MPI_Status status;
            MPI_Recv(local_data.data(), local_size, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        }

        int local_result = searchInChunk(local_data, search_value);

        int global_result = -1;
        if (local_result != -1) {
            global_result = local_offset + local_result;
            cout << "[Process " << world_rank << "] Found value " << search_value << " at index " << global_result << endl;
        }

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
        return -1;
    }
};



//=============================================================================
// Algorithm 2: Bitonic Sort
//=============================================================================


class BitonicSort {
private:
    static int nextPowerOfTwo(int n) {
        int power = 1;
        while (power < n) {
            power *= 2;
        }
        return power;
    }

    static void ascendingSwap(int index1, int index2, vector<int>& data) {
        if (data[index2] < data[index1]) {
            swap(data[index1], data[index2]);
        }
    }

    static void descendingSwap(int index1, int index2, vector<int>& data) {
        if (data[index1] < data[index2]) {
            swap(data[index1], data[index2]);
        }
    }

    static void bitonicSortFromBitonicSequence(int startIndex, int lastIndex, int dir, vector<int>& data) {
        if (startIndex >= lastIndex) return;

        int counter = 0;
        int noOfElements = lastIndex - startIndex + 1;
        for (int j = noOfElements / 2; j > 0; j = j / 2) {
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
                    i = i + j - 1;
                }
            }
        }
    }

    static void bitonicSequenceGenerator(int startIndex, int lastIndex, vector<int>& data) {
        int noOfElements = lastIndex - startIndex + 1;
        for (int j = 2; j <= noOfElements; j = j * 2) {
#pragma omp parallel for
            for (int i = startIndex; i < startIndex + noOfElements; i = i + j) {
                int end = min(i + j - 1, startIndex + noOfElements - 1);
                if (((i / j) % 2) == 0) {
                    bitonicSortFromBitonicSequence(i, end, 1, data);
                }
                else {
                    bitonicSortFromBitonicSequence(i, end, 0, data);
                }
            }
        }
    }

public:
    static vector<int> parallelSort(const vector<int>& data, int world_size) {
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        vector<int> result;

        if (world_rank == 0) {
            cout << "[Master] Bitonic sort initiated with " << world_size << " processes." << endl;

            int power_of_two = 1;
            while (power_of_two < world_size) {
                power_of_two *= 2;
            }

            if (power_of_two != world_size) {
                cout << "[Master] Warning: Bitonic sort is optimal with power-of-two process counts." << endl;
            }

            vector<int> input_data = data;
            int original_size = input_data.size();
            int padded_size = nextPowerOfTwo(original_size);
            input_data.resize(padded_size, INT_MAX);

            MPI_Bcast(&padded_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&original_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

            int elements_per_proc = padded_size / world_size;
            if (elements_per_proc < 1) elements_per_proc = 1;

            for (int i = 1; i < world_size; i++) {
                int start_idx = i * elements_per_proc;
                int chunk_size = (start_idx < padded_size) ? min(elements_per_proc, padded_size - start_idx) : 0;
                MPI_Send(&chunk_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                if (chunk_size > 0) {
                    MPI_Send(&input_data[start_idx], chunk_size, MPI_INT, i, 0, MPI_COMM_WORLD);
                }
            }

            int root_chunk_size = min(elements_per_proc, padded_size);
            vector<int> local_data(root_chunk_size);
            copy(input_data.begin(), input_data.begin() + root_chunk_size, local_data.begin());

            if (root_chunk_size > 1) {
                bitonicSequenceGenerator(0, root_chunk_size - 1, local_data);
            }

            vector<int> gathered_data(padded_size);
            copy(local_data.begin(), local_data.end(), gathered_data.begin());

            for (int i = 1; i < world_size; i++) {
                int start_idx = i * elements_per_proc;
                if (start_idx < padded_size) {
                    int chunk_size = min(elements_per_proc, padded_size - start_idx);
                    MPI_Recv(&gathered_data[start_idx], chunk_size, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }

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
            result.assign(gathered_data.begin(), gathered_data.begin() + original_size);
            cout << "[Master] Bitonic sort completed. Final result size: " << result.size() << "." << endl;
        }
        else {
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

                MPI_Send(local_data.data(), chunk_size, MPI_INT, 0, 1, MPI_COMM_WORLD);
            }
        }

        return result;
    }
};


//=============================================================================
// Algorithm 3: Radix Sort
//=============================================================================

class RadixSort {
public:
    static vector<int> parallelSort(const vector<int>& data, int world_size) {
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        vector<int> result;

        if (world_rank == 0) {

            cout << endl << "[Master] Finding maximum value to determine digit count...\n";
            int max_num = getMax(data);
            cout << endl << "[Master] Distributing data to worker processes...\n";
            int chunk_size = data.size() / world_size;
            int remainder = data.size() % world_size;

            vector<int> local_chunk;
            vector<vector<int>> chunks(world_size);

            for (int i = 0; i < world_size; i++) {
                int start = i * chunk_size + min(i, remainder);
                int end = (i + 1) * chunk_size + min(i + 1, remainder);
                chunks[i].assign(data.begin() + start, data.begin() + end);
            }

            local_chunk = chunks[0];

            for (int i = 1; i < world_size; i++) {
                int size = chunks[i].size();
                MPI_Send(&size, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(chunks[i].data(), size, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&max_num, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            }

            for (int exp = 1; max_num / exp > 0; exp *= 10) {
                cout << "[Master] Performing count sort for digit place " << exp << "...\n";
                countSort(local_chunk, exp);

                for (int i = 1; i < world_size; i++) {
                    MPI_Send(&exp, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
                }
            }

            int end_signal = -1;
            for (int i = 1; i < world_size; i++) {
                MPI_Send(&end_signal, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            }
            cout << "[Master] Gathering sorted chunks from worker processes...\n";
            vector<int> sorted_chunks_sizes(world_size);
            sorted_chunks_sizes[0] = local_chunk.size();

            int total_size = local_chunk.size();
            for (int i = 1; i < world_size; i++) {
                int size;
                MPI_Recv(&size, 1, MPI_INT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                sorted_chunks_sizes[i] = size;
                total_size += size;
            }

            result.resize(total_size);
            copy(local_chunk.begin(), local_chunk.end(), result.begin());

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
            cout << "[Worker " << world_rank << "] Waiting to receive data chunk...\n";

            int chunk_size;
            MPI_Recv(&chunk_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            vector<int> local_chunk(chunk_size);
            MPI_Recv(local_chunk.data(), chunk_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int max_num;
            MPI_Recv(&max_num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int exp;
            while (true) {
                MPI_Recv(&exp, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (exp == -1) break;

                countSort(local_chunk, exp);
            }

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

        for (int i = 1; i < 10; i++) {
            count[i] += count[i - 1];
        }
        for (int i = n - 1; i >= 0; i--) {
            output[count[(arr[i] / exp) % 10] - 1] = arr[i];
            count[(arr[i] / exp) % 10]--;
        }

        for (int i = 0; i < n; i++) {
            arr[i] = output[i];
        }
    }

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

    static void mergeSortRecursive(vector<int>& arr, int l, int r) {
        if (l < r) {
            int m = l + (r - l) / 2;
            mergeSortRecursive(arr, l, m);
            mergeSortRecursive(arr, m + 1, r);
            merge(arr, l, m, r);
        }
    }
    static vector<int> mergeSort(vector<int> arr) {
        mergeSortRecursive(arr, 0, arr.size() - 1);
        return arr;
    }
};


//=============================================================================
// Algorithm 4: Sample Sort
//=============================================================================

class SampleSort {
public:
    static vector<int> parallelSort(const vector<int>& data, int world_size) {
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        int data_size = data.size();
        MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        int chunk_size = data_size / world_size;
        int remainder = data_size % world_size;
        int local_size = (world_rank < remainder) ? chunk_size + 1 : chunk_size;
        vector<int> local_chunk(local_size);

        if (world_rank == 0) {
            copy(data.begin(), data.begin() + local_size, local_chunk.begin());

            for (int i = 1; i < world_size; i++) {
                int recv_size = (i < remainder) ? chunk_size + 1 : chunk_size;
                int recv_start = i * chunk_size + min(i, remainder);
                MPI_Send(&data[recv_start], recv_size, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        }
        else {
            MPI_Recv(local_chunk.data(), local_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        sort(local_chunk.begin(), local_chunk.end());
        vector<int> local_samples;
        if (local_size >= world_size) {
            local_samples.resize(world_size - 1);
            double step = static_cast<double>(local_size) / world_size;

            for (int i = 1; i < world_size; i++) {
                int idx = static_cast<int>(i * step);
                local_samples[i - 1] = (idx < local_size) ? local_chunk[idx] : local_chunk.back();
            }
        }
        else {
            local_samples = local_chunk;
        }

        int local_sample_count = local_samples.size();
        vector<int> all_sample_counts(world_size);
        MPI_Gather(&local_sample_count, 1, MPI_INT,
            all_sample_counts.data(), 1, MPI_INT,
            0, MPI_COMM_WORLD);

        vector<int> all_samples, displacements;
        if (world_rank == 0) {
            displacements.resize(world_size);
            int total_samples = 0;

            for (int i = 0; i < world_size; i++) {
                displacements[i] = total_samples;
                total_samples += all_sample_counts[i];
            }

            all_samples.resize(total_samples);
        }

        MPI_Gatherv(local_samples.data(), local_sample_count, MPI_INT,
            all_samples.data(), all_sample_counts.data(), displacements.data(),
            MPI_INT, 0, MPI_COMM_WORLD);

        vector<int> pivots;
        if (world_rank == 0) {
            sort(all_samples.begin(), all_samples.end());

            pivots.resize(world_size - 1);
            double step = static_cast<double>(all_samples.size()) / world_size;

            for (int i = 1; i < world_size; i++) {
                int idx = static_cast<int>(i * step);
                pivots[i - 1] = (idx < all_samples.size()) ? all_samples[idx] : all_samples.back();
            }
        }

        int pivot_count = (world_rank == 0) ? pivots.size() : 0;
        MPI_Bcast(&pivot_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (world_rank != 0) {
            pivots.resize(pivot_count);
        }
        MPI_Bcast(pivots.data(), pivot_count, MPI_INT, 0, MPI_COMM_WORLD);

        vector<vector<int>> buckets(world_size);
        for (int value : local_chunk) {
            int bucket = 0;
            while (bucket < pivots.size() && value > pivots[bucket]) {
                bucket++;
            }
            buckets[bucket].push_back(value);
        }
        vector<int> local_bucket_sizes(world_size);
        for (int i = 0; i < world_size; i++) {
            local_bucket_sizes[i] = buckets[i].size();
        }

        vector<int> all_bucket_sizes(world_size * world_size);
        MPI_Allgather(local_bucket_sizes.data(), world_size, MPI_INT,
            all_bucket_sizes.data(), world_size, MPI_INT,
            MPI_COMM_WORLD);

        vector<int> recv_counts(world_size);
        for (int i = 0; i < world_size; i++) {
            recv_counts[i] = all_bucket_sizes[i * world_size + world_rank];
        }

        int total_recv_size = 0;
        for (int count : recv_counts) {
            total_recv_size += count;
        }

        vector<int> sdispls(world_size), rdispls(world_size);
        int send_displ = 0, recv_displ = 0;

        for (int i = 0; i < world_size; i++) {
            sdispls[i] = send_displ;
            send_displ += local_bucket_sizes[i];

            rdispls[i] = recv_displ;
            recv_displ += recv_counts[i];
        }
        vector<int> send_buffer;
        for (const auto& bucket : buckets) {
            send_buffer.insert(send_buffer.end(), bucket.begin(), bucket.end());
        }

        vector<int> recv_buffer(total_recv_size);
        MPI_Alltoallv(send_buffer.data(), local_bucket_sizes.data(), sdispls.data(), MPI_INT,
            recv_buffer.data(), recv_counts.data(), rdispls.data(), MPI_INT,
            MPI_COMM_WORLD);

        sort(recv_buffer.begin(), recv_buffer.end());

        vector<int> result;
        int local_sorted_size = recv_buffer.size();
        vector<int> sorted_sizes(world_size);

        MPI_Gather(&local_sorted_size, 1, MPI_INT,
            sorted_sizes.data(), 1, MPI_INT,
            0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            displacements.resize(world_size);
            int total_size = 0;

            for (int i = 0; i < world_size; i++) {
                displacements[i] = total_size;
                total_size += sorted_sizes[i];
            }

            result.resize(total_size);
        }

        MPI_Gatherv(recv_buffer.data(), local_sorted_size, MPI_INT,
            result.data(), sorted_sizes.data(), displacements.data(),
            MPI_INT, 0, MPI_COMM_WORLD);

        return result;
    }
};


//=============================================================================
// Algorithm 5: Parallel Prime Finder in range
//=============================================================================
class ParallelPrimeFinder {
private:
    int world_rank, world_size;
    int global_start = 1, global_end = 100000;
    std::vector<int> local_primes;
    std::vector<int> global_primes;

    bool isPrime(int num) {
        if (num < 2) return false;
        for (int i = 2; i <= std::sqrt(num); ++i)
            if (num % i == 0) return false;
        return true;
    }

    void calculateLocalRange(int& local_start, int& local_end) {
        int total_range = global_end - global_start + 1;
        int base = total_range / world_size;
        int remainder = total_range % world_size;

        local_start = global_start + world_rank * base + std::min(world_rank, remainder);
        local_end = local_start + base - 1;
        if (world_rank < remainder) local_end += 1;
    }

public:
    void setRange(int start, int end) {
        global_start = start;
        global_end = end;
    }

    void initialize() {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    }

    vector<int> run() {
        initialize();
        findLocalPrimes();
        gatherResults();

        if (world_rank == 0) {
            unordered_set<int> unique(global_primes.begin(), global_primes.end());
            vector<int> final_primes(unique.begin(), unique.end());
            sort(final_primes.begin(), final_primes.end());
            return final_primes;
        }
        return vector<int>();
    }

private:
    void findLocalPrimes() {
        int local_start, local_end;
        calculateLocalRange(local_start, local_end);

        for (int num = local_start; num <= local_end; ++num) {
            if (isPrime(num)) {
                local_primes.push_back(num);
            }
        }
    }

    void gatherResults() {
        int local_size = local_primes.size();
        std::vector<int> recvcounts(world_size);
        MPI_Gather(&local_size, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

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

        MPI_Gatherv(
            local_primes.data(), local_size, MPI_INT,
            global_primes.data(), recvcounts.data(), displs.data(), MPI_INT,
            0, MPI_COMM_WORLD
        );
    }
};


//=============================================================================
// Main Function
//=============================================================================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);


    if (world_rank == 0) {
        cout << "===============================================" << endl;
        cout << "Welcome to Parallel Algorithm Simulation with MPI" << endl;
        cout << "===============================================" << endl;
        cout << "Please choose an algorithm to execute: " << endl;
        cout << "01 - Quick Search" << endl;
        cout << "02 - Bitonic Sort" << endl;
        cout << "03 - Radix Sort" << endl;
        cout << "04 - Sample Sort" << endl;
        cout << "05 - Prime Number in Range" << endl;
        cout << "Enter the number of the algorithm to run: ";

        int algorithm_choice;
        cin >> algorithm_choice;



        MPI_Bcast(&algorithm_choice, 1, MPI_INT, 0, MPI_COMM_WORLD);

        double elapsed = 0.0;
        Timer timer;

        switch (algorithm_choice) {
        case 1:
        {
            cout << endl << "-----------------------------" << endl;
            cout << " Quick Search Selected " << endl;
            cout << "-----------------------------" << endl;

            cout << "Enter the full path to the input file: ";
            string filepath;
            cin >> filepath;

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

            MPI_Bcast(&search_value, 1, MPI_INT, 0, MPI_COMM_WORLD);

            timer.start();
            int index = QuickSearch::parallelSearch(data, search_value, world_size);
            elapsed = timer.stop();

            if (index == -1) {
                cout << endl << "Result: Value " << search_value << " not found" << endl;
            }
        }
        break;

        case 2:
        {
            cout << endl << "-----------------------------" << endl;
            cout << " Bitonic Sort Selected " << endl;
            cout << "-----------------------------" << endl;
            cout << "Enter the full path to the input file: ";
            string filepath;
            cin >> filepath;
            vector<int> data = IOHandler::readInputFile(filepath);

            if (data.empty()) {
                cout << "Error: Input data is empty or file could not be read." << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
                MPI_Finalize();
                return 1;
            }
            timer.start();
            vector<int> sorted_data = BitonicSort::parallelSort(data, world_size);
            elapsed = timer.stop();

            cout << endl << "Sorted " << sorted_data.size() << " elements using Bitonic Sort" << endl;

            string output_path;
            cout << endl << "Enter path to save results: ";
            cin >> output_path;
            IOHandler::writeOutputFile(output_path, sorted_data);
        }
        break;

        case 3:
        {
            cout << endl << "-----------------------------" << endl;
            cout << " Radix Sort Selected " << endl;
            cout << "-----------------------------" << endl;
            cout << "Enter the full path to the input file: ";
            string filepath;
            cin >> filepath;
            vector<int> data = IOHandler::readInputFile(filepath);

            if (data.empty()) {
                cout << "Error: Input data is empty or file could not be read." << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
                MPI_Finalize();
                return 1;
            }
            timer.start();
            vector<int> sorted_data = RadixSort::parallelSort(data, world_size);
            elapsed = timer.stop();

            cout << endl << "Sorted " << sorted_data.size() << " elements using Radix Sort" << endl;
            string output_path;
            cout << endl << "Enter path to save results: ";
            cin >> output_path;
            IOHandler::writeOutputFile(output_path, sorted_data);
        }
        break;

        case 4:
        {
            cout << endl << "-----------------------------" << endl;
            cout << " Sample Sort Selected " << endl;
            cout << "-----------------------------" << endl;
            cout << "Enter the full path to the input file: ";
            string filepath;
            cin >> filepath;
            vector<int> data = IOHandler::readInputFile(filepath);

            if (data.empty()) {
                cout << "Error: Input data is empty or file could not be read." << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
                MPI_Finalize();
                return 1;
            }
            timer.start();
            vector<int> sorted_data = SampleSort::parallelSort(data, world_size);
            elapsed = timer.stop();
            cout << endl << "Sorted " << sorted_data.size() << " elements using Sample Sort" << endl;
            string output_path;
            cout << endl << "Enter path to save results: ";
            cin >> output_path;
            IOHandler::writeOutputFile(output_path, sorted_data);
        }
        break;

        case 5:
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
        default:
            cout << endl << "Invalid algorithm choice!" << endl;
            break;
        }
        cout << endl << "Algorithm execution time: " << elapsed << " seconds" << endl;
    }
    else {
        int algorithm_choice;
        MPI_Bcast(&algorithm_choice, 1, MPI_INT, 0, MPI_COMM_WORLD);
        switch (algorithm_choice) {
        case 1:
        {
            int search_value;
            MPI_Bcast(&search_value, 1, MPI_INT, 0, MPI_COMM_WORLD);
            QuickSearch::parallelSearch(vector<int>(), search_value, world_size);
        }
        break;

        break;

        case 2:
            BitonicSort::parallelSort(vector<int>(), world_size);
            break;

        case 3:
            RadixSort::parallelSort(vector<int>(), world_size);
            break;

        case 4:
            SampleSort::parallelSort(vector<int>(), world_size);
            break;

        case 5:
        {
            int start_range, end_range;
            MPI_Bcast(&start_range, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&end_range, 1, MPI_INT, 0, MPI_COMM_WORLD);
            ParallelPrimeFinder finder;
            finder.setRange(start_range, end_range);
            finder.run();
        }
        break;

        default:
            break;
        }
    }

    MPI_Finalize();
    return 0;
}