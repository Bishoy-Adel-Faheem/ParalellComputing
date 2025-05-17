# Parallel Algorithms Implementation

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)
![OpenMP](https://img.shields.io/badge/OpenMP-4.5-red.svg)
![MPI](https://img.shields.io/badge/MPI-4.1-green.svg)

A collection of five efficient parallel algorithms implemented in C++. This project demonstrates parallel computing techniques to achieve significant performance improvements over sequential implementations.

## 📋 Table of Contents
- [Overview](#overview)
- [Algorithms](#algorithms)
  - [Parallel Quick Search](#parallel-quick-search)
  - [Parallel Bitonic Sort](#parallel-bitonic-sort)
  - [Parallel Sample Sort](#parallel-sample-sort)
  - [Parallel Radix Sort](#parallel-radix-sort)
  - [Parallel Prime Finder](#parallel-prime-finder)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Performance Benchmarks](#performance-benchmarks)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project implements five classic algorithms optimized for parallel execution using C++ with both OpenMP and MPI. Each implementation is designed to take advantage of multi-core processors and distributed computing environments to achieve significant speedups compared to their sequential counterparts. The algorithms are tested with a large dataset of approximately 1 million records to demonstrate real-world performance benefits.

### Dataset

The project includes a large input file containing approximately 1 million records used for benchmarking the search and sorting algorithms. This dataset serves as a realistic workload to demonstrate the efficiency and scalability of the parallel implementations.

## 🚀 Algorithms

### Parallel Quick Search

An optimized parallel implementation of the quick search algorithm, which distributes the search workload across multiple threads and processes. The implementation uses a hybrid approach with OpenMP for shared-memory parallelism and MPI for distributed computing, allowing efficient execution on both multi-core machines and compute clusters.

**Key features:**
- Recursive task decomposition for balanced workload
- Efficient thread management to minimize overhead
- Dynamic work stealing for improved load balancing
- MPI communication for distributed search across multiple nodes

### Parallel Bitonic Sort

A parallel implementation of the bitonic sort algorithm, which is well-suited for parallel architectures. The implementation leverages both OpenMP and MPI to efficiently sort large arrays by repeatedly merging bitonic sequences, with work distributed across multiple compute nodes.

**Key features:**
- O(log²n) parallel time complexity
- Perfect for power-of-two sized arrays
- Good locality of reference for cache efficiency
- Deterministic execution pattern regardless of input data
- MPI-based decomposition for distributed sorting

### Parallel Sample Sort

A highly scalable parallel sorting algorithm that extends the concept of quicksort to parallel environments. The implementation uses sampling techniques to determine effective splitters for data partition, and combines OpenMP for intra-node parallelism with MPI for inter-node distribution.

**Key features:**
- Irregular splitting for balanced partition sizes
- Two-phase approach: sampling and actual sorting
- Efficient for large datasets with uneven distribution
- Dynamic load balancing between worker threads
- MPI-based data distribution and result collection

### Parallel Radix Sort

A parallel implementation of radix sort that performs digit-by-digit sorting and leverages both OpenMP threads and MPI processes to handle different buckets concurrently across multiple computing nodes.

**Key features:**
- Non-comparative integer sorting algorithm
- Linear time complexity O(kn) where k is the number of digits
- Highly parallelizable bucket operations
- Efficient for large datasets with fixed-size keys
- MPI-based bucket distribution for distributed execution

### Parallel Prime Finder

A parallel algorithm that finds all prime numbers within a specified range using both OpenMP and MPI to distribute the workload across multiple threads and processes, significantly accelerating the process compared to sequential implementations.

**Key features:**
- Segmented sieve implementation for large ranges
- Even distribution of workload across threads and compute nodes
- Optimized memory usage for large prime ranges
- Thread coordination for minimal synchronization overhead
- MPI-based range partitioning for distributed execution

## 🛠️ Getting Started

### Prerequisites

- C++17 compatible compiler (GCC 7+, Clang 5+, or MSVC 19.14+)
- OpenMP support (usually included with modern C++ compilers)
- MPI implementation (MPICH 3.3+, Open MPI 4.0+, or Intel MPI)
- CMake 3.10+ (for building)
- Git (for cloning the repository)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/parallel-algorithms.git
   cd parallel-algorithms
