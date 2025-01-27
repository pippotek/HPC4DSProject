# HPC4DS Project

This repository contains the code for the High Performance for Data Science project by Filippo Costamagna and Stefano Romeo. In this project  we investigate the parallelization of the PageRank algorithm using different olutions. The aim is to explore how different levels of parallelism affect the efficiency and scalability of the algorithm when applied to very large graphs.

## Implementations

### 1. **Sequential Implementation**

- **File:** `pagerank_sequential.c`
- **Description:**
  - A straightforward single-threaded implementation of the PageRank algorithm.
  - Processes the graph iteratively and calculates the rank of each node until convergence.
  - **Key Features:**
    - Handles dangling nodes.
    - Outputs the top 10 nodes with the highest PageRank.
    - Suitable for smaller datasets due to lack of parallelism.
- **Use Case:** Baseline implementation to compare against parallelized approaches.

---

### 2. **Basic Multithreading Implementation**

- **File:** `pagerank_multithread.c`
- **Description:**
  - Implements the PageRank algorithm using **OpenMP** for basic multithreading.
  - Parallelizes key computations such as:
    - Initialization of ranks.
    - Accumulation of rank contributions.
    - Handling dangling nodes.
  - **Key Features:**
    - Utilizes OpenMP's shared memory model.
    - Efficiently divides work across threads with simple parallel loops.
    - Supports thread-safe operations for shared data updates (e.g., atomic operations).
  - **Advantages:**
    - Significant speedup for large datasets compared to the sequential version.
    - Minimal code complexity, making it easier to understand and adapt.
  - **Limitations:**
    - Threads directly update shared memory, which may cause contention.

---

### 3. **Optimized Multithreading with Local Accumulation**

- **File:** `pagerank_multithread_optimized.c`
- **Description:**
  - Extends the basic multithreaded version by introducing **thread-local storage** for intermediate contributions, reducing contention.
  - Thread-local contributions are merged into the final rank array after parallel computations.
  - **Key Features:**
    - Uses OpenMP for parallelism with improved load balancing.
    - Local contributions reduce contention for shared memory access during edge processing.
    - Higher performance on large datasets, particularly with high node and edge counts.
  - **Advantages:**
    - More scalable than the basic multithreaded version.
    - Handles larger graphs efficiently by reducing memory bottlenecks.
  - **Limitations:**
    - Slightly more complex implementation due to thread-local arrays and merging logic.

---

## Key Differences Among Implementations

| **Feature**      | **Sequential**  | **Basic Multithreading**                  | **Optimized Multithreading**         |
| ---------------------- | --------------------- | ----------------------------------------------- | ------------------------------------------ |
| **Parallelism**  | None                  | OpenMP parallel loops                           | OpenMP with thread-local contributions     |
| **Performance**  | Slow for large graphs | Faster, but limited by shared memory contention | Best performance due to reduced contention |
| **Complexity**   | Simple                | Moderate                                        | Higher due to local contribution merging   |
| **Memory Usage** | Minimal               | Moderate (shared arrays)                        | Higher (per-thread local arrays)           |
| **Scalability**  | Poor                  | Good                                            | Excellent                                  |

---

## Usage

### Compilation

All implementations can be compiled using `gcc`. For the multithreaded versions, ensure that OpenMP is supported on your system.

```bash
# Compile the sequential version
gcc -o pagerank_sequential pagerank_sequential.c -lm

# Compile the basic multithreading version
gcc -o pagerank_multithread pagerank_multithread.c -lm -fopenmp

# Compile the optimized multithreading version
gcc -o pagerank_multithread_optimized pagerank_multithread_optimized.c -lm -fopenmp
```
