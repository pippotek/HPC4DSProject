# HPC4DS Project

This repository contains the project code for the **High-Performance Computing for Data Science (HPC4DS)** course **@UniTN** by **Filippo Costamagna** and **Stefano Romeo**. The project investigates the parallelization of the **PageRank** algorithm using different parallel computing solutions. The goal is to analyze how different levels of parallelism affect the efficiency and scalability of the algorithm when applied to very large graphs.

## Implementations

### 1. **Basic Multithreading Implementation**

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
  - Significant speedup for large datasets compared to sequential execution.
  - Minimal code complexity, making it easy to understand and adapt.
- **Limitations:**
  - Threads directly update shared memory, which may cause contention.

---

### 2. **Optimized Multithreading with Local Accumulation**

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

### 3. **MPI-based Distributed Parallel Implementation**

- **File:** `mpi_pagerank.c`
- **Description:**
  - Implements **distributed parallelization** of the PageRank algorithm using **MPI**.
  - Efficiently distributes computation across multiple processes to handle extremely large graphs.
- **Key Features:**
  - Uses **MPI** to distribute work across multiple nodes in a cluster.
  - Implements **edge partitioning** and **distributed rank accumulation**.
  - Reduces memory usage per node, enabling scaling to billions of nodes and edges.
  - Handles dangling nodes and ensures efficient communication across processes.
- **Advantages:**
  - Scales well across multiple machines.
  - Capable of handling extremely large datasets.
  - Reduces contention compared to shared-memory approaches.
- **Limitations:**
  - Requires MPI setup and execution on a distributed system.
  - Increased communication overhead compared to shared-memory approaches.

---

## Key Differences Among Implementations

| **Feature**         | **Basic Multithreading** | **Optimized Multithreading** | **MPI-based Implementation** |
|---------------------|-------------------------|-----------------------------|------------------------------|
| **Parallelism**     | OpenMP parallel loops   | OpenMP with local storage  | MPI for distributed memory   |
| **Performance**     | Faster than sequential  | Improved scalability       | Best for massive graphs      |
| **Complexity**      | Moderate                | Higher due to local storage | High due to communication    |
| **Memory Usage**    | Moderate (shared arrays) | Higher (per-thread local arrays) | Lower per node (distributed) |
| **Scalability**     | Good                    | Excellent                  | Excellent (multi-node)       |

---

## Usage

### Compilation

All implementations can be compiled using `gcc` (for OpenMP versions) and `mpicc` (for the MPI version).

```bash
# Compile the basic multithreaded version
gcc -o pagerank_multithread pagerank_multithread.c -lm -fopenmp

# Compile the optimized multithreaded version
gcc -o pagerank_multithread_optimized pagerank_multithread_optimized.c -lm -fopenmp

# Compile the MPI version
mpicc -O3 -o mpi_pagerank mpi_pagerank.c -lm
```

### Running

```bash
# Run the multithreaded version (basic or optimized)
./pagerank_multithread edge_list.txt

# Run the MPI-based version with 4 processes
mpirun -np 4 ./mpi_pagerank edge_list.txt
```

### Edge List Format

The input file should contain one edge per line in the format:

`<source_node> <destination_node>`

#### Example
```plaintext
0 1
1 2
2 0
```

### Output

Each implementation outputs the following information:
- **Total number of nodes and edges**
- **Time taken for convergence**
- **Top 10 nodes with the highest PageRank**

#### Example Output
```plaintext
Number of edges: 3
Number of nodes: 3
Converged after 20 iterations
Time taken to converge: 0.123456 seconds

Top 10 nodes by PageRank:
Node 2: 0.3876543210
Node 0: 0.3065432109
Node 1: 0.3058024671
```

---

## Authors
- **Filippo Costamagna**
- **Stefano Romeo**

## License
This project is licensed under the **MIT License** – see the LICENSE file for details.

---

### 🚀 Happy Computing! 🚀

