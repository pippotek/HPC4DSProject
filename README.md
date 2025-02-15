# HPC4DS Project

This repository contains the code for the **High-Performance Computing for Data Science (HPC4DS)** project by **Filippo Costamagna** and **Stefano Romeo**. The project investigates the parallelization of the **PageRank** algorithm using different parallel computing solutions. The goal is to analyze how different levels of parallelism affect the efficiency and scalability of the algorithm when applied to very large graphs.

## Overview

The **PageRank algorithm**, originally developed by Larry Page and Sergey Brin, is a fundamental method for ranking web pages based on link structures. The algorithm models the web as a directed graph, where nodes represent web pages, and edges signify hyperlinks. The iterative process propagates rank scores throughout the network, converging towards a steady-state distribution that reflects the importance of each node.

With the growth of the internet and social networks, single-threaded implementations of PageRank become computationally infeasible due to the vast amount of data involved. This project explores parallel implementations using **OpenMP** and **MPI**, optimizing computational performance through shared and distributed memory paradigms. 

## Implementations

### 1. **Basic Multithreading Implementation (OpenMP)**

- **File:** `pagerank_multithread.c`
- **Description:**
  - Implements the PageRank algorithm using **OpenMP** for shared-memory parallelism.
  - Key parallelized steps include:
    - Rank initialization.
    - Parallel processing of edges.
    - Handling of dangling nodes.
  - Uses atomic operations to manage shared data updates efficiently.
- **Performance Trade-offs:**
  - Improves speedup but experiences contention in shared-memory operations.
  - Effective for multi-core architectures with moderate dataset sizes.

---

### 2. **Optimized Multithreading with Local Accumulation**

- **File:** `pagerank_multithread_optimized.c`
- **Description:**
  - Improves OpenMP efficiency by introducing **thread-local storage** for rank accumulation.
  - Reduces contention by merging thread-local contributions into the final rank array at the end of each iteration.
- **Performance Trade-offs:**
  - Achieves better scalability for large graphs by reducing memory contention.
  - Higher memory overhead due to per-thread local arrays.
  - Less efficient on smaller datasets due to synchronization costs.

---

### 3. **MPI-based Distributed Parallel Implementation**

- **File:** `mpi_pagerank.c`
- **Description:**
  - Implements **distributed parallelization** using **MPI** to handle very large-scale graphs.
  - Distributes edge partitions across multiple computing nodes to optimize workload.
  - Synchronizes global rank values across processes to ensure convergence.
- **Performance Trade-offs:**
  - **Strong scalability up to 16-32 processes**, but communication overhead increases with more nodes.
  - More efficient than OpenMP for extremely large datasets but requires careful tuning of communication strategies.
  - Suitable for high-performance computing (HPC) clusters.

---

## Performance Evaluation

Experiments were conducted on the **HPC cluster at the University of Trento** using the **Friendster social network dataset**:
- **65.6 million nodes**
- **1.8 billion edges**
- **30.14 GB dataset size**

The results show:
- **OpenMP achieves good speedup up to 16-32 threads** but faces synchronization bottlenecks beyond that.
- **MPI scales well up to 32 nodes** but suffers from increasing communication overhead at higher counts.
- **Hybrid OpenMP+MPI strategies were tested but did not yield consistent improvements due to additional synchronization complexity.**

---

## Key Differences Among Implementations

| **Feature**         | **Basic Multithreading (OpenMP)** | **Optimized Multithreading (OpenMP)** | **MPI-based Implementation** |
|---------------------|---------------------------------|-----------------------------------|------------------------------|
| **Parallelism**     | OpenMP parallel loops          | OpenMP with local storage        | MPI for distributed memory   |
| **Performance**     | Faster than sequential         | More scalable with large graphs  | Best for extremely large datasets |
| **Memory Usage**    | Moderate (shared arrays)       | Higher (per-thread local arrays) | Lower per node (distributed) |
| **Scalability**     | Good                           | Excellent                         | Excellent with sufficient nodes |
| **Communication Overhead** | Low                 | Moderate                          | High                         |

---

## Compilation and Execution

### Compilation

```bash
# Compile the OpenMP versions
gcc -o pagerank_multithread pagerank_multithread.c -lm -fopenmp
gcc -o pagerank_multithread_optimized pagerank_multithread_optimized.c -lm -fopenmp

# Compile the MPI version
mpicc -O3 -o mpi_pagerank mpi_pagerank.c -lm
```

### Running

```bash
# Run the multithreaded version
./pagerank_multithread edge_list.txt

# Run the MPI-based version with 4 processes
mpirun -np 4 ./mpi_pagerank edge_list.txt
```

### Edge List Format

The input file should contain one edge per line in the format:

`<source_node> <destination_node>`

Example:
```plaintext
0 1
1 2
2 0
```

### Output

Each implementation outputs:
- **Total number of nodes and edges**
- **Time taken for convergence**
- **Top 10 nodes with the highest PageRank**

Example output:
```plaintext
Number of edges: 3
Number of nodes: 3
Converged after 20 iterations
Time taken: 0.123456 seconds

Top 10 nodes by PageRank:
Node 2: 0.3876543210
Node 0: 0.3065432109
Node 1: 0.3058024671
```

---

## Authors
- **Filippo Costamagna** (University of Trento)
- **Stefano Romeo** (University of Trento)

## License
This project is licensed under the **MIT License** – see the LICENSE file for details.

---
