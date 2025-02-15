# HPC4DS Project - High-Performance Computing for Data Science

## Authors:
- **Filippo Costamagna** (University of Trento)
- **Stefano Romeo** (University of Trento)

## Overview

This repository contains the code for the **High-Performance Computing for Data Science (HPC4DS)** project, which investigates the parallelization of the **PageRank algorithm** using different parallel computing paradigms. The primary goal is to analyze the impact of parallelization on the efficiency and scalability of PageRank when applied to large-scale graphs.

The project compares **shared-memory (OpenMP)** and **distributed-memory (MPI)** approaches and evaluates **two hybrid implementations** that leverage both. The research findings are based on extensive benchmarking performed on the **HPC cluster at the University of Trento**, using the **Friendster social network dataset** (65.6 million nodes, 1.8 billion edges).

---

## PageRank Algorithm

The **PageRank algorithm**, developed by **Larry Page and Sergey Brin**, is an iterative ranking method that assigns importance to nodes in a directed graph based on the structure of incoming links. The fundamental equation is:

\[ PR(P_i) = \frac{1 - d}{N} + d \sum_{P_j \in M(P_i)} \frac{PR(P_j)}{L(P_j)} \]

where:
- **d** = damping factor (default: 0.85)
- **N** = total number of nodes
- **M(P_i)** = set of pages linking to **P_i**
- **L(P_j)** = number of outgoing links from **P_j**

This iterative process continues until convergence, making it computationally expensive, especially for large-scale graphs.

---

## Implementations

### **1. Basic Multithreading Implementation (OpenMP)**

- **File:** `pagerank_multithread.c`
- **Approach:** Uses OpenMP to parallelize the rank propagation step in a shared-memory environment.
- **Optimizations:**
  - Uses atomic operations to prevent race conditions.
  - Reduces contention by carefully scheduling iterations.
- **Trade-offs:**
  - Improves performance for moderate dataset sizes.
  - Faces bottlenecks due to shared-memory contention beyond **16 threads**.

### **2. Optimized Multithreading with Local Accumulation (OpenMP)**

- **File:** `pagerank_multithread_optimized.c`
- **Approach:**
  - Uses **thread-local buffers** for rank updates before merging results.
  - Reduces memory contention by deferring shared-memory writes.
- **Trade-offs:**
  - More scalable for large graphs due to reduced atomic operations.
  - Increased memory overhead per thread.
  - Performance deteriorates beyond **16 threads** due to merge overhead.

### **3. MPI-Based Distributed Parallel Implementation**

- **File:** `mpi_pagerank.c`
- **Approach:**
  - Distributes edge partitions across multiple MPI processes.
  - Uses **MPI Allreduce** for global rank synchronization.
- **Trade-offs:**
  - Scales well up to **16 MPI processes**.
  - Communication overhead becomes dominant beyond **16 processes**.

### **4. Hybrid OpenMP + MPI Implementations**

There are **two hybrid implementations**, each corresponding to one of the OpenMP versions:

- **Hybrid Basic OpenMP + MPI:**
  - Uses MPI for inter-node parallelism and OpenMP for intra-node parallelism.
  - Faces contention due to atomic operations, limiting scalability beyond **16 processes/threads**.

- **Hybrid Optimized OpenMP + MPI:**
  - Implements thread-local storage optimizations within OpenMP alongside MPI parallelism.
  - Reduces atomic contention but introduces overhead from merging local accumulations.
  - Scalability declines beyond **16 processes/threads** due to synchronization overhead.

---

## Performance Evaluation

### **Experimental Setup**

- **Cluster:** HPC @ University of Trento
- **Dataset:** Friendster Social Network
  - **Nodes:** 65.6M
  - **Edges:** 1.8B
  - **Size:** 30.14GB
- **Testing:**
  - Strong and weak scaling experiments
  - Evaluated on **1/8, 1/4, 1/2, and full dataset**

### **Results Summary**

| Implementation | Max Speedup | Best Scalability | Bottleneck |
|---------------|------------|------------------|------------|
| **OpenMP (Basic)** | ~13.7× (16 threads) | Up to 16 threads | Atomic contention |
| **OpenMP (Optimized)** | ~19.3× (16 threads) | Up to 16 threads | Merge overhead |
| **MPI (Distributed)** | ~7.78× (16 processes) | Up to 16 processes | Communication latency |
| **Hybrid (Basic)** | ~8.55× (16 processes) | Similar to MPI | Synchronization complexity |
| **Hybrid (Optimized)** | ~10.88× (16 processes) | Similar to MPI | Merge and sync overhead |

### **Key Observations**
- **Scalability significantly deteriorates beyond 16 threads/processes across all implementations.**
- **MPI-based implementation scales better for large datasets** but has increasing communication overhead.
- **Hybrid approaches did not provide substantial improvements beyond 16 processes/threads.**
- **Future improvements could focus on asynchronous MPI communication and GPU acceleration.**

---

## Conclusion

This project provides a **comprehensive performance analysis of parallel PageRank implementations**. It highlights the trade-offs between OpenMP and MPI approaches and discusses the challenges in achieving optimal parallelism. Future research could explore **GPU acceleration, NUMA-aware optimizations, and asynchronous MPI communication** to further enhance performance.

---

## License
This project is licensed under the **MIT License** – see the LICENSE file for details.

---

## References
For a detailed discussion, see the corresponding research paper: [Parallel Implementations of the PageRank Algorithm] by **Filippo Costamagna** and **Stefano Romeo** (University of Trento).

