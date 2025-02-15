/******************************************************************************
 * Parallel MPI  PageRank Implementation
 *
 * This code reads an edge-list file (skipping comment lines beginning with '#'),
 * computes the PageRank scores in parallel using MPI, and then prints the top
 * 10 nodes by rank.
 *
 * Compile with (for example):
 *     mpicc -O3 -o mpi_pagerank mpi_pagerank.c -lm
 *
 * Run with (for example):
 *     mpirun -np 4 ./mpi_pagerank graph.txt
 ******************************************************************************/

 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <time.h>
 #include <mpi.h>
 #include <stddef.h>   // for offsetof
 
 #define MAX_NODES 1000000000      // Maximum number of nodes in the graph
 #define MAX_EDGES 2000000000      // Maximum number of edges in the graph
 #define DAMPING_FACTOR 0.85       // Damping factor for PageRank calculation
 #define MAX_ITER 100              // Maximum number of iterations for convergence
 #define TOLERANCE 1e-6            // Convergence tolerance
 #define TOP_K 10                  // Number of top nodes to display by PageRank
 
 // Structure to hold an edge from "from" node to "to" node.
 typedef struct {
     int from;
     int to;
 } Edge;
 
 // Structure to hold a node id and its PageRank value.
 typedef struct {
     int node;
     double rank;
 } NodeRank;
 
 
 /* ============================ Utility Functions ============================ */
 
 /**
  * update_top_nodes
  *
  * Given an array of the current top nodes (of size TOP_K), update it if the
  * given node’s rank_value is higher than the minimum in the current top nodes.
  */
 void update_top_nodes(NodeRank top_nodes[TOP_K], int node, double rank_value) {
     int min_index = 0;
     for (int i = 1; i < TOP_K; i++) {
         if (top_nodes[i].rank < top_nodes[min_index].rank) {
             min_index = i;
         }
     }
     if (rank_value > top_nodes[min_index].rank || top_nodes[min_index].node == -1) {
         top_nodes[min_index].node = node;
         top_nodes[min_index].rank = rank_value;
     }
 }
 
 /**
  * print_top_10_ranks
  *
  * Bubble sorts and prints the top 10 nodes (in descending order) by PageRank.
  */
 void print_top_10_ranks(NodeRank top_nodes[TOP_K]) {
     // Bubble sort in descending order
     for (int i = 0; i < TOP_K - 1; i++) {
         for (int j = i + 1; j < TOP_K; j++) {
             if (top_nodes[j].rank > top_nodes[i].rank) {
                 NodeRank temp = top_nodes[i];
                 top_nodes[i] = top_nodes[j];
                 top_nodes[j] = temp;
             }
         }
     }
     printf("Top 10 nodes by PageRank:\n");
     for (int i = 0; i < TOP_K && top_nodes[i].node != -1; i++) {
         printf("Node %d: %.10f\n", top_nodes[i].node, top_nodes[i].rank);
     }
 }
 
 /**
  * read_edges
  *
  * Called only on the master (rank 0), this function reads the entire edge list
  * from file, builds the global edge array and computes the out-degree of each node.
  *
  * Parameters:
  *   filename      - the name of the edge list file.
  *   edges_ptr     - pointer to the (allocated) edge array.
  *   edge_count_ptr- pointer to the number of edges found.
  *   node_count_ptr- pointer to the total number of nodes (assumed to be max_node_id+1).
  *   out_degree_ptr- pointer to the (allocated) out-degree array.
  */
 void read_edges(const char *filename, Edge **edges_ptr, int *edge_count_ptr, int *node_count_ptr, int **out_degree_ptr) {
     Edge *edges = (Edge *) malloc(sizeof(Edge) * MAX_EDGES);
     if (!edges) {
         fprintf(stderr, "Memory allocation failed for edges\n");
         exit(EXIT_FAILURE);
     }
     int *out_degree = (int *) calloc(MAX_NODES, sizeof(int));
     if (!out_degree) {
         fprintf(stderr, "Memory allocation failed for out_degree\n");
         exit(EXIT_FAILURE);
     }
     
     FILE *file = fopen(filename, "r");
     if (!file) {
         perror("Error opening file");
         exit(EXIT_FAILURE);
     }
     
     char line[256];
     int edge_count = 0;
     int max_node = 0;
     while (fgets(line, sizeof(line), file)) {
         if (line[0] == '#') {
             continue;
         }
         int from, to;
         if (sscanf(line, "%d %d", &from, &to) == 2) {
             if (from >= MAX_NODES || to >= MAX_NODES) {
                 fprintf(stderr, "Node id exceeds maximum allowed value.\n");
                 exit(EXIT_FAILURE);
             }
             edges[edge_count].from = from;
             edges[edge_count].to   = to;
             out_degree[from]++;
             if (from > max_node) max_node = from;
             if (to   > max_node) max_node = to;
             edge_count++;
             if (edge_count >= 900000000){
                break;
             }
         }
     }
     fclose(file);
     
     int node_count = max_node + 1;
     *edges_ptr     = edges;
     *edge_count_ptr = edge_count;
     *node_count_ptr = node_count;
     *out_degree_ptr = out_degree;
 }
 
 /* ============================== Main Function ============================== */
 
 int main(int argc, char *argv[]) {
     MPI_Init(&argc, &argv);
     
     int mpi_rank, mpi_size;
     MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
     MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
     
     if (argc != 2) {
         if (mpi_rank == 0)
             fprintf(stderr, "Usage: %s <edge_list_file>\n", argv[0]);
         MPI_Finalize();
         return EXIT_FAILURE;
     }
     
     int node_count, edge_count;
     int *out_degree = NULL;  // global out-degree array (replicated on all processes)
     Edge *edges = NULL;      // only allocated on process 0
     double start_time, end_time;
     
     /* Rank 0 reads the graph from file */
     if (mpi_rank == 0) {
         read_edges(argv[1], &edges, &edge_count, &node_count, &out_degree);
         printf("Number of edges: %d\n", edge_count);
         printf("Number of nodes: %d\n", node_count);
     }
     
     /* Broadcast node_count and edge_count so all processes know the graph size */
     MPI_Bcast(&node_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&edge_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
     
     /* Allocate out_degree on non-root processes and broadcast it */
     if (mpi_rank != 0) {
         out_degree = (int *) malloc(sizeof(int) * node_count);
         if (!out_degree) {
             fprintf(stderr, "Memory allocation failed for out_degree on process %d\n", mpi_rank);
             MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
         }
     }
     MPI_Bcast(out_degree, node_count, MPI_INT, 0, MPI_COMM_WORLD);
     
     /* --- Create an MPI datatype for the Edge structure --- */
     MPI_Datatype MPI_EDGE;
     int lengths[2] = {1, 1};
     MPI_Aint displacements[2];
     displacements[0] = offsetof(Edge, from);
     displacements[1] = offsetof(Edge, to);
     MPI_Datatype types[2] = {MPI_INT, MPI_INT};
     MPI_Type_create_struct(2, lengths, displacements, types, &MPI_EDGE);
     MPI_Type_commit(&MPI_EDGE);
     
     /* --- Distribute the edge list among processes using MPI_Scatterv --- */
     int *sendcounts = NULL;
     int *displs = NULL;
     int local_edge_count;
     Edge *local_edges = NULL;
     if (mpi_rank == 0) {
         sendcounts = (int *) malloc(mpi_size * sizeof(int));
         displs = (int *) malloc(mpi_size * sizeof(int));
         int base = edge_count / mpi_size;
         int rem  = edge_count % mpi_size;
         int sum = 0;
         for (int i = 0; i < mpi_size; i++) {
             sendcounts[i] = base + (i < rem ? 1 : 0);
             displs[i] = sum;
             sum += sendcounts[i];
         }
     }
     int my_edge_count;
     /* First scatter the count of edges each process will receive */
     MPI_Scatter(sendcounts, 1, MPI_INT, &my_edge_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
     local_edge_count = my_edge_count;
     
     /* Allocate space for local edges */
     local_edges = (Edge *) malloc(sizeof(Edge) * local_edge_count);
     if (!local_edges) {
         fprintf(stderr, "Memory allocation failed for local_edges on process %d\n", mpi_rank);
         MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
     }
     
     /* Scatter the edges */
     MPI_Scatterv(edges, sendcounts, displs, MPI_EDGE, local_edges, local_edge_count, MPI_EDGE, 0, MPI_COMM_WORLD);
     
     /* Free the global edge array (and helper arrays) on the master */
     if (mpi_rank == 0) {
         free(edges);
         free(sendcounts);
         free(displs);
     }
     
     /* --- Initialize the rank vectors (replicated on all processes) --- */
     double *rank_vals = (double *) malloc(sizeof(double) * node_count);
     double *temp_rank = (double *) malloc(sizeof(double) * node_count);
     if (!rank_vals || !temp_rank) {
         fprintf(stderr, "Memory allocation failed for rank arrays on process %d\n", mpi_rank);
         MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
     }
     double initial_rank = 1.0 / node_count;
     for (int i = 0; i < node_count; i++) {
         rank_vals[i] = initial_rank;
         temp_rank[i] = 0.0;
     }
     
     /* --- Determine the portion (range) of nodes each process will use for
      * computing the dangling sum and convergence diff.
      * (This is just to avoid redundant work.) */
     int local_node_start, local_node_end;
     int nodes_per_proc = node_count / mpi_size;
     int rem_nodes = node_count % mpi_size;
     if (mpi_rank < rem_nodes) {
         local_node_start = mpi_rank * (nodes_per_proc + 1);
         local_node_end = local_node_start + nodes_per_proc + 1;
     } else {
         local_node_start = mpi_rank * nodes_per_proc + rem_nodes;
         local_node_end = local_node_start + nodes_per_proc;
     }
     
     /* --- Allocate temporary contribution vectors (of length node_count) ---
      * Each process will compute contributions from its local edges into a local
      * vector and then the vectors are summed with MPI_Allreduce. */
     double *local_contrib = (double *) malloc(sizeof(double) * node_count);
     double *global_contrib = (double *) malloc(sizeof(double) * node_count);
     if (!local_contrib || !global_contrib) {
         fprintf(stderr, "Memory allocation failed for contribution vectors on process %d\n", mpi_rank);
         MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
     }
     
     /* --- Begin PageRank iterations --- */
     start_time = MPI_Wtime();
     for (int iter = 0; iter < MAX_ITER; iter++) {
         /* 1) Compute the sum of ranks for dangling nodes (nodes with no out links)
          * over the local node range. */
         double local_dangling_sum = 0.0;
         for (int i = local_node_start; i < local_node_end; i++) {
             if (out_degree[i] == 0)
                 local_dangling_sum += rank_vals[i];
         }
         double global_dangling_sum = 0.0;
         MPI_Allreduce(&local_dangling_sum, &global_dangling_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
         
         double base_rank = (1.0 - DAMPING_FACTOR) / node_count;
         double dangling_contrib = DAMPING_FACTOR * (global_dangling_sum / node_count);
         
         /* 2) Zero out the local contribution vector */
         for (int i = 0; i < node_count; i++) {
             local_contrib[i] = 0.0;
         }
         /* 3) Process each local edge: add contribution from the source node's rank */
         for (int i = 0; i < local_edge_count; i++) {
             int from = local_edges[i].from;
             int to   = local_edges[i].to;
             if (out_degree[from] > 0) {
                 local_contrib[to] += DAMPING_FACTOR * rank_vals[from] / out_degree[from];
             }
         }
         /* 4) Sum all the per-edge contributions across processes */
         MPI_Allreduce(local_contrib, global_contrib, node_count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
         
         /* 5) Combine the base, dangling, and edge contributions to update rank */
         for (int i = 0; i < node_count; i++) {
             temp_rank[i] = base_rank + dangling_contrib + global_contrib[i];
         }
         
         /* 6) Compute the change (diff) over the local node range */
         double local_diff = 0.0;
         for (int i = local_node_start; i < local_node_end; i++) {
             local_diff += fabs(temp_rank[i] - rank_vals[i]);
         }
         double global_diff = 0.0;
         MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
         
         /* 7) Update the rank vector */
         for (int i = 0; i < node_count; i++) {
             rank_vals[i] = temp_rank[i];
         }
         
         if (mpi_rank == 0)
             printf("Iteration %d: diff = %.10f\n", iter + 1, global_diff);
         
         if (global_diff < TOLERANCE) {
             if (mpi_rank == 0)
                 printf("Converged after %d iterations\n", iter + 1);
             break;
         }
     }
     end_time = MPI_Wtime();
     if (mpi_rank == 0)
         printf("Time taken to converge: %.6f seconds\n", end_time - start_time);
     
     /* --- Compute and print the top 10 nodes by PageRank (only done on rank 0) --- */
     if (mpi_rank == 0) {
         NodeRank top_nodes[TOP_K];
         for (int i = 0; i < TOP_K; i++) {
             top_nodes[i].node = -1;
             top_nodes[i].rank = -1.0;
         }
         for (int i = 0; i < node_count; i++) {
             update_top_nodes(top_nodes, i, rank_vals[i]);
         }
         print_top_10_ranks(top_nodes);
     }
     
     /* --- Free dynamically allocated memory and MPI types --- */
     free(local_edges);
     free(rank_vals);
     free(temp_rank);
     free(local_contrib);
     free(global_contrib);
     free(out_degree);
     
     MPI_Type_free(&MPI_EDGE);
     MPI_Finalize();
     return EXIT_SUCCESS;
 }