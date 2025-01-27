#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>  // Include OpenMP header

#define MAX_NODES 1000000000      // Maximum number of nodes in the graph
#define MAX_EDGES 2000000000      // Maximum number of edges in the graph
#define DAMPING_FACTOR 0.85       // Damping factor for PageRank calculation
#define MAX_ITER 100              // Maximum number of iterations for convergence
#define TOLERANCE 1e-6            // Convergence tolerance
#define TOP_K 10                  // Number of top nodes to display by PageRank

typedef struct {
    int from;
    int to;
} Edge;

typedef struct {
    int node;
    double rank;
} NodeRank;

Edge   *edges       = NULL;  // Dynamic array to store edges
int    *out_degree  = NULL;  // Out-degree for each node
double *rank_array  = NULL;  // Current rank for each node
double *temp_rank   = NULL;  // Temporary array for next iteration
NodeRank top_nodes[TOP_K];   // Track the top 10 nodes by rank

int node_count = 0;  // Will be max node ID + 1
int edge_count = 0;  // Number of edges read

//------------------------------------------------------------------------
// Read edges from file; update out_degree[] and node_count
//------------------------------------------------------------------------
void read_edges(const char *filename) {
    edges = (Edge*) malloc(sizeof(Edge) * MAX_EDGES);
    out_degree = (int*) calloc(MAX_NODES, sizeof(int));
    if (!edges || !out_degree) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    while (fscanf(file, "%d %d", &edges[edge_count].from, &edges[edge_count].to) == 2) {
        int from = edges[edge_count].from;
        int to   = edges[edge_count].to;

        if (from >= MAX_NODES || to >= MAX_NODES) {
            fprintf(stderr, "Node id exceeds maximum allowed value.\n");
            exit(EXIT_FAILURE);
        }

        out_degree[from]++;

        // Track highest node ID to compute node_count
        if (from > node_count) node_count = from;
        if (to   > node_count) node_count = to;

        edge_count++;
    }
    fclose(file);

    // node_count was tracking the max node ID; +1 to get the actual count
    node_count++;
}

//------------------------------------------------------------------------
// Initialize rank arrays
//------------------------------------------------------------------------
void initialize_ranks() {
    rank_array = (double*) malloc(sizeof(double) * node_count);
    temp_rank  = (double*) malloc(sizeof(double) * node_count);
    if (!rank_array || !temp_rank) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    double initial_rank = 1.0 / node_count;

    // Initialize rank_array in parallel
    #pragma omp parallel for
    for (int i = 0; i < node_count; i++) {
        rank_array[i] = initial_rank;
    }
}

//------------------------------------------------------------------------
// Compute the final top 10 nodes after ranks have converged
//------------------------------------------------------------------------
void compute_top_10_ranks() {
    // Initialize top_nodes
    for (int i = 0; i < TOP_K; i++) {
        top_nodes[i].node = -1;
        top_nodes[i].rank = -1.0;
    }

    // Check each node's rank against the current top-10
    for (int i = 0; i < node_count; i++) {
        double rank_value = rank_array[i];

        // Find the position of the lowest rank in top_nodes
        int min_index = 0;
        for (int j = 1; j < TOP_K; j++) {
            if (top_nodes[j].rank < top_nodes[min_index].rank) {
                min_index = j;
            }
        }
        // Update the top_nodes array if current rank is higher
        if (rank_value > top_nodes[min_index].rank) {
            // If multiple threads used, the #pragma omp critical would be safe
            // but here we do it sequentially after convergence, so no conflict.
            top_nodes[min_index].node = i;
            top_nodes[min_index].rank = rank_value;
        }
    }
}

//------------------------------------------------------------------------
// Sort and print the top 10 nodes by PageRank
//------------------------------------------------------------------------
void print_top_10_ranks() {
    // Sort in descending order
    for (int i = 0; i < TOP_K - 1; i++) {
        for (int j = i + 1; j < TOP_K; j++) {
            if (top_nodes[j].rank > top_nodes[i].rank) {
                NodeRank temp = top_nodes[i];
                top_nodes[i]  = top_nodes[j];
                top_nodes[j]  = temp;
            }
        }
    }

    printf("Top 10 nodes by PageRank:\n");
    for (int i = 0; i < TOP_K && top_nodes[i].node != -1; i++) {
        printf("Node %d: %.10f\n", top_nodes[i].node, top_nodes[i].rank);
    }
}

//------------------------------------------------------------------------
// Calculate PageRank with dangling-node handling until convergence
//------------------------------------------------------------------------
void calculate_pagerank() {
    // Use 8 threads by default
    omp_set_num_threads(8);

    // Inform about the number of threads
    #pragma omp parallel
    {
        #pragma omp master
        {
            printf("PageRank will run with %d threads\n", omp_get_num_threads());
        }
    }

    double start_time = omp_get_wtime();

    for (int iter = 0; iter < MAX_ITER; iter++) {
        // 1) Reset temp_rank to base = (1 - damping) / N
        double base = (1.0 - DAMPING_FACTOR) / node_count;
        #pragma omp parallel for
        for (int i = 0; i < node_count; i++) {
            temp_rank[i] = base;
        }

        // 2) Compute total rank of dangling nodes
        double dangling_sum = 0.0;
        #pragma omp parallel for reduction(+:dangling_sum)
        for (int i = 0; i < node_count; i++) {
            if (out_degree[i] == 0) {
                // Node i is dangling
                dangling_sum += rank_array[i];
            }
        }
        double dangling_contrib = (DAMPING_FACTOR * dangling_sum) / node_count;

        // 3) Add dangling contribution to every node
        #pragma omp parallel for
        for (int i = 0; i < node_count; i++) {
            temp_rank[i] += dangling_contrib;
        }

        // 4) Distribute contributions from each edge
        //    Using an atomic update to temp_rank[to]
        #pragma omp parallel for
        for (int e = 0; e < edge_count; e++) {
            int from = edges[e].from;
            int to   = edges[e].to;
            if (out_degree[from] > 0) {
                double contrib = (DAMPING_FACTOR * rank_array[from]) / out_degree[from];
                #pragma omp atomic
                temp_rank[to] += contrib;
            }
        }

        // 5) Update rank_array and check convergence
        double diff_local = 0.0;
        #pragma omp parallel for reduction(+:diff_local)
        for (int i = 0; i < node_count; i++) {
            diff_local += fabs(temp_rank[i] - rank_array[i]);
            rank_array[i] = temp_rank[i];
        }

        if (diff_local < TOLERANCE) {
            printf("Converged after %d iterations\n", iter + 1);
            break;
        }
    }

    double end_time = omp_get_wtime();
    printf("Time taken to converge: %.6f seconds\n", end_time - start_time);
}

//------------------------------------------------------------------------
// Main function
//------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <edge_list_file>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // 1) Read edges
    read_edges(argv[1]);
    printf("Number of edges: %d\n", edge_count);
    printf("Number of nodes: %d\n", node_count);

    // 2) Initialize rank arrays
    initialize_ranks();

    // 3) Calculate PageRank (with dangling node handling)
    calculate_pagerank();

    // 4) Compute and print top 10
    compute_top_10_ranks();
    print_top_10_ranks();

    // 5) Cleanup
    free(edges);
    free(out_degree);
    free(rank_array);
    free(temp_rank);

    return EXIT_SUCCESS;
}
