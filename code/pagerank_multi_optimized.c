#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define MAX_NODES       1000000000  // Adjust as needed
#define MAX_EDGES       2000000000  // Adjust as needed
#define DAMPING_FACTOR  0.85
#define MAX_ITER        100
#define TOLERANCE       1e-6
#define TOP_K           10

typedef struct {
    int from;
    int to;
} Edge;

typedef struct {
    int node;
    double rank;
} NodeRank;

// Global arrays
Edge    *edges        = NULL;
int     *out_degree   = NULL;
double  *rank_array   = NULL;  // current iteration rank
double  *temp_rank    = NULL;  // next iteration rank
NodeRank top_nodes[TOP_K];

int node_count = 0;  // highest node id + 1
int edge_count = 0;

//-------------------------------------------------------------------------
// Read the edge list from a file, track max node ID => node_count
//-------------------------------------------------------------------------
void read_edges(const char *filename)
{
    edges = (Edge*) malloc(sizeof(Edge) * MAX_EDGES);
    out_degree = (int*) calloc(MAX_NODES, sizeof(int));
    if (!edges || !out_degree) {
        fprintf(stderr, "Memory allocation failed for edges/out_degree.\n");
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
        
        // Track the max node id
        if (from > node_count) node_count = from;
        if (to   > node_count) node_count = to;

        edge_count++;
    }
    fclose(file);

    // node_count is max ID; +1 to get count
    node_count++;
}

//-------------------------------------------------------------------------
// Initialize PageRank arrays
//-------------------------------------------------------------------------
void initialize_ranks()
{
    rank_array = (double*) malloc(sizeof(double) * node_count);
    temp_rank  = (double*) malloc(sizeof(double) * node_count);
    if (!rank_array || !temp_rank) {
        fprintf(stderr, "Memory allocation failed for rank arrays.\n");
        exit(EXIT_FAILURE);
    }

    double initial_rank = 1.0 / node_count;

    // Initialize rank_array
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < node_count; i++) {
        rank_array[i] = initial_rank;
    }
}

//-------------------------------------------------------------------------
// Compute and store the top K nodes after PageRank has converged
//-------------------------------------------------------------------------
void compute_top_k_ranks()
{
    // Initialize top_nodes
    for (int i = 0; i < TOP_K; i++) {
        top_nodes[i].node = -1;
        top_nodes[i].rank = -1.0;
    }

    // Find the top K via a simple pass
    for (int i = 0; i < node_count; i++) {
        double r = rank_array[i];

        // Find the index of the smallest rank in the current top_nodes
        int min_index = 0;
        for (int j = 1; j < TOP_K; j++) {
            if (top_nodes[j].rank < top_nodes[min_index].rank) {
                min_index = j;
            }
        }
        if (r > top_nodes[min_index].rank) {
            top_nodes[min_index].node = i;
            top_nodes[min_index].rank = r;
        }
    }
}

//-------------------------------------------------------------------------
// Sort and print the top K nodes by rank
//-------------------------------------------------------------------------
void print_top_k_ranks()
{
    // Sort top_nodes in descending order
    for (int i = 0; i < TOP_K - 1; i++) {
        for (int j = i + 1; j < TOP_K; j++) {
            if (top_nodes[j].rank > top_nodes[i].rank) {
                NodeRank temp = top_nodes[i];
                top_nodes[i]  = top_nodes[j];
                top_nodes[j]  = temp;
            }
        }
    }

    printf("\nTop %d nodes by PageRank:\n", TOP_K);
    for (int i = 0; i < TOP_K && top_nodes[i].node != -1; i++) {
        printf("Node %d: %.10f\n", top_nodes[i].node, top_nodes[i].rank);
    }
}

//-------------------------------------------------------------------------
// Calculate PageRank using local accumulation + dangling node fix
//-------------------------------------------------------------------------
void calculate_pagerank()
{
    // Choose the number of threads
    omp_set_num_threads(8);

    int num_threads = 0;
    #pragma omp parallel
    {
        #pragma omp master
        {
            num_threads = omp_get_num_threads();
        }
    }
    printf("PageRank will run with %d threads.\n", num_threads);

    // Allocate local_contrib arrays: local_contrib[t][i] = partial rank for node i
    double **local_contrib = (double**) malloc(num_threads * sizeof(double*));
    if (!local_contrib) {
        fprintf(stderr, "Failed to allocate local_contrib.\n");
        exit(EXIT_FAILURE);
    }
    for (int t = 0; t < num_threads; t++) {
        local_contrib[t] = (double*) calloc(node_count, sizeof(double));
        if (!local_contrib[t]) {
            fprintf(stderr, "Failed to allocate local_contrib[%d].\n", t);
            exit(EXIT_FAILURE);
        }
    }

    double start_time = omp_get_wtime();

    for (int iter = 0; iter < MAX_ITER; iter++) {
        // 1) Initialize temp_rank with base teleportation
        double base = (1.0 - DAMPING_FACTOR) / node_count;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < node_count; i++) {
            temp_rank[i] = base;
        }

        // 2) Handle dangling nodes: sum rank of all out_degree=0, distribute that
        double dangling_sum = 0.0;
        #pragma omp parallel for reduction(+:dangling_sum)
        for (int i = 0; i < node_count; i++) {
            if (out_degree[i] == 0) {
                // This node doesn't distribute to anyone, so its rank should
                // be spread out among all nodes
                dangling_sum += rank_array[i];
            }
        }
        double dangling_contrib = DAMPING_FACTOR * dangling_sum / node_count;

        // Add that contribution to every node in temp_rank
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < node_count; i++) {
            temp_rank[i] += dangling_contrib;
        }

        // 3) Zero out local_contrib arrays
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < num_threads; t++) {
            for (int i = 0; i < node_count; i++) {
                local_contrib[t][i] = 0.0;
            }
        }

        // 4) Accumulate edge contributions in thread-local arrays
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for schedule(static)
            for (int e = 0; e < edge_count; e++) {
                int from = edges[e].from;
                int to   = edges[e].to;
                if (out_degree[from] > 0) {
                    double contrib = (DAMPING_FACTOR * rank_array[from]) / out_degree[from];
                    local_contrib[tid][to] += contrib;
                }
            }
        }

        // 5) Merge local contributions into temp_rank (serial merge for consistency)
        for (int t = 0; t < num_threads; t++) {
            for (int i = 0; i < node_count; i++) {
                temp_rank[i] += local_contrib[t][i];
            }
        }

        // 6) Check for convergence and update rank_array
        double diff = 0.0;
        #pragma omp parallel for reduction(+:diff)
        for (int i = 0; i < node_count; i++) {
            diff += fabs(temp_rank[i] - rank_array[i]);
            rank_array[i] = temp_rank[i];
        }

        
        // 7) Convergence check
        if (diff < TOLERANCE) {
            printf("Converged after %d iterations\n", iter + 1);
            break;
        }
    }

    double end_time = omp_get_wtime();
    printf("Time taken to converge: %.6f seconds\n", end_time - start_time);

    // Free local_contrib arrays
    for (int t = 0; t < num_threads; t++) {
        free(local_contrib[t]);
    }
    free(local_contrib);
}

//-------------------------------------------------------------------------
// Main
//-------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <edge_list_file>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // 1) Read edges
    read_edges(argv[1]);
    printf("Number of edges: %d\n", edge_count);
    printf("Number of nodes: %d\n", node_count);

    // 2) Initialize
    initialize_ranks();

    // 3) Calculate PageRank
    calculate_pagerank();

    // 4) Identify and print the top K nodes
    compute_top_k_ranks();
    print_top_k_ranks();

    // 5) Free arrays
    free(edges);
    free(out_degree);
    free(rank_array);
    free(temp_rank);

    return EXIT_SUCCESS;
}
