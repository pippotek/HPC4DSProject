#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
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

Edge *edges;            // Dynamic array to store edges
int *out_degree;        // Dynamic array to store the out-degree of each node
double *rank;           // Dynamic array to store the current rank of each node
double *temp_rank;      // Dynamic array to store the temporary rank of each node during updates
NodeRank top_nodes[TOP_K]; // Array to track the top 10 nodes by rank
int node_count = 0;     // Tracks the number of unique nodes in the graph
int edge_count = 0;     // Tracks the number of edges in the graph

// Function to read edges from a file and populate edge and out-degree arrays
void read_edges(const char *filename) {
    edges = malloc(sizeof(Edge) * MAX_EDGES);
    out_degree = calloc(MAX_NODES, sizeof(int)); // calloc initializes all elements to 0
    if (!edges || !out_degree) { // Check if memory allocation was successful
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
        
        if (from > node_count) node_count = from;
        if (to > node_count) node_count = to;

        edge_count++;
    }
    fclose(file);

    // node_count was tracking max node ID; increment to get actual count
    node_count++;
}

// Function to initialize the ranks of all nodes
void initialize_ranks() {
    rank = malloc(sizeof(double) * node_count);
    temp_rank = malloc(sizeof(double) * node_count);
    if (!rank || !temp_rank) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    double initial_rank = 1.0 / node_count;

    #pragma omp parallel for
    for (int i = 0; i < node_count; i++) {
        rank[i] = initial_rank;
    }
}

// Compute the final top 10 nodes after ranks have converged
void compute_top_10_ranks() {
    // Initialize top_nodes
    for (int i = 0; i < TOP_K; i++) {
        top_nodes[i].node = -1;
        top_nodes[i].rank = -1.0;
    }

    // A simple approach: for each node, see if it belongs in the top 10
    for (int i = 0; i < node_count; i++) {
        double rank_value = rank[i];

        // Find the position of the minimum rank in top_nodes
        int min_index = 0;
        for (int j = 1; j < TOP_K; j++) {
            if (top_nodes[j].rank < top_nodes[min_index].rank) {
                min_index = j;
            }
        }
        // Update the top_nodes array only if the current rank is higher than the minimum
        if (rank_value > top_nodes[min_index].rank) {
            #pragma omp critical
            {
                if (rank_value > top_nodes[min_index].rank) {
                    top_nodes[min_index].node = i;
                    top_nodes[min_index].rank = rank_value;
                }
            }
        }
    }
}

// Sort and print the top 10 nodes by PageRank
void print_top_10_ranks() {
    // Sort the top_nodes array in descending order
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
        printf("Node %d: %.6f\n", top_nodes[i].node, top_nodes[i].rank);
    }
}

// Function to calculate PageRank iteratively until convergence or max iterations
void calculate_pagerank() {
    #pragma omp parallel
    {
        #pragma omp master
        {
            printf("PageRank will run with %d threads\n", omp_get_num_threads());
        }
    }
    double start_time = omp_get_wtime();  // Use OpenMP's wall-time

    for (int iter = 0; iter < MAX_ITER; iter++) {
        // 1. Reset temp_rank for all nodes
        #pragma omp parallel for
        for (int i = 0; i < node_count; i++) {
            temp_rank[i] = (1.0 - DAMPING_FACTOR) / node_count;
        }

        // 2. Distribute contributions from each edge
        #pragma omp parallel for
        for (int i = 0; i < edge_count; i++) {
            int from = edges[i].from;
            int to   = edges[i].to;
            if (out_degree[from] > 0) {
                double contrib = DAMPING_FACTOR * rank[from] / out_degree[from];
                #pragma omp atomic
                temp_rank[to] += contrib;
            }
        }

        // 3. Compute the difference (for convergence check) and update rank
        double diff_local = 0.0;

        #pragma omp parallel for reduction(+:diff_local)
        for (int i = 0; i < node_count; i++) {
            diff_local += fabs(temp_rank[i] - rank[i]);
            rank[i] = temp_rank[i];
        }

        if (diff_local < TOLERANCE) {
            printf("Converged after %d iterations\n", iter + 1);
            break;
        }
    }

    double end_time = omp_get_wtime();
    printf("Time taken to converge: %.6f seconds\n", end_time - start_time);
}

int main(int argc, char *argv[]) {
    omp_set_num_threads(8);  // Set the number of threads

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <edge_list_file>\n", argv[0]);
        return EXIT_FAILURE;
    }

    read_edges(argv[1]);
    
    printf("Number of edges: %d\n", edge_count);
    printf("Number of nodes: %d\n", node_count);

    initialize_ranks();

    calculate_pagerank();

    // Now that ranks have converged, identify the top 10 nodes:
    compute_top_10_ranks();
    print_top_10_ranks();

    // Free dynamically allocated memory
    free(edges);
    free(out_degree);
    free(rank);
    free(temp_rank);

    return EXIT_SUCCESS;
}