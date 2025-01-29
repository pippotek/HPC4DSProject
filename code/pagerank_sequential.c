#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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

// Global arrays/pointers
Edge *edges;            // Dynamic array to store edges
int *out_degree;        // Dynamic array to store the out-degree of each node
double *rank;           // Dynamic array to store the current rank of each node
double *temp_rank;      // Dynamic array to store the temporary rank of each node during updates
NodeRank top_nodes[TOP_K]; // Array to track the top 10 nodes by rank
int node_count = 0;     // Tracks the number of unique nodes in the graph
int edge_count = 0;     // Tracks the number of edges in the graph

/**
 * Read edges from a file and populate the 'edges' and 'out_degree' arrays.
 */
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

    char line[256];
    while (fgets(line, sizeof(line), file)) {
        // Ignore lines starting with '#'
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
            edges[edge_count].to = to;

            out_degree[from]++;

            // Track highest node ID to compute node_count
            if (from > node_count) node_count = from;
            if (to > node_count) node_count = to;

            edge_count++;
        }
    }
    fclose(file);

    // node_count was tracking the max node ID; +1 to get the actual count
    node_count++;
}

/**
 * Initialize ranks of all nodes to 1/node_count and reset the top_nodes array.
 */
void initialize_ranks() {
    rank      = malloc(sizeof(double) * node_count);
    temp_rank = malloc(sizeof(double) * node_count);
    if (!rank || !temp_rank) {
        fprintf(stderr, "Memory allocation failed for rank arrays\n");
        exit(EXIT_FAILURE);
    }

    double initial_rank = 1.0 / node_count;
    for (int i = 0; i < node_count; i++) {
        rank[i]      = initial_rank;
        temp_rank[i] = 0.0;
    }

    // Initialize top nodes to invalid values
    for (int i = 0; i < TOP_K; i++) {
        top_nodes[i].node = -1;
        top_nodes[i].rank = -1.0;
    }
}

/**
 * Update the array of top nodes if the given node's rank is high enough.
 * Finds the slot with the lowest rank and replaces it if 'rank_value' is larger.
 */
void update_top_nodes(int node, double rank_value) {
    int min_index = 0;
    for (int i = 1; i < TOP_K; i++) {
        if (top_nodes[i].rank < top_nodes[min_index].rank) {
            min_index = i;
        }
    }
    // Only replace if the new rank is larger than the current min,
    // or if the slot is unused (node == -1).
    if (rank_value > top_nodes[min_index].rank || top_nodes[min_index].node == -1) {
        top_nodes[min_index].node = node;
        top_nodes[min_index].rank = rank_value;
    }
}

/**
 * Calculate PageRank iteratively with dangling-node handling.
 */
void calculate_pagerank() {
    clock_t start_time = clock();

    for (int iter = 0; iter < MAX_ITER; iter++) {
        // 1) Reset temp_rank with the base "teleportation" factor
        for (int i = 0; i < node_count; i++) {
            temp_rank[i] = (1.0 - DAMPING_FACTOR) / node_count;
        }

        // 2) Calculate dangling node contribution
        double dangling_sum = 0.0;
        for (int i = 0; i < node_count; i++) {
            if (out_degree[i] == 0) {
                dangling_sum += rank[i];
            }
        }
        double dangling_contrib = DAMPING_FACTOR * (dangling_sum / node_count);

        // 3) Distribute dangling contribution to all nodes
        for (int i = 0; i < node_count; i++) {
            temp_rank[i] += dangling_contrib;
        }

        // 4) Distribute rank from each edge
        for (int i = 0; i < edge_count; i++) {
            int from = edges[i].from;
            int to   = edges[i].to;
            if (out_degree[from] > 0) {
                temp_rank[to] += DAMPING_FACTOR * rank[from] / out_degree[from];
            }
        }

        // 5) Update rank values and check for convergence
        double diff = 0.0;
        for (int i = 0; i < node_count; i++) {
            diff    += fabs(temp_rank[i] - rank[i]);
            rank[i]  = temp_rank[i];
        }

        if (diff < TOLERANCE) {
            printf("Converged after %d iterations\n", iter + 1);
            break;
        }
    }

    clock_t end_time = clock();
    double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Time taken to converge: %.6f seconds\n", time_taken);
}

/**
 * Sort and print the top 10 nodes by PageRank (descending order).
 */
void print_top_10_ranks() {
    // Simple bubble sort to order top_nodes in descending rank
    for (int i = 0; i < TOP_K - 1; i++) {
        for (int j = i + 1; j < TOP_K; j++) {
            if (top_nodes[j].rank > top_nodes[i].rank) {
                NodeRank temp = top_nodes[i];
                top_nodes[i]  = top_nodes[j];
                top_nodes[j]  = temp;
            }
        }
    }

    // Print out the final top 10
    printf("Top 10 nodes by PageRank:\n");
    for (int i = 0; i < TOP_K && top_nodes[i].node != -1; i++) {
        printf("Node %d: %.10f\n", top_nodes[i].node, top_nodes[i].rank);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <edge_list_file>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // 1) Read edges
    read_edges(argv[1]);
    printf("Number of edges: %d\n", edge_count);
    printf("Number of nodes: %d\n", node_count);

    // 2) Initialize ranks
    initialize_ranks();

    // 3) Calculate PageRank
    calculate_pagerank();

    // 4) Now that the final ranks are established, do ONE pass
    //    to fill in the top 10 nodes array.
    for (int i = 0; i < node_count; i++) {
        update_top_nodes(i, rank[i]);
    }

    // 5) Print the top 10
    print_top_10_ranks();

    // 6) Free dynamically allocated memory
    free(edges);
    free(out_degree);
    free(rank);
    free(temp_rank);

    return EXIT_SUCCESS;
}
