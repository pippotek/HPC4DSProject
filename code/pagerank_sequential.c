#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_NODES 1000000000      // Maximum number of nodes in the graph
#define MAX_EDGES 2000000000      // Maximum number of edges in the graph
#define DAMPING_FACTOR 0.85       // Damping factor for PageRank calculation
#define MAX_ITER 100              // Maximum number of iterations for convergence
#define TOLERANCE 1e-6           // Convergence tolerance
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
        int to = edges[edge_count].to;

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

    node_count++;
}

// Function to initialize the ranks of all nodes and the top nodes array
void initialize_ranks() {
    rank = malloc(sizeof(double) * node_count);
    temp_rank = malloc(sizeof(double) * node_count);
    if (!rank || !temp_rank) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    
    double initial_rank = 1.0 / node_count;
    for (int i = 0; i < node_count; i++) {
        rank[i] = initial_rank;
    }

    for (int i = 0; i < TOP_K; i++) {
        top_nodes[i].node = -1;
        top_nodes[i].rank = -1.0;
    }
}

// Helper function to check if a node is in the top nodes
int is_in_top_nodes(int node) {
    for (int i = 0; i < TOP_K; i++) {
        if (top_nodes[i].node == node) {
            return 1; // Node is already in the top list
        }
    }
    return 0;
}

// Function to update the top nodes with the given node and its rank
void update_top_nodes(int node, double rank_value) {
    // Skip if the node is already in the top list
    if (is_in_top_nodes(node)) return;

    // Find the position of the minimum rank in top_nodes
    int min_index = 0;
    for (int i = 1; i < TOP_K; i++) {
        if (top_nodes[i].rank < top_nodes[min_index].rank) {
            min_index = i;
        }
    }

    // Update the top_nodes array only if the current rank is higher than the minimum
    if (rank_value > top_nodes[min_index].rank || top_nodes[min_index].node == -1) {
        top_nodes[min_index].node = node;
        top_nodes[min_index].rank = rank_value;
    }
}

// Function to calculate PageRank iteratively until convergence or max iterations
void calculate_pagerank() {
    clock_t start_time = clock();

    for (int iter = 0; iter < MAX_ITER; iter++) {
        for (int i = 0; i < node_count; i++) {
            temp_rank[i] = (1.0 - DAMPING_FACTOR) / node_count;
        }

        for (int i = 0; i < edge_count; i++) {
            int from = edges[i].from;
            int to = edges[i].to;

            if (out_degree[from] > 0) {
                temp_rank[to] += DAMPING_FACTOR * rank[from] / out_degree[from];
            }
        }

        double diff = 0;
        for (int i = 0; i < node_count; i++) {
            diff += fabs(temp_rank[i] - rank[i]);
            rank[i] = temp_rank[i];
            
            // Update the top nodes with the new rank value
            update_top_nodes(i, rank[i]);
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

// Function to print the top 10 nodes by PageRank
void print_top_10_ranks() {
    printf("Top 10 nodes by PageRank:\n");

    // Sort the top_nodes array for displaying in descending order
    for (int i = 0; i < TOP_K - 1; i++) {
        for (int j = i + 1; j < TOP_K; j++) {
            if (top_nodes[j].rank > top_nodes[i].rank) {
                NodeRank temp = top_nodes[i];
                top_nodes[i] = top_nodes[j];
                top_nodes[j] = temp;
            }
        }
    }

    // Print the sorted top nodes
    for (int i = 0; i < TOP_K && top_nodes[i].node != -1; i++) {
        printf("Node %d: %.6f\n", top_nodes[i].node, top_nodes[i].rank);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <edge_list_file>\n", argv[0]);
        return EXIT_FAILURE;
    }

    read_edges(argv[1]);
    
    printf("Number of edges: %d\n", edge_count);
    printf("Number of nodes: %d\n", node_count);

    initialize_ranks();
    calculate_pagerank();
    print_top_10_ranks();

    // Free dynamically allocated memory
    free(edges);
    free(out_degree);
    free(rank);
    free(temp_rank);

    return EXIT_SUCCESS;
}
