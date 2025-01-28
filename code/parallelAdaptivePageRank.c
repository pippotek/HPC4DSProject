#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mpi.h"

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))

#define MAX_NODES 2000000000      // Maximum number of nodes in the graph
#define MAX_EDGES 2000000000      // Maximum number of edges in the graph
#define DAMPING_FACTOR 0.85       // Damping factor for PageRank calculation
#define MAX_ITER 100              // Maximum number of iterations for convergence
#define TOLERANCE 1e-6           // Convergence tolerance
#define TOP_K 10                  // Number of top nodes to display by PageRank
#define COMM 1
#define RANK 2

typedef struct {
    int from;
    int to;
} Edge;

typedef struct {
    int node;
    double rank;
} NodeRank;

typedef struct {
    int dest_id;
    int src_id[100];
    int out_degree;
    int in_deg;
} LinkStructure;

Edge *edges;            // Dynamic array to store edges
int *out_degree;        // Dynamic array to store the out-degree of each node
double *rank;           // Dynamic array to store the current rank of each node
double *temp_rank;      // Dynamic array to store the temporary rank of each node during updates
NodeRank top_nodes[TOP_K]; // Array to track the top 10 nodes by rank
int node_count = 0;     // Tracks the number of unique nodes in the graph
int edge_count = 0;     // Tracks the number of edges in the graph
int *edge_array;
int rankId;

// Function to read edges from a file and populate edge and out-degree arrays
int read_edges(const char *filename) {
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

    edge_array = (int *)malloc(sizeof(int) * edge_count*2);

    for(int i=0;i<edge_count*2;i+=2){
        edge_array[i] = edges[i/2].from;
        edge_array[i+1] = edges[i/2].to;
    }

    node_count++;
    /*for(int i = 0;i<edge_count*2;i+=2){
        printf("E[%d], E[%d+1] : (%d, %d)\n",i,i, edge_array[i], edge_array[i+1]);
    }
    */
    return edge_count;
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
void calculate_pagerank(int *recv_array, int chunk_size) {
    clock_t start_time = clock();

    for (int iter = 0; iter < MAX_ITER; iter++) {
        for (int i = 0; i < node_count; i++) {
            temp_rank[i] = (1.0 - DAMPING_FACTOR) / node_count;
        }

        for (int i = 0; i < chunk_size; i+=2) {
            int from = recv_array[i];
            int to = recv_array[i+1];

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
    //printf("Time taken to converge: %.6f seconds\n", time_taken);
}

// Function to print the top 10 nodes by PageRank
void print_top_10_ranks() {
    printf("Top 10 nodes by PageRank in node %d:\n", rankId);

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

void count_out_degree(int *recv_array, int chunk_size){

    for(int i = 0;i<chunk_size;i+=2){
        out_degree[recv_array[i]]++;
    }
}

//DA ottimizzare se possibile
int count_max_nodes(int *recv_array, int chunk_size){
    int max_node = 0;

    for(int i = 0;i<chunk_size;i++){
        if(recv_array[i] > max_node)
            max_node = recv_array[i];
    }

    return max_node;
}

int count_min_nodes(int *recv_array, int chunk_size){
    int min_node = recv_array[0];

    for(int i = 0;i<chunk_size;i++){
        if(recv_array[i] < min_node)
            min_node = recv_array[i];
    }

    return min_node;
}

void communicate_pagerank(int *recv_array, int *chunk_size, int neighbour, int *min_node, int *max_node){
    int neigh_min = 0;
    int neigh_max = 0;
    MPI_Request req_neigh, req_node;

    //Out degree e rank sono compresi all'interno di min - max elementi
    MPI_Sendrecv(min_node, 1, MPI_INT, neighbour, COMM, &neigh_min, 1, MPI_INT, neighbour, COMM, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(max_node, 1, MPI_INT, neighbour, COMM, &neigh_max, 1, MPI_INT, neighbour, COMM, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("Rank : %d recieves from neighbour : %d, MIN : %d, MAX : %d \n", rankId, neighbour, neigh_min, neigh_max);

    int *neigh_array = (int *)malloc(sizeof(int) * (*chunk_size));
    double *neigh_rank = (double *)malloc(sizeof(double)*(neigh_max-neigh_min+1));
    int *neigh_degree = (int *)malloc(sizeof(int) * MAX_NODES);

    MPI_Sendrecv(recv_array, *chunk_size, MPI_INT, neighbour, COMM, neigh_array, *chunk_size, MPI_INT, neighbour, COMM, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(out_degree+(*min_node), (*max_node-*min_node + 1), MPI_INT, neighbour, COMM, neigh_degree, (neigh_max - neigh_min + 1), MPI_INT, neighbour, COMM, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(rank, (node_count), MPI_DOUBLE, neighbour, COMM, neigh_rank, (neigh_max - neigh_min + 1), MPI_DOUBLE, neighbour, COMM, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("Rank : %d recieved from neighbour %d all the data, merging...\n", rankId, neighbour);

    merge_data(neigh_array, neigh_rank, neigh_degree, chunk_size, recv_array, neigh_min, neigh_max, max_node, min_node);

    *min_node = MIN(*min_node, neigh_min);
    *max_node = MAX(*max_node, neigh_max);

    free(neigh_array);
    free(neigh_rank);
    free(neigh_degree);
}

void merge_data(int *neigh_array, double *neigh_rank, int *neigh_degree, int *chunk_size, int *recv_array, int neigh_min, int neigh_max, int *max_node, int *min_node){

    *chunk_size = (*chunk_size)*2

    recv_array = realloc(recv_array, sizeof(int) * chunk_size);
    memcpy(recv_array+chunk_size/2, neigh_array);

    for(int i = 0;i<neigh_max-neigh_min+1;i++)
        out_degree[i]+=neigh_degree[i];

    int new_max = MAX(*max_node, neigh_max)
    int new_min = MIN(*min_node, neigh_max)

    //Da ottimizzare
    *new_rank = (int *)malloc(sizeof(int) * (new_max - new_min + 1));

    for(int i = 0;i<new_max - new_min + 1;i++){
        if((i > *min_node && i < neigh_min) || (i < *max_node && i > neigh_max)){
            new_rank[i] = rank[i];
        }else{
            if(i > *min_node && i > neigh_min) || (i < *max_node && i < neigh_max){
                new_rank[i] = (rank[i - (*min_node != new_min)*(new_min-(*node_min))] + neigh_rank[i - (neigh_min != new_min)*(new_min-neigh_min)])/2 // Occhio a questa formula 
            }else{
               new_rank[i] = neigh_rank[i];
            }
        }
    }

    printf("Data successfully merged in rank: %d\n", rankId)
}

int main(int argc, char *argv[]) {

    MPI_Init(NULL, NULL);

    int size, chunk_size, problemDim, neighbour, groupId;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    groupId = rankId;

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <edge_list_file>\n", argv[0]);
        return EXIT_FAILURE;
    }

    if(rankId == 0){
        problemDim = read_edges(argv[1]);
        chunk_size = (problemDim/size)*2;
        printf("RANK 0 calculates chunk_size of : %d with a ProblemDim of %d\n", chunk_size, problemDim);
    }

    out_degree = (int *)calloc(MAX_NODES, sizeof(int));

    MPI_Bcast(&chunk_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    printf("Rank : %d has %d elements\n", rankId, chunk_size/2);

    if(rankId == 0)
        printf("Scattering %d\n", chunk_size*size);

    int *recv_array = (int *)malloc(sizeof(int) * chunk_size);
    MPI_Scatter(edge_array, chunk_size, MPI_INT, recv_array, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    count_out_degree(recv_array, chunk_size);

    int max_node = count_max_nodes(recv_array, chunk_size);
    int min_node = count_min_nodes(recv_array, chunk_size);
    printf("Rank : %d has a min of : %d and a max of : %d \n", rankId, min_node, max_node);

    node_count = max_node-min_node+1;

    initialize_ranks();
    for(int i = 1;i<size;i*=2){
        calculate_pagerank(recv_array, chunk_size);
        if(groupId % 2 == 0){
            neighbour = rankId + i;
        }else{
            neighbour = rankId - i;
        }
        communicate_pagerank(recv_array, &chunk_size, neighbour, &min_node, &max_node);
        groupId = groupId / 2;
        printf("Rank : %d is now in group %d\n", rankId, groupId);
        break;
    }
    //print_top_10_ranks();

    //Free dynamically allocated memory
    if(rankId == 0){
        free(edges);
        free(edge_array);
    }
    free(out_degree);
    free(temp_rank);
    free(rank);
    free(recv_array);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
