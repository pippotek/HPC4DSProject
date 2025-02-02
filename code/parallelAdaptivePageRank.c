#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
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

    Edge *edges = malloc(sizeof(Edge) * MAX_EDGES);
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

    edge_array = (int *)malloc(sizeof(int) * edge_count*2);

    for(int i=0;i<edge_count*2;i+=2){
        edge_array[i] = edges[i/2].from;
        edge_array[i+1] = edges[i/2].to;
    }

    node_count++;
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
        printf("Rank: %d at iteration %d\n", rankId, iter);
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

void count_statistics(int *recv_array, int chunk_size, int *max_node, int *min_node, int halfed){
    if(halfed == 0){
        *max_node = 0;
        *min_node = recv_array[0];
    } 

    for(int i = 0 + ((chunk_size/2)*halfed);i < chunk_size;i++){

        if(recv_array[i] < *min_node){
            *min_node = recv_array[i];
        }

        if(recv_array[i] > *max_node){
            *max_node = recv_array[i];
        }

        if(i%2 == 0){
            out_degree[recv_array[i]]++;
        }
    }
}

int *merge_data(int *neigh_array, double *neigh_rank, int *chunk_size, int *recv_array, int neigh_min, int neigh_max, int *max_node, int *min_node){

    (*chunk_size)*=2;

    recv_array = reallocarray(recv_array, *chunk_size, sizeof(int));
    //memcpy(recv_array+((*chunk_size)/2)*sizeof(int), neigh_array, sizeof(int) * ((*chunk_size)/2));
    for(int i = *chunk_size/2;i<*chunk_size;i++){
        recv_array[i] = neigh_array[i-*chunk_size/2];
    }
    
    int new_max = MAX(*max_node, neigh_max);
    int new_min = MIN(*min_node, neigh_min);

    printf("Node %d, max: %d, min %d\n",rankId, new_max, new_min);

    //Da ottimizzare
    double *new_rank = (double *)malloc(sizeof(double) * (new_max - new_min + 1));

    for(int i = 0;i<new_max - new_min + 1;i++){
        new_rank[i] = 0;
        if((i+new_min >= *min_node && i+new_min < neigh_min) || (i+new_min <= *max_node && i+new_min > neigh_max)){
            new_rank[i] = rank[i];
        }
        else{
            if((i+new_min >= *min_node && i+new_min >= neigh_min && i+new_min <= *max_node && i+new_min <= neigh_max)){
                new_rank[i] = (rank[i - (*min_node != new_min)*(new_min-(*min_node)-1)] + neigh_rank[i - (neigh_min != new_min)*(new_min-neigh_min-1)])/2; // Occhio a questa formula 
            }
            else{
               new_rank[i] = neigh_rank[i];
            }
        }
    }

    count_statistics(recv_array, *chunk_size, max_node, min_node, 1); 


    rank = new_rank;
    node_count = new_max-new_min+1;
    printf("Data successfully merged in rank: %d\n", rankId);
    return recv_array;
}

int *communicate_pagerank(int *recv_array, int *chunk_size, int neighbour, int *min_node, int *max_node){
    int neigh_min = 0;
    int neigh_max = 0;
    MPI_Request req[4];
    MPI_Status stat[4];

    printf("Rank :%d, communicates with :%d\n",rankId, neighbour);

    //Out degree e rank sono compresi all'interno di min - max elementi
    MPI_Isend(min_node, 1, MPI_INT, neighbour, COMM, MPI_COMM_WORLD, req);
    MPI_Irecv(&neigh_min, 1, MPI_INT, neighbour, COMM, MPI_COMM_WORLD, req+1);
    MPI_Isend(max_node, 1, MPI_INT, neighbour, COMM, MPI_COMM_WORLD, req+2);
    MPI_Irecv(&neigh_max, 1, MPI_INT, neighbour, COMM, MPI_COMM_WORLD, req+3);

    MPI_Waitall(4, req, stat);

    printf("Rank : %d recieves from neighbour : %d, MIN : %d, MAX : %d \n", rankId, neighbour, neigh_min, neigh_max);

    int *neigh_array = (int *)malloc(sizeof(int) * (*chunk_size));
    double *neigh_rank = (double *)malloc(sizeof(double)*(neigh_max-neigh_min+1));

    MPI_Isend(recv_array, *chunk_size, MPI_INT, neighbour, COMM, MPI_COMM_WORLD, req);
    MPI_Irecv(neigh_array, *chunk_size, MPI_INT, neighbour, COMM, MPI_COMM_WORLD, req+1);
    MPI_Isend(rank, node_count, MPI_DOUBLE, neighbour, COMM, MPI_COMM_WORLD, req+2);
    MPI_Irecv(neigh_rank, (neigh_max - neigh_min + 1), MPI_DOUBLE, neighbour, COMM, MPI_COMM_WORLD, req+3);

    MPI_Waitall(4, req, stat);

    printf("Rank : %d recieved from neighbour %d all the data, merging...\n", rankId, neighbour);

    recv_array = merge_data(neigh_array, neigh_rank, chunk_size, recv_array, neigh_min, neigh_max, max_node, min_node);


    free(neigh_array);
    free(neigh_rank);
    return recv_array;
}

void print_final_ranks(int min_node, int max_node){
    int node_count = max_node-min_node+1;

    printf("Rank : %d declares final ranks:\n", rankId);
    for(int i = 0;i<node_count;i++){
        printf("%d: %d --> %f\n",rankId, (i+min_node), rank[i]);
    }

    printf("Rank %d has declared all pageranks\n", rankId);
}

void check_differences(int node_count){

    if(rankId == 0){

        double *neigh_rank = (double *)malloc(sizeof(double) * node_count);
        MPI_Recv(neigh_rank, node_count, MPI_DOUBLE, MPI_ANY_SOURCE, COMM, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Differences : \n");
        for(int i = 0;i< node_count;i++){
            printf("%d --> %f - %f = %f\n", i+1, rank[i], neigh_rank[i], rank[i]-neigh_rank[i]);
        }
    }else{
        MPI_Send(rank, node_count, MPI_DOUBLE, 0, COMM, MPI_COMM_WORLD);
    }
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
    }

    out_degree = (int *)calloc(MAX_NODES, sizeof(int));

    MPI_Bcast(&chunk_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *recv_array = (int *)malloc(sizeof(int) * chunk_size);
    MPI_Scatter(edge_array, chunk_size, MPI_INT, recv_array, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    int max_node = 0;
    int min_node = 0;

    count_statistics(recv_array, chunk_size, &max_node, &min_node, 0);

    node_count = max_node-min_node+1;

    initialize_ranks();
    for(int i = 1;i<size;i*=2){
        calculate_pagerank(recv_array, chunk_size);
        if(groupId % 2 == 0){
            neighbour = rankId + i;
        }else{
            neighbour = rankId - i;
        }
        recv_array = communicate_pagerank(recv_array, &chunk_size, neighbour, &min_node, &max_node);

        groupId = groupId / 2;
        node_count = max_node-min_node+1;
        printf("Rank : %d is now in group %d\n", rankId, groupId);
        printf("Rank : %d has now %d elements\n", rankId, max_node-min_node+1);

        MPI_Barrier(MPI_COMM_WORLD);
    }

    //print_top_10_ranks();

    calculate_pagerank(recv_array, chunk_size);
    printf("Final iteration complete in rank %d, exiting...\n", rankId);

    if(rankId == 0)
        print_final_ranks(min_node, max_node);

    //Free dynamically allocated memory
    if(rankId == 0){
        free(edge_array);
    }

    free(out_degree);
    free(temp_rank);
    free(rank);

    printf("Rank : %d exits...\n", rankId);

    MPI_Finalize();

    return 0;
}
