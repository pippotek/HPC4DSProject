#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <omp.h>
#include "mpi.h"

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))

#define MAX_NODES 2000000000     // Maximum number of nodes in the graph
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


int *out_degree;        // Dynamic array to store the out-degree of each node
Edge *edges;
double *rank;           // Dynamic array to store the current rank of each node
int node_count = 0;     // Tracks the number of unique nodes in the graph
int edge_count = 0;     // Tracks the number of edges in the graph
int *edge_array;
int rankId;
double *temp_rank;

// Function to read edges from a file and populate edge and out-degree arrays
int read_edges(const char *filename) { 


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

int parprocess(MPI_File *in, int size, const int overlap, int *recv_array) {
    MPI_Offset globalstart;
    int mysize;
    char *chunk;
    
    /* read in relevant chunk of file into "chunk",
     * which starts at location in the file globalstart
     * and has size mysize 
     */
        MPI_Offset globalend;
        MPI_Offset filesize;
    
        /* figure out who reads what */
        MPI_File_get_size(*in, &filesize);
        filesize--;  /* get rid of text file eof */

        mysize = filesize/size;
        globalstart = rankId * mysize + 128;
        globalend   = globalstart + mysize - 1;
        if (rankId == size-1) globalend = filesize;
    
        /* add overlap to the end of everyone's chunk except last proc... */
        if (rankId != size-1)
            globalend += overlap;
    
        mysize =  globalend - globalstart + 1;
    
        /* allocate memory */
        chunk = malloc( (mysize + 1)*sizeof(char));
    
        /* everyone reads in their part */
        if(MPI_File_read_at_all(*in, globalstart, chunk, mysize, MPI_CHAR, MPI_STATUS_IGNORE) == -1){
            printf("Rank : %d failed", rankId);
        }
        chunk[mysize] = '\0';
    
        printf("Reading the file\n");
    
    /*
     * everyone calculate what their start and end *really* are by going 
     * from the first newline after start to the first newline after the
     * overlap region starts (eg, after end - overlap + 1)
     */
    
    int locstart=0, locend=mysize;
    if (rankId != 0) {
        while(chunk[locstart++] != '\n');
    }

    if (rankId != size-1) {
        locend-=overlap;
        while(chunk[locend] != '\n') locend++;
    }

    int true_start = (globalstart + locstart);
    int neigh_beginning = 0;
    int true_size = 0;

    if(rankId != 0)
        MPI_Send(&true_start, 1, MPI_INT, rankId-1, 1, MPI_COMM_WORLD);

    if(rankId != size-1){
        MPI_Recv(&neigh_beginning, 1, MPI_INT, rankId+1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        true_size = neigh_beginning - globalstart + 1;
    }
    
    if(rankId != size-1){
        locend = neigh_beginning - 1;
        chunk[true_size] = '\0';
    }
    mysize = locend-locstart+1;
    
    /* "Process" our chunk by replacing non-space characters with '1' for
     * rank 1, '2' for rank 2, etc... 
     */

   int line_counter = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    printf("After Barrier\n");
    
   char *line = strtok(chunk, "\n");
    while (line) {
        int from, to;
        if (sscanf(line, "%d %d", &from, &to) == 2) {
            recv_array[line_counter++] = from;
            recv_array[line_counter++] = to;
        }
        line = strtok(NULL, "\n");
    }

    
    return line_counter;
}

// Function to initialize the ranks of all nodes and the top nodes array
void initialize_ranks() {
    printf("Node count: %d\n",node_count);
    rank = (double *)malloc(sizeof(double) * node_count);
    if (!rank) {
        fprintf(stderr, "Memory allocation of rank failed\n");
        exit(EXIT_FAILURE);
    }
    
    double initial_rank = 1.0 / node_count;
    for (int i = 0; i < node_count; i++) {
        rank[i] = initial_rank;
    }
}

// Function to calculate PageRank iteratively until convergence or max iterations
void calculate_pagerank(int *recv_array, int chunk_size, int min_node) {
    clock_t start_time = clock();
    double * temp_rank = (double *)malloc(sizeof(double) * node_count);

    for (int iter = 0; iter < MAX_ITER; iter++) {
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

        //Possibile Vettorizzazione o parallelismo
        for (int i = 0; i < chunk_size; i+=2) {
            int from = recv_array[i];
            int to = recv_array[i+1];
                        
            temp_rank[to-min_node] += DAMPING_FACTOR * rank[from-min_node] / out_degree[from];
        }

        double diff = 0;
        for (int i = 0; i < node_count; i++) {
            diff += fabs(temp_rank[i] - rank[i]);
            rank[i] = temp_rank[i];
        }

        if(rankId == 0)
            printf("Iteration %d with diff = %f\n", iter, diff);

        if (diff < TOLERANCE) {
            printf("Converged after %d iterations\n", iter + 1);
            break;
        }
    }
    clock_t end_time = clock();
    double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    free(temp_rank);
    printf("Time taken to converge: %.6f seconds\n", time_taken);
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

int *merge_data(int *neigh_array, double *neigh_rank, int *edge_count, int neigh_edge, int *recv_array, int neigh_min, int neigh_max, int *max_node, int *min_node){

    (*edge_count) += neigh_edge;

    int *new_array = (int *)malloc(sizeof(int)*(*edge_count));
    for(int i = 0;i<*edge_count - neigh_edge;i++){
        new_array[i] = recv_array[i];
    }

    for(int i = *edge_count - neigh_edge; i < *edge_count;i++){
        new_array[i] = neigh_array[i-(*edge_count - neigh_edge)];
    }
    
    int new_max = MAX(*max_node, neigh_max);
    int new_min = MIN(*min_node, neigh_min);

    printf("Node %d, max: %d, min %d\n",rankId, new_max, new_min);

    //Da ottimizzare
    double *new_rank = (double *)malloc(sizeof(double) * (new_max - new_min + 1));
    for(int i = 0;i<new_max - new_min+1 ;i++){
        new_rank[i] = 0;
        if((i+new_min >= *min_node && i+new_min < neigh_min) || (i+new_min <= *max_node && i+new_min > neigh_max)){
            new_rank[i] = rank[i];
        }
        else{
            if((i+new_min >= *min_node && i+new_min >= neigh_min && i+new_min <= *max_node && i+new_min <= neigh_max)){
                double my_rank = rank[i + (*min_node != new_min)*(new_min-(*min_node))];
                double neigh = neigh_rank[i + (neigh_min != new_min)*(new_min-neigh_min)];
                new_rank[i] = (my_rank + neigh)/2;
            }
            else{
               new_rank[i] = neigh_rank[i];
            }
        }
    }

    count_statistics(new_array, *edge_count, max_node, min_node, 0); 

    node_count = new_max-new_min;
    rank = new_rank;
 
    free(recv_array);
    printf("Data successfully merged in rank: %d\n", rankId);
    return new_array;
}

int *communicate_pagerank(int *recv_array, int *edge_count, int neighbour, int *min_node, int *max_node){
    int neigh_min = 0;
    int neigh_max = 0;
    int neigh_edge = 0;
    MPI_Request req[6];
    MPI_Request req2[4];
    MPI_Status stat[6];
    MPI_Status stat2[4];

    printf("Rank :%d, communicates with :%d\n",rankId, neighbour);

    //Out degree e rank sono compresi all'interno di min - max elementi
    MPI_Isend(min_node, 1, MPI_INT, neighbour, COMM, MPI_COMM_WORLD, req);
    MPI_Irecv(&neigh_min, 1, MPI_INT, neighbour, COMM, MPI_COMM_WORLD, req+1);
    MPI_Isend(max_node, 1, MPI_INT, neighbour, COMM, MPI_COMM_WORLD, req+2);
    MPI_Irecv(&neigh_max, 1, MPI_INT, neighbour, COMM, MPI_COMM_WORLD, req+3);
    MPI_Isend(edge_count, 1, MPI_INT, neighbour, COMM, MPI_COMM_WORLD, req+4);
    MPI_Irecv(&neigh_edge, 1, MPI_INT, neighbour, COMM, MPI_COMM_WORLD, req+5);


    MPI_Waitall(6, req, stat);

    printf("Rank : %d recieves from neighbour : %d, MIN : %d, MAX : %d with %d edges\n", rankId, neighbour, neigh_min, neigh_max, neigh_edge);

    int *neigh_array = (int *)malloc(sizeof(int) * (neigh_edge));
    double *neigh_rank = (double *)malloc(sizeof(double)*(neigh_max-neigh_min+1));

    MPI_Isend(recv_array, *edge_count, MPI_INT, neighbour, COMM, MPI_COMM_WORLD, req2);
    MPI_Irecv(neigh_array, neigh_edge, MPI_INT, neighbour, COMM, MPI_COMM_WORLD, req2+1);
    MPI_Isend(rank, node_count, MPI_DOUBLE, neighbour, COMM, MPI_COMM_WORLD, req2+2);
    MPI_Irecv(neigh_rank, (neigh_max - neigh_min + 1), MPI_DOUBLE, neighbour, COMM, MPI_COMM_WORLD, req2+3);

    MPI_Waitall(4, req2, stat2);
    printf("Rank : %d recieved from neighbour %d all the data, merging...\n", rankId, neighbour);

    recv_array = merge_data(neigh_array, neigh_rank, edge_count, neigh_edge, recv_array, neigh_min, neigh_max, max_node, min_node);


    free(neigh_array);
    free(neigh_rank);
    return recv_array;
}

void print_final_ranks(int min_node){
    printf("Rank : %d declares final ranks:\n", rankId);
    for(int i = 0;i<node_count;i++){
            printf("%d: %d --> %f\n",rankId, (i+min_node), rank[i]);
    }

    printf("Rank %d has declared all pageranks\n", rankId);
}

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int size, problemDim, neighbour, groupId;
    MPI_File in;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    groupId = rankId;

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <edge_list_file>\n", argv[0]);
        return EXIT_FAILURE;
    }

    out_degree = (int *)calloc(MAX_NODES, sizeof(int));

    MPI_File_open(MPI_COMM_WORLD, argv[1], MPI_MODE_RDONLY, MPI_INFO_NULL, &in);

    int *recv_array = (int *)malloc(sizeof(int) * MAX_NODES);
    int edge_count = parprocess(&in, size, 10, recv_array);
    printf("Beginning pageRank\n");

    int max_node = 0;
    int min_node = 0;
    printf("Edge_count is %d\n", edge_count);

    count_statistics(recv_array, edge_count, &max_node, &min_node, 0);

    node_count = max_node-min_node+1;
    printf("Node count in rank %d is %d\n", min_node, max_node);

    initialize_ranks();
    for(int i = 1;i<size;i*=2){
        calculate_pagerank(recv_array, edge_count, min_node);
        if(groupId % 2 == 0){
            neighbour = rankId + i;
        }else{
            neighbour = rankId - i;
        }

        printf("Rank %d beginning communication\n",rankId);
        recv_array = communicate_pagerank(recv_array, &edge_count, neighbour, &min_node, &max_node);

        groupId = groupId / 2;
        node_count = max_node-min_node+1;
        printf("Rank : %d is now in group %d\n", rankId, groupId);
        printf("Rank : %d has now %d elements\n", rankId, max_node-min_node);
    }

    calculate_pagerank(recv_array, edge_count, min_node);
    printf("Final iteration complete in rank %d, exiting...\n", rankId);

    //Free dynamically allocated memory
    if(rankId == 0){
        free(edge_array);
    }

    free(out_degree);
    free(rank);

    printf("Rank : %d exits...\n", rankId);

    MPI_Finalize();
}
