#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define DAMPING_FACTOR 0.85
#define MAX_ITER       100
#define TOLERANCE      1e-6

// Adjust these if you want smaller/larger arrays for demonstration.
// In a real-world scenario, you'd likely parse them from the file or
// define them based on maximum node IDs encountered.
#define MAX_NODES 1000000000
#define MAX_EDGES 2000000000

typedef struct {
    int from;
    int to;
} Edge;

Edge   *edges       = NULL;
int    *out_degree  = NULL;
double *rank_array  = NULL;
double *temp_rank   = NULL;

int node_count = 0;
int edge_count = 0;

//-------------------------------------------------------------------------
// Read the edge list from a file, track max node ID => node_count
//-------------------------------------------------------------------------
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

// -------------------------------------------------------------------------
// Initialize rank arrays
// -------------------------------------------------------------------------
void initialize_ranks() {
    rank_array = (double*) malloc(sizeof(double) * node_count);
    temp_rank  = (double*) malloc(sizeof(double) * node_count);

    if (!rank_array || !temp_rank) {
        fprintf(stderr, "Failed to allocate rank arrays.\n");
        exit(EXIT_FAILURE);
    }

    double init_val = 1.0 / node_count;
    #pragma omp parallel for
    for (int i = 0; i < node_count; i++) {
        rank_array[i] = init_val;
        temp_rank[i]  = 0.0;  // Just to initialize
    }
}

// -------------------------------------------------------------------------
// Calculate PageRank using per-thread local accumulation via calloc().
// -------------------------------------------------------------------------
void calculate_pagerank() {
    printf("Running PageRank with local per-thread accumulators.\n");

    double start = omp_get_wtime();

    for (int iter = 0; iter < MAX_ITER; iter++) {
        // 1) Reset temp_rank to the base factor: (1 - d)/N
        double base = (1.0 - DAMPING_FACTOR) / node_count;
        #pragma omp parallel for
        for (int i = 0; i < node_count; i++) {
            temp_rank[i] = base;
        }

        // 2) Handle dangling nodes (out_degree=0)
        double dangling_sum = 0.0;
        #pragma omp parallel for reduction(+:dangling_sum)
        for (int i = 0; i < node_count; i++) {
            if (out_degree[i] == 0) {
                dangling_sum += rank_array[i];
            }
        }
        double dangling_contrib = (DAMPING_FACTOR * dangling_sum) / node_count;
        #pragma omp parallel for
        for (int i = 0; i < node_count; i++) {
            temp_rank[i] += dangling_contrib;
        }

        // 3) Use local arrays allocated via calloc() for partial sums
        //    Each thread accumulates edge contributions *privately*,
        //    then we'll merge in a second pass.
        int num_threads = 0;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();

            // We only discover total threads after entering parallel region:
            #pragma omp master
            {
                num_threads = omp_get_num_threads();
            }
        }

        // We need an array of pointers, one per thread.
        double **local_contrib = (double**) malloc(num_threads * sizeof(double*));
        if (!local_contrib) {
            fprintf(stderr, "Failed to allocate local_contrib pointers.\n");
            exit(EXIT_FAILURE);
        }

        // --------------------------
        // PART A: Compute local sums
        // --------------------------
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            // calloc() => OS can map pages lazily (demand paging).
            local_contrib[tid] = (double*) calloc(node_count, sizeof(double));
            if (!local_contrib[tid]) {
                fprintf(stderr, "Failed to allocate local_contrib[%d].\n", tid);
                exit(EXIT_FAILURE);
            }

            // Accumulate contributions from edges handled by this thread
            #pragma omp for
            for (int e = 0; e < edge_count; e++) {
                int from = edges[e].from;
                int to   = edges[e].to;
                if (out_degree[from] > 0) {
                    double c = (DAMPING_FACTOR * rank_array[from]) / out_degree[from];
                    local_contrib[tid][to] += c;
                }
            }
        }

        // ----------------------------------------------------------
        // PART B: Merge local_contrib back into temp_rank in parallel
        // ----------------------------------------------------------
        #pragma omp parallel for
        for (int i = 0; i < node_count; i++) {
            double sum = 0.0;
            // Sum contributions from all threads
            for (int t = 0; t < num_threads; t++) {
                sum += local_contrib[t][i];
            }
            temp_rank[i] += sum;
        }

        // Free each thread's local array
        for (int t = 0; t < num_threads; t++) {
            free(local_contrib[t]);
        }
        free(local_contrib);

        // 4) Check for convergence by computing L1 norm difference
        double diff = 0.0;
        #pragma omp parallel for reduction(+:diff)
        for (int i = 0; i < node_count; i++) {
            diff += fabs(temp_rank[i] - rank_array[i]);
        }

        // 5) Copy temp_rank -> rank_array
        #pragma omp parallel for
        for (int i = 0; i < node_count; i++) {
            rank_array[i] = temp_rank[i];
        }

        // 6) Check tolerance
        if (diff < TOLERANCE) {
            printf("Converged after %d iterations (diff = %g)\n", iter + 1, diff);
            break;
        }
    }

    double end = omp_get_wtime();
    printf("PageRank finished in %.3f seconds.\n", end - start);
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    // 1) Read edges (stub)
    if (argc < 2) {
        printf("Usage: %s <edge_file>\n(Using synthetic data for demo)\n", argv[0]);
    } else {
        read_edges(argv[1]);
    }

    printf("node_count = %d, edge_count = %d\n", node_count, edge_count);

    // 2) Initialize rank arrays
    initialize_ranks();

    // 3) Calculate PageRank
    calculate_pagerank();

    // 4) Print final ranks (for small node_count)
    if (node_count <= 20) {
        for (int i = 0; i < node_count; i++) {
            printf("Node %d => Rank: %f\n", i, rank_array[i]);
        }
    }

    // Cleanup
    free(edges);
    free(out_degree);
    free(rank_array);
    free(temp_rank);

    return 0;
}
