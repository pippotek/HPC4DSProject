/******************************************************************************
 * Parallel MPI PageRank (Improved)
 *
 * - Reads an edge list from a file on rank 0.
 * - Builds an adjacency list (still on rank 0).
 * - Distributes node ownership and adjacency to the appropriate rank.
 * - Performs PageRank iterations in parallel.
 * - Gathers final ranks on rank 0 and prints the top 10.
 *
 * Compile with (for example):
 *     mpicc -O3 -o mpi_pagerank_optimized mpi_pagerank_optimized.c -lm
 *
 * Run with (for example):
 *     mpirun -np 4 ./mpi_pagerank_optimized graph.txt
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <stddef.h>   // for offsetof

#define DAMPING_FACTOR 0.85
#define MAX_ITER       100
#define TOLERANCE      1e-6
#define TOP_K          10

/* A small structure for adjacency. We store each node's outgoing neighbors. */
typedef struct {
    int *neighbors;
    int  count;
} AdjacencyList;

/* Structure to hold a node id and its PageRank value (used for top-K). */
typedef struct {
    int node;
    double rank;
} NodeRank;

/* --------------------------------------------------------------------------
   update_top_nodes

   Keep track of the top K nodes in an array of NodeRank size K.
   This is a simple method that updates the minimum element if a new
   node's rank is bigger.
   -------------------------------------------------------------------------- */
static void update_top_nodes(NodeRank top_nodes[TOP_K], int node, double rank_value)
{
    /* Find index of the minimum rank among the current top_nodes */
    int min_index = 0;
    for (int i = 1; i < TOP_K; i++) {
        if (top_nodes[i].rank < top_nodes[min_index].rank) {
            min_index = i;
        }
    }
    /* If rank_value is bigger than that minimum, replace it */
    if (rank_value > top_nodes[min_index].rank || top_nodes[min_index].node == -1) {
        top_nodes[min_index].node = node;
        top_nodes[min_index].rank = rank_value;
    }
}

/* --------------------------------------------------------------------------
   print_top_10_ranks

   Sorts the top K nodes in descending order by rank (bubble sort since K=10).
   Then prints them.
   -------------------------------------------------------------------------- */
static void print_top_10_ranks(NodeRank top_nodes[TOP_K])
{
    /* Bubble sort in descending order by rank */
    for (int i = 0; i < TOP_K - 1; i++) {
        for (int j = i + 1; j < TOP_K; j++) {
            if (top_nodes[j].rank > top_nodes[i].rank) {
                NodeRank temp = top_nodes[i];
                top_nodes[i] = top_nodes[j];
                top_nodes[j] = temp;
            }
        }
    }
    printf("Top %d nodes by PageRank:\n", TOP_K);
    for (int i = 0; i < TOP_K && top_nodes[i].node != -1; i++) {
        printf("  Node %d: %.10f\n", top_nodes[i].node, top_nodes[i].rank);
    }
}

/* ==========================================================================
   MAIN
   ========================================================================== */
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <edge_list_file>\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    double start_time, end_time;
    double *rank_vals  = NULL; /* PageRank vector, size = node_count */
    double *temp_rank  = NULL; /* Temporary copy of rank vector */
    int node_count     = 0;    /* total number of nodes = max_node_id + 1 */
    int *out_degree    = NULL; /* global out-degree array, size = node_count */

    /* For distributing adjacency, we'll flatten all adjacency into a single array. */
    int *adj_flat = NULL;      /* array of all neighbors for all nodes (built on rank 0) */
    int *prefix   = NULL;      /* prefix[i] = starting index in adj_flat for node i */

    /* -- Rank 0 reads the entire file and builds adjacency -- */
    if (rank == 0)
    {
        /* 1) First pass: determine max_node and count out-degrees. */
        FILE *fp = fopen(argv[1], "r");
        if (!fp) {
            perror("Error opening file");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        const int LINE_SZ = 256;
        char line[LINE_SZ];
        int max_node_id = -1;

        /* We'll store out_degree in a dynamic array. We might not know how big the nodes can go,
           but let's guess or reallocate as needed. For simplicity, we'll store them in a
           dynamically grown array if needed. However, we can also do a single pass to find max_node
           first, then allocate exactly. */

        /* We'll do two passes:
           (a) find max_node,
           (b) then allocate out_degree and adjacency structures,
           (c) then read edges again.
        */

        while (fgets(line, LINE_SZ, fp)) {
            if (line[0] == '#') continue;  /* skip comments */
            int from, to;
            if (sscanf(line, "%d %d", &from, &to) == 2) {
                if (from > max_node_id) max_node_id = from;
                if (to   > max_node_id) max_node_id = to;
            }
        }
        fclose(fp);

        if (max_node_id < 0) {
            fprintf(stderr, "No valid edges found in file.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        node_count = max_node_id + 1;

        /* 2) Allocate out_degree array and adjacency storage. */
        out_degree = (int *) calloc(node_count, sizeof(int));
        if (!out_degree) {
            fprintf(stderr, "Failed to allocate out_degree.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        /* Second pass to count out_degrees and build a list of edges for each node. */
        /* We'll store adjacency in a "flattened" form eventually, but first we can gather
           neighbors in a temporary array-of-vectors style. */
        AdjacencyList *adj_temp = (AdjacencyList*) calloc(node_count, sizeof(AdjacencyList));
        if (!adj_temp) {
            fprintf(stderr, "Failed to allocate adjacency temp array.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        /* Re-open the file and fill adjacency. */
        fp = fopen(argv[1], "r");
        if (!fp) {
            perror("Error opening file (second pass)");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        while (fgets(line, LINE_SZ, fp)) {
            if (line[0] == '#') continue;  /* skip comments */
            int from, to;
            if (sscanf(line, "%d %d", &from, &to) == 2) {
                out_degree[from]++;
            }
        }
        fclose(fp);

        /* Now allocate adjacency arrays for each node. */
        for (int i = 0; i < node_count; i++) {
            adj_temp[i].count = 0;
            if (out_degree[i] > 0) {
                adj_temp[i].neighbors = (int*) malloc(out_degree[i] * sizeof(int));
                if (!adj_temp[i].neighbors) {
                    fprintf(stderr, "Malloc failed for adjacency node %d\n", i);
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
            } else {
                adj_temp[i].neighbors = NULL;
            }
        }

        /* Third pass: fill the adjacency lists. We need to reset out_degree counts as "cursors". */
        memset(out_degree, 0, node_count * sizeof(int));

        fp = fopen(argv[1], "r");
        if (!fp) {
            perror("Error opening file (third pass)");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        while (fgets(line, LINE_SZ, fp)) {
            if (line[0] == '#') continue;
            int from, to;
            if (sscanf(line, "%d %d", &from, &to) == 2) {
                int idx = out_degree[from]++;  /* current insertion index */
                adj_temp[from].neighbors[idx] = to;
            }
        }
        fclose(fp);

        /* Now out_degree[] is correct (reset to final) and adjacency is built. */

        /* Flatten adjacency: compute prefix array. */
        prefix = (int*) malloc((node_count + 1) * sizeof(int));
        if (!prefix) {
            fprintf(stderr, "Failed to allocate prefix array.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        prefix[0] = 0;
        for (int i = 1; i <= node_count; i++) {
            prefix[i] = prefix[i - 1] + out_degree[i - 1];
        }
        int total_neighbors = prefix[node_count];

        /* Allocate flat adjacency array. */
        adj_flat = (int*) malloc(total_neighbors * sizeof(int));
        if (!adj_flat) {
            fprintf(stderr, "Failed to allocate adj_flat array.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        /* Fill adj_flat by copying from adj_temp. */
        for (int i = 0; i < node_count; i++) {
            int offset = prefix[i];
            for (int j = 0; j < out_degree[i]; j++) {
                adj_flat[offset + j] = adj_temp[i].neighbors[j];
            }
        }

        /* Cleanup the temporary adjacency structure. */
        for (int i = 0; i < node_count; i++) {
            free(adj_temp[i].neighbors);
        }
        free(adj_temp);

        /* rank 0 can now print some basic info */
        printf("Number of nodes: %d\n", node_count);
        printf("Total edges: %d\n", prefix[node_count]);
    }

    /* Broadcast node_count so everyone knows how large to allocate. */
    MPI_Bcast(&node_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (node_count <= 0) {
        /* No graph to process. */
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* Distribute out_degree, prefix, and adj_flat to all processes. */
    if (rank != 0) {
        out_degree = (int*) malloc(node_count * sizeof(int));
        if (!out_degree) {
            fprintf(stderr, "Failed to allocate out_degree on rank %d\n", rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        prefix = (int*) malloc((node_count + 1) * sizeof(int));
        if (!prefix) {
            fprintf(stderr, "Failed to allocate prefix on rank %d\n", rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    /* Bcast out_degree and prefix. */
    MPI_Bcast(out_degree, node_count, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(prefix, node_count + 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Figure out local node range (block partition). */
    int base_nodes = node_count / size;
    int remainder  = node_count % size;
    int local_start, local_end;
    if (rank < remainder) {
        local_start = rank * (base_nodes + 1);
        local_end   = local_start + (base_nodes + 1);
    } else {
        local_start = rank * base_nodes + remainder;
        local_end   = local_start + base_nodes;
    }
    if (local_end > node_count) local_end = node_count; /* safety check */
    int local_count = local_end - local_start;

    /* Each rank needs only the adjacency for its local nodes. We'll gather that
       from adj_flat. The relevant portion in adj_flat is from prefix[local_start]
       to prefix[local_end]. */

    int start_offset = 0, end_offset = 0, my_adj_count = 0;
    if (rank == 0) {
        start_offset = prefix[local_start];
        end_offset   = prefix[local_end];
        my_adj_count = end_offset - start_offset;
    }
    /* We'll gather these offset bounds on *all* ranks so they know how many
       adjacency entries to receive. */
    MPI_Bcast(&start_offset, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&end_offset,   1, MPI_INT, 0, MPI_COMM_WORLD);
    my_adj_count = end_offset - start_offset;

    /* Now we can allocate local adjacency for each rank. */
    int *local_adj = (int*) malloc(my_adj_count * sizeof(int));
    if (!local_adj) {
        fprintf(stderr, "Rank %d failed to allocate local_adj\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* Scatter the relevant portion of adj_flat to each rank:
       We can do this with MPI_Scatterv. First, rank 0 sets up sendcounts/displs for the adjacency. */
    int *sendcounts = NULL;
    int *displs     = NULL;
    if (rank == 0) {
        sendcounts = (int*) malloc(size * sizeof(int));
        displs     = (int*) malloc(size * sizeof(int));
        int running_offset = 0;
        for (int p = 0; p < size; p++) {
            /* compute that p's local node range */
            int p_start, p_end;
            if (p < remainder) {
                p_start = p * (base_nodes + 1);
                p_end   = p_start + (base_nodes + 1);
            } else {
                p_start = p * base_nodes + remainder;
                p_end   = p_start + base_nodes;
            }
            if (p_end > node_count) p_end = node_count;

            int off1 = prefix[p_start];
            int off2 = prefix[p_end];
            sendcounts[p] = off2 - off1;   /* adjacency slice size */
        }
        /* compute displacements */
        int sum = 0;
        for (int p = 0; p < size; p++) {
            displs[p] = sum;
            sum += sendcounts[p];
        }
    }

    /* Scatterv the adjacency slice. */
    MPI_Scatterv(
        adj_flat, sendcounts, displs, MPI_INT,
        local_adj, my_adj_count, MPI_INT,
        0, MPI_COMM_WORLD
    );

    /* rank 0 can now free the big adj_flat if it wants */
    if (rank == 0) {
        free(adj_flat);
        free(sendcounts);
        free(displs);
    }

    /* Allocate PageRank vectors (full size for simplicity). */
    rank_vals = (double*) malloc(node_count * sizeof(double));
    temp_rank = (double*) malloc(node_count * sizeof(double));
    if (!rank_vals || !temp_rank) {
        fprintf(stderr, "Rank %d failed to allocate rank arrays.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* Initialize the rank values */
    double init_rank = 1.0 / (double) node_count;
    for (int i = 0; i < node_count; i++) {
        rank_vals[i] = init_rank;
        temp_rank[i] = 0.0;
    }

    /* We'll store local contributions in an array, also size = node_count (to be combined). */
    double *local_contrib  = (double*) calloc(node_count, sizeof(double));
    double *global_contrib = (double*) calloc(node_count, sizeof(double));
    if (!local_contrib || !global_contrib) {
        fprintf(stderr, "Rank %d failed to allocate contribution arrays.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* Prepare to iterate */
    start_time = MPI_Wtime();
    const double base_rank = (1.0 - DAMPING_FACTOR) / (double) node_count;

    for (int iter = 1; iter <= MAX_ITER; iter++)
    {
        /* 1) Compute global dangling sum (sum of ranks of nodes with out_degree=0). */
        double local_dangling_sum = 0.0;
        for (int i = local_start; i < local_end; i++) {
            if (out_degree[i] == 0) {
                local_dangling_sum += rank_vals[i];
            }
        }
        double global_dangling_sum = 0.0;
        MPI_Allreduce(&local_dangling_sum, &global_dangling_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double dang_contrib = DAMPING_FACTOR * (global_dangling_sum / (double) node_count);

        /* 2) Prepare node-based contribution: rank[i] / out_degree[i] (if out_degree>0). */
        /*    We'll do this only for i in [local_start, local_end], but store globally. */
        for (int i = local_start; i < local_end; i++) {
            if (out_degree[i] > 0) {
                temp_rank[i] = DAMPING_FACTOR * rank_vals[i] / (double) out_degree[i];
            } else {
                temp_rank[i] = 0.0;
            }
        }

        /* 3) Clear local_contrib. We'll add partial sums for adjacency. */
        memset(local_contrib, 0, node_count * sizeof(double));

        /* 4) For each node i in [local_start, local_end], add its contribution to each neighbor. */
        for (int i = local_start; i < local_end; i++) {
            double c = temp_rank[i];  /* node-based contribution (DAMPING_FACTOR * rank[i]/out_degree[i]) */
            if (c > 0.0) {
                int offset_start = prefix[i] - prefix[local_start]; /* local offset in local_adj */
                int offset_end   = offset_start + out_degree[i];
                for (int pos = offset_start; pos < offset_end; pos++) {
                    int nbr = local_adj[pos];
                    local_contrib[nbr] += c;
                }
            }
        }

        /* 5) Sum up local contributions across all ranks to get global contributions. */
        MPI_Allreduce(local_contrib, global_contrib, node_count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        /* 6) Now compute new rank vector in temp_rank, also track difference for convergence. */
        double local_diff = 0.0;
        for (int i = local_start; i < local_end; i++) {
            double oldval = rank_vals[i];
            double newval = base_rank + dang_contrib + global_contrib[i];
            temp_rank[i]  = newval;
            local_diff += fabs(newval - oldval);
        }

        double global_diff = 0.0;
        MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        /* 7) Copy temp_rank to rank_vals for next iteration. */
        for (int i = local_start; i < local_end; i++) {
            rank_vals[i] = temp_rank[i];
        }

        /* Optionally print progress every few iterations */
        if (rank == 0 && (iter % 5 == 0)) {
            printf("Iteration %d, diff = %.8f\n", iter, global_diff);
        }

        /* 8) Check convergence */
        if (global_diff < TOLERANCE) {
            if (rank == 0) {
                printf("Converged after %d iterations (diff=%.8f)\n", iter, global_diff);
            }
            break;
        }
    } /* end of iteration loop */

    end_time = MPI_Wtime();
    if (rank == 0) {
        printf("Time to converge: %.6f seconds\n", end_time - start_time);
    }

    /* --- Gather final ranks on rank 0 to determine top 10. --- */
    /*   (In this example, each rank already holds the entire rank_vals array, so no gather is needed.) */

    if (rank == 0) {
        /* Find top 10. */
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

    /* --- Cleanup --- */
    free(rank_vals);
    free(temp_rank);
    free(local_contrib);
    free(global_contrib);
    free(local_adj);
    free(out_degree);
    free(prefix);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
