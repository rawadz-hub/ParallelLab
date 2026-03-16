#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "sorting.h"

/*
   Merge two sorted chunks of array T!
   The two chunks are of size size
   First chunck starts at T[0], second chunck starts at T[size]
*/
void merge(uint64_t *T, const uint64_t size) {
    uint64_t *X = (uint64_t *)malloc(2 * size * sizeof(uint64_t));

    uint64_t i = 0;
    uint64_t j = size;
    uint64_t k = 0;

    while ((i < size) && (j < 2 * size)) {
        if (T[i] < T[j]) {
            X[k] = T[i];
            i = i + 1;
        } else {
            X[k] = T[j];
            j = j + 1;
        }
        k = k + 1;
    }

    if (i < size) {
        for (; i < size; i++, k++) {
            X[k] = T[i];
        }
    } else {
        for (; j < 2 * size; j++, k++) {
            X[k] = T[j];
        }
    }

    memcpy(T, X, 2 * size * sizeof(uint64_t));
    free(X);

    return;
}

/*
   merge sort -- sequential, parallel --
*/

/*
   ------------------------------------------------------------------------
   Sequential recursive MergeSort
   ------------------------------------------------------------------------
   This follows the PDF specification:

   (a) If the input has fewer than two elements, return.
   (b) Partition the input sequence into two halves.
   (c) Apply mergesort to the two subsequences.
   (d) Merge the two sorted subsequences.
*/
static void sequential_merge_sort(uint64_t *T, uint64_t size) {
    if (size < 2) {
        return;
    }

    uint64_t mid = size / 2;

    /* Sort left half */
    sequential_merge_sort(T, mid);

    /* Sort right half */
    sequential_merge_sort(T + mid, mid);

    /* Merge the 2 sorted halves */
    merge(T, mid);
}



// /*
//    Parallel MergeSort with OpenMP tasks
//    This is the direct parallelization version for question 4(b).
// */
// static void parallel_mergesort_rec(uint64_t *T, uint64_t size) {
//     if (size < 2) {
//         return;
//     }

//     uint64_t mid = size / 2;

//     #pragma omp task shared(T)
//     parallel_mergesort_rec(T, mid);

//     #pragma omp task shared(T)
//     parallel_mergesort_rec(T + mid, mid);

//     #pragma omp taskwait
//     merge(T, mid);
// }

// void parallel_merge_sort(uint64_t *T, const uint64_t size) {
//     if (size < 2) {
//         return;
//     }

//     #pragma omp parallel
//     {
//         #pragma omp single
//         {
//             parallel_mergesort_rec(T, size);
//         }
//     }
// }

/*
   Performance improvement:
   Creating OpenMP tasks has a cost. If the subarray is very small,
   that cost can be larger than the work of sorting it. Therefore,
   we stop creating tasks for small subproblems and sort them
   sequentially instead.
*/

/*
   ------------------------------------------------------------------------
   Parallel recursive MergeSort with OpenMP tasks
   ------------------------------------------------------------------------
*/
static void parallel_mergesort_rec(uint64_t *T, uint64_t size) {
    if (size < 2) {
        return;
    }

    uint64_t mid = size / 2;
    uint64_t stop = 2048;
    /*
       If the subproblem is small, solve it sequentially to avoid the
       overhead of creating too many tiny tasks.
    */
    if (size <= stop) {
        sequential_merge_sort(T, mid);
        sequential_merge_sort(T + mid, mid);
    } else {
        /* Sort left half in a task */
        #pragma omp task shared(T)
        parallel_mergesort_rec(T, mid);

        /* Sort right half in a task */
        #pragma omp task shared(T)
        parallel_mergesort_rec(T + mid, mid);

        /* Wait until both halves are sorted before merging */
        #pragma omp taskwait
    }

    /* Merge the two sorted halves */
    merge(T, mid);
}

/*
   ------------------------------------------------------------------------
   Top-level parallel MergeSort
   ------------------------------------------------------------------------
   Name matches what main() calls.
*/
void parallel_merge_sort(uint64_t *T, const uint64_t size) {
    if (size < 2) {
        return;
    }

    #pragma omp parallel
    {
        #pragma omp single
        {
            parallel_mergesort_rec(T, size);
        }
    }
}



int main(int argc, char **argv) {
    // Init cpu_stats to measure CPU cycles and elapsed time
    struct cpu_stats *stats = cpu_stats_init();

    unsigned int exp;

    /* the program takes one parameter N which is the size of the array to
       be sorted. The array will have size 2^N */
    if (argc != 2) {
        fprintf(stderr, "Usage: merge.run N \n");
        exit(-1);
    }

    uint64_t N = 1 << (atoi(argv[1]));
    /* the array to be sorted */
    uint64_t *X = (uint64_t *)malloc(N * sizeof(uint64_t));

    printf("--> Sorting an array of size %lu\n", N);
#ifdef RINIT
    printf("--> The array is initialized randomly\n");
#endif

    for (exp = 0; exp < NB_EXPERIMENTS; exp++) {
#ifdef RINIT
        init_array_random(X, N);
#else
        init_array_sequence(X, N);
#endif

        cpu_stats_begin(stats);

        sequential_merge_sort(X, N);

        experiments[exp] = cpu_stats_end(stats);

        /* verifying that X is properly sorted */
#ifdef RINIT
        if (!is_sorted(X, N)) {
            print_array(X, N);
            fprintf(stderr, "ERROR: the sequential sorting of the array failed\n");
            exit(-1);
        }
#else
        if (!is_sorted_sequence(X, N)) {
            print_array(X, N);
            fprintf(stderr, "ERROR: the sequential sorting of the array failed\n");
            exit(-1);
        }
#endif
    }

    println_cpu_stats_report("mergesort serial", average_report(experiments, NB_EXPERIMENTS));

    for (exp = 0; exp < NB_EXPERIMENTS; exp++) {
#ifdef RINIT
        init_array_random(X, N);
#else
        init_array_sequence(X, N);
#endif

        cpu_stats_begin(stats);

        parallel_merge_sort(X, N);

        experiments[exp] = cpu_stats_end(stats);

        /* verifying that X is properly sorted */
#ifdef RINIT
        if (!is_sorted(X, N)) {
            print_array(X, N);
            fprintf(stderr, "ERROR: the parallel sorting of the array failed\n");
            exit(-1);
        }
#else
        if (!is_sorted_sequence(X, N)) {
            print_array(X, N);
            fprintf(stderr, "ERROR: the parallel sorting of the array failed\n");
            exit(-1);
        }
#endif
    }

    println_cpu_stats_report("mergesort parallel", average_report(experiments, NB_EXPERIMENTS));

    /* print_array (X, N) ; */

    /* before terminating, we run one extra test of the algorithm */
    uint64_t *Y = (uint64_t *)malloc(N * sizeof(uint64_t));
    uint64_t *Z = (uint64_t *)malloc(N * sizeof(uint64_t));

#ifdef RINIT
    init_array_random(Y, N);
#else
    init_array_sequence(Y, N);
#endif

    memcpy(Z, Y, N * sizeof(uint64_t));

    sequential_merge_sort(Y, N);
    parallel_merge_sort(Z, N);

    if (!are_vector_equals(Y, Z, N)) {
        fprintf(stderr, "ERROR: sorting with the sequential and the parallel algorithm does not give the same result\n");
        exit(-1);
    }

    free(X);
    free(Y);
    free(Z);
}
