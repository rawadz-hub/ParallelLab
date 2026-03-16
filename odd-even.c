#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <stdbool.h>

#include "sorting.h"

/*
   odd-even sort -- sequential, parallel --
*/

/*
   ------------------------------------------------------------------------
   Sequential odd-even sort
   ------------------------------------------------------------------------
   This follows exactly the algorithm given in the PDF:

   do
       sorted = true
       for i in 0..N-2 with step 2:
           if T[i] > T[i+1]:
               swap
               sorted = false
       for i in 1..N-2 with step 2:
           if T[i] > T[i+1]:
               swap
               sorted = false
   while (sorted == false)  :contentReference[oaicite:0]{index=0}
*/
void sequential_oddeven_sort(uint64_t *T, const uint64_t size) {
    if (size < 2) {
        return;
    }

    int sorted;

    do {
        sorted = 1;  // "sorted = true" in the PDF

        // First pass: even indices, step 2
        for (uint64_t i = 0; i < size - 1; i += 2) {
            if (T[i] > T[i + 1]) {
                uint64_t tmp = T[i];
                T[i] = T[i + 1];
                T[i + 1] = tmp;
                sorted = 0;  // "sorted = false" in the PDF
            }
        }

        // Second pass: odd indices, step 2
        for (uint64_t i = 1; i < size - 1; i += 2) {
            if (T[i] > T[i + 1]) {
                uint64_t tmp = T[i];
                T[i] = T[i + 1];
                T[i + 1] = tmp;
                sorted = 0;  // "sorted = false" in the PDF
            }
        }

    } while (!sorted);  // "while (sorted == false)" in the PDF
}

/*
   ------------------------------------------------------------------------
   Parallel odd-even sort
   ------------------------------------------------------------------------
   This also follows the PDF algorithm exactly, but parallelizes each of the
   two passes. The key reason this works well is that, inside one pass, the
   compared pairs do NOT overlap:

   Even pass:
       (0,1), (2,3), (4,5), ...
   Odd pass:
       (1,2), (3,4), (5,6), ...

   So different iterations can run in parallel safely. This is exactly why
   the PDF says odd-even sort is more adapted than bubble sort for parallel
   implementation. :contentReference[oaicite:1]{index=1}
*/
void parallel_oddeven_sort(uint64_t *T, const uint64_t size) {
    if (size < 2) {
        return;
    }

    int sorted;

    do {
        sorted = 1;  // corresponds to "sorted = true" in the PDF

        /*
           First loop of the PDF:
               for i in 0..N-2 with step 2

           We parallelize it with OpenMP.
           The pairs are disjoint:
               (0,1), (2,3), (4,5), ...
           so no two iterations write the same array cell.
        */
        #pragma omp parallel for reduction(&&:sorted) schedule(static)
        for (uint64_t i = 0; i < size - 1; i += 2) {
            if (T[i] > T[i + 1]) {
                uint64_t tmp = T[i];
                T[i] = T[i + 1];
                T[i + 1] = tmp;
                sorted = 0;  // corresponds to "sorted = false" in the PDF
            }
        }

        /*
           Second loop of the PDF:
               for i in 1..N-2 with step 2

           Again we parallelize it.
           The compared pairs are:
               (1,2), (3,4), (5,6), ...
           which are also disjoint inside this phase.
        */
        #pragma omp parallel for reduction(&&:sorted) schedule(static)
        for (uint64_t i = 1; i < size - 1; i += 2) {
            if (T[i] > T[i + 1]) {
                uint64_t tmp = T[i];
                T[i] = T[i + 1];
                T[i + 1] = tmp;
                sorted = 0;  // corresponds to "sorted = false" in the PDF
            }
        }

        /*
           This matches the last line of the PDF:
               while (sorted == false)

           If neither the even phase nor the odd phase performed any swap,
           then sorted stays true and the algorithm stops.
        */
    } while (!sorted);
}

int main(int argc, char **argv) {
    // Init cpu_stats to measure CPU cycles and elapsed time
    struct cpu_stats *stats = cpu_stats_init();

    unsigned int exp;

    /* the program takes one parameter N which is the size of the array to
       be sorted. The array will have size 2^N */
    if (argc != 2) {
        fprintf(stderr, "Usage: odd-even.run N \n");
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

        sequential_oddeven_sort(X, N);

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

    println_cpu_stats_report("odd-even serial", average_report(experiments, NB_EXPERIMENTS));

    for (exp = 0; exp < NB_EXPERIMENTS; exp++) {
#ifdef RINIT
        init_array_random(X, N);
#else
        init_array_sequence(X, N);
#endif

        cpu_stats_begin(stats);

        parallel_oddeven_sort(X, N);

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

    println_cpu_stats_report("odd-even parallel", average_report(experiments, NB_EXPERIMENTS));

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

    sequential_oddeven_sort(Y, N);
    parallel_oddeven_sort(Z, N);

    if (!are_vector_equals(Y, Z, N)) {
        fprintf(stderr, "ERROR: sorting with the sequential and the parallel algorithm does not give the same result\n");
        exit(-1);
    }

    free(X);
    free(Y);
    free(Z);
}
