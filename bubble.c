#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "stdbool.h"


#include "sorting.h"

/*
   bubble sort -- sequential, parallel --
*/

void sequential_bubble_sort(uint64_t *T, const uint64_t size) {
    /* TODO: sequential implementation of bubble sort */
    uint64_t sorted;

    do {
        sorted = 1;

        for (uint64_t i = 0; i < size - 1; ++i) {
            if (T[i] > T[i + 1]) {
                uint64_t tmp = T[i];
                T[i] = T[i + 1];
                T[i + 1] = tmp;
                sorted = 0;
            }
        }

    } while (!sorted);
}



void parallel_bubble_sort(uint64_t *T, const uint64_t size) {
    
    /* "sorted" is true only if no swaps occurred in chunks or at chunk borders */
    uint64_t sorted;

    /* We repeat the whole process until no swap occurs anywhere in the array. (no swaos meaning the array is fully sorted) */
    do {
        sorted = 1;

        /*
        Each thread processes a separate chunk of the array in parallel.
        We avoid "#pragma omp for" because adjacent iterations could access
        the same elements (e.g., T[i+1]), causing data conflicts.(why? because the end of a thread's chunk is the start of the next thread's chunk)
        */
        #pragma omp parallel shared(T, size, sorted)
        {
            int tid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();
            
            uint64_t start = (uint64_t)tid * size / nthreads;
            uint64_t end   = (uint64_t)(tid + 1) * size / nthreads;

            /*
               local_sorted is private to the thread.
               It stays 1 if this thread made no swap inside its own chunk.
               If the thread swaps at least once, it becomes 0.
            */
            int local_sorted = 1;

            /* Bubble sort on the thread's chunk. */
            int done;
            do {
                done = 1;

                for (uint64_t i = start; i + 1 < end; ++i) {
                    if (T[i] > T[i + 1]) {
                        uint64_t tmp = T[i];
                        T[i] = T[i + 1];
                        T[i + 1] = tmp;

                        done = 0;
                        local_sorted = 0;
                    }
                }
            } while (!done);

            /*Update the global "sorted" variable in critical section because multiple threads may update it at the same time */
            #pragma omp critical
            {
                sorted = sorted && local_sorted;
            }
        }

        /*NOw we make a sequential pass to handle borders */
        int number_of_threads = omp_get_max_threads();
        for (int t = 0; t < number_of_threads - 1; ++t) {
            uint64_t end_left    = (uint64_t)(t + 1) * size / number_of_threads;
            uint64_t start_right = end_left;
            if (T[end_left - 1] > T[start_right]) {
                uint64_t tmp = T[end_left - 1];
                T[end_left - 1] = T[start_right];
                T[start_right] = tmp;

                //a border swap means the whole array was not yet sorted, so we put sorted = 0 for another itration
                
                sorted = 0;
            }
        }

    } while (!sorted);
}

int main(int argc, char **argv) {
    // Init cpu_stats to measure CPU cycles and elapsed time
    struct cpu_stats *stats = cpu_stats_init();

    unsigned int exp;

    /* the program takes one parameter N which is the size of the array to
       be sorted. The array will have size 2^N */
    if (argc != 2) {
        fprintf(stderr, "Usage: bubble.run N \n");
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

        sequential_bubble_sort(X, N);

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

    println_cpu_stats_report("bubble serial", average_report(experiments, NB_EXPERIMENTS));

    for (exp = 0; exp < NB_EXPERIMENTS; exp++) {
#ifdef RINIT
        init_array_random(X, N);
#else
        init_array_sequence(X, N);
#endif

        cpu_stats_begin(stats);

        parallel_bubble_sort(X, N);

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

    println_cpu_stats_report("bubble parallel", average_report(experiments, NB_EXPERIMENTS));

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

    sequential_bubble_sort(Y, N);
    parallel_bubble_sort(Z, N);

    if (!are_vector_equals(Y, Z, N)) {
        fprintf(stderr, "ERROR: sorting with the sequential and the parallel algorithm does not give the same result\n");
        exit(-1);
    }

    free(X);
    free(Y);
    free(Z);
}
