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
    
    if (size < 2) {
        return;
    }

    /* 
       "sorted" is the global stop condition of the whole algorithm.
       It becomes 1 only if:
       - every thread made no internal swap in its own chunk
       - and the border pass made no swap between chunks
    */
    uint64_t sorted;

    /*
       We keep repeating until the whole array is sorted.
       The idea is:
       1) each thread bubble-sorts only its own chunk
       2) then we do a sequential pass on the chunk borders
       3) if no border swap happened, and chunks were internally sorted,
          then the whole array is sorted
    */
    do {
        sorted = 1;

        /*
           Parallel region:
           each thread computes its own chunk manually.

           We do NOT use "#pragma omp for" here, because we want full control
           over the chunk boundaries. If we used omp for on i = 0..size-2,
           consecutive iterations could be given to different threads and they
           would touch the same array cell, causing races.

           Example of race we want to avoid:
           - iteration i     uses T[i]   and T[i+1]
           - iteration i + 1 uses T[i+1] and T[i+2]

           So instead, each thread gets a disjoint chunk of the ARRAY and only
           works strictly inside that chunk.
        */
        #pragma omp parallel shared(T, size, sorted)
        {
            int tid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();

            /*
               Compute chunk boundaries for this thread.

               We split the ARRAY elements, not the loop iterations.

               start = first element index of the chunk
               end   = one past the last element index of the chunk

               This is the usual [start, end) convention.

               We use proportional splitting instead of:
                   chunk = size / nthreads
               because this formula distributes the remainder more cleanly.
            */
            uint64_t start = (uint64_t)tid * size / nthreads;
            uint64_t end   = (uint64_t)(tid + 1) * size / nthreads;

            /*
               local_sorted is private to the thread.

               It stays 1 if this thread made no swap inside its own chunk.
               If the thread swaps at least once, it becomes 0.
            */
            int local_sorted = 1;

            /*
               Standard bubble sort INSIDE the chunk only.

               Why "while (!done)"?
               Because bubble sort needs repeated passes until no swap occurs.

               Why i + 1 < end?
               Because the comparison uses T[i] and T[i+1], and both must stay
               inside the same chunk.

               If the chunk is [start, end), the last safe comparison is:
                   T[end - 2] with T[end - 1]

               We deliberately DO NOT compare across the border:
                   T[end - 1] with T[end]
               because T[end] belongs to the next chunk.
               Those border comparisons are handled later, sequentially.
            */
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

            /*
               Combine the result of this thread into the global flag.

               We use critical because several threads may try to update
               "sorted" at the same time.

               Global logic:
               sorted = sorted AND local_sorted

               So if even one thread did a swap in its chunk,
               the global "sorted" becomes 0.
            */
            #pragma omp critical
            {
                sorted = sorted && local_sorted;
            }
        }

        /*
           Sequential border pass.

           At this point, every chunk is internally sorted.
           The only possible disorder left is between neighboring chunks.

           So we compare each border pair:
               end_of_chunk_k  with  start_of_chunk_(k+1)

           If that border is inverted, we swap it.

           Why sequential?
           Because this is the simplest safe way to handle boundaries,
           and it matches the chunk-based method you were discussing.

           Why repeat the outer loop after a border swap?
           Because after swapping a border pair, the two affected chunks may
           no longer be internally sorted. So we must re-sort chunks again
           in the next iteration.
        */
        int number_of_threads = omp_get_max_threads();

        /*
           We only have meaningful borders if there is more than one chunk.
           This loop reconstructs the same chunk boundaries and checks the
           border between chunk k and chunk k+1.
        */
        for (int t = 0; t < number_of_threads - 1; ++t) {
            uint64_t end_left    = (uint64_t)(t + 1) * size / number_of_threads;
            uint64_t start_right = end_left;

            /*
               Skip degenerate cases where a chunk may be empty,
               which can happen if there are more threads than elements.
            */
            if (end_left == 0 || start_right >= size) {
                continue;
            }

            /*
               Border pair is:
               T[end_left - 1] and T[start_right]
            */
            if (T[end_left - 1] > T[start_right]) {
                uint64_t tmp = T[end_left - 1];
                T[end_left - 1] = T[start_right];
                T[start_right] = tmp;

                /*
                   A border swap means the whole array was not yet sorted,
                   so force another outer iteration.
                */
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
