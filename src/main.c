#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "pooler.h"
#include "distributions.h"
#include "assertf.h"

#include <unistd.h> // For sleep function


#include "data_manager.h"

void populate_sparse_vector(u8* vec, u32 length, u32 num_ones) {
    for(u32 it = 0; it < num_ones; ++it) vec[it] = 1;
    for(u32 it = num_ones; it < length; ++it) vec[it] = 0;
}

f32 calculate_sparsity(u8* vec, u32 len) {
    u32 ones = 0;
    for(u32 it = 0; it < len; ++it)
        ones += (vec[it] == 1);

    return ((f32) ones) / (f32) len;
}

void print_vector(u8* vec, u32 len) {
    f32 sparsity = calculate_sparsity(vec, len);

    for(u32 it = 0; it < len; ++it) 
        printf("%u", vec[it]);
    printf(" [sparsity=%f%%]", sparsity * 100);
}

void generate_dataset(mat_u8* dataset, u32 num_samples, u32 bits_per_sample, f32 min_sparsity, f32 max_sparsity) {
    matrix_u8_init(dataset, num_samples, bits_per_sample);

    for(u32 sample = 0; sample < num_samples; ++sample) {
        f32 sparsity = unif_rand_range_f32(min_sparsity, max_sparsity);
        populate_sparse_vector(MATRIX_AXIS1(*dataset, sample), bits_per_sample, sparsity * bits_per_sample);
        shuffle_array_u8(MATRIX_AXIS1(*dataset, sample), bits_per_sample);

        f32 recomp_spsity = calculate_sparsity(MATRIX_AXIS1(*dataset, sample), bits_per_sample);
        assertf(fabs(recomp_spsity - sparsity) < 0.01, "wrong sparsity [expec: %f got: %f]", sparsity, recomp_spsity);
    }
}

void test_dataset() {
    u32 m = 100, n = 100;
    mat_u8 dataset, dataset_copy;
    generate_dataset(&dataset, m, n, 0.02, 0.2);

    for(u32 sample = 0; sample < 5; ++sample) {
        print_vector(MATRIX_AXIS1(dataset, sample), n);
    }

    matrix_u8_init(&dataset_copy, m, n);
    memcpy(dataset_copy.data, dataset.data, m * n * sizeof(u8));

    int res = memcmp(dataset.data, dataset_copy.data, m * n * sizeof(u8));
    printf("memcopy + memcmp: (%i)\n", res);

    write_dataset("data/dataset.data", &dataset, m, n);

    u32 num_samples = 100;
    u32 bits_per = 100;
    read_dataset("data/dataset.data", &dataset, &num_samples, &bits_per);

    assertf(num_samples == m && bits_per == n, "wrong num samples or bits");

    printf("\n");

    for(u32 sample = 0; sample < 5; ++sample) {
        print_vector(MATRIX_AXIS1(dataset, sample), n);
    }

    res = memcmp(dataset.data, dataset_copy.data, m * n * sizeof(u8));
    assertf(res == 0, "not the same!!");

    printf("WE GOOD! (%i)\n", res);
}

void test_pooler() {
    u32 num_inputs = 30;
    u32 num_columns = 128;

    pooler_t p;
    pooling_init(&p, num_inputs, num_columns, 1.0);

    printf("%u\n", p.params.top_k);

    u8* vec = calloc(num_inputs, sizeof(*vec));
    populate_sparse_vector(vec, num_inputs, 0.2 * num_inputs);

    for(u32 it = 0; it < 10; ++it) {
        shuffle_array_u8(vec, num_inputs);

        pooling_step(&p, vec, num_inputs);

        print_vector(vec, num_inputs);
        print_vector(p.column_activations, num_columns);
    }
}

int main(int argc, char *argv[]) {  

// Put the error message in a char array:
    const char error_message[] = "Error: usage: %s X\n\t \
        0 is for training from scratch\n";


    /* Error Checking */
    if(argc < 0) {
        printf(error_message, argv[0]);
        exit(1);
    }

    u32 num_datapoints = 10, num_bits_per = 100;
    mat_u8 dataset;
    matrix_u8_init(&dataset, num_datapoints, num_bits_per);
    // generate_dataset(&dataset, m, n, 0.02, 0.2);
    u32 dataset_max_size;
    read_dataset_partial("data/dataset.data", &dataset, num_datapoints, &dataset_max_size, &num_bits_per);
    
    u32 num_columns = 128;

    pooler_t p;
    pooling_init(&p, num_bits_per, num_columns, 1.0);

    pooling_print(&p);

    f32 input_sparsity = 0.0f, column_activation_sparsity = 0.0f;

    printf("First pass\n");
    for (u32 it = 0; it < num_datapoints; ++it) {
        pooling_step(&p, MATRIX_AXIS1(dataset, it), num_bits_per);

        input_sparsity = calculate_sparsity(MATRIX_AXIS1(dataset, it), num_bits_per);
        column_activation_sparsity = calculate_sparsity(p.column_activations, num_columns);

        printf("%f%%\t->\t%f%%\n", input_sparsity * 100, column_activation_sparsity * 100);
    }

    u32 N = 200;
    printf("\n\nMake it see the dataset %u times\n", N);
    for(u32 super_it = 0; super_it < N; ++super_it) {
        f32 max_sparsity = 0.0f, min_sparsity = 1.0f;
        f64 mean_sparsity_acc = 0.0;
        for (u32 it = 0; it < num_datapoints; ++it) {
            pooling_step(&p, MATRIX_AXIS1(dataset, it), num_bits_per);

            column_activation_sparsity = calculate_sparsity(p.column_activations, num_columns);
            if(column_activation_sparsity > max_sparsity) max_sparsity = column_activation_sparsity;
            if(column_activation_sparsity < min_sparsity) min_sparsity = column_activation_sparsity;
            mean_sparsity_acc += column_activation_sparsity;
        }
        mean_sparsity_acc /= num_datapoints;

        if(super_it % 100 == 0)
            printf("\t%u\t[min = %.3f] [avg = %.3lf] [max = %.3f]\n", super_it, min_sparsity, mean_sparsity_acc, max_sparsity);
    }

    printf("\nLast pass\n");
    for (u32 it = 0; it < num_datapoints; ++it) {
        pooling_step(&p, MATRIX_AXIS1(dataset, it), num_bits_per);

        // Print input/output pair with a carriage return at the beginning
        input_sparsity = calculate_sparsity(MATRIX_AXIS1(dataset, it), num_bits_per);
        column_activation_sparsity = calculate_sparsity(p.column_activations, num_columns);

        printf("%f%%\t->\t%f%%\n", input_sparsity * 100, column_activation_sparsity * 100);

        // fflush(stdout); // Ensure the output is displayed immediately
        
        // sleep(1); // Simulate processing time between outputs

        // Move cursor up two lines
        // if(it < num_datapoints-1) printf("\033[F");
    }

    printf("\n\n");

    return 0;
}
