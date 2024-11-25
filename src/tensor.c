#include "tensor.h"

#define INSTANTIATE_TENSOR_INIT(symbol) \
    void tensor_##symbol##_init(TENSOR_TYPE(symbol)* t, u32 shape1, u32 shape2, u32 shape3) { \
        TENSOR_INIT(t, shape1, shape2, shape3, DATA_TYPE(symbol)); \
    }

INSTANTIATE_TENSOR_INIT(u16)
INSTANTIATE_TENSOR_INIT(u8)

#define INSTANTIATE_MATRIX_INIT(symbol) \
    void matrix_##symbol##_init(MAT_TYPE(symbol)* m, u32 rows, u32 cols) { \
        MATRIX_INIT(m, rows, cols, DATA_TYPE(symbol)); \
    }

INSTANTIATE_MATRIX_INIT(u32)
INSTANTIATE_MATRIX_INIT(u16)
INSTANTIATE_MATRIX_INIT(u8)

void mat_u8_mean(
    f64* mean, mat_u8 dataset, 
    u32 sample_size, u32 num_samples) {
    for(u32 offset_it = 0; offset_it < sample_size; ++offset_it) 
        mean[offset_it] = 0;

    for(u32 sample_it = 0; sample_it < num_samples; ++sample_it) {
        for(u32 offset_it = 0; offset_it < sample_size; ++offset_it) {
            mean[offset_it] += *MATRIX(dataset, sample_it, offset_it);
        }
    }

    for(u32 offset_it = 0; offset_it < sample_size; ++offset_it) 
        mean[offset_it] /= num_samples;
}

void mat_u8_variance(
    f64* variance, mat_u8 dataset, 
    u32 sample_size, u32 num_samples, 
    f64* mean) {
    for(u32 offset_it = 0; offset_it < sample_size; ++offset_it) 
        variance[offset_it] = 0;

    for(u32 sample_it = 0; sample_it < num_samples; ++sample_it) {
        for(u32 offset_it = 0; offset_it < sample_size; ++offset_it) {
            variance[offset_it] += pow(*MATRIX(dataset, sample_it, offset_it) - mean[offset_it], 2);
        }
    }

    for(u32 offset_it = 0; offset_it < sample_size; ++offset_it) 
        variance[offset_it] /= (num_samples - 1);
}
