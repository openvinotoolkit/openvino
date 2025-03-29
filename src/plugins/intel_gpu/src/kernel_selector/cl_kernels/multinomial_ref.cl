// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if WITH_REPLACEMENT

KERNEL (multinomial_ref)(  __global INPUT0_TYPE* cdf
                           , __global INPUT1_TYPE* uniform_samples
                           , __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                           , FUSED_OPS_DECLS
#endif
)
{
    unsigned i = get_global_id(0) * SAMPLES_SIZE;
    unsigned j = get_global_id(1);
    if (i >= BATCH_NUM * SAMPLES_SIZE || j >= SAMPLES_SIZE)
        return;
    unsigned i_translated = i / SAMPLES_SIZE * CLASS_SIZE;
    unsigned selected_class_idx = CLASS_SIZE;
    INPUT1_TYPE sample_value = uniform_samples[i + j];
    for (unsigned k = 0; k < CLASS_SIZE; ++k) {
        if (sample_value <= cdf[i_translated + k]) {
            output[i + j] = (OUTPUT_TYPE) k;
            selected_class_idx = k;
            break;
        }
    }
}

#else

KERNEL (multinomial_ref)(  __global INPUT0_TYPE* cdf
                           , __global INPUT1_TYPE* uniform_samples
                           , __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                           , FUSED_OPS_DECLS
#endif
)
{
    unsigned i = get_global_id(0) * SAMPLES_SIZE;
    if (i >= BATCH_NUM * SAMPLES_SIZE)
        return;
    for (unsigned j = 0; j < SAMPLES_SIZE; ++j) {
        unsigned i_translated = i / SAMPLES_SIZE * CLASS_SIZE;
        unsigned selected_class_idx = CLASS_SIZE;
        INPUT1_TYPE sample_value = uniform_samples[i + j];
        for (unsigned k = 0; k < CLASS_SIZE; ++k) {
            if (sample_value <= cdf[i_translated + k]) {
                output[i + j] = (OUTPUT_TYPE) k;
                selected_class_idx = k;
                break;
            }
        }
        // Additional step without replacement - change probability of a given class to 0, and update the cdf
        INPUT0_TYPE class_probability = selected_class_idx ? cdf[i_translated + selected_class_idx] -
                                                             cdf[i_translated + selected_class_idx - 1]
                                                           : cdf[i_translated + selected_class_idx];
        INPUT0_TYPE divisor = 1 - class_probability;
        for (unsigned k = 0; k < CLASS_SIZE; ++k) {
            if (k >= selected_class_idx)
                cdf[i_translated + k] -= class_probability;
            cdf[i_translated + k] /= divisor;
        }
    }
}

#endif
