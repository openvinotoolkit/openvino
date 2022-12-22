// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"
#include "include/batch_headers/fetch_data.cl"

KERNEL(ctc_greedy_decoder_ref)(const __global INPUT0_TYPE* probabilities
                              ,const __global INPUT1_TYPE* sequence_indicators
                                    ,__global OUTPUT_TYPE* output_sequences
#ifdef SECOND_OUTPUT_EXIST
                                    ,__global INPUT2_TYPE* second_output
#endif
                              )
{
    // Fill output_sequences with -1
    for (int ii = 0; ii < T_ * N_; ii++) {
        output_sequences[ii] = (OUTPUT_TYPE)(-1.0f);
    }

    for (int n = 0; n < N_; ++n) {
        int prev_class_idx = -1;
        int output_index = n * T_;

        for (int t = 0; t < T_; ++t) {
            // get maximum probability and its index
#ifdef SECOND_OUTPUT_EXIST
            if (t >= sequence_indicators[n]) break;
#else
            if (sequence_indicators[t * N_ + n] == 0) break;
#endif
            int max_class_idx = 0;
#ifdef SECOND_OUTPUT_EXIST
            const __global INPUT0_TYPE* probs = probabilities + n * C_ * T_ + t * C_;
#else
            const __global INPUT0_TYPE* probs = probabilities + t * C_ * N_ + n * C_;
#endif
            INPUT0_TYPE max_prob = probs[0];
            ++probs;

            for (int c = 1; c < C_; ++c, ++probs) {
                if (*probs > max_prob) {
                    max_class_idx = c;
                    max_prob = *probs;
                }
            }

            if (max_class_idx != blank_index_ && !(ctc_merge_repeated_ && max_class_idx == prev_class_idx)) {
                output_sequences[output_index] = max_class_idx;
                output_index++;
            }

            prev_class_idx = max_class_idx;
        }
#ifdef SECOND_OUTPUT_EXIST
        second_output[n] = output_index - n * T_;
#endif
    }
}
