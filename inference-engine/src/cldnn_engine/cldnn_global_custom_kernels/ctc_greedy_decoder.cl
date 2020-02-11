// Copyright (C) 2018-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void ctc_greedy_decoder(const __global INPUT0_TYPE*  probabilities,
                                 const __global INPUT1_TYPE*  sequence_indicators,
                                       __global OUTPUT0_TYPE* output_sequences)
{
    const int dims = sizeof(INPUT0_DIMS) / sizeof(INPUT0_DIMS[0]);
    int T_ = INPUT0_DIMS[0];
    int N_ = INPUT0_DIMS[1];
    int C_ = INPUT0_DIMS[2];

    // Fill output_sequences with -1
    for (int ii = 0; ii < T_*N_; ii++) {
        output_sequences[ii] = (OUTPUT0_TYPE)(-1.0f);
    }

    for (int n = 0; n < N_; ++n) {
        int prev_class_idx = -1;
        int output_index = n*T_;

        for (int t = 0; /* check at end */; ++t) {
            // get maximum probability and its index
            int max_class_idx = 0;

            const __global INPUT0_TYPE* probs = probabilities + t*C_*N_ + n*C_;
            INPUT0_TYPE max_prob = probs[0];
            ++probs;

            for (int c = 1; c < C_; ++c, ++probs) {
                if (*probs > max_prob) {
                    max_class_idx = c;
                    max_prob = *probs;
                }
            }

            if (max_class_idx != C_-1 && !(ctc_merge_repeated_ && max_class_idx == prev_class_idx)) {
                output_sequences[output_index] =  max_class_idx;
                output_index++;
            }

            prev_class_idx = max_class_idx;

            if (t + 1 == T_ || sequence_indicators[(t + 1)*N_ + n] == 0) {
                break;
            }
        }
    }
}
