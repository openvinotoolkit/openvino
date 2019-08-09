// Copyright (C) 2019 Intel Corporation
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

__kernel void ctc_ocl(__global half* probabilities,
                     __global half* output_sequences,
                     int C_)
{
    size_t t = get_global_id(0);

    __global half* probs = probabilities + t * C_;

    int max_class_idx = 0;
    half max_prob = probs[0];
    ++probs;
    for (int c = 1 ; c < C_ ; c++, ++probs)
    {
        if (*probs > max_prob)
        {
            max_prob  = *probs;
            max_class_idx = c;
        }
    }
    output_sequences[t] = (half)max_class_idx;
}

__kernel void postProcess(__global half* input,
                          __global half* output,
                          __global half* seq_ind,
                          int height,
                          int width,
                          int classes)
{
    int wr_index = 0;
    int rd_index = 0;

    half update_data;
    int update_index;

    for (int i = 0; i < classes; i++)
    {
        output[i] = (half)(-1);
    }

    for (int n = 0; n < height; ++n)
    {
        int prev_class_id = -1;
        for (int t = 0; t < classes; ++t)
        {
            int class_id = (int)input[rd_index++];
            update_index = wr_index;
            update_data = output[update_index];

            if ((class_id < (width - 1)) && !(1 && class_id == prev_class_id))
            {
                update_data = (half)class_id;
                wr_index++;

            }
            output[update_index] = update_data;
            prev_class_id = class_id;

            if (seq_ind[t + 1] == 0 ) {
                break;
            }
        }
    }
}

__kernel void ctc_ref_fp16(__global half* probabilities, __global half* seq_ind, __global half* output_sequences, int C, int H, int W)
{
    int T_ = C;
    int N_ = H;
    int C_ = W;

    // Fill output_sequences with -1
    for (int i = 0; i < T_; i++)
    {
        output_sequences[i] = (half)(-1.0);
    }
    int output_index = 0;

    // Caffe impl
    for(int n = 0; n < N_; ++n)
    {
        int prev_class_idx = -1;

        for (int t = 0; t < T_; ++t)
        {
            // get maximum probability and its index
            int max_class_idx = 0;
            __global half* probs;
            half max_prob;

            probs = probabilities + t*C_;
            max_prob = probs[0];
            ++probs;

            for (int c = 1; c < C_; ++c, ++probs)
            {
                if (*probs > max_prob)
                {
                    max_class_idx = c;
                    max_prob = *probs;
                }
            }

            //if (max_class_idx != blank_index_
            //        && !(merge_repeated_&& max_class_idx == prev_class_idx))
            if (max_class_idx < C_-1 && !(1 && max_class_idx == prev_class_idx))
            {
                output_sequences[output_index] = (half)max_class_idx;
                output_index++;
            }

            prev_class_idx = max_class_idx;

            // Assume sequence_indicators is always 1
            if (seq_ind[t + 1] == 0)
            {
                break;
            }
        }
    }
}
