// Copyright (c) 2020 Intel Corporation
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


#include "include/include_all.cl"

#ifdef PACKED_SUM
KERNEL(embedding_bag_ref)(
        const __global INPUT0_TYPE* emb_table,
        const __global INPUT1_TYPE* indices,
#ifdef INPUT2_TYPE
        const __global INPUT2_TYPE* weights,
#endif
        __global OUTPUT_TYPE* output)
{
    const uint batch = get_global_id(0);
    const uint emb_dim1 = get_global_id(1);
    const uint emb_dim2 = (uint)get_global_id(2) / OUTPUT_SIZE_X;
    const uint emb_dim3 = (uint)get_global_id(2) % OUTPUT_SIZE_X;

    OUTPUT_TYPE res = OUTPUT_VAL_ZERO;
    for (int i = 0; i < INPUT1_FEATURE_NUM; ++i)
    {
        uint indices_index = INPUT1_GET_INDEX(batch, i, 0, 0);
        uint emb_index = INPUT0_GET_INDEX(indices[indices_index], emb_dim1, emb_dim2, emb_dim3);
        OUTPUT_TYPE val = emb_table[emb_index];
#ifdef INPUT2_TYPE
        {
            uint weight_index = INPUT2_GET_INDEX(batch, i, 0, 0);
            val *= weights[weight_index];
        }
#endif
        res += val;
    }
    uint out_ind = OUTPUT_GET_INDEX(batch, emb_dim1, emb_dim2, emb_dim3);
    output[out_ind] = ACTIVATION(TO_OUTPUT_TYPE(res), ACTIVATION_PARAMS);
}
#endif // PACKED_SUM

#ifdef OFFSETS_SUM
KERNEL(embedding_bag_ref)(
        const __global INPUT0_TYPE* emb_table,
        const __global INPUT1_TYPE* indices,
        const __global INPUT2_TYPE* offsets,
#ifdef INPUT3_TYPE
        const __global INPUT3_TYPE* weights,
#endif
        __global OUTPUT_TYPE* output)
{
    const uint batch = get_global_id(0);
    const uint emb_dim1 = get_global_id(1);
    const uint emb_dim2 = (uint)get_global_id(2) / OUTPUT_SIZE_X;
    const uint emb_dim3 = (uint)get_global_id(2) % OUTPUT_SIZE_X;

    uint offsets_ind = INPUT2_OFFSET + batch;
    uint start_indices = INPUT1_OFFSET + offsets[offsets_ind];
    offsets_ind = INPUT2_OFFSET + batch + 1;
    uint end_indices = (batch < OUTPUT_BATCH_NUM - 1) ?
                            INPUT1_OFFSET + offsets[offsets_ind] :
                            INPUT1_LENGTH;

    OUTPUT_TYPE res = OUTPUT_VAL_ZERO;
    for (int i = start_indices; i < end_indices; ++i)
    {
        uint indices_index = INPUT1_OFFSET + i;
        uint emb_index = INPUT0_GET_INDEX(indices[indices_index], emb_dim1, emb_dim2, emb_dim3);
        OUTPUT_TYPE val = emb_table[emb_index];
#ifdef INPUT3_TYPE
        {
            uint weight_index = INPUT3_OFFSET + i;
            val *= weights[weight_index];
        }
#endif
        res += val;
    }

#ifdef DEFAULT_INDEX
    if (start_indices == end_indices) {
        {
            uint emb_index = INPUT0_GET_INDEX(DEFAULT_INDEX, emb_dim1, emb_dim2, emb_dim3);
            res = emb_table[emb_index];
        }
    }
#endif

    uint out_ind = OUTPUT_GET_INDEX(batch, emb_dim1, emb_dim2, emb_dim3);
    output[out_ind] = ACTIVATION(TO_OUTPUT_TYPE(res), ACTIVATION_PARAMS);
}
#endif // OFFSETS_SUM

#ifdef SEGMENTS_SUM
KERNEL(embedding_bag_ref)(
        const __global INPUT0_TYPE* emb_table,
        const __global INPUT1_TYPE* indices,
        const __global INPUT2_TYPE* segment_ids,
#ifdef INPUT3_TYPE
        const __global INPUT3_TYPE* weights,
#endif
        __global OUTPUT_TYPE* output)
{
    const uint batch = get_global_id(0);
    const uint emb_dim1 = get_global_id(1);
    const uint emb_dim2 = (uint)get_global_id(2) / OUTPUT_SIZE_X;
    const uint emb_dim3 = (uint)get_global_id(2) % OUTPUT_SIZE_X;

    OUTPUT_TYPE res = OUTPUT_VAL_ZERO;
    bool found = false;
    for (int i = 0; i < INPUT2_LENGTH; ++i) {
        uint id = segment_ids[INPUT2_OFFSET + i];
        if (id > batch)
            break;
        if (id == batch) {
            found = true;
            uint index = indices[INPUT1_OFFSET + i];
            uint emb_index = INPUT0_GET_INDEX(index, emb_dim1, emb_dim2, emb_dim3);
            OUTPUT_TYPE val = emb_table[emb_index];
#ifdef INPUT3_TYPE
            {
                uint weight_index = INPUT3_OFFSET + i;
                val *= weights[weight_index];
            }
#endif
            res += val;
        }
    }

#ifdef DEFAULT_INDEX
    if (!found) {
        uint emb_index = INPUT0_GET_INDEX(DEFAULT_INDEX, emb_dim1, emb_dim2, emb_dim3);
        res = emb_table[emb_index];
    }
#endif

    uint out_ind = OUTPUT_GET_INDEX(batch, emb_dim1, emb_dim2, emb_dim3);
    output[out_ind] = ACTIVATION(TO_OUTPUT_TYPE(res), ACTIVATION_PARAMS);
}
#endif // SEGMENTS_SUM
