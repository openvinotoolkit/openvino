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

KERNEL(batch_to_space_ref)(const __global INPUT0_TYPE* input, __global OUTPUT_TYPE* output)
{
    const uint batch = get_global_id(0);
    const uint feature = get_global_id(1);

#ifdef OUTPUT_LAYOUT_BFYX
    const uint w = 0;
    const uint z = 0;
    const uint y = (uint)get_global_id(2) / OUTPUT_SIZE_X;
    const uint x = (uint)get_global_id(2) % OUTPUT_SIZE_X;
    const uint input_w = 0;
    const uint input_z = 0;
    const uint offset_w = 0;
    const uint offset_z = 0;
#elif OUTPUT_LAYOUT_BFZYX
    const uint w = 0;
    const uint z = (uint)get_global_id(2) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const uint yx = (uint)get_global_id(2) % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const uint y = yx / OUTPUT_SIZE_X;
    const uint x = yx % OUTPUT_SIZE_X;
    const uint input_w = 0;
    const uint input_z = (z + CROPS_BEGIN_Z) / BLOCK_SHAPE_Z;
    const uint offset_w = 0;
    const uint offset_z = (z + CROPS_BEGIN_Z) % BLOCK_SHAPE_Z;
#elif OUTPUT_LAYOUT_BFWZYX
    const uint w = (uint)get_global_id(2) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z);
    const uint zyx = (uint)get_global_id(2) % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z);
    const uint yx = zyx % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const uint z = zyx / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const uint y = yx / OUTPUT_SIZE_X;
    const uint x = yx % OUTPUT_SIZE_X;
    const uint input_w = (w + CROPS_BEGIN_W) / BLOCK_SHAPE_W;
    const uint input_z = (z + CROPS_BEGIN_Z) / BLOCK_SHAPE_Z;
    const uint offset_w = (w + CROPS_BEGIN_W) % BLOCK_SHAPE_W;
    const uint offset_z = (z + CROPS_BEGIN_Z) % BLOCK_SHAPE_Z;
#endif

    const uint input_feature = (feature + CROPS_BEGIN_FEATURE) / BLOCK_SHAPE_FEATURE;
    const uint offset_feature = (feature + CROPS_BEGIN_FEATURE) % BLOCK_SHAPE_FEATURE;

    const uint input_y = (y + CROPS_BEGIN_Y) / BLOCK_SHAPE_Y;
    const uint offset_y = (y + CROPS_BEGIN_Y) % BLOCK_SHAPE_Y;

    const uint input_x = (x + CROPS_BEGIN_X) / BLOCK_SHAPE_X;
    const uint offset_x = (x + CROPS_BEGIN_X) % BLOCK_SHAPE_X;

    const uint offset_batch = ((offset_feature * BLOCK_SHAPE_W * BLOCK_SHAPE_Z * BLOCK_SHAPE_Y +
                               offset_w * BLOCK_SHAPE_Z * BLOCK_SHAPE_Y +
                               offset_z * BLOCK_SHAPE_Y +
                               offset_y) * BLOCK_SHAPE_X +
                               offset_x) * OUTPUT_BATCH_NUM;
    const uint input_batch = batch + offset_batch;

    const uint input_index = INPUT0_OFFSET +
                             input_batch * INPUT0_BATCH_PITCH +
                             input_feature * INPUT0_FEATURE_PITCH +
                             input_w * INPUT0_W_PITCH +
                             input_z * INPUT0_Z_PITCH +
                             input_y * INPUT0_Y_PITCH +
                             input_x;

    const uint output_index = OUTPUT_OFFSET +
                              batch * OUTPUT_BATCH_PITCH +
                              feature * OUTPUT_FEATURE_PITCH +
                              w * OUTPUT_W_PITCH +
                              z * OUTPUT_Z_PITCH +
                              y * OUTPUT_Y_PITCH +
                              x;

    output[output_index] = ACTIVATION(input[input_index], ACTIVATION_PARAMS);
}
