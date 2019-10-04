// Copyright (c) 2019 Intel Corporation
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

KERNEL(gather_tree_gpu_ref.cl)(
    const __global UNIT_TYPE* step_input,
    const __global UNIT_TYPE* parent_input,
    const __global UNIT_TYPE* max_seq_len_input,
    const __global UNIT_TYPE* end_token,
    __global UNIT_TYPE* output)
{
    const uint beam = get_global_id(0);
    const uint batch = get_global_id(1);
    /*
         b -> time
         f -> batch
         y -> beam
    */
    uint parent = beam;
    for(int time = INPUT0_BATCH_NUM - 1; time >= 0; time--) {

        while (time >= (uint)max_seq_len_input[batch]) {
            output[OUTPUT_GET_INDEX(time, batch, beam, 0)] = end_token[0];
            time--;
        }
        output[OUTPUT_GET_INDEX(time, batch, beam, 0)] =
            step_input[INPUT0_GET_INDEX(time, batch, parent, 0)];
        parent = (uint)parent_input[INPUT0_GET_INDEX(time, batch, parent, 0)];
    }

}
