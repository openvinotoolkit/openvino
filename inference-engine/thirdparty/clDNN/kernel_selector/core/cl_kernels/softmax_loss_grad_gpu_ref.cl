// Copyright (c) 2018 Intel Corporation
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

KERNEL(softmax_loss_grad_gpu_ref)(
    const __global INPUT0_TYPE* input_pred,
    __global OUTPUT_TYPE* output,
    const __global INPUT1_TYPE* labels
    )
{
    const uint b_x          = get_global_id(0);
    const uint batch_id     = b_x / OUTPUT_SIZE_X;
    const uint x            = b_x % OUTPUT_SIZE_X;

    const uint input_pred_idx = GET_DATA_INDEX(INPUT0, batch_id, 0, 0, x);
    const uint labels_idx = GET_DATA_INDEX(INPUT1, batch_id, 0, 0, 0);

    UNIT_TYPE label = labels[labels_idx];
    const uint output_idx = GET_DATA_INDEX(OUTPUT, batch_id, 0, 0, x);

    if(label == x)
        output[output_idx] = input_pred[input_pred_idx] - 1;
    else
        output[output_idx] = input_pred[input_pred_idx];
}