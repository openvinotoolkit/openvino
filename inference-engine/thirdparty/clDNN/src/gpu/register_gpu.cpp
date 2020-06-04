/*
// Copyright (c) 2016-2020 Intel Corporation
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
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "register_gpu.hpp"

namespace cldnn { namespace gpu {

#define REGISTER_GPU(prim)                      \
    static detail::attach_##prim##_gpu attach_##prim

void register_implementations_gpu() {
    REGISTER_GPU(activation);
    REGISTER_GPU(activation_grad);
    REGISTER_GPU(apply_adam);
    REGISTER_GPU(arg_max_min);
    REGISTER_GPU(average_unpooling);
    REGISTER_GPU(batch_norm);
    REGISTER_GPU(batch_norm_grad);
    REGISTER_GPU(binary_convolution);
    REGISTER_GPU(border);
    REGISTER_GPU(broadcast);
    REGISTER_GPU(concatenation);
    REGISTER_GPU(condition);
    REGISTER_GPU(contract);
    REGISTER_GPU(convolution);
    REGISTER_GPU(convolution_grad_weights);
    REGISTER_GPU(crop);
    REGISTER_GPU(custom_gpu_primitive);
    REGISTER_GPU(data);
    REGISTER_GPU(deconvolution);
    REGISTER_GPU(deformable_conv);
    REGISTER_GPU(deformable_interp);
    REGISTER_GPU(depth_to_space);
    REGISTER_GPU(detection_output);
    REGISTER_GPU(eltwise);
    REGISTER_GPU(embed);
    REGISTER_GPU(fully_connected);
    REGISTER_GPU(fully_connected_grad_input);
    REGISTER_GPU(fully_connected_grad_weights);
    REGISTER_GPU(gather);
    REGISTER_GPU(gemm);
    REGISTER_GPU(index_select);
    REGISTER_GPU(input_layout);
    REGISTER_GPU(lookup_table);
    REGISTER_GPU(lrn);
    REGISTER_GPU(lstm_gemm);
    REGISTER_GPU(lstm_elt);
    REGISTER_GPU(max_unpooling);
    REGISTER_GPU(mutable_data);
    REGISTER_GPU(mvn);
    REGISTER_GPU(normalize);
    REGISTER_GPU(one_hot);
    REGISTER_GPU(permute);
    REGISTER_GPU(pooling);
    REGISTER_GPU(prior_box);
    REGISTER_GPU(proposal);
    REGISTER_GPU(pyramid_roi_align);
    REGISTER_GPU(quantize);
    REGISTER_GPU(reduce);
    REGISTER_GPU(region_yolo);
    REGISTER_GPU(reorder);
    REGISTER_GPU(reorg_yolo);
    REGISTER_GPU(reshape);
    REGISTER_GPU(reverse_sequence);
    REGISTER_GPU(roi_pooling);
    REGISTER_GPU(scale);
    REGISTER_GPU(scale_grad_input);
    REGISTER_GPU(scale_grad_weights);
    REGISTER_GPU(select);
    REGISTER_GPU(shuffle_channels);
    REGISTER_GPU(softmax);
    REGISTER_GPU(softmax_loss_grad);
    REGISTER_GPU(space_to_depth);
    REGISTER_GPU(strided_slice);
    REGISTER_GPU(tile);
    REGISTER_GPU(fused_conv_bn_scale);
    REGISTER_GPU(fused_conv_eltwise);
    REGISTER_GPU(lstm_dynamic_input);
    REGISTER_GPU(lstm_dynamic_timeloop);
    REGISTER_GPU(generic_layer);
    REGISTER_GPU(gather_tree);
    REGISTER_GPU(resample);
    REGISTER_GPU(non_max_suppression);
    REGISTER_GPU(grn);
    REGISTER_GPU(ctc_greedy_decoder);
    REGISTER_GPU(cum_sum);
    REGISTER_GPU(embedding_bag);
}

}  // namespace gpu
}  // namespace cldnn
