// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "register_gpu.hpp"

namespace cldnn { namespace gpu {

#define REGISTER_GPU(prim)                      \
    static detail::attach_##prim##_gpu attach_##prim

void register_implementations_gpu() {
    REGISTER_GPU(activation);
    REGISTER_GPU(arg_max_min);
    REGISTER_GPU(average_unpooling);
    REGISTER_GPU(binary_convolution);
    REGISTER_GPU(border);
    REGISTER_GPU(broadcast);
    REGISTER_GPU(concatenation);
    REGISTER_GPU(condition);
    REGISTER_GPU(convolution);
    REGISTER_GPU(crop);
    REGISTER_GPU(custom_gpu_primitive);
    REGISTER_GPU(data);
    REGISTER_GPU(deconvolution);
    REGISTER_GPU(deformable_conv);
    REGISTER_GPU(deformable_interp);
    REGISTER_GPU(depth_to_space);
    REGISTER_GPU(batch_to_space);
    REGISTER_GPU(detection_output);
    REGISTER_GPU(eltwise);
    REGISTER_GPU(fully_connected);
    REGISTER_GPU(gather);
    REGISTER_GPU(gather_nd);
    REGISTER_GPU(gemm);
    REGISTER_GPU(input_layout);
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
    REGISTER_GPU(scatter_update);
    REGISTER_GPU(scatter_nd_update);
    REGISTER_GPU(scatter_elements_update);
    REGISTER_GPU(select);
    REGISTER_GPU(shuffle_channels);
    REGISTER_GPU(softmax);
    REGISTER_GPU(space_to_batch);
    REGISTER_GPU(space_to_depth);
    REGISTER_GPU(strided_slice);
    REGISTER_GPU(tile);
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
    REGISTER_GPU(extract_image_patches);
    REGISTER_GPU(loop);
}

}  // namespace gpu
}  // namespace cldnn
