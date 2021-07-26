// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "register.hpp"

namespace cldnn {
namespace ocl {

#define REGISTER_OCL(prim)                      \
    static detail::attach_##prim##_impl attach_##prim

void register_implementations() {
    REGISTER_OCL(activation);
    REGISTER_OCL(arg_max_min);
    REGISTER_OCL(average_unpooling);
    REGISTER_OCL(binary_convolution);
    REGISTER_OCL(border);
    REGISTER_OCL(broadcast);
    REGISTER_OCL(concatenation);
    REGISTER_OCL(convolution);
    REGISTER_OCL(crop);
    REGISTER_OCL(custom_gpu_primitive);
    REGISTER_OCL(deconvolution);
    REGISTER_OCL(deformable_conv);
    REGISTER_OCL(deformable_interp);
    REGISTER_OCL(depth_to_space);
    REGISTER_OCL(batch_to_space);
    REGISTER_OCL(eltwise);
    REGISTER_OCL(fully_connected);
    REGISTER_OCL(gather);
    REGISTER_OCL(gather_nd);
    REGISTER_OCL(gemm);
    REGISTER_OCL(lrn);
    REGISTER_OCL(lstm_gemm);
    REGISTER_OCL(lstm_elt);
    REGISTER_OCL(max_unpooling);
    REGISTER_OCL(mutable_data);
    REGISTER_OCL(mvn);
    REGISTER_OCL(normalize);
    REGISTER_OCL(one_hot);
    REGISTER_OCL(permute);
    REGISTER_OCL(pooling);
    REGISTER_OCL(pyramid_roi_align);
    REGISTER_OCL(quantize);
    REGISTER_OCL(reduce);
    REGISTER_OCL(region_yolo);
    REGISTER_OCL(reorder);
    REGISTER_OCL(reorg_yolo);
    REGISTER_OCL(reshape);
    REGISTER_OCL(reverse_sequence);
    REGISTER_OCL(roi_pooling);
    REGISTER_OCL(scale);
    REGISTER_OCL(scatter_update);
    REGISTER_OCL(scatter_nd_update);
    REGISTER_OCL(scatter_elements_update);
    REGISTER_OCL(select);
    REGISTER_OCL(shuffle_channels);
    REGISTER_OCL(softmax);
    REGISTER_OCL(space_to_batch);
    REGISTER_OCL(space_to_depth);
    REGISTER_OCL(strided_slice);
    REGISTER_OCL(tile);
    REGISTER_OCL(fused_conv_eltwise);
    REGISTER_OCL(lstm_dynamic_input);
    REGISTER_OCL(lstm_dynamic_timeloop);
    REGISTER_OCL(generic_layer);
    REGISTER_OCL(gather_tree);
    REGISTER_OCL(resample);
    REGISTER_OCL(grn);
    REGISTER_OCL(ctc_greedy_decoder);
    REGISTER_OCL(cum_sum);
    REGISTER_OCL(embedding_bag);
    REGISTER_OCL(extract_image_patches);
}

}  // namespace ocl
}  // namespace cldnn
