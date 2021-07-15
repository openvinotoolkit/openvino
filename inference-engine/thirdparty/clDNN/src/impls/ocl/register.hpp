// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "cldnn/primitives/activation.hpp"
#include "cldnn/primitives/arg_max_min.hpp"
#include "cldnn/primitives/average_unpooling.hpp"
#include "cldnn/primitives/batch_to_space.hpp"
#include "cldnn/primitives/binary_convolution.hpp"
#include "cldnn/primitives/border.hpp"
#include "cldnn/primitives/broadcast.hpp"
#include "cldnn/primitives/concatenation.hpp"
#include "cldnn/primitives/convolution.hpp"
#include "cldnn/primitives/crop.hpp"
#include "cldnn/primitives/custom_gpu_primitive.hpp"
#include "cldnn/primitives/deconvolution.hpp"
#include "cldnn/primitives/depth_to_space.hpp"
#include "cldnn/primitives/eltwise.hpp"
#include "cldnn/primitives/fully_connected.hpp"
#include "cldnn/primitives/gather.hpp"
#include "cldnn/primitives/gather_nd.hpp"
#include "cldnn/primitives/gemm.hpp"
#include "cldnn/primitives/lrn.hpp"
#include "cldnn/primitives/lstm.hpp"
#include "cldnn/primitives/lstm_dynamic.hpp"
#include "cldnn/primitives/max_unpooling.hpp"
#include "cldnn/primitives/mutable_data.hpp"
#include "cldnn/primitives/mvn.hpp"
#include "cldnn/primitives/normalize.hpp"
#include "cldnn/primitives/one_hot.hpp"
#include "cldnn/primitives/permute.hpp"
#include "cldnn/primitives/pooling.hpp"
#include "cldnn/primitives/pyramid_roi_align.hpp"
#include "cldnn/primitives/quantize.hpp"
#include "cldnn/primitives/reduce.hpp"
#include "cldnn/primitives/region_yolo.hpp"
#include "cldnn/primitives/reorder.hpp"
#include "cldnn/primitives/reorg_yolo.hpp"
#include "cldnn/primitives/reshape.hpp"
#include "cldnn/primitives/reverse_sequence.hpp"
#include "cldnn/primitives/roi_pooling.hpp"
#include "cldnn/primitives/scale.hpp"
#include "cldnn/primitives/scatter_update.hpp"
#include "cldnn/primitives/scatter_elements_update.hpp"
#include "cldnn/primitives/scatter_nd_update.hpp"
#include "cldnn/primitives/select.hpp"
#include "cldnn/primitives/shuffle_channels.hpp"
#include "cldnn/primitives/softmax.hpp"
#include "cldnn/primitives/space_to_batch.hpp"
#include "cldnn/primitives/strided_slice.hpp"
#include "cldnn/primitives/tile.hpp"
#include "cldnn/primitives/resample.hpp"
#include "cldnn/primitives/gather_tree.hpp"
#include "cldnn/primitives/fused_conv_eltwise.hpp"
#include "cldnn/primitives/lstm_dynamic_input.hpp"
#include "cldnn/primitives/lstm_dynamic_timeloop.hpp"
#include "cldnn/primitives/grn.hpp"
#include "cldnn/primitives/ctc_greedy_decoder.hpp"
#include "generic_layer.hpp"


namespace cldnn {
namespace ocl {
void register_implementations();

namespace detail {

#define REGISTER_OCL(prim)              \
    struct attach_##prim##_impl {        \
        attach_##prim##_impl();          \
    }

REGISTER_OCL(activation);
REGISTER_OCL(arg_max_min);
REGISTER_OCL(average_unpooling);
REGISTER_OCL(batch_to_space);
REGISTER_OCL(binary_convolution);
REGISTER_OCL(border);
REGISTER_OCL(broadcast);
REGISTER_OCL(concatenation);
REGISTER_OCL(convolution);
REGISTER_OCL(crop);
REGISTER_OCL(custom_gpu_primitive);
REGISTER_OCL(data);
REGISTER_OCL(deconvolution);
REGISTER_OCL(deformable_conv);
REGISTER_OCL(deformable_interp);
REGISTER_OCL(depth_to_space);
REGISTER_OCL(eltwise);
REGISTER_OCL(embed);
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
REGISTER_OCL(scatter_elements_update);
REGISTER_OCL(scatter_nd_update);
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

#undef REGISTER_OCL

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
