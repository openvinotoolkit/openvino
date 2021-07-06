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
#include "cldnn/primitives/condition.hpp"
#include "cldnn/primitives/convolution.hpp"
#include "cldnn/primitives/crop.hpp"
#include "cldnn/primitives/custom_gpu_primitive.hpp"
#include "cldnn/primitives/data.hpp"
#include "cldnn/primitives/deconvolution.hpp"
#include "cldnn/primitives/depth_to_space.hpp"
#include "cldnn/primitives/detection_output.hpp"
#include "cldnn/primitives/eltwise.hpp"
#include "cldnn/primitives/fully_connected.hpp"
#include "cldnn/primitives/gather.hpp"
#include "cldnn/primitives/gather_nd.hpp"
#include "cldnn/primitives/gemm.hpp"
#include "cldnn/primitives/input_layout.hpp"
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
#include "cldnn/primitives/prior_box.hpp"
#include "cldnn/primitives/proposal.hpp"
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
#include "cldnn/primitives/non_max_suppression.hpp"
#include "cldnn/primitives/grn.hpp"
#include "cldnn/primitives/ctc_greedy_decoder.hpp"
#include "cldnn/primitives/loop.hpp"
#include "generic_layer.hpp"


namespace cldnn { namespace gpu {
void register_implementations_gpu();

namespace detail {

#define REGISTER_GPU(prim)              \
    struct attach_##prim##_gpu {        \
        attach_##prim##_gpu();          \
    }

REGISTER_GPU(activation);
REGISTER_GPU(arg_max_min);
REGISTER_GPU(average_unpooling);
REGISTER_GPU(batch_to_space);
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
REGISTER_GPU(detection_output);
REGISTER_GPU(eltwise);
REGISTER_GPU(embed);
REGISTER_GPU(fully_connected);
REGISTER_GPU(gather);
REGISTER_GPU(gather_nd);
REGISTER_GPU(gemm);
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
REGISTER_GPU(scatter_update);
REGISTER_GPU(scatter_elements_update);
REGISTER_GPU(scatter_nd_update);
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

#undef REGISTER_GPU

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
