// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/arg_max_min.hpp"
#include "intel_gpu/primitives/batch_to_space.hpp"
#include "intel_gpu/primitives/border.hpp"
#include "intel_gpu/primitives/broadcast.hpp"
#include "intel_gpu/primitives/bucketize.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/convert_color.hpp"
#include "intel_gpu/primitives/convolution.hpp"
#include "intel_gpu/primitives/crop.hpp"
#include "intel_gpu/primitives/ctc_greedy_decoder.hpp"
#include "intel_gpu/primitives/ctc_loss.hpp"
#include "intel_gpu/primitives/custom_gpu_primitive.hpp"
#include "intel_gpu/primitives/deconvolution.hpp"
#include "intel_gpu/primitives/depth_to_space.hpp"
#include "intel_gpu/primitives/detection_output.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/experimental_detectron_detection_output.hpp"
#include "intel_gpu/primitives/experimental_detectron_prior_grid_generator.hpp"
#include "intel_gpu/primitives/experimental_detectron_roi_feature_extractor.hpp"
#include "intel_gpu/primitives/experimental_detectron_topk_rois.hpp"
#include "intel_gpu/primitives/eye.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/gather.hpp"
#include "intel_gpu/primitives/gather_elements.hpp"
#include "intel_gpu/primitives/gather_nd.hpp"
#include "intel_gpu/primitives/gather_tree.hpp"
#include "intel_gpu/primitives/gemm.hpp"
#include "intel_gpu/primitives/grid_sample.hpp"
#include "intel_gpu/primitives/grn.hpp"
#include "intel_gpu/primitives/group_normalization.hpp"
#include "intel_gpu/primitives/lrn.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/multinomial.hpp"
#include "intel_gpu/primitives/mvn.hpp"
#include "intel_gpu/primitives/non_max_suppression.hpp"
#include "intel_gpu/primitives/normalize.hpp"
#include "intel_gpu/primitives/one_hot.hpp"
#include "intel_gpu/primitives/permute.hpp"
#include "intel_gpu/primitives/pooling.hpp"
#include "intel_gpu/primitives/quantize.hpp"
#include "intel_gpu/primitives/random_uniform.hpp"
#include "intel_gpu/primitives/range.hpp"
#include "intel_gpu/primitives/reduce.hpp"
#include "intel_gpu/primitives/region_yolo.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/reorg_yolo.hpp"
#include "intel_gpu/primitives/resample.hpp"
#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/primitives/reverse_sequence.hpp"
#include "intel_gpu/primitives/rms.hpp"
#include "intel_gpu/primitives/roi_align.hpp"
#include "intel_gpu/primitives/roi_pooling.hpp"
#include "intel_gpu/primitives/roll.hpp"
#include "intel_gpu/primitives/scatter_elements_update.hpp"
#include "intel_gpu/primitives/scatter_nd_update.hpp"
#include "intel_gpu/primitives/scatter_update.hpp"
#include "intel_gpu/primitives/select.hpp"
#include "intel_gpu/primitives/shape_of.hpp"
#include "intel_gpu/primitives/shuffle_channels.hpp"
#include "intel_gpu/primitives/slice.hpp"
#include "intel_gpu/primitives/softmax.hpp"
#include "intel_gpu/primitives/space_to_batch.hpp"
#include "intel_gpu/primitives/strided_slice.hpp"
#include "intel_gpu/primitives/swiglu.hpp"
#include "intel_gpu/primitives/tile.hpp"
#include "intel_gpu/primitives/non_zero.hpp"
#include "intel_gpu/primitives/eye.hpp"
#include "intel_gpu/primitives/unique.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
#include "intel_gpu/primitives/kv_cache.hpp"

namespace cldnn {
namespace ocl {
void register_implementations();

namespace detail {

#define REGISTER_OCL(prim)        \
    struct attach_##prim##_impl { \
        attach_##prim##_impl();   \
    }

REGISTER_OCL(activation);
REGISTER_OCL(adaptive_pooling);
REGISTER_OCL(arg_max_min);
REGISTER_OCL(batch_to_space);
REGISTER_OCL(border);
REGISTER_OCL(broadcast);
REGISTER_OCL(bucketize);
REGISTER_OCL(concatenation);
REGISTER_OCL(convolution);
REGISTER_OCL(crop);
REGISTER_OCL(custom_gpu_primitive);
REGISTER_OCL(data);
REGISTER_OCL(deconvolution);
REGISTER_OCL(deformable_conv);
REGISTER_OCL(deformable_interp);
REGISTER_OCL(depth_to_space);
REGISTER_OCL(detection_output);
REGISTER_OCL(dft);
REGISTER_OCL(experimental_detectron_detection_output);
REGISTER_OCL(experimental_detectron_generate_proposals_single_image);
REGISTER_OCL(experimental_detectron_prior_grid_generator);
REGISTER_OCL(experimental_detectron_roi_feature_extractor);
REGISTER_OCL(experimental_detectron_topk_rois);
REGISTER_OCL(eltwise);
REGISTER_OCL(embed);
REGISTER_OCL(fully_connected);
REGISTER_OCL(gather);
REGISTER_OCL(gather_nd);
REGISTER_OCL(gather_elements);
REGISTER_OCL(gemm);
REGISTER_OCL(generate_proposals);
REGISTER_OCL(grid_sample);
REGISTER_OCL(group_normalization);
REGISTER_OCL(kv_cache);
REGISTER_OCL(paged_attention);
REGISTER_OCL(lrn);
REGISTER_OCL(lstm_elt);
REGISTER_OCL(multiclass_nms);
REGISTER_OCL(multinomial);
REGISTER_OCL(mutable_data);
REGISTER_OCL(mvn);
REGISTER_OCL(non_max_suppression);
REGISTER_OCL(matrix_nms);
REGISTER_OCL(normalize);
REGISTER_OCL(one_hot);
REGISTER_OCL(permute);
REGISTER_OCL(pooling);
REGISTER_OCL(prior_box);
REGISTER_OCL(quantize);
REGISTER_OCL(random_uniform);
REGISTER_OCL(range);
REGISTER_OCL(reduce);
REGISTER_OCL(region_yolo);
REGISTER_OCL(reorder);
REGISTER_OCL(reorg_yolo);
REGISTER_OCL(reshape);
REGISTER_OCL(reverse);
REGISTER_OCL(reverse_sequence);
REGISTER_OCL(rms);
REGISTER_OCL(roi_align);
REGISTER_OCL(roi_pooling);
REGISTER_OCL(roll);
REGISTER_OCL(scatter_update);
REGISTER_OCL(scatter_elements_update);
REGISTER_OCL(scatter_nd_update);
REGISTER_OCL(select);
REGISTER_OCL(shape_of);
REGISTER_OCL(shuffle_channels);
REGISTER_OCL(slice);
REGISTER_OCL(softmax);
REGISTER_OCL(space_to_batch);
REGISTER_OCL(space_to_depth);
REGISTER_OCL(strided_slice);
REGISTER_OCL(swiglu);
REGISTER_OCL(tile);
REGISTER_OCL(gather_tree);
REGISTER_OCL(resample);
REGISTER_OCL(grn);
REGISTER_OCL(ctc_greedy_decoder);
REGISTER_OCL(ctc_loss);
REGISTER_OCL(cum_sum);
REGISTER_OCL(embedding_bag);
REGISTER_OCL(extract_image_patches);
REGISTER_OCL(convert_color);
REGISTER_OCL(count_nonzero);
REGISTER_OCL(gather_nonzero);
REGISTER_OCL(eye);
REGISTER_OCL(unique_count);
REGISTER_OCL(unique_gather);

#undef REGISTER_OCL

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
