// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "register.hpp"

namespace cldnn {
namespace ocl {

#define REGISTER_OCL(prim) static detail::attach_##prim##_impl attach_##prim

void register_implementations() {
    REGISTER_OCL(activation);
    REGISTER_OCL(adaptive_pooling);
    REGISTER_OCL(arg_max_min);
    REGISTER_OCL(border);
    REGISTER_OCL(broadcast);
    REGISTER_OCL(bucketize);
    REGISTER_OCL(concatenation);
    REGISTER_OCL(convolution);
    REGISTER_OCL(crop);
    REGISTER_OCL(custom_gpu_primitive);
    REGISTER_OCL(deconvolution);
    REGISTER_OCL(deformable_conv);
    REGISTER_OCL(deformable_interp);
    REGISTER_OCL(depth_to_space);
    REGISTER_OCL(detection_output);
    REGISTER_OCL(dft);
    REGISTER_OCL(batch_to_space);
    REGISTER_OCL(experimental_detectron_detection_output);
    REGISTER_OCL(experimental_detectron_generate_proposals_single_image);
    REGISTER_OCL(experimental_detectron_prior_grid_generator);
    REGISTER_OCL(experimental_detectron_roi_feature_extractor);
    REGISTER_OCL(experimental_detectron_topk_rois);
    REGISTER_OCL(eltwise);
    REGISTER_OCL(fully_connected);
    REGISTER_OCL(gather);
    REGISTER_OCL(gather_elements);
    REGISTER_OCL(gather_nd);
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
    REGISTER_OCL(scatter_nd_update);
    REGISTER_OCL(scatter_elements_update);
    REGISTER_OCL(select);
    REGISTER_OCL(shape_of);
    REGISTER_OCL(shuffle_channels);
    REGISTER_OCL(softmax);
    REGISTER_OCL(space_to_batch);
    REGISTER_OCL(space_to_depth);
    REGISTER_OCL(slice);
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
}

}  // namespace ocl
}  // namespace cldnn
