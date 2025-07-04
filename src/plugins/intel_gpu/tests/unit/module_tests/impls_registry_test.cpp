// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "registry/registry.hpp"
#include "intel_gpu/primitives/adaptive_pooling.hpp"
#include "intel_gpu/primitives/arg_max_min.hpp"
#include "intel_gpu/primitives/assign.hpp"
#include "intel_gpu/primitives/batch_to_space.hpp"
#include "intel_gpu/primitives/border.hpp"
#include "intel_gpu/primitives/broadcast.hpp"
#include "intel_gpu/primitives/bucketize.hpp"
#include "intel_gpu/primitives/condition.hpp"
#include "intel_gpu/primitives/convert_color.hpp"
#include "intel_gpu/primitives/crop.hpp"
#include "intel_gpu/primitives/ctc_greedy_decoder.hpp"
#include "intel_gpu/primitives/ctc_loss.hpp"
#include "intel_gpu/primitives/cum_sum.hpp"
#include "intel_gpu/primitives/custom_gpu_primitive.hpp"
#include "intel_gpu/primitives/deconvolution.hpp"
#include "intel_gpu/primitives/depth_to_space.hpp"
#include "intel_gpu/primitives/detection_output.hpp"
#include "intel_gpu/primitives/dft.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/embedding_bag.hpp"
#include "intel_gpu/primitives/experimental_detectron_detection_output.hpp"
#include "intel_gpu/primitives/experimental_detectron_generate_proposals_single_image.hpp"
#include "intel_gpu/primitives/experimental_detectron_prior_grid_generator.hpp"
#include "intel_gpu/primitives/experimental_detectron_roi_feature_extractor.hpp"
#include "intel_gpu/primitives/experimental_detectron_topk_rois.hpp"
#include "intel_gpu/primitives/extract_image_patches.hpp"
#include "intel_gpu/primitives/eye.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/gather.hpp"
#include "intel_gpu/primitives/gather_elements.hpp"
#include "intel_gpu/primitives/gather_nd.hpp"
#include "intel_gpu/primitives/gather_tree.hpp"
#include "intel_gpu/primitives/gemm.hpp"
#include "intel_gpu/primitives/generate_proposals.hpp"
#include "intel_gpu/primitives/grid_sample.hpp"
#include "intel_gpu/primitives/grn.hpp"
#include "intel_gpu/primitives/group_normalization.hpp"
#include "intel_gpu/primitives/kv_cache.hpp"
#include "intel_gpu/primitives/loop.hpp"
#include "intel_gpu/primitives/matrix_nms.hpp"
#include "intel_gpu/primitives/multiclass_nms.hpp"
#include "intel_gpu/primitives/multinomial.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/mvn.hpp"
#include "intel_gpu/primitives/non_max_suppression.hpp"
#include "intel_gpu/primitives/non_zero.hpp"
#include "intel_gpu/primitives/one_hot.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
#include "intel_gpu/primitives/permute.hpp"
#include "intel_gpu/primitives/prior_box.hpp"
#include "intel_gpu/primitives/proposal.hpp"
#include "intel_gpu/primitives/quantize.hpp"
#include "intel_gpu/primitives/random_uniform.hpp"
#include "intel_gpu/primitives/range.hpp"
#include "intel_gpu/primitives/read_value.hpp"
#include "intel_gpu/primitives/reduce.hpp"
#include "intel_gpu/primitives/region_yolo.hpp"
#include "intel_gpu/primitives/reorg_yolo.hpp"
#include "intel_gpu/primitives/resample.hpp"
#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/primitives/reverse.hpp"
#include "intel_gpu/primitives/reverse_sequence.hpp"
#include "intel_gpu/primitives/rms.hpp"
#include "intel_gpu/primitives/roi_align.hpp"
#include "intel_gpu/primitives/roll.hpp"
#include "intel_gpu/primitives/rope.hpp"
#include "intel_gpu/primitives/scaled_dot_product_attention.hpp"
#include "intel_gpu/primitives/scatter_elements_update.hpp"
#include "intel_gpu/primitives/scatter_nd_update.hpp"
#include "intel_gpu/primitives/scatter_update.hpp"
#include "intel_gpu/primitives/select.hpp"
#include "intel_gpu/primitives/shape_of.hpp"
#include "intel_gpu/primitives/shuffle_channels.hpp"
#include "intel_gpu/primitives/slice.hpp"
#include "intel_gpu/primitives/space_to_batch.hpp"
#include "intel_gpu/primitives/space_to_depth.hpp"
#include "intel_gpu/primitives/strided_slice.hpp"
#include "intel_gpu/primitives/swiglu.hpp"
#include "intel_gpu/primitives/tile.hpp"
#include "intel_gpu/primitives/unique.hpp"
#include "intel_gpu/primitives/fake_convert.hpp"
#include "primitive_inst.h"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

template <typename PType, typename... Args, typename std::enable_if<(sizeof...(Args) == 0), bool>::type = true>
void check_impl() {
    const auto& all_impls = ov::intel_gpu::Registry<PType>::get_implementations();
    ASSERT_GT(all_impls.size(), 0);
    size_t actual_impls_count = 0;
    for (size_t i = 0; i < all_impls.size(); i++) {
        ASSERT_NE(all_impls[i], nullptr) << " Implementation " << i << " of " << PType().type_string();
        if (std::dynamic_pointer_cast<ImplementationManagerLegacy<PType>>(all_impls[i]) != nullptr)
            actual_impls_count++;
    }

    std::vector<shape_types> shapes = {shape_types::static_shape, shape_types::dynamic_shape};
    std::vector<impl_types> impls = {impl_types::ocl, impl_types::cpu, impl_types::common, impl_types::onednn};

    size_t expected_impls_count = 0;
    for (auto& impl : impls) {
        for (auto& shape : shapes) {
            if (implementation_map<PType>::get(impl, shape) != nullptr)
                expected_impls_count++;
        }
    }

    ASSERT_EQ(expected_impls_count, actual_impls_count) << " for " << PType().type_string();
}

template <typename PType, typename... Args, typename std::enable_if<(sizeof...(Args) > 0), bool>::type = true>
void check_impl() {
    check_impl<PType>();
    check_impl<Args...>();
}

template <typename... Args>
void check_impls() {
    check_impl<Args...>();
}

}  // namespace

TEST(registry_test, no_null_impls) {
    program p(get_test_engine(), get_test_default_config(get_test_engine()));  // dummy program to register impls
    check_impls<cldnn::concatenation,
                cldnn::convolution,
                cldnn::deconvolution,
                cldnn::fully_connected,
                cldnn::gemm,
                cldnn::pooling,
                cldnn::reduce,
                cldnn::reorder,
                cldnn::assign,
                cldnn::read_value,
                cldnn::condition,
                cldnn::loop,
                cldnn::input_layout,
                cldnn::non_max_suppression_gather,
                cldnn::proposal,
                cldnn::activation,
                cldnn::adaptive_pooling,
                cldnn::arg_max_min,
                cldnn::batch_to_space,
                cldnn::border,
                cldnn::broadcast,
                cldnn::bucketize,
                cldnn::crop,
                cldnn::custom_gpu_primitive,
                cldnn::data,
                cldnn::depth_to_space,
                cldnn::detection_output,
                cldnn::dft,
                cldnn::experimental_detectron_detection_output,
                cldnn::experimental_detectron_generate_proposals_single_image,
                cldnn::experimental_detectron_prior_grid_generator,
                cldnn::experimental_detectron_roi_feature_extractor,
                cldnn::experimental_detectron_topk_rois,
                cldnn::eltwise,
                cldnn::gather,
                cldnn::gather_nd,
                cldnn::gather_elements,
                cldnn::generate_proposals,
                cldnn::grid_sample,
                cldnn::group_normalization,
                cldnn::kv_cache,
                cldnn::lrn,
                cldnn::multiclass_nms,
                cldnn::multinomial,
                cldnn::mutable_data,
                cldnn::mvn,
                cldnn::non_max_suppression,
                cldnn::matrix_nms,
                cldnn::normalize,
                cldnn::one_hot,
                cldnn::permute,
                cldnn::paged_attention,
                cldnn::prior_box,
                cldnn::quantize,
                cldnn::random_uniform,
                cldnn::range,
                cldnn::region_yolo,
                cldnn::reorg_yolo,
                cldnn::reshape,
                cldnn::reverse,
                cldnn::reverse_sequence,
                cldnn::rms,
                cldnn::roi_align,
                cldnn::roi_pooling,
                cldnn::roll,
                cldnn::scatter_update,
                cldnn::scatter_elements_update,
                cldnn::scatter_nd_update,
                cldnn::select,
                cldnn::shape_of,
                cldnn::shuffle_channels,
                cldnn::slice,
                cldnn::softmax,
                cldnn::space_to_batch,
                cldnn::space_to_depth,
                cldnn::strided_slice,
                cldnn::swiglu,
                cldnn::tile,
                cldnn::gather_tree,
                cldnn::resample,
                cldnn::grn,
                cldnn::ctc_greedy_decoder,
                cldnn::ctc_loss,
                cldnn::cum_sum,
                cldnn::embedding_bag,
                cldnn::extract_image_patches,
                cldnn::convert_color,
                cldnn::count_nonzero,
                cldnn::gather_nonzero,
                cldnn::eye,
                cldnn::unique_count,
                cldnn::unique_gather,
                cldnn::scaled_dot_product_attention,
                cldnn::rope,
                cldnn::fake_convert>();
}
