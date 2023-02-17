// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/roi_align.hpp"
#include "openvino/op/squeeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_roi_align(NodeContext& context) {
    num_inputs_check(context, 7, 7);
    auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    try {
        auto input_tmp = context.get_input(0);
        auto boxes_tmp = context.get_input(1);

        auto input = std::make_shared<v0::Convert>(input_tmp, element::f32);
        auto boxes = std::make_shared<v1::ConvertLike>(boxes_tmp, input);

        auto spatial_scale = context.const_input<float>(2);
        auto output_size_h = context.const_input<int64_t>(3);
        auto output_size_w = context.const_input<int64_t>(4);
        auto sampling_ratio = context.const_input<int64_t>(5);
        auto aligned = context.const_input<bool>(6);
        v9::ROIAlign::AlignedMode aligned_mode = aligned ? v9::ROIAlign::AlignedMode::HALF_PIXEL_FOR_NN : 
            v9::ROIAlign::AlignedMode::ASYMMETRIC;

        auto const_1 = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {1}));
        auto const_neg_1 = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {-1}));
        auto const_rois_indices = context.mark_node(v0::Constant::create(element::i64, Shape{4}, {1, 2, 3, 4}));
        auto rois = std::make_shared<v8::Gather>(boxes, const_rois_indices, const_1);

        auto const_0 = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0}));
        auto batch_indices_tmp = std::make_shared<v8::Gather>(boxes, const_0, const_1);
        auto batch_indices_tmp_2 = std::make_shared<v1::Reshape>(batch_indices_tmp, const_neg_1, false);
        auto batch_indices = std::make_shared<v0::Convert>(batch_indices_tmp_2, element::i64);

        auto roi_align = std::make_shared<v9::ROIAlign>(input, rois, batch_indices, 
            output_size_h, output_size_w, sampling_ratio, spatial_scale,
            v9::ROIAlign::PoolingMode::AVG, aligned_mode);

        return {context.mark_node(roi_align)};
    } catch(const ov::frontend::GeneralFailure& ex) {
        auto a = ex.what();
        return {zero};
    } catch (const std::exception &exc) {
        std::cerr << exc.what();
        return {zero};
    }
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
