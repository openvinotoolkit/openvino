// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/roi_align_rotated.hpp"
#include "openvino/op/slice.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector mmdeploy_roi_align_rotated(const ov::frontend::onnx::Node& node) {
    const auto inputs = node.get_ov_inputs();

    FRONT_END_GENERAL_CHECK(inputs.size() == 2,
                            "The mmdeploy.MMCVRoIAlignRotated operator expects 2 inputs. Got: ",
                            inputs.size());

    const auto& data = inputs[0];
    const auto& rois_data = inputs[1];

    // Slice the rois_data to get the rois and rois_batch_idx.
    const auto step = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{1}, 1);
    const auto axes = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{1}, 1);

    const auto start_rois = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{1}, 1);
    const auto stop_rois =
        std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{1}, std::numeric_limits<int32_t>::max());
    const auto rois = std::make_shared<v8::Slice>(rois_data, start_rois, stop_rois, step, axes);

    const auto start_rois_batch = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{1}, 0);
    const auto stop_rois_batch = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{1}, 1);
    const auto rois_batch = std::make_shared<v8::Slice>(rois_data, start_rois_batch, stop_rois_batch, step, axes);

    const auto lin_shape = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{1}, -1);
    const auto rois_batch_reshaped = std::make_shared<v1::Reshape>(rois_batch, lin_shape, false);

    const auto rois_batch_idx = std::make_shared<v0::Convert>(rois_batch_reshaped, ov::element::i32);

    // Read op attributes:
    const auto aligned = static_cast<bool>(node.get_attribute_value<int64_t>("aligned", 1));
    FRONT_END_GENERAL_CHECK(aligned == true, "The mmdeploy.RoiAlignRotated only supports aligned = True.");

    const auto pooled_h = node.get_attribute_value<int64_t>("output_height", 1);
    const auto pooled_w = node.get_attribute_value<int64_t>("output_width", 1);
    const auto sampling_ratio = node.get_attribute_value<int64_t>("sampling_ratio", 1);
    const auto spatial_scale = node.get_attribute_value<float>("spatial_scale", 1.0f);
    const auto clockwise = static_cast<bool>(node.get_attribute_value<int64_t>("clockwise", 0));

    return {std::make_shared<v15::ROIAlignRotated>(data,
                                                   rois,
                                                   rois_batch_idx,
                                                   static_cast<int>(pooled_h),
                                                   static_cast<int>(pooled_w),
                                                   static_cast<int>(sampling_ratio),
                                                   spatial_scale,
                                                   clockwise)};
}
ONNX_OP("MMCVRoIAlignRotated", OPSET_SINCE(1), ai_onnx::opset_1::mmdeploy_roi_align_rotated, MMDEPLOY_DOMAIN);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
