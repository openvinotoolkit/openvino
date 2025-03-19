// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/roi_align.hpp"

#include "core/operator_set.hpp"
#include "openvino/frontend/exception.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector roi_align(const ov::frontend::onnx::Node& node) {
    const auto inputs = node.get_ov_inputs();

    FRONT_END_GENERAL_CHECK(inputs.size() == 3, "The RoiAlign operator expects 3 inputs. Got: ", inputs.size());

    const auto& data = inputs[0];
    const auto& rois = inputs[1];
    const auto& num_rois = inputs[2];

    const auto pooled_h = static_cast<int>(node.get_attribute_value<int64_t>("output_height", 1));
    const auto pooled_w = static_cast<int>(node.get_attribute_value<int64_t>("output_width", 1));
    const auto sampling_ratio = static_cast<int>(node.get_attribute_value<int64_t>("sampling_ratio", 1));
    const auto spatial_scale = node.get_attribute_value<float>("spatial_scale", 1.0f);
    const auto mode = node.get_attribute_value<std::string>("mode", "avg");
    const auto pooling_mode = ov::EnumNames<v9::ROIAlign::PoolingMode>::as_enum(mode);
    const auto aligned_mode = v9::ROIAlign::AlignedMode::ASYMMETRIC;  // Compatible up to ONNX-opset16

    return {std::make_shared<v9::ROIAlign>(data,
                                           rois,
                                           num_rois,
                                           pooled_h,
                                           pooled_w,
                                           sampling_ratio,
                                           spatial_scale,
                                           pooling_mode,
                                           aligned_mode)};
}
ONNX_OP("RoiAlign", OPSET_RANGE(1, 15), ai_onnx::opset_1::roi_align);
}  // namespace opset_1
namespace opset_16 {
ov::OutputVector roi_align(const ov::frontend::onnx::Node& node) {
    const auto inputs = node.get_ov_inputs();

    FRONT_END_GENERAL_CHECK(inputs.size() == 3, "The RoiAlign operator expects 3 inputs. Got: ", inputs.size());

    const auto& data = inputs[0];
    const auto& rois = inputs[1];
    const auto& num_rois = inputs[2];

    const auto pooled_h = node.get_attribute_value<int64_t>("output_height", 1);
    const auto pooled_w = node.get_attribute_value<int64_t>("output_width", 1);
    const auto sampling_ratio = node.get_attribute_value<int64_t>("sampling_ratio", 1);
    const auto spatial_scale = node.get_attribute_value<float>("spatial_scale", 1.0f);
    const auto mode = node.get_attribute_value<std::string>("mode", "avg");
    const auto pooling_mode = ov::EnumNames<v9::ROIAlign::PoolingMode>::as_enum(mode);

    const auto coordinate_transformation_mode =
        node.get_attribute_value<std::string>("coordinate_transformation_mode", "");
    auto aligned_mode = v9::ROIAlign::AlignedMode::HALF_PIXEL_FOR_NN;  // Match ONNX ROIAlign-16 default

    if (coordinate_transformation_mode == "output_half_pixel") {
        aligned_mode = v9::ROIAlign::AlignedMode::ASYMMETRIC;
    }

    return {std::make_shared<v9::ROIAlign>(data,
                                           rois,
                                           num_rois,
                                           static_cast<int>(pooled_h),
                                           static_cast<int>(pooled_w),
                                           static_cast<int>(sampling_ratio),
                                           spatial_scale,
                                           pooling_mode,
                                           aligned_mode)};
}
ONNX_OP("RoiAlign", OPSET_SINCE(16), ai_onnx::opset_16::roi_align);
}  // namespace opset_16
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
