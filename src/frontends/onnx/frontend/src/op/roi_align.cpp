// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/roi_align.hpp"

#include <memory>

#include "ngraph/opsets/opset9.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector roi_align(const Node& node) {
    const auto inputs = node.get_ng_inputs();

    NGRAPH_CHECK(inputs.size() == 3, "The RoiAlign operator expects 3 inputs. Got: ", inputs.size());

    const auto& data = inputs[0];
    const auto& rois = inputs[1];
    const auto& num_rois = inputs[2];

    const auto pooled_h = static_cast<int>(node.get_attribute_value<int64_t>("output_height", 1));
    const auto pooled_w = static_cast<int>(node.get_attribute_value<int64_t>("output_width", 1));
    const auto sampling_ratio = static_cast<int>(node.get_attribute_value<int64_t>("sampling_ratio", 1));
    const auto spatial_scale = node.get_attribute_value<float>("spatial_scale", 1.0f);
    const auto mode = node.get_attribute_value<std::string>("mode", "avg");
    const auto pooling_mode = EnumNames<opset9::ROIAlign::PoolingMode>::as_enum(mode);
    const auto aligned_mode = opset9::ROIAlign::AlignedMode::ASYMMETRIC;  // Compatible up to ONNX-opset16

    return {std::make_shared<opset9::ROIAlign>(data,
                                               rois,
                                               num_rois,
                                               pooled_h,
                                               pooled_w,
                                               sampling_ratio,
                                               spatial_scale,
                                               pooling_mode,
                                               aligned_mode)};
}
}  // namespace set_1
namespace set_16 {
OutputVector roi_align(const Node& node) {
    const auto inputs = node.get_ng_inputs();

    NGRAPH_CHECK(inputs.size() == 3, "The RoiAlign operator expects 3 inputs. Got: ", inputs.size());

    const auto& data = inputs[0];
    const auto& rois = inputs[1];
    const auto& num_rois = inputs[2];

    const auto pooled_h = node.get_attribute_value<int64_t>("output_height", 1);
    const auto pooled_w = node.get_attribute_value<int64_t>("output_width", 1);
    const auto sampling_ratio = node.get_attribute_value<int64_t>("sampling_ratio", 1);
    const auto spatial_scale = node.get_attribute_value<float>("spatial_scale", 1.0f);
    const auto mode = node.get_attribute_value<std::string>("mode", "avg");
    const auto pooling_mode = EnumNames<opset9::ROIAlign::PoolingMode>::as_enum(mode);

    const auto coordinate_transformation_mode =
        node.get_attribute_value<std::string>("coordinate_transformation_mode", "");
    auto aligned_mode = opset9::ROIAlign::AlignedMode::HALF_PIXEL_FOR_NN;  // Match ONNX ROIAlign-16 default

    if (coordinate_transformation_mode == "output_half_pixel") {
        aligned_mode = opset9::ROIAlign::AlignedMode::ASYMMETRIC;
    }

    return {std::make_shared<opset9::ROIAlign>(data,
                                               rois,
                                               num_rois,
                                               static_cast<int>(pooled_h),
                                               static_cast<int>(pooled_w),
                                               static_cast<int>(sampling_ratio),
                                               spatial_scale,
                                               pooling_mode,
                                               aligned_mode)};
}
}  // namespace set_16

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
