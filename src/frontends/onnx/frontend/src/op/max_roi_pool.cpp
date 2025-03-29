// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/roi_pooling.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector max_roi_pool(const ov::frontend::onnx::Node& node) {
    const auto& inputs = node.get_ov_inputs();
    const auto X = inputs.at(0);
    const auto rois = inputs.at(1);

    OPENVINO_ASSERT(X.get_element_type() == ov::element::f16 || X.get_element_type() == ov::element::f32 ||
                        X.get_element_type() == ov::element::f64,
                    "MaxRoiPool operator only supports float16, float32 and float64 datatypes.");

    const auto pooled_shape = node.get_attribute_value<std::vector<size_t>>("pooled_shape");
    const auto spatial_scale = node.get_attribute_value<float>("spatial_scale", 1.0);

    return {std::make_shared<v0::ROIPooling>(X, rois, ov::Shape(pooled_shape), spatial_scale, "max")};
}
ONNX_OP("MaxRoiPool", OPSET_SINCE(1), ai_onnx::opset_1::max_roi_pool);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
