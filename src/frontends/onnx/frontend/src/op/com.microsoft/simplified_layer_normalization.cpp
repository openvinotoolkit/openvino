// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sqrt.hpp"
#include "utils/common.hpp"

using namespace ov::op;
using ::ONNX_NAMESPACE::TensorProto_DataType;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {

ov::OutputVector simplified_layer_normalization(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 2);

    const auto inputs = node.get_ov_inputs();
    auto X = inputs[0];
    const auto scale = inputs[1];

    CHECK_VALID_NODE(node,
                     X.get_element_type() == scale.get_element_type(),
                     "X and scale must be of same type, got :",
                     X.get_element_type(),
                     scale.get_element_type());

    float epsilon = node.get_attribute_value<float>("epsilon", 1e-5f);
    int64_t axis = node.get_attribute_value<int64_t>("axis", -1);
    int64_t default_stash_type = static_cast<int64_t>(TensorProto_DataType::TensorProto_DataType_FLOAT);
    int64_t stash_type_i = node.get_attribute_value<int64_t>("stash_type", default_stash_type);
    element::Type stash_type = common::get_ov_element_type(stash_type_i);

    auto rank = std::make_shared<v0::ShapeOf>(X);
    auto axes = std::make_shared<v4::Range>(v0::Constant::create(element::i64, {}, {axis}),
                                            (axis < 0 ? v0::Constant::create(element::i64, {}, {0})->output(0) : rank),
                                            v0::Constant::create(element::i64, {}, {1}),
                                            element::i64);

    bool needs_type_casting = stash_type != X.get_element_type();
    if (needs_type_casting) {
        X = std::make_shared<v0::Convert>(X, stash_type);
    }

    auto squared_X = std::make_shared<v1::Multiply>(X, X);                // X^2
    auto mean = std::make_shared<v1::ReduceMean>(squared_X, axes, true);  // mean = (1/N) * Σ(j=1 to N) X_j^2
    auto rms_value =
        std::make_shared<v0::Sqrt>(std::make_shared<v1::Add>(mean, v0::Constant::create(stash_type, {}, {epsilon})));
    auto inv_std_var = std::make_shared<v1::Divide>(v0::Constant::create(stash_type, {}, {1.0}), rms_value);
    auto normalized = std::make_shared<v1::Multiply>(X, inv_std_var);  // X / RMS(X)

    auto scaled = std::make_shared<v1::Multiply>(normalized, scale);  // (X / RMS(X)) * scale

    return ov::OutputVector{scaled, inv_std_var};
}

/* This operator isn't clearly defined in ONNX documentation:
    - https://github.com/onnx/onnx/blob/main/docs/Operators.md
    - https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md
   Strange, but a SkipSimplifiedLayerNormalization is a part of com.microsoft domain:
    -
   https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.SkipSimplifiedLayerNormalization
   Same time SimplifiedLayerNormalization is described here and in some models it is found as a part of ai.onnx domain:
    - https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webgpu-operators.md
   To align with actual behavior and some documentation - decided to register it as a ai.onnx, but leave
   in a folder with com.microsoft operations, because it isn't defined as a part of ONNX.
*/
ONNX_OP("SimplifiedLayerNormalization", OPSET_SINCE(1), ai_onnx::opset_1::simplified_layer_normalization);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
