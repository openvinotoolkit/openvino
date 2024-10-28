// Copyright (C) 2018-2024 Intel Corporation
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
namespace com_microsoft {
namespace opset_1 {

ov::OutputVector simplified_layer_normalization(const ov::frontend::onnx::Node& node) {
    const auto inputs = node.get_ov_inputs();
    const auto num_inputs = inputs.size();

    FRONT_END_GENERAL_CHECK(num_inputs == 2,
                            "SimplifiedLayerNormalization takes 2 inputs. Provided " + std::to_string(num_inputs));

    auto X = inputs.at(0);            // required input tensor
    const auto scale = inputs.at(1);  // required scale tensor

    CHECK_VALID_NODE(node,
                     X.get_element_type() == ov::element::f16 || X.get_element_type() == ov::element::f32 ||
                         X.get_element_type() == ov::element::f64 || X.get_element_type() == ov::element::bf16,
                     "Unsupported input X type, accepted float16, float, double, bfloat16, got: ",
                     X.get_element_type());
    CHECK_VALID_NODE(node,
                     scale.get_element_type() == ov::element::f16 || scale.get_element_type() == ov::element::f32 ||
                         scale.get_element_type() == ov::element::f64 || scale.get_element_type() == ov::element::bf16,
                     "Unsupported input scale type, accepted float16, float, double, bfloat16, got: ",
                     scale.get_element_type());

    float epsilon = node.get_attribute_value<float>("epsilon", 1e-5f);  // required
    int64_t axis = node.get_attribute_value<int64_t>("axis", -1);       // required
    int64_t default_stash_type = static_cast<int64_t>(TensorProto_DataType::TensorProto_DataType_FLOAT);
    int64_t stash_type_i = node.get_attribute_value<int64_t>("stash_type", default_stash_type);  // required
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

    auto squared_X = std::make_shared<v1::Multiply>(X, X);          // X^2
    auto mean = std::make_shared<v1::ReduceMean>(squared_X, axes);  // mean = (1/N) * Σ(j=1 to N) X_j^2
    auto rms_value =
        std::make_shared<v0::Sqrt>(std::make_shared<v1::Add>(mean, v0::Constant::create(stash_type, {}, {epsilon})));
    auto inv_std_var = std::make_shared<v1::Divide>(v0::Constant::create(stash_type, {}, {1.0}), rms_value);
    auto normalized = std::make_shared<v1::Multiply>(X, inv_std_var);  // X / RMS(X)
    auto scaled = std::make_shared<v1::Multiply>(normalized, scale);   // (X / RMS(X)) * scale

    return ov::OutputVector{scaled, inv_std_var};
}

ONNX_OP("SimplifiedLayerNormalization",
        OPSET_SINCE(1),
        com_microsoft::opset_1::simplified_layer_normalization,
        MICROSOFT_DOMAIN);
}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
