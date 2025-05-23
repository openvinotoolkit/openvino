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
#include "openvino/op/power.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
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

ov::OutputVector skip_simplified_layer_normalization(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 3);

    const auto inputs = node.get_ov_inputs();
    const auto input_tensor = inputs[0];
    const auto skip = inputs[1];

    CHECK_VALID_NODE(node,
                     input_tensor.get_element_type() == skip.get_element_type(),
                     "input tensor and skip must be of same type, got :",
                     input_tensor.get_element_type(),
                     skip.get_element_type());

    // input + skip
    std::shared_ptr<ov::Node> input = std::make_shared<v1::Add>(input_tensor, skip);

    // add bias if available
    if (inputs.size() == 4) {
        const auto bias = inputs[3];
        CHECK_VALID_NODE(node,
                         input_tensor.get_element_type() == bias.get_element_type(),
                         "input tensor and bias must be of same type, got: ",
                         input_tensor.get_element_type(),
                         bias.get_element_type());
        input = std::make_shared<v1::Add>(input, bias);
    }

    float epsilon = node.get_attribute_value<float>("epsilon");
    ov::element::Type element_type = input->get_output_element_type(0);

    auto squared_input = std::make_shared<v1::Multiply>(input, input);
    auto mean = std::make_shared<v1::ReduceMean>(squared_input,
                                                 v0::Constant::create(element::i64, {}, {-1}),
                                                 true);  // mean = (1/N) * Σ(j=1 to N) X_j^2
    auto rms_value =
        std::make_shared<v0::Sqrt>(std::make_shared<v1::Add>(mean, v0::Constant::create(element_type, {}, {epsilon})));
    auto inv_std_var = std::make_shared<v1::Divide>(v0::Constant::create(element_type, {}, {1.0f}), rms_value);
    auto normalized = std::make_shared<v1::Multiply>(input, inv_std_var);  // X / RMS(X) auto scaled =
    auto scaled = std::make_shared<v1::Multiply>(normalized, inputs[2]);   // (X / RMS(X)) * scale

    return ov::OutputVector{scaled, mean, inv_std_var, input};
}
ONNX_OP("SkipSimplifiedLayerNormalization",
        OPSET_SINCE(1),
        com_microsoft::opset_1::skip_simplified_layer_normalization,
        MICROSOFT_DOMAIN);
}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
