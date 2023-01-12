// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "op_translation_utils.hpp"
#include "utils.hpp"

using namespace std;

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

OutputVector softmax(const ov::frontend::tensorflow::NodeContext& node) {
    const auto& decoder = std::dynamic_pointer_cast<DecoderFlatBuffer>(node.get_decoder());
    FRONT_END_GENERAL_CHECK(decoder != nullptr,
                            "Unexpected decoder during operation translation. Expected DecoderFlatBuffer");
    auto data = node.get_input(0);
    auto beta = static_cast<float>(decoder->get_attribute(&tflite::SoftmaxOptions::beta));
    auto beta_const = opset10::Constant::create(element::f32, Shape{}, vector<float>{beta});
    auto mul_const = make_shared<opset10::ConvertLike>(beta_const, data);
    
    auto mul = make_shared<opset10::Multiply>(data, mul_const);
    auto res = make_shared<opset8::Softmax>(mul, -1);
    res->set_friendly_name(decoder->get_op_name());
    return {res};
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
