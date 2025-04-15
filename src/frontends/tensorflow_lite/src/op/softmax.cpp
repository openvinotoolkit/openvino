// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "op_translation_utils.hpp"
#include "utils.hpp"

using namespace std;

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

OutputVector softmax(const ov::frontend::tensorflow_lite::NodeContext& node) {
    const auto& decoder = node.get_decoder();
    auto beta = node.get_attribute<float>("beta");
    Output<Node> output = node.get_input(0);
    if (beta != 1.) {
        auto beta_const = opset10::Constant::create(element::f32, Shape{}, vector<float>{beta});
        auto mul_data = make_shared<opset10::ConvertLike>(beta_const, output);
        output = make_shared<opset10::Multiply>(output, mul_data);
    }
    output = make_shared<opset8::Softmax>(output, -1);
    output.get_node()->set_friendly_name(decoder->get_op_name());
    return {output};
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
