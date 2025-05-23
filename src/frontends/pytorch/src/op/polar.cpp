// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/sin.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_polar(const NodeContext& context) {
    num_inputs_check(context, 2, 3);
    auto abs = context.get_input(0);
    auto angle = context.get_input(1);
    auto cos_node = context.mark_node(std::make_shared<v0::Cos>(angle));
    auto real = context.mark_node(std::make_shared<v1::Multiply>(abs, cos_node));
    auto sin_node = context.mark_node(std::make_shared<v0::Sin>(angle));
    auto imag = context.mark_node(std::make_shared<v1::Multiply>(abs, sin_node));
    auto complex_tensor = context.mark_node(std::make_shared<ComplexTypeMark>(real, imag));
    return {complex_tensor};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
