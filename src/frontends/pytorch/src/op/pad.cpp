// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_pad(NodeContext& context) {
    auto data = context.get_input(0);
    auto paddings = context.const_input<std::vector<int64_t>>(1);
    std::string mode = "constant";
    double value = 0;
    const auto data_rank = data.get_partial_shape().rank().get_length();
    std::vector<int64_t> pad_begin(data_rank, 0);
    std::vector<int64_t> pad_end(data_rank, 0);
    auto pad_mode = ov::op::PadMode::CONSTANT;
    for (size_t i = paddings.size() / 2; i > 0; i--) {
        pad_begin[i] = paddings[2 * i + 1];
        pad_end[i] = paddings[2 * i];
    }
    if (context.get_input_size() >= 3){

        if (!context.input_is_none(2)){
            mode = context.const_input<std::string>(2);
        }
        if (mode == "constant" && context.get_input_size() == 4 && !context.input_is_none(3)){
            value = context.const_input<double>(3);
        }
    }
    if (mode == "constant"){
        pad_mode = ov::op::PadMode::CONSTANT;
    }
    else if (mode == "reflect") {
        pad_mode = ov::op::PadMode::REFLECT;
    }
    else if (mode == "replicate") {
        pad_mode = ov::op::PadMode::EDGE;
    }
    else {
    FRONT_END_OP_CONVERSION_CHECK(false, "aten::pad conversion doesn't support [" + mode + "] padding mode");
    }
    return {std::make_shared<opset8::Pad>(
        data,
        std::make_shared<opset8::Constant>(element::i64, Shape{pad_begin.size()}, pad_begin),
        std::make_shared<opset8::Constant>(element::i64, Shape{pad_end.size()}, pad_end),
        std::make_shared<opset8::Constant>(data.get_element_type(), Shape{}, std::vector<double>{value}),
        pad_mode)};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov