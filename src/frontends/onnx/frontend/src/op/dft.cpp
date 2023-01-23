// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/dft.hpp"

#include "default_opset.hpp"
#include "utils/common.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector dft(const Node& node) {
    const OutputVector ng_inputs{node.get_ng_inputs()};
    ov::Output<ov::Node> data = ng_inputs.at(0);

    if (data.get_partial_shape().rank().is_static()) {
        const auto length = data.get_partial_shape().rank().get_length();
        const auto last_axis_pos = length - 1;
        const auto last_dim = data.get_partial_shape()[last_axis_pos];
        if (last_dim.is_static()) {
            if (last_dim.get_length() == 1) {
                ov::Output<ov::Node> imag_part = default_opset::Constant::create(data.get_element_type(), {}, {0});
                imag_part = std::make_shared<default_opset::Broadcast>(imag_part,
                                                                       std::make_shared<default_opset::ShapeOf>(data));
                data = std::make_shared<default_opset::Concat>(OutputVector{data, imag_part}, last_axis_pos);
            }
        }
    }

    const auto dft_length_provided = ng_inputs.size() > 1;
    const auto axis = node.get_attribute_value<int64_t>("axis", 1);
    const auto axis_const = default_opset::Constant::create(element::i64, {1}, {axis});
    const auto inverse = node.get_attribute_value<int64_t>("inverse", 0);
    const auto onesided = node.get_attribute_value<int64_t>("onesided", 0);

    ov::Output<ov::Node> result;
    if (inverse) {
        if (onesided) {
            result = dft_length_provided ? std::make_shared<default_opset::IRDFT>(data, axis_const, ng_inputs.at(1))
                                         : std::make_shared<default_opset::IRDFT>(data, axis_const);
        } else {
            result = dft_length_provided ? std::make_shared<default_opset::IDFT>(data, axis_const, ng_inputs.at(1))
                                         : std::make_shared<default_opset::IDFT>(data, axis_const);
        }
    } else {
        if (onesided) {
            result = dft_length_provided ? std::make_shared<default_opset::RDFT>(data, axis_const, ng_inputs.at(1))
                                         : std::make_shared<default_opset::RDFT>(data, axis_const);
        } else {
            result = dft_length_provided ? std::make_shared<default_opset::DFT>(data, axis_const, ng_inputs.at(1))
                                         : std::make_shared<default_opset::DFT>(data, axis_const);
        }
    }
    return {result};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
