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
    const auto data = ng_inputs.at(0);

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
