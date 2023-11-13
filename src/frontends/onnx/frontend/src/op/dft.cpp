// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/dft.hpp"

#include "onnx_import/core/null_node.hpp"
#include "utils/common.hpp"
#include "utils/dft.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector dft(const Node& node) {
    const OutputVector ng_inputs{node.get_ng_inputs()};
    const ov::Output<ov::Node> data = ng_inputs.at(0);

    const auto dft_length_provided = ng_inputs.size() > 1 && !ngraph::op::is_null(ng_inputs[1]);
    const auto axis = node.get_attribute_value<int64_t>("axis", 1);
    const auto inverse = node.get_attribute_value<int64_t>("inverse", 0);
    const auto onesided = node.get_attribute_value<int64_t>("onesided", 0);

    return {dft::make_dft(data,
                          dft_length_provided ? ng_inputs.at(1) : std::make_shared<NullNode>(),
                          axis,
                          inverse == 1,
                          onesided == 1)};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
