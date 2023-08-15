// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/reshape.hpp"

#include <cstddef>
#include <memory>
#include <vector>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/shape.hpp"
#include "utils/reshape.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector reshape(const Node& node) {
    OutputVector ng_inputs{node.get_ng_inputs()};
    const auto data = ng_inputs.at(0);

    Output<ngraph::Node> pattern;
    bool special_zero = true;
    // Since opset 5 the target shape is provided as input
    if (ng_inputs.size() == 2) {
        pattern = ng_inputs.at(1);
    } else {
        // Added in onnx reshape version 14
        special_zero = !node.get_attribute_value<int64_t>("allowzero", 0);

        pattern = node.get_attribute_as_constant<std::vector<int64_t>>("shape", {});
    }

    return {std::make_shared<default_opset::Reshape>(data, pattern, special_zero)};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
