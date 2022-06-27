// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/concat.hpp"

#include <cstdint>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/validation_util.hpp"
#include "op/concat.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector concat(const Node& node) {
    OutputVector inputs{node.get_ng_inputs()};
    std::int64_t axis = node.get_attribute_value<std::int64_t>("axis");
    OutputVector valid_inputs;
    std::copy_if(inputs.begin(), inputs.end(), std::back_inserter(valid_inputs), [](ov::Output<ov::Node>& in) -> bool {
        return !(in.get_partial_shape().same_scheme(PartialShape(Shape{})));
    });
    return {std::make_shared<default_opset::Concat>(valid_inputs, axis)};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
