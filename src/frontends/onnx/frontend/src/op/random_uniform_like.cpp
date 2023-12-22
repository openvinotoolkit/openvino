// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/random_uniform_like.hpp"

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/opsets/opset8.hpp"
#include "ngraph/shape.hpp"
#include "utils/common.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {

OutputVector random_uniform_like(const Node& node) {
    OutputVector inputs{node.get_ng_inputs()};
    const auto input = inputs.at(0);

    ngraph::element::Type target_type;
    if (node.has_attribute("dtype")) {
        const auto dtype = node.get_attribute_value<int64_t>("dtype");
        target_type = common::get_ngraph_element_type(dtype);
    } else {
        target_type = input.get_element_type();
    }

    const auto target_shape = std::make_shared<default_opset::ShapeOf>(input);

    const auto high_const = node.get_attribute_as_constant<float>("high", 1.0f);
    const auto low_const = node.get_attribute_as_constant<float>("low", 0.0f);
    const auto seed = node.get_attribute_value<float>("seed", 0.f);

    const uint64_t global_seed = 0;
    const auto seed_uint64 = static_cast<uint64_t>(seed * 1000);

    return {std::make_shared<ngraph::opset8::RandomUniform>(target_shape,
                                                            low_const,
                                                            high_const,
                                                            target_type,
                                                            global_seed,
                                                            seed_uint64)};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
