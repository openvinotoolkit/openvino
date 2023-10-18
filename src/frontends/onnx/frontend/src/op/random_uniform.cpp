// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/random_uniform.hpp"

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

OutputVector random_uniform(const Node& node) {
    CHECK_VALID_NODE(node, node.has_attribute("shape"), "RandomUniform operator must specify a 'shape' attribute.");

    const auto dtype =
        node.get_attribute_value<int64_t>("dtype", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
    const auto high_const = node.get_attribute_as_constant<float>("high", 1.0f);
    const auto low_const = node.get_attribute_as_constant<float>("low", 0.0f);
    const auto seed = node.get_attribute_value<float>("seed", 0.0f);
    const auto target_shape_const = node.get_attribute_as_constant<std::vector<int64_t>>("shape");

    const auto target_type = common::get_ngraph_element_type(dtype);
    const uint64_t global_seed = 0;
    // TODO: This multiplication leads to a mismatch in accuracy. Issue: 123003
    const auto seed_uint64 = static_cast<uint64_t>(seed * 1000);

    return {std::make_shared<ngraph::opset8::RandomUniform>(target_shape_const,
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
