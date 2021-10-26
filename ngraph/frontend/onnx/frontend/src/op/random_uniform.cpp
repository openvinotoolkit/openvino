// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/random_uniform.hpp"

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/opsets/opset8.hpp"
#include "ngraph/shape.hpp"
#include "utils/common.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {

OutputVector random_uniform(const Node& node) {
    CHECK_VALID_NODE(node, node.has_attribute("shape"), "RandomUniform operator must specify a 'shape' attribute.");

    const auto dtype =
        node.get_attribute_value<int64_t>("dtype", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
    const auto high = node.get_attribute_value<float>("high", 1.0f);
    const auto low = node.get_attribute_value<float>("low", 0.0f);
    const auto seed = node.get_attribute_value<int64_t>("seed", 0);
    const auto shape = node.get_attribute_value<std::vector<int64_t>>("shape");

    const auto target_shape_const = default_opset::Constant::create(ngraph::element::i64, Shape{shape.size()}, shape);
    const auto high_const = default_opset::Constant::create(ngraph::element::f32, Shape{1}, {high});
    const auto low_const = default_opset::Constant::create(ngraph::element::f32, Shape{1}, {low});

    const auto target_type = common::get_ngraph_element_type(dtype);
    const uint64_t global_seed = 0;

    return {std::make_shared<ngraph::opset8::RandomUniform>(target_shape_const,
                                                            low_const,
                                                            high_const,
                                                            target_type,
                                                            global_seed,
                                                            seed)};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
