// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset6.hpp>
#include "fill_constant_batch_size_like.hpp"
#include <paddlepaddle_frontend/utility.hpp>

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

NamedOutputs fill_constant_batch_size_like (const NodeContext& node) {
    //TODO to Support other data types other than FP32
    auto input_dim_idx = node.get_attribute<int32_t>("input_dim_idx", 0);
    auto output_dim_idx = node.get_attribute<int32_t>("output_dim_idx", 0);
    auto value = node.get_attribute<float>("value");
    auto shapes = node.get_attribute<std::vector<int32_t> >("shape");
    auto input = node.get_ng_input("Input");
    auto parial_shape = input.get_partial_shape();
    PDPD_ASSERT(parial_shape.is_static(), "fill_constant_batch_size_like: must use static shape.");
    auto static_shape = parial_shape.get_shape();
    PDPD_ASSERT(input_dim_idx < (int32_t)static_shape.size(), "fill_constant_batch_size_like: input_dim_idx should not exceed input dims.");
    PDPD_ASSERT(output_dim_idx < (int32_t)shapes.size(), "fill_constant_batch_size_like: output_dim_idx should not exceed shapes dims.");
    shapes[output_dim_idx] = static_shape[input_dim_idx];
    return node.default_single_output_mapping(
        {std::make_shared<ngraph::opset6::Constant>(ngraph::element::f32, Shape(shapes.begin(), shapes.end()), value)}, 
        {"Out"});
}

}}}}