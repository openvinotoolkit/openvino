// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/frontend/jax/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/range.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace jax {
namespace op {
using namespace ov::op;

OutputVector translate_iota(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    // Get the dtype and size from the input
    auto dtype = context.const_named_param<element::Type>("dtype");
    
    auto size = context.const_named_param<int64_t>("size");

    // Create constant nodes for start, limit, and step
    auto start = v0::Constant::create(element::i64, Shape({}), {0});
    auto limit = v0::Constant::create(element::i64, Shape({}), {size});
    auto step = v0::Constant::create(element::i64, Shape{}, {1});

    // Create the Range operation
    auto range = std::make_shared<v4::Range>(start, limit, step,dtype);

    return {range};
}

}  // namespace op
}  // namespace jax
}  // namespace frontend
}  // namespace ov