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
    num_inputs_check(context, 2, 2);
    auto dtype = context.const_named_param<ov::element::Type>("dtype");
    auto size = context.const_named_param<int64_t>("size");
    auto start = v0::Constant::create(ov::element::i64, Shape{}, {0});
    auto step = v0::Constant::create(ov::element::i64, Shape{}, {1});
    auto stop = v0::Constant::create(ov::element::i64, Shape{}, {size});
    auto res = std::make_shared<ov::op::v4::Range>(start, stop, step, dtype);
    return {res};
};

}  // namespace op
}  // namespace jax
}  // namespace frontend
}  // namespace ov