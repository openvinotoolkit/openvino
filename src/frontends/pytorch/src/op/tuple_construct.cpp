// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_tuple_construct(NodeContext& context) {
    auto n_inputs = context.get_input_size();
    FRONT_END_OP_CONVERSION_CHECK(
        n_inputs == 1,
        "prim::TupleConstruct conversion doesn't support cases when the number of inputs is not one.");
    return {context.get_input(0)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov