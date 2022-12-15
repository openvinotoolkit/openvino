// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>

#include "op_table.hpp"
#include "openvino/core/validation_util.hpp"

using namespace std;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_assert_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Assert"});
    auto cond = node.get_input(0);
    auto cond_const = get_constant_from_source(cond);
    TENSORFLOW_OP_VALIDATION(node,
                             cond_const,
                             "[TensorFlow Frontend] The condition must be constant for further model conversion.");
    auto cond_values = cond_const->cast_vector<bool>();
    TENSORFLOW_OP_VALIDATION(node,
                             cond_values.size() == 1,
                             "[TensorFlow Frontend] Incorrect model - the condition must have one element.");
    TENSORFLOW_OP_VALIDATION(node,
                             cond_values[0],
                             "[TensorFlow Frontend] The condition must be true for further model conversion.");
    return {};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
