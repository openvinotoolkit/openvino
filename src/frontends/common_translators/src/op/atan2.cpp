// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/atan2.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace common_translators {

using namespace ov::op;

OutputVector translate_atan2_util(const NodeContext& context, const Output<Node>& lhs, const Output<Node>& rhs) {
    return {context.mark_node(std::make_shared<v17::Atan2>(lhs, rhs))};
}

OutputVector translate_atan2(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto lhs = context.get_input(0);
    auto rhs = context.get_input(1);
    return translate_atan2_util(context, lhs, rhs);
}

}  // namespace common_translators
}  // namespace frontend
}  // namespace ov
