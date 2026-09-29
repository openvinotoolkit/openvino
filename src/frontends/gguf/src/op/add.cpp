// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/add.hpp"

#include <memory>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/op/convert.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

OutputVector translate_add(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto lhs = context.get_input(0);
    auto rhs = context.get_input(1);
    const auto output_type = lhs.get_element_type();
    if (output_type != rhs.get_element_type()) {
        lhs = std::make_shared<ov::op::v0::Convert>(lhs, ov::element::f32);
        rhs = std::make_shared<ov::op::v0::Convert>(rhs, ov::element::f32);
    }
    Output<Node> result = std::make_shared<ov::op::v1::Add>(lhs, rhs);
    if (result.get_element_type() != output_type) {
        result = std::make_shared<ov::op::v0::Convert>(result, output_type);
    }
    return rename_outputs_with_suffix({std::move(result)}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
