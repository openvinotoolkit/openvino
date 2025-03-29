// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/multinomial.hpp"

#include "common_op_table.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_multinomial_op(const NodeContext& node) {
    default_op_checks(node, 2, {"Multinomial"});
    auto logits = node.get_input(0);
    auto num_samples = node.get_input(1);
    auto global_seed = node.get_attribute<int64_t>("seed", 0);
    auto op_seed = node.get_attribute<int64_t>("seed2", 0);
    auto output_type = node.get_attribute<ov::element::Type>("output_dtype");

    auto res =
        std::make_shared<ov::op::v13::Multinomial>(logits, num_samples, output_type, true, true, global_seed, op_seed);
    set_node_name(node.get_name(), res);
    return res->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
