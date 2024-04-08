// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/round.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "common_op_table.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_round_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Round", "ROUND"}, true);

    auto input = node.get_input(0);
    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr());

    // using default round mode "half_to_even" in openvino,
    // as TF has only that mode
    auto round_mode = v5::Round::RoundMode::HALF_TO_EVEN;

    if (complex_type_mark) {
        // Store the complex part type for the output that will be a complex type tensor
        element::Type complex_part_type = complex_type_mark->get_complex_part_type();

        input = complex_type_mark->input_value(0);
        auto round = make_shared<v5::Round>(input, round_mode);
        set_node_name(node.get_name(), round);
        auto complex_round = make_shared<ComplexTypeMark>(round, complex_part_type);
        return {complex_round->output(0)};
    }

    auto res = make_shared<v5::Round>(input, round_mode);
    set_node_name(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
