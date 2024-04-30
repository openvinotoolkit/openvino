// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reverse_sequence.hpp"

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_reverse_sequence_op(const NodeContext& node) {
    default_op_checks(node, 2, {"ReverseSequence"}, true);
    auto input = node.get_input(0);
    auto seq_lengths = node.get_input(1);

    // retrieve attributes
    auto seq_dim = node.get_attribute<int64_t>("seq_dim");
    auto batch_dim = node.get_attribute<int64_t>("batch_dim", 0);

    // handling negative values
    if (seq_dim < 0) {
        seq_dim -= 1;
    }
    if (batch_dim < 0) {
        batch_dim -= 1;
    }

    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr());
    if (complex_type_mark) {
        auto base_input = complex_type_mark->input_value(0);
        // Reverse sequence for real and imaginary parts
        auto reverse_sequence = make_shared<v0::ReverseSequence>(base_input, seq_lengths, batch_dim, seq_dim);

        auto complex_result =
            make_shared<ComplexTypeMark>(reverse_sequence, complex_type_mark->get_complex_part_type());

        set_node_name(node.get_name(), reverse_sequence);

        return {complex_result};
    }

    auto reverse_sequence = make_shared<v0::ReverseSequence>(input, seq_lengths, batch_dim, seq_dim);
    set_node_name(node.get_name(), reverse_sequence);
    return {reverse_sequence};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
