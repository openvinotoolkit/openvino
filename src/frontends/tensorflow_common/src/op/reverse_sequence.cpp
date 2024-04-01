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
    // Handling negative values by adding rank if needed
    if (seq_dim < 0) {
        seq_dim += input.get_shape().size();
    }
    if (batch_dim < 0) {
        batch_dim += input.get_shape().size();
    }

    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr());
    if (complex_type_mark) {
        auto const_one = make_shared<v0::Constant>(element::i32, Shape{1}, 1);
        auto seq_dim_tensor = make_shared<v1::Select>(
            make_shared<v1::Equal>(make_shared<v1::Rank>(input), const_one),
            make_shared<v0::Constant>(element::i64, Shape{}, seq_dim),
            make_shared<v1::Select>(make_shared<v1::Equal>(make_shared<v1::Rank>(input), const_one),
                                    make_shared<v1::Subtract>(make_shared<v1::Rank>(input), const_one),
                                    make_shared<v0::Constant>(element::i64, Shape{}, -1)));
        auto batch_dim_tensor = make_shared<v1::Select>(
            make_shared<v1::Equal>(make_shared<v1::Rank>(input), const_one),
            make_shared<v0::Constant>(element::i64, Shape{}, batch_dim),
            make_shared<v1::Select>(make_shared<v1::Equal>(make_shared<v0::Rank>(input), const_one),
                                    make_shared<v1::Subtract>(make_shared<v1::Rank>(input), const_one),
                                    make_shared<v0::Constant>(element::i64, Shape{}, -1)));
        auto updated_seq_dim = make_shared<v1::Add>(seq_dim_tensor, const_one);
        auto updated_batch_dim = make_shared<v1::Add>(batch_dim_tensor, const_one);

        auto reverse_sequence =
            make_shared<v0::ReverseSequence>(input, seq_lengths, updated_batch_dim, updated_seq_dim);
        set_node_name(node.get_name(), reverse_sequence);

        auto complex_reverse =
            make_shared<ComplexTypeMark>(reverse_sequence, complex_type_mark->get_complex_part_type());
        return {complex_reverse};
    }

    auto reverse_sequence = make_shared<v0::ReverseSequence>(input, seq_lengths, batch_dim, seq_dim);
    set_node_name(node.get_name(), reverse_sequence);
    return {reverse_sequence};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
