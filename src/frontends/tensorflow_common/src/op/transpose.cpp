// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/transpose.hpp"

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_transpose_op(const NodeContext& node) {
    default_op_checks(node, 2, {"Transpose", "TRANSPOSE"}, true);
    auto x = node.get_input(0);
    auto perm = node.get_input(1);

    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(x.get_node_shared_ptr());
    if (complex_type_mark) {
        element::Type complex_part_type = complex_type_mark->get_complex_part_type();
        x = complex_type_mark->get_data();

        auto input_rank = compute_subgraph_scalar_rank(x, element::i32, false);
        auto const_one = make_shared<v0::Constant>(element::i32, Shape{1}, 1);
        auto input_rank_minus_one = make_shared<v1::Subtract>(input_rank, const_one)->output(0);

        OutputVector concat_inputs;
        concat_inputs.push_back(perm);
        concat_inputs.push_back(input_rank_minus_one);

        auto concat = make_shared<v0::Concat>(concat_inputs, 0);
        auto transpose = make_shared<v1::Transpose>(x, concat);
        set_node_name(node.get_name(), transpose);
        auto complex_transpose = make_shared<ComplexTypeMark>(transpose, complex_part_type);

        return {complex_transpose->output(0)};
    }
    auto transpose = make_shared<v1::Transpose>(x, perm);
    set_node_name(node.get_name(), transpose);
    return {transpose};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
