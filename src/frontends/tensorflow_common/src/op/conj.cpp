// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"

using namespace std;
using namespace ov::op;

std::shared_ptr<ov::op::v0::Concat> get_conj_ptr(const ov::Output<ov::Node>& node) {
    auto real_index = make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape::Shape{1});
    auto imag_index = make_shared<v0::Constant>(ov::element::i32, ov::Shape::Shape{1}, 1);

    auto gather_axis = make_shared<v0::Constant>(ov::element::i32, ov::Shape::Shape{1}, -1);

    auto real = make_shared<v8::Gather>(node, real_index, gather_axis)->output(0);
    auto imag = make_shared<v8::Gather>(node, imag_index, gather_axis)->output(0);

    imag = make_shared<v0::Negative>(imag);

    auto conj = make_shared<v0::Concat>(ov::OutputVector{real, imag}, -1);
    return conj;
}

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_conj_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Conjugate"}, true);

    auto x = node.get_input(0);

    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(x.get_node_shared_ptr());

    std::shared_ptr<Node> conj{x.get_node_shared_ptr()};
    if (complex_type_mark) {
        element::Type complex_part_type = complex_type_mark->get_complex_part_type();
        auto x = complex_type_mark->input_value(0);
        auto conj = get_conj_ptr(x);

        set_node_name(node.get_name(), conj);
        auto complex_conj = make_shared<ComplexTypeMark>(conj, complex_part_type);
        return {complex_conj->output(0)};
    }

    set_node_name(node.get_name(), conj);
    return {conj};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov