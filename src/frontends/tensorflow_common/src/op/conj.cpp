// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"

using namespace std;
using namespace ov::op;


namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_conj_op (const NodeContext& node) {
     default_op_checks(node, 1, {"Conjugate"}, true);

    auto x = node.get_input(0);

    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(x.get_node_shared_ptr());

    std::shared_ptr<Node> conj {x.get_node_shared_ptr()};
    if (complex_type_mark) {
        element::Type complex_part_type = complex_type_mark->get_complex_part_type();
        auto x = complex_type_mark->input_value(0);

        auto real_index = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
        auto imag_index = make_shared<v0::Constant>(element::i32, Shape{1}, 1);

        auto gather_axis = make_shared<v0::Constant>(element::i32, Shape{1}, -1);

        auto real = make_shared<v8::Gather>(x, real_index, gather_axis)->output(0);
        auto imag = make_shared<v8::Gather>(x, imag_index, gather_axis)->output(0);

        imag = make_shared<v0::Negative>(imag);

        auto conj = make_shared<v0::Concat>(OutputVector{real, imag}, -1); 

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