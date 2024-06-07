// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/add.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_add_n_op(const NodeContext& node) {
    default_op_checks(node, 1, {"AddN", "ADD_N"}, true);
    int num_size = static_cast<int>(node.get_input_size());
    auto result = node.get_input(0);
    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(result.get_node_shared_ptr());
    if (complex_type_mark) {
        element::Type complex_part_type = complex_type_mark->get_complex_part_type();
        result = complex_type_mark->input_value(0);

        // converting all the inputs to complex type (simulating complex type) and adding them
        for (int ind = 1; ind < num_size; ++ind) {
            auto complex_type_mark_ind = as_type_ptr<ComplexTypeMark>(node.get_input(ind).get_node_shared_ptr());
            result = make_shared<v1::Add>(result, complex_type_mark_ind->input_value(0));
        }
        auto complex_add_n = make_shared<ComplexTypeMark>(result, complex_part_type);
        set_node_name(node.get_name(), result.get_node_shared_ptr());
        return {complex_add_n->output(0)};
    }

    for (int ind = 1; ind < num_size; ++ind) {
        result = make_shared<v1::Add>(result, node.get_input(ind));
    }

    set_node_name(node.get_name(), result.get_node_shared_ptr());
    return {result};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
