// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/add.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_add_n_op(const NodeContext& node) {
    default_op_checks(node, 1, {"AddN", "ADD_N"});
    int num_size = static_cast<int>(node.get_input_size());


    auto result = node.get_input(0);
    for (int ind = 1; ind < num_size; ++ind) {
        result = make_shared<v1::Add>(result, node.get_input(ind));
    }

    auto complex_type_mark_result = as_type_ptr<ComplexTypeMark>(result.get_node_shared_ptr());
    if (complex_type_mark_result) {
        FRONT_END_GENERAL_CHECK(complex_type_mark_result != nullptr,
                                "AddN/ADD_N got complex and non-complex inputs. Inputs should be of same type.");
        result = complex_type_mark_result->input_value(0);

        element::Type complex_part_type_result = complex_type_mark_result->get_complex_part_type();
        auto complex_result = make_shared<ComplexTypeMark>(result->output(0), complex_part_type_result);
        return {complex_result};
    }

    set_node_name(node.get_name(), result);
    return {result};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov

