// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_reshape_op(const NodeContext& node) {
    default_op_checks(node, 2, {"Reshape"}, true);
    auto tensor = node.get_input(0);
    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(tensor.get_node_shared_ptr());
    auto shape = node.get_input(1);

    if (complex_type_mark) {
        element::Type complex_part_type = complex_type_mark->get_complex_part_type();
        tensor = complex_type_mark->input_value(0);

        OutputVector concat_inputs;
        concat_inputs.push_back(shape);
        concat_inputs.push_back(make_shared<Constant>(shape.get_element_type(), Shape{1}, 2));

        auto concat = make_shared<Concat>(concat_inputs, 0);
        auto reshape = make_shared<Reshape>(tensor, concat, false);
        set_node_name(node.get_name(), reshape);
        auto complex_reshape = make_shared<ComplexTypeMark>(reshape, complex_part_type);
        return {complex_reshape->output(0)};
    }

    auto reshape = make_shared<Reshape>(tensor, shape, false);
    set_node_name(node.get_name(), reshape);
    return {reshape};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
