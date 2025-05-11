// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reshape.hpp"

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"

using namespace std;
using namespace ov::op;

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
        tensor = complex_type_mark->get_data();

        OutputVector concat_inputs;
        concat_inputs.push_back(shape);
        concat_inputs.push_back(make_shared<v0::Constant>(shape.get_element_type(), Shape{1}, 2));

        auto concat = make_shared<v0::Concat>(concat_inputs, 0);
        auto reshape = make_shared<v1::Reshape>(tensor, concat, false);
        set_node_name(node.get_name(), reshape);
        auto complex_reshape = make_shared<ComplexTypeMark>(reshape, complex_part_type);
        return {complex_reshape->output(0)};
    }

    auto reshape = make_shared<v1::Reshape>(tensor, shape, false);
    set_node_name(node.get_name(), reshape);
    return {reshape};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
