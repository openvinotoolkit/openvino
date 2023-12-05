// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/opsets/opset10.hpp"

using namespace std;
using namespace ov::opset10;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_input_arg_op(const NodeContext& node) {
    default_op_checks(node, 0, {"input_arg"});
    auto param_type = node.get_attribute<ov::element::Type>("type");

    element::Type complex_part_type = element::dynamic;

    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(node.get_input(0).get_node_shared_ptr());
    if (complex_type_mark) {
        complex_part_type = complex_type_mark->get_complex_part_type();
    }

    auto param = std::make_shared<Parameter>(param_type, ov::PartialShape::dynamic());
    set_node_name(node.get_name(), param);
    if (complex_type_mark) {
        auto param_complex = make_shared<ComplexTypeMark>(param, complex_part_type);
        return param_complex->outputs();
    }

    return param->outputs();
}

OutputVector translate_output_arg_op(const NodeContext& node) {
    default_op_checks(node, 1, {"output_arg"}, true);

    element::Type complex_part_type = element::dynamic;
    auto input = node.get_input(0);

    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr());

    if (complex_type_mark) {
        complex_part_type = complex_type_mark->get_complex_part_type();
        input = complex_type_mark->input_value(0);
        auto result = std::make_shared<Result>(input);
        set_node_name(node.get_name(), result);
        auto result_complex = make_shared<ComplexTypeMark>(result, complex_part_type);
        return result_complex->outputs();
    }
    auto result = std::make_shared<Result>(input);
    set_node_name(node.get_name(), result);
    return result->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
