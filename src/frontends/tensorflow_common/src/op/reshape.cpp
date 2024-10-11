// Copyright (C) 2018-2024 Intel Corporation
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
        tensor = complex_type_mark->input_value(0);

        OutputVector concat_inputs;
        concat_inputs.push_back(shape);
        concat_inputs.push_back(make_shared<v0::Constant>(shape.get_element_type(), Shape{1}, 2));

        auto concat = make_shared<v0::Concat>(concat_inputs, 0);
        auto reshape = make_shared<v1::Reshape>(tensor, concat, false);
        set_node_name(node.get_name(), reshape);
        auto complex_reshape = make_shared<ComplexTypeMark>(reshape, complex_part_type);
        return {complex_reshape->output(0)};
    }

    std::string output_shape_str = node.get_attribute<std::string>("_output_shapes", "");
    if (!output_shape_str.empty()) {
        ov::PartialShape output_shape = ov::PartialShape(output_shape_str);
        if (tensor.get_partial_shape().is_dynamic()) {
            if(output_shape.is_static()) {
                shape = make_shared<v0::Constant>(shape.get_element_type(), shape.get_shape() , output_shape.to_shape());
            } else if (output_shape.rank().is_static()) {
                if(!output_shape.compatible(tensor.get_partial_shape())) {
                    std::vector<size_t> shape_dimensions;
                    const auto& input_shape = tensor.get_partial_shape();
                    const auto& axis_shape = shape.get_shape();
                    // std::cout << "### " << node.get_name()
                    // << ", output_shape_str=" << output_shape_str
                    // << ", output_shape=" << output_shape
                    // << ", input_shape=" << input_shape
                    // << ", axis_shape=" << axis_shape
                    // << std::endl;
                    bool replace_with_const = true;
                    for (size_t i=0; i < axis_shape[0] && replace_with_const; i++) {
                        if (output_shape[i].is_static()) 
                            shape_dimensions.push_back(output_shape[i].get_length());
                        else if (input_shape[i].is_static())
                            shape_dimensions.push_back(input_shape[i].get_length());
                        else
                            replace_with_const = false;
                    }
                    if (replace_with_const)
                        shape = make_shared<v0::Constant>(shape.get_element_type(), axis_shape , shape_dimensions);
                }
            }
        } 
    }
    auto reshape = make_shared<v1::Reshape>(tensor, shape, false);
    set_node_name(node.get_name(), reshape);
    return {reshape};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
