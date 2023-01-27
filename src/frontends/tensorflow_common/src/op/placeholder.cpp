// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/opsets/opset8.hpp"
#include "helper_ops/str_ops.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_placeholder_op(const NodeContext& node) {
    std::cerr << "[ INFO PT FE ] Start Placeholder conversion\n";

    Any structural_type = node.get_attribute_as_any("dtype");
    // TODO: calling get_attribute_as_any directly instead of
    // templated version skips calling apply_additional_conversion_rules.
    // It may result in unrecognized type in case if it is completely
    // undefined type which are not covered by one of ov::element::StructuralType's

    bool is_element_type = structural_type.is<ov::element::Type>();

    ov::element::Type dtype = is_element_type ?
                                    structural_type.as<ov::element::Type>() :
                                    ov::element::dynamic;

    // END OF DUPLICATED FRAGMENT

    auto shape = node.get_attribute<ov::PartialShape>("shape", ov::PartialShape::dynamic());
    if (shape.rank().is_static() && shape.rank().get_length() == 0 && node.has_attribute("_output_shapes")) {
        // we know some cases when Placeholder operation has empty scalar `shape` attribute value
        // and non-empty `_output_shapes` attribute value.
        // `_output_shapes` attribute value turns to be correct in this case
        auto output_shapes = node.get_attribute<std::vector<ov::PartialShape>>("_output_shapes");
        if (output_shapes.size() == 1 && output_shapes[0].rank().is_static()) {
            shape = output_shapes[0];
        }
    }

    auto res = std::make_shared<Parameter>(dtype, shape);

    if(!is_element_type) {
        // There is not representable type information in tf_dtype, save it to RT info
        res->get_rt_info()["structural_type"] = StructuralTypeAttribute(structural_type);
    }

    set_node_name(node.get_name(), res);
    return res->outputs();
}

OutputVector translate_placeholder_with_default_op(const NodeContext& node) {
    // For parity with legacy frontend, it creates a constant node with the default value
    // As a rule, PlaceholderWithDefault is mainly used for is_training variables in the model
    TENSORFLOW_OP_VALIDATION(node,
                             node.get_input_size() > 0,
                             "PlaceholderWithDefault must have at least one input that is the default value.");
    auto input = node.get_input(0);
    set_out_name(node.get_name(), input);
    return {input};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
