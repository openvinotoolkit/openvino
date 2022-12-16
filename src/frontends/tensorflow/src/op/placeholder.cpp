// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/frontend/tensorflow/frontend.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_placeholder_op(const NodeContext& node) {

    // TODO: Remove this duplicated code: it exists here and in frontend.cpp where TensorPlace for Placeholders is created

    std::cerr << "[ INFO PT FE ] Start Placeholder conversion\n";

    Any structural_type = node.get_attribute_as_any("dtype");
    // TODO: calling get_attribute_as_any directly instead of
    // templated version skips calling apply_additional_conversion_rules.
    // It may result in unrecognized type in case if it is completely
    // undefined type which are not covered by one of ov::element::StructuralType's

    bool is_element_type = structural_type.is<ov::element::Type>();

    ov::element::Type tf_dtype = is_element_type ?
                                    structural_type.as<ov::element::Type>() :
                                    ov::element::dynamic;

    // END OF DUPLICATED FRAGMENT

    auto tf_shape = node.get_attribute<ov::PartialShape>("shape", ov::PartialShape::dynamic());

    std::cerr << "[ INFO PT FE ] Placeholder '" << node.get_name() << "' conversion: shape = " << tf_shape << "\n";

    auto res = std::make_shared<Parameter>(tf_dtype, tf_shape);

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