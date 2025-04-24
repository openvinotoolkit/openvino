// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gather_nd.hpp"

#include "gather_nd_shape_inference.hpp"
#include "itt.hpp"

namespace ov {

// ------------------------------ V5 ------------------------------
op::v5::GatherND::GatherND(const Output<Node>& data, const Output<Node>& indices, const size_t batch_dims)
    : GatherNDBase(data, indices, batch_dims) {
    constructor_validate_and_infer_types();
}

void op::v5::GatherND::validate_and_infer_types() {
    OV_OP_SCOPE(v5_GatherND_validate_and_infer_types);

    const auto& data_type = get_input_element_type(0);
    const auto& indices_type = get_input_element_type(1);

    NODE_VALIDATION_CHECK(this,
                          indices_type.is_integral_number(),
                          "The indices type is expected to be an integer type. Got: ",
                          indices_type);

    const auto out_shapes =
        shape_infer(this, std::vector<PartialShape>{get_input_partial_shape(0), get_input_partial_shape(1)});
    set_output_type(0, data_type, out_shapes[0]);
}

std::shared_ptr<Node> op::v5::GatherND::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v5_GatherND_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<op::v5::GatherND>(new_args.at(0), new_args.at(1), m_batch_dims);
}

// ------------------------------ V8 ------------------------------
op::v8::GatherND::GatherND(const Output<Node>& data, const Output<Node>& indices, const size_t batch_dims)
    : GatherNDBase(data, indices, batch_dims) {
    constructor_validate_and_infer_types();
}

void op::v8::GatherND::validate_and_infer_types() {
    OV_OP_SCOPE(v8_GatherND_validate_and_infer_types);
    const auto& data_type = get_input_element_type(0);
    const auto& indices_type = get_input_element_type(1);

    NODE_VALIDATION_CHECK(this,
                          indices_type.is_integral_number(),
                          "The indices type is expected to be an integer type. Got: ",
                          indices_type);

    const auto out_shapes =
        shape_infer(this, std::vector<PartialShape>{get_input_partial_shape(0), get_input_partial_shape(1)});
    set_output_type(0, data_type, ov::PartialShape(out_shapes[0]));
}

std::shared_ptr<Node> op::v8::GatherND::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v8_GatherND_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<op::v8::GatherND>(new_args.at(0), new_args.at(1), m_batch_dims);
}
}  // namespace ov
