// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/string_tensor_unpack.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/op.hpp"
#include "string_tensor_unpack_shape_inference.hpp"

namespace ov {
namespace op {
namespace v15 {

StringTensorUnpack::StringTensorUnpack(const Output<Node>& data) : Op({data}) {
    constructor_validate_and_infer_types();
}

void StringTensorUnpack::validate_and_infer_types() {
    OV_OP_SCOPE(v15_StringTensorUnpack_validate_and_infer_types);

    const auto& data_element_type = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          data_element_type == ov::element::string,
                          "StringTensorUnpack expects a tensor with string elements. Got: ",
                          data_element_type);

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, ov::element::i32, output_shapes[0]);
    set_output_type(1, ov::element::i32, output_shapes[1]);
    set_output_type(2, ov::element::u8, output_shapes[2]);
}

std::shared_ptr<Node> StringTensorUnpack::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(v15_StringTensorUnpack_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<StringTensorUnpack>(new_args.at(0));
}
}  // namespace v15
}  // namespace op
}  // namespace ov
