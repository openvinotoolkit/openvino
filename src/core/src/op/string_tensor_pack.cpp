// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/string_tensor_unpack.hpp"
#include "string_tensor_pack_shape_inference.hpp"

namespace ov {
namespace op {
namespace v15 {

StringTensorPack::StringTensorPack(const Output<Node>& begins, const Output<Node>& ends, const Output<Node>& symbols)
    : Op({begins, ends, symbols}) {
    constructor_validate_and_infer_types();
}

void StringTensorPack::validate_and_infer_types() {
    OV_OP_SCOPE(v15_StringTensorPack_validate_and_infer_types);

    const auto& begins_element_type = get_input_element_type(0);
    const auto& ends_element_type = get_input_element_type(1);
    const bool is_valid_index_type = (begins_element_type == element::i32 || begins_element_type == element::i64) &&
                                     begins_element_type == ends_element_type;
    NODE_VALIDATION_CHECK(
        this,
        is_valid_index_type,
        "The element types of the begins and ends input tensors must match and be of i32 or i64 type. Got: ",
        begins_element_type,
        " and ",
        ends_element_type);

    const auto& data_element_type = get_input_element_type(2);
    NODE_VALIDATION_CHECK(this,
                          data_element_type == ov::element::u8,
                          "StringTensorPack expects a tensor with ov::element::u8 elements. Got: ",
                          data_element_type);

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, ov::element::string, output_shapes[0]);
}

std::shared_ptr<Node> StringTensorPack::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(v15_StringTensorPack_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<StringTensorPack>(new_args.at(0), new_args.at(1), new_args.at(2));
}
}  // namespace v15
}  // namespace op
}  // namespace ov
