// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/qsa_prepare.hpp"

#include "itt.hpp"
#include "qsa_prepare_shape_inference.hpp"

namespace ov::op::v17 {

QSAPrepare::QSAPrepare(const OutputVector& inputs, int64_t key_block_size, float epsilon)
    : Op(inputs),
      m_key_block_size(key_block_size),
      m_epsilon(epsilon) {
    constructor_validate_and_infer_types();
}

QSAPrepare::QSAPrepare(const Output<Node>& raw_index_keys,
                       const Output<Node>& past_block_index_keys,
                       const Output<Node>& key_length,
                       const Output<Node>& rope_cos,
                       const Output<Node>& rope_sin,
                       const Output<Node>& k_norm_weight,
                       const Output<Node>& past_incomplete_raw_keys,
                       const Output<Node>& past_incomplete_rope,
                       int64_t key_block_size,
                       float epsilon)
    : QSAPrepare(OutputVector{raw_index_keys,
                              past_block_index_keys,
                              key_length,
                              rope_cos,
                              rope_sin,
                              k_norm_weight,
                              past_incomplete_raw_keys,
                              past_incomplete_rope},
                 key_block_size,
                 epsilon) {}

void QSAPrepare::validate_and_infer_types() {
    OV_OP_SCOPE(v17_QSAPrepare_validate_and_infer_types);
    const auto input_count = get_input_size();
    NODE_VALIDATION_CHECK(this, input_count == 8, "The number of inputs must be 8. Got: ", input_count);
    NODE_VALIDATION_CHECK(this,
                          m_epsilon > 0.0f,
                          "The attribute epsilon must be a positive floating point number. Got: ",
                          m_epsilon);

    static constexpr const char* data_input_names[] = {"raw_index_keys",
                                                       "past_block_index_keys",
                                                       nullptr,
                                                       "rope_cos",
                                                       "rope_sin",
                                                       "k_norm_weight",
                                                       "past_incomplete_raw_keys",
                                                       "past_incomplete_rope"};
    element::Type data_et = get_input_element_type(0);
    for (size_t i = 1; i < input_count; ++i) {
        if (data_input_names[i] == nullptr) {
            continue;
        }
        NODE_VALIDATION_CHECK(this,
                              element::Type::merge(data_et, data_et, get_input_element_type(i)),
                              "The element type of ",
                              data_input_names[i],
                              " must be the same as of raw_index_keys. Got: ",
                              get_input_element_type(i));
    }
    NODE_VALIDATION_CHECK(this,
                          data_et.is_dynamic() || data_et.is_real(),
                          "The element type of the index key inputs must be a floating point type. Got: ",
                          data_et);

    const auto& key_length_et = get_input_element_type(2);
    NODE_VALIDATION_CHECK(this,
                          key_length_et.is_dynamic() || key_length_et == element::i32 || key_length_et == element::i64,
                          "The element type of key_length must be i32 or i64. Got: ",
                          key_length_et);

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, data_et, output_shapes[0]);
    set_output_type(1, data_et, output_shapes[1]);
    set_output_type(2, data_et, output_shapes[2]);
}

bool QSAPrepare::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v17_QSAPrepare_visit_attributes);
    visitor.on_attribute("key_block_size", m_key_block_size);
    visitor.on_attribute("epsilon", m_epsilon);
    return true;
}

std::shared_ptr<Node> QSAPrepare::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v17_QSAPrepare_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<QSAPrepare>(new_args, m_key_block_size, m_epsilon);
}

}  // namespace ov::op::v17
