// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/transpose.hpp"

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/reference/transpose.hpp"
#include "transpose_shape_inference.hpp"

namespace ov {
namespace op {
namespace v1 {

Transpose::Transpose(const Output<Node>& arg, const Output<Node>& input_order) : Op({arg, input_order}) {
    constructor_validate_and_infer_types();
}

void Transpose::validate_and_infer_types() {
    OV_OP_SCOPE(v1_Transpose_validate_and_infer_types);
    const auto& input_order_et = get_input_element_type(ORDER);
    NODE_VALIDATION_CHECK(this,
                          input_order_et.is_dynamic() || input_order_et.is_integral_number(),
                          "Input order must have an integral number element type.");

    set_input_is_relevant_to_shape(ORDER);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);

    set_output_type(ARG, get_input_element_type(ARG), output_shapes[ARG_T]);
}

std::shared_ptr<Node> Transpose::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Transpose_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Transpose>(new_args[ARG], new_args[ORDER]);
}

enum class int4_extract_t : uint8_t { low_half = 0, high_half = 4 };

bool Transpose::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v1_Transpose_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 2);

    const auto& order = inputs[ORDER];
    if (order.get_element_type().is_integral()) {
        const auto& arg = inputs[ARG];
        const auto& arg_type = arg.get_element_type();
        auto axes_order = ov::get_tensor_data_as<int64_t>(order);
        const auto out_shape = calc_output_shape(this, arg.get_shape(), axes_order);

        auto& out = outputs[ARG_T];
        out.set_shape(out_shape);

        struct int4_iterator {
            explicit int4_iterator(uint8_t* ptr) : m_ptr(ptr), m_half(int4_extract_t::low_half) {}
            explicit int4_iterator(uint8_t* ptr, int4_extract_t half) : m_ptr(ptr), m_half(half) {}
            void operator++() {
                if (m_half == int4_extract_t::low_half) {
                    m_half = int4_extract_t::high_half;
                } else {
                    m_half = int4_extract_t::low_half;
                    m_ptr += 1;
                }
            }

            int4_iterator operator+(const size_t shift) {
                return int4_iterator{m_ptr + shift / 2,
                                     shift % 2 ? int4_extract_t::high_half : int4_extract_t::low_half};
            }

            void copy_from(const int4_iterator& from) const {
                // TODO: DOUBLE CHECK THIS
                // perhaps that's not entirely accurate
                uint8_t from_val = *from.m_ptr;
                uint8_t mask_from = from.m_half == int4_extract_t::high_half ? 0xF0 : 0x0F;
                uint8_t mask_to = m_half == int4_extract_t::high_half ? 0x0F : 0xF0;

                if (from.m_half < m_half) {
                    from_val <<= 4;
                } else if (from.m_half > m_half) {
                    from_val >>= 4;
                } else {
                    from_val &= mask_from;
                }

                *m_ptr = (*m_ptr & mask_to) | from_val;
            }

            uint8_t* m_ptr;
            int4_extract_t m_half;
        };

        auto out_ptr = int4_iterator(static_cast<uint8_t*>(out.data()));
        auto in_ptr = int4_iterator(static_cast<uint8_t*>(arg.data()));
        if ((arg_type == ov::element::i4 || arg_type == ov::element::u4) && arg.get_shape().size() == 2) {
            for (size_t i = 0; i < out_shape[0]; i++) {
                size_t off = i;
                for (size_t j = 0; j < out_shape[1]; j++) {
                    out_ptr.copy_from(in_ptr + off);
                    ++out_ptr;
                    off += out_shape[0];
                }
            }
        } else {
            reference::transpose(static_cast<const char*>(arg.data()),
                                 static_cast<char*>(out.data()),
                                 arg.get_shape(),
                                 arg.get_element_type().size(),
                                 axes_order,
                                 out_shape);
        }
        return true;
    } else {
        return false;
    }
}

bool Transpose::has_evaluate() const {
    OV_OP_SCOPE(v1_Transpose_has_evaluate);
    return get_input_element_type(ORDER).is_integral_number();
}

bool Transpose::evaluate_lower(ov::TensorVector& output_values) const {
    return get_input_tensor(ORDER).has_and_set_bound() && default_lower_bound_evaluator(this, output_values);
}

bool Transpose::evaluate_upper(ov::TensorVector& output_values) const {
    return get_input_tensor(ORDER).has_and_set_bound() && default_upper_bound_evaluator(this, output_values);
}

bool Transpose::evaluate_symbol(TensorSymbolVector& output_symbols) const {
    return get_input_tensor(ORDER).has_and_set_bound() && ov::util::default_symbol_evaluator(this, output_symbols);
}
}  // namespace v1
}  // namespace op
}  // namespace ov
