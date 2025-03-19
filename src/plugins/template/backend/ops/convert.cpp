// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/convert.hpp"

#include "evaluate_node.hpp"
#include "openvino/core/type/element_iterator.hpp"

namespace convert {
template <ov::element::Type_t ti, ov::element::Type_t to>
bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) {
    outputs[0].set_shape(inputs[0].get_shape());
    const auto element_count = ov::shape_size(outputs[0].get_shape());

    ov::reference::convert(ov::element::iterator<ti>(static_cast<const void*>(inputs[0].data())),
                           ov::element::iterator<to>(outputs[0].data()),
                           element_count);
    return true;
}

template <ov::element::Type_t OUT_ET>
bool evaluate_by_input_type(ov::TensorVector& outputs, const ov::TensorVector& inputs) {
    switch (inputs[0].get_element_type()) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean, OUT_ET>(outputs, inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1, OUT_ET>(outputs, inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4, OUT_ET>(outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8, OUT_ET>(outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16, OUT_ET>(outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32, OUT_ET>(outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64, OUT_ET>(outputs, inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4, OUT_ET>(outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8, OUT_ET>(outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16, OUT_ET>(outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32, OUT_ET>(outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64, OUT_ET>(outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16, OUT_ET>(outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16, OUT_ET>(outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32, OUT_ET>(outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64, OUT_ET>(outputs, inputs);
    case ov::element::nf4:
        return evaluate<ov::element::nf4, OUT_ET>(outputs, inputs);
    default:
        return false;
    }
}

namespace {
bool evaluate_by_output_type(const ov::element::Type& output_et,
                             ov::TensorVector& outputs,
                             const ov::TensorVector& inputs) {
    switch (output_et) {
    case ov::element::boolean:
        return evaluate_by_input_type<ov::element::boolean>(outputs, inputs);
    case ov::element::u1:
        return evaluate_by_input_type<ov::element::u1>(outputs, inputs);
    case ov::element::u4:
        return evaluate_by_input_type<ov::element::u4>(outputs, inputs);
    case ov::element::u8:
        return evaluate_by_input_type<ov::element::u8>(outputs, inputs);
    case ov::element::u16:
        return evaluate_by_input_type<ov::element::u16>(outputs, inputs);
    case ov::element::u32:
        return evaluate_by_input_type<ov::element::u32>(outputs, inputs);
    case ov::element::u64:
        return evaluate_by_input_type<ov::element::u64>(outputs, inputs);
    case ov::element::i4:
        return evaluate_by_input_type<ov::element::i4>(outputs, inputs);
    case ov::element::i8:
        return evaluate_by_input_type<ov::element::i8>(outputs, inputs);
    case ov::element::i16:
        return evaluate_by_input_type<ov::element::i16>(outputs, inputs);
    case ov::element::i32:
        return evaluate_by_input_type<ov::element::i32>(outputs, inputs);
    case ov::element::i64:
        return evaluate_by_input_type<ov::element::i64>(outputs, inputs);
    case ov::element::bf16:
        return evaluate_by_input_type<ov::element::bf16>(outputs, inputs);
    case ov::element::f16:
        return evaluate_by_input_type<ov::element::f16>(outputs, inputs);
    case ov::element::f32:
        return evaluate_by_input_type<ov::element::f32>(outputs, inputs);
    case ov::element::f64:
        return evaluate_by_input_type<ov::element::f64>(outputs, inputs);
    case ov::element::nf4:
        return evaluate_by_input_type<ov::element::nf4>(outputs, inputs);
    default:
        return false;
    }
}
}  // namespace
}  // namespace convert

template <>
bool evaluate_node<ov::op::v1::ConvertLike>(std::shared_ptr<ov::Node> node,
                                            ov::TensorVector& outputs,
                                            const ov::TensorVector& inputs) {
    if (convert::evaluate_by_output_type(node->get_output_element_type(0), outputs, inputs)) {
        return true;
    } else {
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}

template <>
bool evaluate_node<ov::op::v0::Convert>(std::shared_ptr<ov::Node> node,
                                        ov::TensorVector& outputs,
                                        const ov::TensorVector& inputs) {
    if (convert::evaluate_by_output_type(node->get_output_element_type(0), outputs, inputs)) {
        return true;
    } else {
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}
