// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/convert.hpp"

#include "evaluate_node.hpp"

namespace convert_like_v1 {
template <ov::element::Type_t ti, ov::element::Type_t to>
inline void evaluate(const std::shared_ptr<ov::op::v1::ConvertLike>& op,
                     ov::TensorVector& outputs,
                     const ov::TensorVector& inputs) {
    using T_I = typename ov::element_type_traits<ti>::value_type;
    using T_O = typename ov::element_type_traits<to>::value_type;
    outputs[0].set_shape(inputs[0].get_shape());
    size_t element_count = ov::shape_size(outputs[0].get_shape());

    if (((ti == ov::element::u1) || (to == ov::element::u1)) ||
        ((ti == ov::element::u4) || (to == ov::element::u4)) ||
        ((ti == ov::element::i4) || (to == ov::element::i4)) ||
        ((ti == ov::element::nf4) || (to == ov::element::nf4))) {
        ov::reference::detail::lp_convert(inputs[0].data<T_I>(), outputs[0].data<T_O>(), element_count, ti, to);
    } else {
        ov::reference::convert(inputs[0].data<T_I>(), outputs[0].data<T_O>(), element_count);
    }
}
}  // namespace convert_like_v1

template <ov::element::Type_t OUT_ET>
bool evaluate(const std::shared_ptr<ov::op::v1::ConvertLike>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    switch (inputs[0].get_element_type()) {
    case ov::element::boolean:
        convert_like_v1::evaluate<ov::element::boolean, OUT_ET>(op, outputs, inputs);
        break;
    case ov::element::u1:
        convert_like_v1::evaluate<ov::element::u1, OUT_ET>(op, outputs, inputs);
        break;
    case ov::element::u4:
        convert_like_v1::evaluate<ov::element::u4, OUT_ET>(op, outputs, inputs);
        break;
    case ov::element::u8:
        convert_like_v1::evaluate<ov::element::u8, OUT_ET>(op, outputs, inputs);
        break;
    case ov::element::u16:
        convert_like_v1::evaluate<ov::element::u16, OUT_ET>(op, outputs, inputs);
        break;
    case ov::element::u32:
        convert_like_v1::evaluate<ov::element::u32, OUT_ET>(op, outputs, inputs);
        break;
    case ov::element::u64:
        convert_like_v1::evaluate<ov::element::u64, OUT_ET>(op, outputs, inputs);
        break;
    case ov::element::i4:
        convert_like_v1::evaluate<ov::element::i4, OUT_ET>(op, outputs, inputs);
        break;
    case ov::element::i8:
        convert_like_v1::evaluate<ov::element::i8, OUT_ET>(op, outputs, inputs);
        break;
    case ov::element::i16:
        convert_like_v1::evaluate<ov::element::i16, OUT_ET>(op, outputs, inputs);
        break;
    case ov::element::i32:
        convert_like_v1::evaluate<ov::element::i32, OUT_ET>(op, outputs, inputs);
        break;
    case ov::element::i64:
        convert_like_v1::evaluate<ov::element::i64, OUT_ET>(op, outputs, inputs);
        break;
    case ov::element::bf16:
        convert_like_v1::evaluate<ov::element::bf16, OUT_ET>(op, outputs, inputs);
        break;
    case ov::element::f16:
        convert_like_v1::evaluate<ov::element::f16, OUT_ET>(op, outputs, inputs);
        break;
    case ov::element::f32:
        convert_like_v1::evaluate<ov::element::f32, OUT_ET>(op, outputs, inputs);
        break;
    case ov::element::f64:
        convert_like_v1::evaluate<ov::element::f64, OUT_ET>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}

template <>
bool evaluate_node<ov::op::v1::ConvertLike>(std::shared_ptr<ov::Node> node,
                                            ov::TensorVector& outputs,
                                            const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::v1::ConvertLike>(node), outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v1::ConvertLike>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v1::ConvertLike>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v1::ConvertLike>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v1::ConvertLike>(node), outputs, inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::v1::ConvertLike>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v1::ConvertLike>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v1::ConvertLike>(node), outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v1::ConvertLike>(node), outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v1::ConvertLike>(node), outputs, inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::v1::ConvertLike>(node), outputs, inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::v1::ConvertLike>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v1::ConvertLike>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v1::ConvertLike>(node), outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v1::ConvertLike>(node), outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v1::ConvertLike>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}
