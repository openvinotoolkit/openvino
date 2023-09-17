// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/convert.hpp"

#include "evaluate_node.hpp"

namespace convert_like_v1 {
template <ngraph::element::Type_t ti, ngraph::element::Type_t to>
inline void evaluate(const std::shared_ptr<ngraph::op::v1::ConvertLike>& op,
                     const ngraph::HostTensorVector& outputs,
                     const ngraph::HostTensorVector& inputs) {
    outputs[0]->set_shape(inputs[0]->get_shape());
    size_t element_count = ngraph::shape_size(outputs[0]->get_shape());

    if (((ti == ngraph::element::u1) || (to == ngraph::element::u1)) ||
        ((ti == ngraph::element::u4) || (to == ngraph::element::u4)) ||
        ((ti == ngraph::element::i4) || (to == ngraph::element::i4))) {
        ov::reference::detail::lp_convert(inputs[0]->get_data_ptr<ti>(),
                                          outputs[0]->get_data_ptr<to>(),
                                          element_count,
                                          ti,
                                          to);
    } else {
        ov::reference::convert(inputs[0]->get_data_ptr<ti>(), outputs[0]->get_data_ptr<to>(), element_count);
    }
}
}  // namespace convert_like_v1

template <ngraph::element::Type_t OUT_ET>
bool evaluate(const std::shared_ptr<ngraph::op::v1::ConvertLike>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    switch (inputs[0]->get_element_type()) {
    case ngraph::element::Type_t::boolean:
        convert_like_v1::evaluate<ngraph::element::Type_t::boolean, OUT_ET>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::u1:
        convert_like_v1::evaluate<ngraph::element::Type_t::u1, OUT_ET>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::u4:
        convert_like_v1::evaluate<ngraph::element::Type_t::u4, OUT_ET>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::u8:
        convert_like_v1::evaluate<ngraph::element::Type_t::u8, OUT_ET>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::u16:
        convert_like_v1::evaluate<ngraph::element::Type_t::u16, OUT_ET>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::u32:
        convert_like_v1::evaluate<ngraph::element::Type_t::u32, OUT_ET>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::u64:
        convert_like_v1::evaluate<ngraph::element::Type_t::u64, OUT_ET>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::i4:
        convert_like_v1::evaluate<ngraph::element::Type_t::i4, OUT_ET>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::i8:
        convert_like_v1::evaluate<ngraph::element::Type_t::i8, OUT_ET>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::i16:
        convert_like_v1::evaluate<ngraph::element::Type_t::i16, OUT_ET>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::i32:
        convert_like_v1::evaluate<ngraph::element::Type_t::i32, OUT_ET>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::i64:
        convert_like_v1::evaluate<ngraph::element::Type_t::i64, OUT_ET>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::bf16:
        convert_like_v1::evaluate<ngraph::element::Type_t::bf16, OUT_ET>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::f16:
        convert_like_v1::evaluate<ngraph::element::Type_t::f16, OUT_ET>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::f32:
        convert_like_v1::evaluate<ngraph::element::Type_t::f32, OUT_ET>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::f64:
        convert_like_v1::evaluate<ngraph::element::Type_t::f64, OUT_ET>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}

template <>
bool evaluate_node<ngraph::op::v1::ConvertLike>(std::shared_ptr<ngraph::Node> node,
                                                const ngraph::HostTensorVector& outputs,
                                                const ngraph::HostTensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ngraph::op::v1::Select>(node) || ov::is_type<ngraph::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ngraph::element::Type_t::boolean:
        return evaluate<ngraph::element::Type_t::boolean>(ov::as_type_ptr<ngraph::op::v1::ConvertLike>(node),
                                                          outputs,
                                                          inputs);
    case ngraph::element::Type_t::bf16:
        return evaluate<ngraph::element::Type_t::bf16>(ov::as_type_ptr<ngraph::op::v1::ConvertLike>(node),
                                                       outputs,
                                                       inputs);
    case ngraph::element::Type_t::f16:
        return evaluate<ngraph::element::Type_t::f16>(ov::as_type_ptr<ngraph::op::v1::ConvertLike>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::f64:
        return evaluate<ngraph::element::Type_t::f64>(ov::as_type_ptr<ngraph::op::v1::ConvertLike>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::f32:
        return evaluate<ngraph::element::Type_t::f32>(ov::as_type_ptr<ngraph::op::v1::ConvertLike>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i4:
        return evaluate<ngraph::element::Type_t::i4>(ov::as_type_ptr<ngraph::op::v1::ConvertLike>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::i8:
        return evaluate<ngraph::element::Type_t::i8>(ov::as_type_ptr<ngraph::op::v1::ConvertLike>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::i16:
        return evaluate<ngraph::element::Type_t::i16>(ov::as_type_ptr<ngraph::op::v1::ConvertLike>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i32:
        return evaluate<ngraph::element::Type_t::i32>(ov::as_type_ptr<ngraph::op::v1::ConvertLike>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i64:
        return evaluate<ngraph::element::Type_t::i64>(ov::as_type_ptr<ngraph::op::v1::ConvertLike>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u1:
        return evaluate<ngraph::element::Type_t::u1>(ov::as_type_ptr<ngraph::op::v1::ConvertLike>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::u4:
        return evaluate<ngraph::element::Type_t::u4>(ov::as_type_ptr<ngraph::op::v1::ConvertLike>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::u8:
        return evaluate<ngraph::element::Type_t::u8>(ov::as_type_ptr<ngraph::op::v1::ConvertLike>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::u16:
        return evaluate<ngraph::element::Type_t::u16>(ov::as_type_ptr<ngraph::op::v1::ConvertLike>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u32:
        return evaluate<ngraph::element::Type_t::u32>(ov::as_type_ptr<ngraph::op::v1::ConvertLike>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u64:
        return evaluate<ngraph::element::Type_t::u64>(ov::as_type_ptr<ngraph::op::v1::ConvertLike>(node),
                                                      outputs,
                                                      inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                       std::string("in evaluate_node()"));
    }
}
