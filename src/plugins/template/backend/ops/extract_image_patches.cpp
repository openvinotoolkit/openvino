// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/extract_image_patches.hpp"

#include "evaluate_node.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v3::ExtractImagePatches>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    ov::reference::extract_image_patches<T>(op,
                                            inputs[0].data<T>(),
                                            outputs[0].data<T>(),
                                            inputs[0].get_shape(),
                                            outputs[0].get_shape());
    return true;
}

template <>
bool evaluate_node<ov::op::v3::ExtractImagePatches>(std::shared_ptr<ov::Node> node,
                                                    ov::TensorVector& outputs,
                                                    const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::Type_t::boolean:
        return evaluate<ov::element::Type_t::boolean>(ov::as_type_ptr<ov::op::v3::ExtractImagePatches>(node),
                                                      outputs,
                                                      inputs);
    case ov::element::Type_t::bf16:
        return evaluate<ov::element::Type_t::bf16>(ov::as_type_ptr<ov::op::v3::ExtractImagePatches>(node),
                                                   outputs,
                                                   inputs);
    case ov::element::Type_t::f16:
        return evaluate<ov::element::Type_t::f16>(ov::as_type_ptr<ov::op::v3::ExtractImagePatches>(node),
                                                  outputs,
                                                  inputs);
    case ov::element::Type_t::f64:
        return evaluate<ov::element::Type_t::f64>(ov::as_type_ptr<ov::op::v3::ExtractImagePatches>(node),
                                                  outputs,
                                                  inputs);
    case ov::element::Type_t::f32:
        return evaluate<ov::element::Type_t::f32>(ov::as_type_ptr<ov::op::v3::ExtractImagePatches>(node),
                                                  outputs,
                                                  inputs);
    case ov::element::Type_t::i4:
        return evaluate<ov::element::Type_t::i4>(ov::as_type_ptr<ov::op::v3::ExtractImagePatches>(node),
                                                 outputs,
                                                 inputs);
    case ov::element::Type_t::i8:
        return evaluate<ov::element::Type_t::i8>(ov::as_type_ptr<ov::op::v3::ExtractImagePatches>(node),
                                                 outputs,
                                                 inputs);
    case ov::element::Type_t::i16:
        return evaluate<ov::element::Type_t::i16>(ov::as_type_ptr<ov::op::v3::ExtractImagePatches>(node),
                                                  outputs,
                                                  inputs);
    case ov::element::Type_t::i32:
        return evaluate<ov::element::Type_t::i32>(ov::as_type_ptr<ov::op::v3::ExtractImagePatches>(node),
                                                  outputs,
                                                  inputs);
    case ov::element::Type_t::i64:
        return evaluate<ov::element::Type_t::i64>(ov::as_type_ptr<ov::op::v3::ExtractImagePatches>(node),
                                                  outputs,
                                                  inputs);
    case ov::element::Type_t::u1:
        return evaluate<ov::element::Type_t::u1>(ov::as_type_ptr<ov::op::v3::ExtractImagePatches>(node),
                                                 outputs,
                                                 inputs);
    case ov::element::Type_t::u4:
        return evaluate<ov::element::Type_t::u4>(ov::as_type_ptr<ov::op::v3::ExtractImagePatches>(node),
                                                 outputs,
                                                 inputs);
    case ov::element::Type_t::u8:
        return evaluate<ov::element::Type_t::u8>(ov::as_type_ptr<ov::op::v3::ExtractImagePatches>(node),
                                                 outputs,
                                                 inputs);
    case ov::element::Type_t::u16:
        return evaluate<ov::element::Type_t::u16>(ov::as_type_ptr<ov::op::v3::ExtractImagePatches>(node),
                                                  outputs,
                                                  inputs);
    case ov::element::Type_t::u32:
        return evaluate<ov::element::Type_t::u32>(ov::as_type_ptr<ov::op::v3::ExtractImagePatches>(node),
                                                  outputs,
                                                  inputs);
    case ov::element::Type_t::u64:
        return evaluate<ov::element::Type_t::u64>(ov::as_type_ptr<ov::op::v3::ExtractImagePatches>(node),
                                                  outputs,
                                                  inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}
