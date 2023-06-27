// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluate_node.hpp"
#include "ngraph/runtime/reference/group_normalization_.hpp"

template <ngraph::element::Type_t DATA_ET>
bool evaluate(const std::shared_ptr<ngraph::op::v12::GroupNormalization>& node,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    ngraph::runtime::reference::group_normalization(inputs[0]->get_data_ptr<DATA_ET>(),
                                                    inputs[1]->get_data_ptr<DATA_ET>(),
                                                    inputs[2]->get_data_ptr<DATA_ET>(),
                                                    outputs[0]->get_data_ptr<DATA_ET>(),
                                                    inputs[0]->get_shape(),
                                                    static_cast<size_t>(node->get_num_groups()),
                                                    node->get_epsilon());
    return true;
}

template <>
bool evaluate_node<ngraph::op::v12::GroupNormalization>(std::shared_ptr<ngraph::Node> node,
                                                        const ngraph::HostTensorVector& outputs,
                                                        const ngraph::HostTensorVector& inputs) {
    switch (node->get_input_element_type(0)) {
    // TODO uncomment below
    // case ngraph::element::Type_t::bf16:
    //     return evaluate<ngraph::element::Type_t::bf16>(ov::as_type_ptr<ngraph::op::v12::GroupNormalization>(node),
    //                                                    outputs,
    //                                                    inputs);
    // case ngraph::element::Type_t::f16:
    //     return evaluate<ngraph::element::Type_t::f16>(ov::as_type_ptr<ngraph::op::v12::GroupNormalization>(node),
    //                                                   outputs,
    //                                                   inputs);
    // case ngraph::element::Type_t::f64:
    //     return evaluate<ngraph::element::Type_t::f64>(ov::as_type_ptr<ngraph::op::v12::GroupNormalization>(node),
    //                                                   outputs,
    //                                                   inputs);
    case ngraph::element::Type_t::f32:
        return evaluate<ngraph::element::Type_t::f32>(ov::as_type_ptr<ngraph::op::v12::GroupNormalization>(node),
                                                      outputs,
                                                      inputs);
    // TODO uncomment below
    // case ngraph::element::Type_t::i4:
    //     return evaluate<ngraph::element::Type_t::i4>(ov::as_type_ptr<ngraph::op::v12::GroupNormalization>(node),
    //                                                  outputs,
    //                                                  inputs);
    // case ngraph::element::Type_t::i8:
    //     return evaluate<ngraph::element::Type_t::i8>(ov::as_type_ptr<ngraph::op::v12::GroupNormalization>(node),
    //                                                  outputs,
    //                                                  inputs);
    // case ngraph::element::Type_t::i16:
    //     return evaluate<ngraph::element::Type_t::i16>(ov::as_type_ptr<ngraph::op::v12::GroupNormalization>(node),
    //                                                   outputs,
    //                                                   inputs);
    // case ngraph::element::Type_t::i32:
    //     return evaluate<ngraph::element::Type_t::i32>(ov::as_type_ptr<ngraph::op::v12::GroupNormalization>(node),
    //                                                   outputs,
    //                                                   inputs);
    // case ngraph::element::Type_t::i64:
    //     return evaluate<ngraph::element::Type_t::i64>(ov::as_type_ptr<ngraph::op::v12::GroupNormalization>(node),
    //                                                   outputs,
    //                                                   inputs);
    // case ngraph::element::Type_t::u1:
    //     return evaluate<ngraph::element::Type_t::u1>(ov::as_type_ptr<ngraph::op::v12::GroupNormalization>(node),
    //                                                  outputs,
    //                                                  inputs);
    // case ngraph::element::Type_t::u4:
    //     return evaluate<ngraph::element::Type_t::u4>(ov::as_type_ptr<ngraph::op::v12::GroupNormalization>(node),
    //                                                  outputs,
    //                                                  inputs);
    // case ngraph::element::Type_t::u8:
    //     return evaluate<ngraph::element::Type_t::u8>(ov::as_type_ptr<ngraph::op::v12::GroupNormalization>(node),
    //                                                  outputs,
    //                                                  inputs);
    // case ngraph::element::Type_t::u16:
    //     return evaluate<ngraph::element::Type_t::u16>(ov::as_type_ptr<ngraph::op::v12::GroupNormalization>(node),
    //                                                   outputs,
    //                                                   inputs);
    // case ngraph::element::Type_t::u32:
    //     return evaluate<ngraph::element::Type_t::u32>(ov::as_type_ptr<ngraph::op::v12::GroupNormalization>(node),
    //                                                   outputs,
    //                                                   inputs);
    // case ngraph::element::Type_t::u64:
    //     return evaluate<ngraph::element::Type_t::u64>(ov::as_type_ptr<ngraph::op::v12::GroupNormalization>(node),
    //                                                   outputs,
    //                                                   inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                       std::string("in evaluate_node()"));
    }
}
