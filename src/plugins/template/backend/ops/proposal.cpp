// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"
#include "openvino/op/proposal.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v0::Proposal>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::proposal_v0<T>(inputs[0]->get_data_ptr<T>(),
                                               inputs[1]->get_data_ptr<T>(),
                                               inputs[2]->get_data_ptr<T>(),
                                               outputs[0]->get_data_ptr<T>(),
                                               inputs[0]->get_shape(),
                                               inputs[1]->get_shape(),
                                               inputs[2]->get_shape(),
                                               outputs[0]->get_shape(),
                                               op.get()->get_attrs());
    return true;
}

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v4::Proposal>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::proposal_v4<T>(inputs[0]->get_data_ptr<T>(),
                                               inputs[1]->get_data_ptr<T>(),
                                               inputs[2]->get_data_ptr<T>(),
                                               outputs[0]->get_data_ptr<T>(),
                                               outputs[1]->get_data_ptr<T>(),
                                               inputs[0]->get_shape(),
                                               inputs[1]->get_shape(),
                                               inputs[2]->get_shape(),
                                               outputs[0]->get_shape(),
                                               outputs[1]->get_shape(),
                                               op.get()->get_attrs());
    return true;
}
