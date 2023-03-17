// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/node.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/runtime/reference/proposal.hpp"

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v0::Proposal>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    using T = typename ngraph::element_type_traits<ET>::value_type;
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

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v4::Proposal>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    using T = typename ngraph::element_type_traits<ET>::value_type;
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