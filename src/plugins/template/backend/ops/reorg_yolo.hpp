// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/node.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/runtime/reference/reorg_yolo.hpp"

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v0::ReorgYolo>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    ngraph::runtime::reference::reorg_yolo(inputs[0]->get_data_ptr<char>(),
                                           outputs[0]->get_data_ptr<char>(),
                                           inputs[0]->get_shape(),
                                           op->get_strides().at(0),
                                           inputs[0]->get_element_type().size());
    return true;
}