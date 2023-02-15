// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"
#include "ngraph/runtime/reference/reorg_yolo.hpp"
#include "openvino/op/reorg_yolo.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v0::ReorgYolo>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    ngraph::runtime::reference::reorg_yolo(inputs[0]->get_data_ptr<char>(),
                                   outputs[0]->get_data_ptr<char>(),
                                   inputs[0]->get_shape(),
                                   op->get_strides().at(0),
                                   inputs[0]->get_element_type().size());
    return true;
}
