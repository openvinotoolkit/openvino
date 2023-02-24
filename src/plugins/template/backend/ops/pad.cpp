// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/pad.hpp"

#include "evaluates_map.hpp"
#include "openvino/op/pad.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v1::Pad>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    ngraph::runtime::reference::pad(inputs[0]->get_data_ptr<char>(),
                                    inputs[1]->get_data_ptr<char>(),
                                    outputs[0]->get_data_ptr<char>(),
                                    shape_size(inputs[0]->get_shape()),
                                    inputs[1]->get_shape(),
                                    outputs[0]->get_shape(),
                                    op->get_pads_end(),
                                    op->get_pads_begin(),
                                    op->get_pad_mode());
    return true;
}
