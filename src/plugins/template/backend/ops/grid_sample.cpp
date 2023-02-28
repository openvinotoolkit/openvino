// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"
#include "openvino/op/grid_sample.hpp"

template <ov::element::Type_t DATA_ET>
bool evaluate(const std::shared_ptr<ov::op::v9::GridSample>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    const auto& attributes = op->get_attributes();
    ov::element::Type grid_et = op->get_input_element_type(1);
    switch (grid_et) {
    case ov::element::Type_t::f32:
        ngraph::runtime::reference::grid_sample(outputs[0]->get_data_ptr<DATA_ET>(),
                                                inputs[0]->get_data_ptr<DATA_ET>(),
                                                inputs[1]->get_data_ptr<ov::element::Type_t::f32>(),
                                                inputs[0]->get_shape(),
                                                inputs[1]->get_shape(),
                                                attributes.align_corners,
                                                attributes.mode,
                                                attributes.padding_mode);
        break;
    default:
        return false;
    }
    return true;
}
