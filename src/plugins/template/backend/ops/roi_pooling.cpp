// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/roi_pooling.hpp"

#include "evaluates_map.hpp"
#include "openvino/op/roi_pooling.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v0::ROIPooling>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::roi_pooling<T>(inputs[0]->get_data_ptr<const T>(),
                                               inputs[1]->get_data_ptr<const T>(),
                                               outputs[0]->get_data_ptr<T>(),
                                               op->get_input_shape(0),
                                               op->get_input_shape(1),
                                               op->get_output_shape(0),
                                               op->get_spatial_scale(),
                                               op->get_method());
    return true;
}