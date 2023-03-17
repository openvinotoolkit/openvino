// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/node.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/runtime/reference/roi_pooling.hpp"

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v0::ROIPooling>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    using T = typename ngraph::element_type_traits<ET>::value_type;
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