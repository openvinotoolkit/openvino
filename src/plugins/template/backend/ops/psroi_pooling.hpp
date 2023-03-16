// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::PSROIPooling>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    using T = typename ngraph::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::psroi_pooling<T>(inputs[0]->get_data_ptr<T>(),
                                                 inputs[0]->get_shape(),
                                                 inputs[1]->get_data_ptr<T>(),
                                                 inputs[1]->get_shape(),
                                                 outputs[0]->get_data_ptr<T>(),
                                                 outputs[0]->get_shape(),
                                                 op->get_mode(),
                                                 op->get_spatial_scale(),
                                                 op->get_spatial_bins_x(),
                                                 op->get_spatial_bins_y());

    return true;
}