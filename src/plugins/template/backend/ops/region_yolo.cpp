// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v0::RegionYolo>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    using T = typename ngraph::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::region_yolo<T>(inputs[0]->get_data_ptr<const T>(),
                                               outputs[0]->get_data_ptr<T>(),
                                               inputs[0]->get_shape(),
                                               static_cast<int>(op->get_num_coords()),
                                               static_cast<int>(op->get_num_classes()),
                                               static_cast<int>(op->get_num_regions()),
                                               op->get_do_softmax(),
                                               op->get_mask());
    return true;
}