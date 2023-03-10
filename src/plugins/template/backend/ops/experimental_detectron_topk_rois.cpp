// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v6::ExperimentalDetectronTopKROIs>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    using T = typename ngraph::element_type_traits<ET>::value_type;
    size_t max_rois = op->get_max_rois();
    outputs[0]->set_shape(ngraph::Shape{max_rois, 4});
    ngraph::runtime::reference::experimental_detectron_topk_rois<T>(inputs[0]->get_data_ptr<const T>(),
                                                                    inputs[1]->get_data_ptr<const T>(),
                                                                    inputs[0]->get_shape(),
                                                                    inputs[1]->get_shape(),
                                                                    max_rois,
                                                                    outputs[0]->get_data_ptr<T>());
    return true;
}