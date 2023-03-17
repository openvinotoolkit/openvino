// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/node.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/runtime/reference/avg_pool.hpp"

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v1::AvgPool>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    using T = typename ngraph::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::avg_pool<T>(inputs[0]->get_data_ptr<T>(),
                                            outputs[0]->get_data_ptr<T>(),
                                            inputs[0]->get_shape(),
                                            op->get_output_shape(0),
                                            op->get_kernel(),
                                            op->get_strides(),
                                            op->get_pads_begin(),
                                            op->get_pads_end(),
                                            !op->get_exclude_pad());
    return true;
}