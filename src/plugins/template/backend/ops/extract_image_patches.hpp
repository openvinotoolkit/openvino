// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/node.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/runtime/reference/extract_image_patches.hpp"

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v3::ExtractImagePatches>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    using T = typename ngraph::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::extract_image_patches<T>(op,
                                                         inputs[0]->get_data_ptr<T>(),
                                                         outputs[0]->get_data_ptr<T>(),
                                                         inputs[0]->get_shape(),
                                                         outputs[0]->get_shape());
    return true;
}