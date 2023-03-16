// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v0::MVN>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    using T = typename ngraph::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::mvn<T>(inputs[0]->get_data_ptr<ET>(),
                                       outputs[0]->get_data_ptr<ET>(),
                                       inputs[0]->get_shape(),
                                       op->get_normalize_variance(),
                                       op->get_reduction_axes(),
                                       op->get_eps());
    return true;
}

namespace mvn_6_axes {
template <typename T>
ngraph::AxisSet mvn_6_reduction_axes(const ngraph::HostTensorPtr& axes_input, size_t rank) {
    T* a = axes_input->get_data_ptr<T>();
    auto v = std::vector<T>(a, a + axes_input->get_shape()[0]);
    std::vector<size_t> axes(v.size(), 0);
    for (size_t i = 0; i < v.size(); i++) {
        if (v[i] < 0) {
            if (rank + v[i] < 0) {
                throw ngraph::ngraph_error("Unexpected axis");
            }
            axes[i] = (size_t)(rank + v[i]);
        } else {
            axes[i] = (size_t)(v[i]);
        }
    }
    return ngraph::AxisSet(axes);
}
}  // namespace mvn_6_axes

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v6::MVN>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    using T = typename ngraph::element_type_traits<ET>::value_type;
    ngraph::AxisSet reduction_axes;
    auto rank = inputs[0]->get_shape().size();
    if (inputs[1]->get_element_type() == ngraph::element::i64) {
        reduction_axes = mvn_6_axes::mvn_6_reduction_axes<int64_t>(inputs[1], rank);
    } else if (inputs[1]->get_element_type() == ngraph::element::i32) {
        reduction_axes = mvn_6_axes::mvn_6_reduction_axes<int32_t>(inputs[1], rank);
    } else {
        throw ngraph::ngraph_error("Unexpected indices type");
    }
    ngraph::runtime::reference::mvn_6<T>(inputs[0]->get_data_ptr<ET>(),
                                         outputs[0]->get_data_ptr<ET>(),
                                         inputs[0]->get_shape(),
                                         reduction_axes,
                                         op->get_normalize_variance(),
                                         op->get_eps(),
                                         op->get_eps_mode());
    return true;
}
