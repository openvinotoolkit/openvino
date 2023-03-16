// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"

namespace cum_sum_v0 {
template <ngraph::element::Type_t t1, ngraph::element::Type_t t2>
inline void evaluate(const std::shared_ptr<ngraph::op::v0::CumSum>& op,
                     const ngraph::HostTensorVector& outputs,
                     const ngraph::HostTensorVector& inputs) {
    using T1 = typename ngraph::element_type_traits<t1>::value_type;
    using T2 = typename ngraph::element_type_traits<t2>::value_type;
    ngraph::runtime::reference::cumsum<T1, T2>(inputs[0]->get_data_ptr<T1>(),
                                               inputs[1]->get_data_ptr<T2>(),
                                               outputs[0]->get_data_ptr<T1>(),
                                               inputs[0]->get_shape(),
                                               op->is_exclusive(),
                                               op->is_reverse());
}
}  // namespace cum_sum_v0

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v0::CumSum>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    switch (inputs[1]->get_element_type()) {
    case ngraph::element::Type_t::i64:
        cum_sum_v0::evaluate<ET, ngraph::element::Type_t::i64>(op, outputs, inputs);
        break;
    default:
        cum_sum_v0::evaluate<ET, ngraph::element::Type_t::i32>(op, outputs, inputs);
        break;
    }
    return true;
}