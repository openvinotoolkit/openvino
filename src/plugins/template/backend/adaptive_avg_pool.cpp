// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"
#include <ngraph/runtime/reference/adaptive_avg_pool.hpp>

using namespace ngraph;
using namespace std;

namespace {
template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v8::AdaptiveAvgPool>& op,
              const HostTensorVector& outputs,
              const HostTensorVector& inputs) {
    using T = typename element_type_traits<ET>::value_type;
    runtime::reference::adaptive_avg_pool(inputs[0]->get_data_ptr<T>(),
                                          outputs[0]->get_data_ptr<T>(),
                                          inputs[0]->get_shape(),
                                          op->get_output_shape(0));
    return true;
}
}