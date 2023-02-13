// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"
#include <ngraph/runtime/reference/abs.hpp>

using namespace ngraph;
using namespace std;

namespace {
template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v0::Abs>& op, const HostTensorVector& outputs, const HostTensorVector& inputs) {
    using T = typename element_type_traits<ET>::value_type;
    runtime::reference::abs<T>(inputs[0]->get_data_ptr<T>(),
                               outputs[0]->get_data_ptr<T>(),
                               shape_size(inputs[0]->get_shape()));
    return true;
}
}
