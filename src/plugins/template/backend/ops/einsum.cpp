// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"
#include "ngraph/runtime/reference/einsum.hpp"
#include "openvino/op/einsum.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v7::Einsum>& op, const ov::HostTensorVector& outputs, const ov::HostTensorVector& inputs) {
    const auto equation = op->get_equation();
    ngraph::runtime::reference::einsum(outputs, inputs, equation);
    return true;
}