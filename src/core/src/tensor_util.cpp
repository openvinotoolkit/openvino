// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/tensor_util.hpp"

#include "openvino/op/greater_eq.hpp"
#include "openvino/op/reduce_logical_and.hpp"

ov::Tensor ov::util::greater_equal(const ov::Tensor& lhs, const ov::Tensor& rhs) {
    if (!lhs || !rhs)
        return {};
    TensorVector outputs{{element::boolean, {}}};
    if (ov::op::v1::GreaterEqual().evaluate(outputs, {lhs, rhs}))
        return std::move(outputs[0]);
    else
        return {};
}

bool ov::util::reduce_and(const ov::Tensor& t) {
    if (!t)
        return false;

    auto outputs = TensorVector{{element::boolean, Shape{}}};
    auto axes = Tensor(element::i64, Shape{t.get_shape().size()});
    std::iota(axes.data<int64_t>(), axes.data<int64_t>() + t.get_shape().size(), 0);
    if (!ov::op::v1::ReduceLogicalAnd().evaluate(outputs, {t, std::move(axes)}))
        return false;
    return outputs[0].data<char>();
}
