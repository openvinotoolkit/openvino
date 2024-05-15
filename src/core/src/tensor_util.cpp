// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/tensor_util.hpp"

#include "openvino/op/greater_eq.hpp"
#include "openvino/op/reduce_logical_and.hpp"

ov::Tensor ov::util::ge(const ov::Tensor& lhs, const ov::Tensor& rhs) {
    if (!lhs || !rhs)
        return {};
    Tensor result(element::boolean, {});
    TensorVector outputs = {result};
    if (ov::op::v1::GreaterEqual().evaluate(outputs, {lhs, rhs}))
        return outputs[0];
    else
        return {};
}

bool ov::util::all(const ov::Tensor& t) {
    if (!t)
        return false;

    Tensor result(element::boolean, {});
    TensorVector outputs = {result};

    auto axes_vector = std::vector<int64_t>(t.get_shape().size());
    std::iota(axes_vector.begin(), axes_vector.end(), 0);
    auto axes = make_tensor_of_value(element::i64, Shape(axes_vector.size()), axes_vector);

    if (!ov::op::v1::ReduceLogicalAnd().evaluate(outputs, {t, axes}))
        return false;
    auto result_as_vector = to_vector<bool>(result);
    if (result_as_vector.size() != 1)
        return false;
    return result_as_vector[0];
}
