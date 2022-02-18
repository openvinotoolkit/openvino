// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "ngraph/except.hpp"
#include "ngraph/node.hpp"
#include "openvino/op/util/activation_functions.hpp"

namespace ngraph {
namespace op {
namespace util {
namespace error {
using ov::op::util::error::UnknownActivationFunction;
}  // namespace error

namespace detail {
using ov::op::util::detail::hardsigmoid;
using ov::op::util::detail::relu;
using ov::op::util::detail::sigmoid;
using ov::op::util::detail::tanh;
}  // namespace detail

using ov::op::util::ActivationFunction;
using ov::op::util::ActivationFunctionType;
using ov::op::util::get_activation_func_by_name;
}  // namespace util
}  // namespace op
}  // namespace ngraph
