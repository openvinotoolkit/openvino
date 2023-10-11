// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(NGRAPH_LEGACY_HEADER_INCLUDED)
#    define NGRAPH_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

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
