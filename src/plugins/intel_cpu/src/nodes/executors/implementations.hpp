// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <type_traits>
#include <vector>

#include "nodes/executors/convolution_config.hpp"
#include "nodes/executors/eltwise_config.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/matmul_config.hpp"
#include "nodes/executors/interpolate_config.hpp"

namespace ov::intel_cpu {

// @todo move to separate header file
template <typename... T>
struct always_false : std::false_type {};

template <typename Attrs>
const std::vector<ExecutorImplementation<Attrs>>& getImplementations() {
    static_assert(always_false<Attrs>::value, "Only specialization instantiations are allowed");
    return {};
}

// FullyConnected
template <>
const std::vector<ExecutorImplementation<FCAttrs>>& getImplementations();

// Convolution
template <>
const std::vector<ExecutorImplementation<ConvAttrs>>& getImplementations();

// Eltwise
template <>
const std::vector<ExecutorImplementation<EltwiseAttrs>>& getImplementations();

// MatMul
template <>
const std::vector<ExecutorImplementation<MatMulAttrs>>& getImplementations();

// Interpolate
template <>
const std::vector<ExecutorImplementation<InterpolateAttrs>>& getImplementations();

}  // namespace ov::intel_cpu
