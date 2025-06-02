// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0mvn
//

#include "eltwise.hpp"

#include <utility>

#include "nodes/executors/executor.hpp"

namespace ov::intel_cpu {

EltwiseExecutor::EltwiseExecutor(ExecutorContext::CPtr context) : context(std::move(context)) {}

}  // namespace ov::intel_cpu
