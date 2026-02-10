// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "concat.hpp"

#include <utility>

#include "nodes/executors/executor.hpp"

namespace ov::intel_cpu {

ConcatExecutor::ConcatExecutor(ExecutorContext::CPtr context) : context(std::move(context)) {}

}  // namespace ov::intel_cpu
