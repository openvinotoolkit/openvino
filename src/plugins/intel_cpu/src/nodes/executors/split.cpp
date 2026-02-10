// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "split.hpp"

#include <utility>

#include "executor.hpp"

namespace ov::intel_cpu {

SplitExecutor::SplitExecutor(ExecutorContext::CPtr context) : context(std::move(context)) {}

}  // namespace ov::intel_cpu
