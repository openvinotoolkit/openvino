// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce.hpp"

#include <utility>

namespace ov::intel_cpu {

ReduceExecutor::ReduceExecutor(ExecutorContext::CPtr context) : context(std::move(context)) {}

}  // namespace ov::intel_cpu
