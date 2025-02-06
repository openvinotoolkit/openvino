// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce.hpp"

#include <utility>

namespace ov {
namespace intel_cpu {

ReduceExecutor::ReduceExecutor(ExecutorContext::CPtr context) : context(std::move(context)) {}

}  // namespace intel_cpu
}  // namespace ov
