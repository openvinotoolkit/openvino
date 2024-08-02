// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce.hpp"

namespace ov {
namespace intel_cpu {

ReduceExecutor::ReduceExecutor(const ExecutorContext::CPtr context) : context(context) {}

}   // namespace intel_cpu
}   // namespace ov