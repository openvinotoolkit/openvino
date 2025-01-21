// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0mvn
//

#include "eltwise.hpp"

#include <utility>

namespace ov {
namespace intel_cpu {

EltwiseExecutor::EltwiseExecutor(ExecutorContext::CPtr context) : context(std::move(context)) {}

}  // namespace intel_cpu
}  // namespace ov
