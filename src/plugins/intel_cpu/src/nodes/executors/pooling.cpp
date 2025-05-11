// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pooling.hpp"

#include <utility>

namespace ov::intel_cpu {

PoolingExecutor::PoolingExecutor(ExecutorContext::CPtr context) : context(std::move(context)) {}

}  // namespace ov::intel_cpu
