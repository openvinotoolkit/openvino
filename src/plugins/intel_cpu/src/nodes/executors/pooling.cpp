// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pooling.hpp"

namespace ov {
namespace intel_cpu {

PoolingExecutor::PoolingExecutor(const ExecutorContext::CPtr context) : context(context) {}

}   // namespace intel_cpu
}   // namespace ov