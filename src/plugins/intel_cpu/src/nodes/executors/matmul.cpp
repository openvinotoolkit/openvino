// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matmul.hpp"

namespace ov {
namespace intel_cpu {

using namespace InferenceEngine;

MatMulExecutor::MatMulExecutor(const ExecutorContext::CPtr context) : context(context) {}

}   // namespace intel_cpu
}   // namespace ov
