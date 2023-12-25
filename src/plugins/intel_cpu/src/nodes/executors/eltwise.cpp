// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0mvn
//

#include "eltwise.hpp"

namespace ov {
namespace intel_cpu {

EltwiseExecutor::EltwiseExecutor(const ExecutorContext::CPtr context) : context(context) {}

}   // namespace intel_cpu
}   // namespace ov