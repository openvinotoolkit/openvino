// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert.hpp"

#include <utility>

ov::intel_cpu::ConvertExecutor::ConvertExecutor(ov::intel_cpu::ExecutorContext::CPtr context)
    : convertContext(std::move(context)) {}
