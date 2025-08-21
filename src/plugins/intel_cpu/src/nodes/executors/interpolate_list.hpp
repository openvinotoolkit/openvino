// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "executor.hpp"
#include "executor_factory.hpp"
#include "implementations.hpp"
#include "interpolate.hpp"
#include "interpolate_config.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "openvino/core/except.hpp"

namespace ov::intel_cpu {

using namespace executor;

// Factory for creating interpolate executors
using InterpolateExecutorFactory = ExecutorFactory<InterpolateAttrs>;
using InterpolateExecutorFactoryPtr = std::shared_ptr<InterpolateExecutorFactory>;
using InterpolateExecutorFactoryCPtr = std::shared_ptr<const InterpolateExecutorFactory>;

}  // namespace ov::intel_cpu