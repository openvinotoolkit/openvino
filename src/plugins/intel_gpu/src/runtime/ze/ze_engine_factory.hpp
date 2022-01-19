// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/device.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/stream.hpp"

namespace cldnn {
namespace ze {

// Factory for ze_engine creation. It's moved outside of ze_engine class to avoid possible CL includes conflict
// between different engines in engine.cpp file
std::shared_ptr<cldnn::engine> create_ze_engine(const device::ptr device, runtime_types runtime_type,
                        const engine_configuration& configuration, const InferenceEngine::ITaskExecutor::Ptr task_executor);

}  // namespace ze
}  // namespace cldnn
