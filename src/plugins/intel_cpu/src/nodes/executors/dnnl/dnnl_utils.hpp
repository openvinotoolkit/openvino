// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// @file dnnl_utils.hpp
// Contains utility methods used by oneDNN backend executors
//

#pragma once

#include "cpu_memory.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/executors/executor.hpp"
#include "utils/clone_original_blob.h"

namespace ov {
namespace intel_cpu {
namespace utils {
MemoryPtr prepareWeightsMemory(const DnnlMemoryDescPtr srcWeightDesc,
                               const DnnlMemoryDescPtr dstWeightDesc,
                               const MemoryCPtr weightsMem,
                               const ExecutorContext::CPtr context,
                               const bool needShiftSignedToUnsigned = false,
                               const InputPrepType preprocessing = InputPrepType::None);
}  // namespace utils
}  // namespace intel_cpu
}  // namespace ov
