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

namespace ov::intel_cpu::utils {
MemoryPtr prepareWeightsMemory(DnnlMemoryDescPtr srcWeightDesc,
                               DnnlMemoryDescPtr dstWeightDesc,
                               MemoryCPtr weightsMem,
                               ExecutorContext::CPtr context,
                               bool needShiftSignedToUnsigned = false);
}  // namespace ov::intel_cpu::utils
