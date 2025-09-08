// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// @file dnnl_utils.hpp
// Contains utility methods used by oneDNN backend executors
//

#pragma once

#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <unordered_map>

#include "cache/multi_cache.h"
#include "cpu_memory.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/executors/executor.hpp"
#include "weights_cache.hpp"

namespace ov::intel_cpu::utils {
MemoryPtr prepareWeightsMemory(const DnnlMemoryDescPtr& srcWeightDesc,
                               const DnnlMemoryDescPtr& dstWeightDesc,
                               const MemoryCPtr& weightsMem,
                               const ExecutorContext::CPtr& context,
                               bool needShiftSignedToUnsigned = false);

MemoryPtr prepareWeightsMemory(const DnnlMemoryDescPtr& srcWeightDesc,
                               const DnnlMemoryDescPtr& dstWeightDesc,
                               const MemoryCPtr& weightsMem,
                               const dnnl::engine& eng,
                               const MultiCachePtr& rtCache,
                               const WeightsSharing::Ptr& globalWeightCache,
                               const std::shared_ptr<std::unordered_map<std::string, MemoryPtr>>& privateWeightCache,
                               bool needShiftSignedToUnsigned = false);
}  // namespace ov::intel_cpu::utils
