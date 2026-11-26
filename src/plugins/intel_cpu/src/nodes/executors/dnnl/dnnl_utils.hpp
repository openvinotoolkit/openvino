// Copyright (C) 2018-2026 Intel Corporation
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
#include "utils/general_utils.h"
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
                               const std::shared_ptr<ThreadPool>& threadPool,
                               bool needShiftSignedToUnsigned = false);

bool isNonNegativeI8(const IMemory& mem);

/// Rounds a row count up to a bucket, bounding the number of distinct GEMM shapes (cached primitives, JIT builds).
inline size_t normalizeRowsBucket(size_t m) {
    if (m < 512) {
        return rnd_up(m, 16);
    }
    if (m < 1024) {
        return rnd_up(m, 32);
    }
    return rnd_up(m, 256);
}
}  // namespace ov::intel_cpu::utils
