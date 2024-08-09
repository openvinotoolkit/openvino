// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "cpu_memory.h"
#include "graph_context.h"

namespace ov {
namespace intel_cpu {

enum InputPrepType {
    FTZ,
    PutToNumaLocalCache,
    SimpleClone,
    None
};

MemoryPtr cloneBlob(const IMemory& blob, const dnnl::engine& engine, bool needFlushDenormalsToZero);
InputPrepType requiresPreProcessing(const IMemory& blob,
                                    GraphContext::CPtr context,
                                    const dnnl::engine& engine);

}  // namespace intel_cpu
}   // namespace ov
