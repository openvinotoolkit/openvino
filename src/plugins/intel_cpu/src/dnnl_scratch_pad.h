// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc.h"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

// A scratchpad is one memory block plus child scratchpads, created on first use. The children form a tree; each child
// is an ordinary DnnlScratchPad. The tree belongs to the graph context (one per stream): the next node reuses the same
// pads, and each block grows to the largest request.
//
// Contract:
//  1. One buffer per block. A pad's block holds one buffer at a time. Several views of that buffer are fine, e.g. the
//     full and partial tiles or the src/dst views of a nested executor.
//  2. Ownership. The pad in an executor's context belongs to that executor, together with all its children, while its
//     node executes.
//  3. Composite executors (executors that create nested executors) may use their own block for one buffer and keep
//     the other buffers in children. They give each nested executor a context whose pad is a child they do not use
//     themselves, may give one child to several nested executors that run one after another, and assert that each
//     nested context got the expected pad.
//  4. Pointers. A later request on a pad may reallocate its block. Memory objects follow it (Memory::update calls
//     set_data_handle on the shared handle); raw pointers must be read again in execute.
//  5. Threads. child() is not thread-safe: call it only during node initialization or prepareParams, which run on one
//     thread per graph.
class DnnlScratchPad {
    MemoryBlockPtr blockPtr;
    MemoryBlockWithReuse* baseBlockPtr = nullptr;
    dnnl::engine eng;
    int numaNode;
    std::vector<std::shared_ptr<DnnlScratchPad>> children;

public:
    explicit DnnlScratchPad(dnnl::engine eng, int numa_node = -1) : eng(std::move(eng)), numaNode(numa_node) {
        auto baseMemoryBlock = std::make_unique<MemoryBlockWithReuse>(numa_node);
        baseBlockPtr = baseMemoryBlock.get();
        blockPtr = std::make_shared<DnnlMemoryBlock>(std::move(baseMemoryBlock));
    }

    MemoryPtr createScratchPadMem(const MemoryDescPtr& md) {
        return std::make_shared<Memory>(eng, md, blockPtr);
    }

    // A pad whose memory never overlaps this pad's block or its other children. The same index always returns the same
    // child, so executors created again reuse it.
    std::shared_ptr<DnnlScratchPad> child(size_t idx) {
        if (idx >= children.size()) {
            children.resize(idx + 1);
        }
        if (!children[idx]) {
            children[idx] = std::make_shared<DnnlScratchPad>(eng, numaNode);
        }
        return children[idx];
    }

    // Bytes held by this pad and all its children.
    [[nodiscard]] size_t size() const {
        size_t total = baseBlockPtr ? baseBlockPtr->size() : 0;
        for (const auto& c : children) {
            total += c ? c->size() : 0;
        }
        return total;
    }
};

using DnnlScratchPadPtr = std::shared_ptr<DnnlScratchPad>;

}  // namespace ov::intel_cpu
