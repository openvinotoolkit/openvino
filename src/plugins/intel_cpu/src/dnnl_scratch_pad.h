// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <utility>

#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc.h"
#include "partitioned_mem_blk.h"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

class DnnlScratchPad {
    MemoryBlockPtr blockPtr;
    MemoryBlockWithReuse* baseBlockPtr = nullptr;
    dnnl::engine eng;
    size_t baseOffset = 0;

public:
    explicit DnnlScratchPad(dnnl::engine eng, int numa_node = -1) : eng(std::move(eng)) {
        auto baseMemoryBlock = std::make_unique<MemoryBlockWithReuse>(numa_node);
        baseBlockPtr = baseMemoryBlock.get();
        blockPtr = std::make_shared<DnnlMemoryBlock>(std::move(baseMemoryBlock));
    }

    // A scratchpad that shares the block of `parent` and starts `offset` bytes into it.
    DnnlScratchPad(const DnnlScratchPad& parent, size_t offset)
        : blockPtr(parent.blockPtr),
          eng(parent.eng),
          baseOffset(parent.baseOffset + offset) {}

    MemoryPtr createScratchPadMem(const MemoryDescPtr& md) {
        return createScratchPadMem(md, 0);
    }

    // The memory is placed `offset` bytes after the start of this scratchpad. It grows the shared block to
    // offset + size and follows its reallocations.
    MemoryPtr createScratchPadMem(const MemoryDescPtr& md, size_t offset) {
        const size_t memOffset = baseOffset + offset;
        if (memOffset == 0) {
            return std::make_shared<Memory>(eng, md, blockPtr);
        }
        // Partitions are counted in bytes: with the memory's size as the partition size, the offsets do not depend on
        // it, so each block serves one memory of one size.
        const size_t size = std::max<size_t>(md->getCurrentMemSize(), 1);
        return std::make_shared<Memory>(
            eng,
            md,
            std::make_shared<PartitionedMemoryBlock>(blockPtr, memOffset + size, memOffset, size));
    }

    [[nodiscard]] size_t size() const {
        if (baseBlockPtr) {
            return baseBlockPtr->size();
        }
        return 0;
    }
};

using DnnlScratchPadPtr = std::shared_ptr<DnnlScratchPad>;

}  // namespace ov::intel_cpu
