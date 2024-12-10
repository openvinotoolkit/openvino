// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "cpu_memory.h"
#include "utils/general_utils.h"

namespace ov {
namespace intel_cpu {

class DnnlScratchPad {
    MemoryBlockPtr blockPtr;
    MemoryBlockWithReuse* baseBlockPtr = nullptr;
    dnnl::engine eng;

public:
    DnnlScratchPad(const dnnl::engine& eng, int numa_node = -1) : eng(eng) {
        auto baseMemoryBlock = make_unique<MemoryBlockWithReuse>(numa_node);
        baseBlockPtr = baseMemoryBlock.get();
        blockPtr = std::make_shared<DnnlMemoryBlock>(std::move(baseMemoryBlock));
    }

    MemoryPtr createScratchPadMem(const MemoryDescPtr& md) {
        return std::make_shared<Memory>(eng, md, blockPtr);
    }

    size_t size() const {
        if (baseBlockPtr)
            return baseBlockPtr->size();
        return 0;
    }
};

using DnnlScratchPadPtr = std::shared_ptr<DnnlScratchPad>;

}  // namespace intel_cpu
}  // namespace ov
