// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <utility>

#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_control.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

class DnnlScratchPad {
    MemoryBlockPtr blockPtr;
    MemoryBlockWithReuse* baseBlockPtr = nullptr;
    dnnl::engine eng;

public:
    explicit DnnlScratchPad(dnnl::engine eng,
                            int numa_node = -1,
                            const std::shared_ptr<IAuxiliaryMemoryControl>& memoryControl = nullptr)
        : eng(std::move(eng)) {
        auto baseMemoryBlock = std::make_unique<MemoryBlockWithReuse>(numa_node);
        baseBlockPtr = baseMemoryBlock.get();
        blockPtr = std::make_shared<DnnlMemoryBlock>(std::move(baseMemoryBlock));
        if (memoryControl) {
            memoryControl->registerMemoryBlock(blockPtr);
        }
    }

    MemoryPtr createScratchPadMem(const MemoryDescPtr& md) {
        return std::make_shared<Memory>(eng, md, blockPtr);
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
