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
    dnnl::engine eng;

public:
    DnnlScratchPad(const dnnl::engine& eng, int numa_node = -1) : eng(eng) {
        blockPtr = std::make_shared<DnnlMemoryBlock>(make_unique<MemoryBlockWithReuse>(numa_node));
    }

    MemoryPtr createScratchPadMem(const MemoryDescPtr& md) {
        return std::make_shared<Memory>(eng, md, blockPtr);
    }
};

using DnnlScratchPadPtr = std::shared_ptr<DnnlScratchPad>;

}  // namespace intel_cpu
}  // namespace ov
