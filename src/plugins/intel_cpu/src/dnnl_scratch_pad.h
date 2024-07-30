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
    MemoryMngrPtr mgrPtr;
    dnnl::engine eng;

public:
    DnnlScratchPad(const dnnl::engine& eng) : eng(eng) {
        mgrPtr = std::make_shared<DnnlMemoryMngr>(make_unique<MemoryMngrWithReuse>());
    }

    MemoryPtr createScratchPadMem(const MemoryDescPtr& md) {
        return std::make_shared<Memory>(eng, md, mgrPtr);
    }
};

using DnnlScratchPadPtr = std::shared_ptr<DnnlScratchPad>;

}  // namespace intel_cpu
}  // namespace ov
