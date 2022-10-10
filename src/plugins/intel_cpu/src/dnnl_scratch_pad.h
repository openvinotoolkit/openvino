// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "common/memory.hpp"
#include "cpu_memory.h"
#include "dnnl_extension_utils.h"

namespace ov {
namespace intel_cpu {

class DnnlScratchPad {
    Memory curScratchpadMem;

public:
    DnnlScratchPad(dnnl::engine eng) : curScratchpadMem(eng) {}

    void setScratchPad(std::unordered_map<int, dnnl::memory>& args, const DnnlMemoryDescPtr& md) {
        // Memory re-allocation is free from multi-threading safety issue
        // as long as it's called from stream's scheduling thread
        curScratchpadMem.redefineDesc(md);
        args[DNNL_ARG_SCRATCHPAD] = curScratchpadMem.GetPrimitive();
    }
};

using DnnlScratchPadPtr = std::shared_ptr<DnnlScratchPad>;

}  // namespace intel_cpu
}  // namespace ov
