// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common/memory.hpp"
#include <memory>
#include "cpu_memory.h"
#include "dnnl_extension_utils.h"

namespace ov {
namespace intel_cpu {

class ScratchPad {
    dnnl::engine eng;
    Memory m_curScratchpadMem;
    size_t m_curScratchpadSize = 0;

public:
    ScratchPad(dnnl::engine eng) : eng(eng), m_curScratchpadMem(eng) {}

    void setScratchPad(std::unordered_map<int, dnnl::memory> & args, const dnnl::memory::desc & md) {
        // Scratch pad memory only increase for now
        // Memory re-allocation is free from multi-threading safety issue
        // as long as it's called from stream's scheduling thread
        auto requiredSize = md.get_size();
        if (m_curScratchpadSize < requiredSize) {
            m_curScratchpadMem.redefineDesc(DnnlExtensionUtils::makeDescriptor(md));
            m_curScratchpadSize = requiredSize;
        }
        args[DNNL_ARG_SCRATCHPAD] = m_curScratchpadMem.GetPrimitive();
    }
};

using ScratchPadPtr = std::shared_ptr<ScratchPad>;

}   // namespace intel_cpu
}   // namespace ov
