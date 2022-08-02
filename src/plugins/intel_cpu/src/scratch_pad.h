// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common/memory.hpp"
#include <memory>

namespace ov {
namespace intel_cpu {

class ScratchPad {
    size_t m_curScratchpadSize = 0;
    std::shared_ptr<void> m_curScratchpadMem = nullptr;
    dnnl::engine eng;

public:
    ScratchPad(dnnl::engine eng) : eng(eng) {}

    void setScratchPad(std::unordered_map<int, dnnl::memory> & args, const dnnl::memory::desc & md) {
        auto sz = md.get_size();
        // Scratch pad memory only increase for now
        // Memory re-allocation is free from multi-threading safety issue
        // as long as it's called from stream's scheduling thread
        if (m_curScratchpadSize < sz) {
            ResizeScratchpad(sz);
        }
        args[DNNL_ARG_SCRATCHPAD] =
            dnnl::memory(md, eng, m_curScratchpadMem.get());
    }

private:
    void ResizeScratchpad(size_t sz) {
        m_curScratchpadMem.reset();

        void * ptr = dnnl::impl::malloc(sz, 4096);
        if (ptr == nullptr)
            IE_THROW() << "dnnl::impl::malloc failed for scratchpad size " << sz;

        m_curScratchpadMem = std::shared_ptr<void>(ptr, [sz](void * p){
                dnnl::impl::free(p);
        });
        m_curScratchpadSize = sz;
    }
};

using ScratchPadPtr = std::shared_ptr<ScratchPad>;

}   // namespace intel_cpu
}   // namespace ov
