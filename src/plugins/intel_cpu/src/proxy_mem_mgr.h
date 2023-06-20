// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"

namespace ov {
namespace intel_cpu {

class ProxyMemoryMngr : public IMemoryMngrObserver {
public:
    explicit ProxyMemoryMngr(MemoryMngrPtr pMngr) : m_pOrigMngr(pMngr), m_pMngr(pMngr) {
        OPENVINO_ASSERT(m_pOrigMngr, "Memory manager is uninitialized");
    }

    void* getRawPtr() const noexcept override;
    void setExtBuff(void* ptr, size_t size) override;
    bool resize(size_t size) override;
    bool hasExtBuffer() const noexcept override;

    void registerMemory(Memory* memPtr) override;
    void unregisterMemory(Memory* memPtr) override;

    void setManager(MemoryMngrPtr _pMngr);

private:
    // We keep the original MemMngr as may fallback to copy output.
    const MemoryMngrPtr m_pOrigMngr;
    MemoryMngrPtr m_pMngr;

    // WA: resize stage might not work because there is no shape change,
    // but the underlying actual memory manager changes.
    mutable size_t m_Size = 0ul;
};
using ProxyMemoryMngrPtr = std::shared_ptr<ProxyMemoryMngr>;
using ProxyMemoryMngrCPtr = std::shared_ptr<const ProxyMemoryMngr>;

}   // namespace intel_cpu
}   // namespace ov