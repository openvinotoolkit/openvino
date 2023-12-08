// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"

namespace ov {
namespace intel_cpu {

/**
 * @brief A proxy object that additionally implements observer pattern
 */
class ProxyMemoryMngr : public IMemoryMngrObserver {
public:
    ProxyMemoryMngr() : m_pOrigMngr(std::make_shared<MemoryMngrWithReuse>()), m_pMngr(m_pOrigMngr) {}
    explicit ProxyMemoryMngr(std::shared_ptr<IMemoryMngr> pMngr) {
        OPENVINO_ASSERT(pMngr, "Memory manager is uninitialized");
        m_pMngr = pMngr;
    }

    void* getRawPtr() const noexcept override;
    void setExtBuff(void* ptr, size_t size) override;
    bool resize(size_t size) override;
    bool hasExtBuffer() const noexcept override;

    void registerMemory(Memory* memPtr) override;
    void unregisterMemory(Memory* memPtr) override;

    void setMemMngr(std::shared_ptr<IMemoryMngr> pMngr);
    void setMemMngrResize(std::shared_ptr<IMemoryMngr> pMngr);
    void reset();

private:
    void notifyUpdate();

    // We keep the original MemMngr as may fallback to copy output.
    std::shared_ptr<IMemoryMngr> m_pOrigMngr = nullptr;
    std::shared_ptr<IMemoryMngr> m_pMngr = nullptr;

    std::unordered_set<Memory*> m_setMemPtrs;

    // WA: resize stage might not work because there is no shape change,
    // but the underlying actual memory manager changes.
    size_t m_size = 0ul;
};

using ProxyMemoryMngrPtr = std::shared_ptr<ProxyMemoryMngr>;
using ProxyMemoryMngrCPtr = std::shared_ptr<const ProxyMemoryMngr>;

}   // namespace intel_cpu
}   // namespace ov