// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "proxy_mem_mgr.h"
#include "utils/debug_capabilities.h"

using namespace ov::intel_cpu;

void ProxyMemoryMngr::setMemMngr(std::shared_ptr<IMemoryMngr> pMngr) {
    OPENVINO_ASSERT(pMngr, "Attempt to set null memory manager to a ProxyMemoryMngr object");
    if (m_pMngr == pMngr) {
        return;
    }

    m_pMngr = pMngr;
    notifyUpdate();
}

void ProxyMemoryMngr::setMemMngrResize(std::shared_ptr<IMemoryMngr> pMngr) {
    OPENVINO_ASSERT(pMngr, "Attempt to set null memory manager to a ProxyMemoryMngr object");
    if (m_pMngr == pMngr) {
        return;
    }

    m_pMngr = pMngr;
    m_pMngr->resize(m_size);
    notifyUpdate();
}

void ProxyMemoryMngr::reset() {
    if (!m_pOrigMngr) {
        m_pOrigMngr = std::make_shared<MemoryMngrWithReuse>();
    }

    if (m_pMngr == m_pOrigMngr) {
        return;
    }

    m_pMngr = m_pOrigMngr;
    m_pMngr->resize(m_size);
    notifyUpdate();
}

void* ProxyMemoryMngr::getRawPtr() const noexcept {
    return m_pMngr->getRawPtr();
}

void ProxyMemoryMngr::setExtBuff(void* ptr, size_t size) {
    m_pMngr->setExtBuff(ptr, size);
    notifyUpdate();
}

bool ProxyMemoryMngr::resize(size_t size) {
    auto res = m_pMngr->resize(size);
    DEBUG_LOG(this, ", ", m_pMngr, " size ", m_size, " -> ", size, " resized? ", res, " RawPtr ", getRawPtr());
    m_size = size;
    notifyUpdate();
    return res;
}

bool ProxyMemoryMngr::hasExtBuffer() const noexcept {
    return m_pMngr->hasExtBuffer();
}

void ProxyMemoryMngr::registerMemory(Memory* memPtr) {
    if (memPtr) {
        m_setMemPtrs.insert(memPtr);
    }
}

void ProxyMemoryMngr::unregisterMemory(Memory* memPtr) {
    if (memPtr) {
        m_setMemPtrs.erase(memPtr);
    }
}

void ProxyMemoryMngr::notifyUpdate() {
    for (auto& item : m_setMemPtrs) {
        if (item) {
            item->update();
        }
    }
}