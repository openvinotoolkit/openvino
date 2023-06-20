// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "proxy_mem_mgr.h"
#include "utils/debug_capabilities.h"

using namespace ov::intel_cpu;

void ProxyMemoryMngr::setManager(MemoryMngrPtr _pMngr) {
    auto _validated = (_pMngr != m_pMngr);
    if (_pMngr) {
        m_pMngr = _pMngr;
    } else {
        m_pMngr = m_pOrigMngr;
    }

    // WA: unconditionally resize to last size
    if (_validated)
        m_pMngr->resize(m_Size);
}

void* ProxyMemoryMngr::getRawPtr() const noexcept {
    return m_pMngr->getRawPtr();
}

void ProxyMemoryMngr::setExtBuff(void* ptr, size_t size) {
    return m_pMngr->setExtBuff(ptr, size);
}

bool ProxyMemoryMngr::resize(size_t size) {
    m_Size = size;
    DEBUG_LOG(m_pMngr, ", size ", size);
    return m_pMngr->resize(size);
}

bool ProxyMemoryMngr::hasExtBuffer() const noexcept {
    return m_pMngr->hasExtBuffer();
}

void ProxyMemoryMngr::registerMemory(Memory* memPtr) {
    m_pMngr->registerMemory(memPtr);
}

void ProxyMemoryMngr::unregisterMemory(Memory* memPtr) {
    m_pMngr->unregisterMemory(memPtr);
}