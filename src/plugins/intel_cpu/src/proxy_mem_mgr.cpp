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
    if (_validated) {
        auto res = m_pMngr->resize(m_Size);
        DEBUG_LOG(this, ", ", m_pMngr, " size ", m_Size, " -> ", m_Size, " resized? ", res, " RawPtr ", getRawPtr());
    }
}

void* ProxyMemoryMngr::getRawPtr() const noexcept {
    return m_pMngr->getRawPtr();
}

void ProxyMemoryMngr::setExtBuff(void* ptr, size_t size) {
    return m_pMngr->setExtBuff(ptr, size);
}

bool ProxyMemoryMngr::resize(size_t size) {
    auto res = m_pMngr->resize(size);
    DEBUG_LOG(this, ", ", m_pMngr, " size ", m_Size, " -> ", size, " resized? ", res, " RawPtr ", getRawPtr());
    m_Size = size;
    return res;
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