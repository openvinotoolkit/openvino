// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "proxy_mem_mgr.h"
#include "utils/debug_capabilities.h"

using namespace ov::intel_cpu;

void ProxyMemoryMngr::reset(std::shared_ptr<IMemoryMngr> _pMngr) {
    auto _validated = (_pMngr != m_pMngr);
    if (_pMngr) {
        m_pMngr = _pMngr;
    } else {
        m_pMngr = m_pOrigMngr;
    }

    // WA: unconditionally resize to last size
    if (_validated) {
        auto res = m_pMngr->resize(m_size);
        DEBUG_LOG(this, ", ", m_pMngr, " size ", m_size, " -> ", m_size, " resized? ", res, " RawPtr ", getRawPtr());

        notifyUpdate();
    }
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