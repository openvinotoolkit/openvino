// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "proxy_mem_blk.h"

#include "utils/debug_capabilities.h"

using namespace ov::intel_cpu;

void ProxyMemoryBlock::setMemBlock(std::shared_ptr<IMemoryBlock> pBlock) {
    OPENVINO_ASSERT(pBlock, "Attempt to set null memory block to a ProxyMemoryBlock object");
    if (m_pMemBlock == pBlock) {
        return;
    }

    m_pMemBlock = std::move(pBlock);
    notifyUpdate();
}

void ProxyMemoryBlock::setMemBlockResize(std::shared_ptr<IMemoryBlock> pBlock) {
    OPENVINO_ASSERT(pBlock, "Attempt to set null memory block to a ProxyMemoryBlock object");
    if (m_pMemBlock == pBlock) {
        return;
    }

    m_pMemBlock = std::move(pBlock);
    m_pMemBlock->resize(m_size);
    notifyUpdate();
}

void ProxyMemoryBlock::reset() {
    if (!m_pOrigBlock) {
        m_pOrigBlock = std::make_shared<MemoryBlockWithReuse>();
    }

    if (m_pMemBlock == m_pOrigBlock) {
        return;
    }

    m_pMemBlock = m_pOrigBlock;
    m_pMemBlock->resize(m_size);
    notifyUpdate();
}

void* ProxyMemoryBlock::getRawPtr() const noexcept {
    return m_pMemBlock->getRawPtr();
}

void ProxyMemoryBlock::setExtBuff(void* ptr, size_t size) {
    m_pMemBlock->setExtBuff(ptr, size);
    notifyUpdate();
}

bool ProxyMemoryBlock::resize(size_t size) {
    auto res = m_pMemBlock->resize(size);
    DEBUG_LOG(this, ", ", m_pMemBlock, " size ", m_size, " -> ", size, " resized? ", res, " RawPtr ", getRawPtr());
    m_size = size;
    notifyUpdate();
    return res;
}

bool ProxyMemoryBlock::hasExtBuffer() const noexcept {
    return m_pMemBlock->hasExtBuffer();
}

void ProxyMemoryBlock::registerMemory(Memory* memPtr) {
    if (memPtr) {
        m_setMemPtrs.insert(memPtr);
    }
}

void ProxyMemoryBlock::unregisterMemory(Memory* memPtr) {
    if (memPtr) {
        m_setMemPtrs.erase(memPtr);
    }
}

void ProxyMemoryBlock::notifyUpdate() {
    for (auto& item : m_setMemPtrs) {
        if (item) {
            item->update();
        }
    }
}
