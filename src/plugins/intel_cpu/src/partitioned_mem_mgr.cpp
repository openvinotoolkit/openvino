// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "partitioned_mem_mgr.h"

using namespace ov::intel_cpu;

MemoryMngrPtr PartitionedMemoryMngr::sourceMemMngrNoThrow() const noexcept {
    if (auto pEdge = m_wEdge.lock()) {
        MemoryPtr pMem = nullptr;
        try {
            pMem = pEdge->getMemoryPtr();
        }
        catch(...) {
            return nullptr;
        }
        if (pMem) {
            if (auto memMngr = pMem->getMemoryMngr()) {
                return memMngr;
            }
        }
    }
    return nullptr;
}

MemoryMngrPtr PartitionedMemoryMngr::sourceMemMngr() const {
    auto memMngr = sourceMemMngrNoThrow();
    IE_ASSERT(memMngr != nullptr) << "PartitionedMemoryMngr references nullptr";
    return memMngr;
}

void* PartitionedMemoryMngr::getRawPtr() const noexcept {
    if (auto memMngr = sourceMemMngrNoThrow()) {
        return static_cast<uint8_t*>(memMngr->getRawPtr()) + m_offset_blocks * m_size;
    }
    return nullptr;
}

void PartitionedMemoryMngr::setExtBuff(void* ptr, size_t size) {
    auto memMngr = sourceMemMngr();
    memMngr->setExtBuff(ptr, size);
}

bool PartitionedMemoryMngr::resize(size_t size) {
    auto memMngr = sourceMemMngr();
    m_size = size;
    return memMngr->resize(size * m_part);
}

bool PartitionedMemoryMngr::hasExtBuffer() const noexcept {
    if (auto memMngr = sourceMemMngrNoThrow()) {
        return memMngr->hasExtBuffer();
    }
    return false;
}

void PartitionedMemoryMngr::registerMemory(Memory* memPtr) {
    auto memMngr = sourceMemMngr();
    memMngr->registerMemory(memPtr);
}

void PartitionedMemoryMngr::unregisterMemory(Memory* memPtr) {
    auto memMngr = sourceMemMngr();
    memMngr->unregisterMemory(memPtr);
}

