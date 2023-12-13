// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "partitioned_mem_mgr.h"

using namespace ov::intel_cpu;

void* PartitionedMemoryMngr::getRawPtr() const noexcept {
    return static_cast<uint8_t*>(m_pMngr->getRawPtr()) + m_offset_blocks * m_size / m_size_blocks;
}

void PartitionedMemoryMngr::setExtBuff(void* ptr, size_t size) {
    m_pMngr->setExtBuff(ptr, size);
}

bool PartitionedMemoryMngr::resize(size_t size) {
    m_size = size;
    return m_pMngr->resize(m_size * m_total_blocks / m_size_blocks);
}

bool PartitionedMemoryMngr::hasExtBuffer() const noexcept {
    return m_pMngr->hasExtBuffer();
}

void PartitionedMemoryMngr::registerMemory(Memory* memPtr) {
    m_pMngr->registerMemory(memPtr);
}

void PartitionedMemoryMngr::unregisterMemory(Memory* memPtr) {
    m_pMngr->unregisterMemory(memPtr);
}

