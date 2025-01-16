// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "partitioned_mem_blk.h"

using namespace ov::intel_cpu;

void* PartitionedMemoryBlock::getRawPtr() const noexcept {
    return static_cast<uint8_t*>(m_pBlock->getRawPtr()) + m_offset_chunks * m_size / m_size_chunks;
}

void PartitionedMemoryBlock::setExtBuff(void* ptr, size_t size) {
    m_pBlock->setExtBuff(ptr, size);
}

bool PartitionedMemoryBlock::resize(size_t size) {
    m_size = size;
    return m_pBlock->resize(m_size * m_total_chunks / m_size_chunks);
}

bool PartitionedMemoryBlock::hasExtBuffer() const noexcept {
    return m_pBlock->hasExtBuffer();
}

void PartitionedMemoryBlock::registerMemory(Memory* memPtr) {
    m_pBlock->registerMemory(memPtr);
}

void PartitionedMemoryBlock::unregisterMemory(Memory* memPtr) {
    m_pBlock->unregisterMemory(memPtr);
}

