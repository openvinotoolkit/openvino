// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"

namespace ov {
namespace intel_cpu {

class PartitionedMemoryMngr : public IMemoryMngrObserver {
public:
    PartitionedMemoryMngr(MemoryMngrPtr pMngr, size_t total_blocks = 1, ptrdiff_t offset_blocks = 0, size_t size_blocks = 1)
        : m_pMngr(pMngr), m_total_blocks(total_blocks), m_offset_blocks(offset_blocks), m_size_blocks(size_blocks) {
        IE_ASSERT(m_pMngr) << "Memory manager is uninitialized";
    }

    void* getRawPtr() const noexcept override;
    void setExtBuff(void* ptr, size_t size) override;
    bool resize(size_t size) override;
    bool hasExtBuffer() const noexcept override;
    void registerMemory(Memory* memPtr) override;
    void unregisterMemory(Memory* memPtr) override;

private:
    MemoryMngrPtr m_pMngr;
    size_t m_total_blocks = 1; // size of the parent memory in blocks
    ptrdiff_t m_offset_blocks = 0; // offset from the base pointer in blocks
    size_t m_size_blocks = 1; // size of the partition in blocks
    size_t m_size = 0; // self size in bytes
};

}   // namespace intel_cpu
}   // namespace ov