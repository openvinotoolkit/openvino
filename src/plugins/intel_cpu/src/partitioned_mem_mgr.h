// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "edge.h"

namespace ov {
namespace intel_cpu {

class PartitionedMemoryMngr : public IMemoryMngrObserver {
public:
    PartitionedMemoryMngr(EdgePtr pEdge, size_t part = 1, ptrdiff_t offset_blocks = 0)
        : m_wEdge(pEdge), m_part(part), m_offset_blocks(offset_blocks) {}

    void* getRawPtr() const noexcept override;
    void setExtBuff(void* ptr, size_t size) override;
    bool resize(size_t size) override;
    bool hasExtBuffer() const noexcept override;
    void registerMemory(Memory* memPtr) override;
    void unregisterMemory(Memory* memPtr) override;

private:
    MemoryMngrPtr sourceMemMngr() const;
    MemoryMngrPtr sourceMemMngrNoThrow() const noexcept;

private:
    EdgeWeakPtr m_wEdge;
    size_t m_part = 1; // the size of the block as a fraction of the reference memory size
    ptrdiff_t m_offset_blocks = 0; // offset from the reference memory beginning in blocks
    size_t m_size = 0; // self size in bytes
};

}   // namespace intel_cpu
}   // namespace ov