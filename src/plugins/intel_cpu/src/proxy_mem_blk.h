// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"

namespace ov {
namespace intel_cpu {

/**
 * @brief A proxy object that additionally implements observer pattern
 */
class ProxyMemoryBlock : public IMemoryBlockObserver {
public:
    ProxyMemoryBlock() : m_pOrigBlock(std::make_shared<MemoryBlockWithReuse>()), m_pMemBlock(m_pOrigBlock) {}
    explicit ProxyMemoryBlock(const std::shared_ptr<IMemoryBlock>& pBlock) {
        OPENVINO_ASSERT(pBlock, "Memory block is uninitialized");
        m_pMemBlock = pBlock;
    }

    void* getRawPtr() const noexcept override;
    void setExtBuff(void* ptr, size_t size) override;
    bool resize(size_t size) override;
    bool hasExtBuffer() const noexcept override;

    void registerMemory(Memory* memPtr) override;
    void unregisterMemory(Memory* memPtr) override;

    void setMemBlock(std::shared_ptr<IMemoryBlock> pBlock);
    void setMemBlockResize(std::shared_ptr<IMemoryBlock> pBlock);
    void reset();

private:
    void notifyUpdate();

    // We keep the original MemBlock as may fallback to copy output.
    std::shared_ptr<IMemoryBlock> m_pOrigBlock = nullptr;
    std::shared_ptr<IMemoryBlock> m_pMemBlock = nullptr;

    std::unordered_set<Memory*> m_setMemPtrs;

    // WA: resize stage might not work because there is no shape change,
    // but the underlying actual memory block changes.
    size_t m_size = 0ul;
};

using ProxyMemoryBlockPtr = std::shared_ptr<ProxyMemoryBlock>;
using ProxyMemoryBlockCPtr = std::shared_ptr<const ProxyMemoryBlock>;

}  // namespace intel_cpu
}  // namespace ov
