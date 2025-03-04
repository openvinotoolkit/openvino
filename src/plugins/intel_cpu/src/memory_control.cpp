// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_control.hpp"

#include <cstddef>
#include <memory>
#include <utility>

#include "openvino/runtime/memory_solver.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

namespace {

class StaticPartitionMemoryBlock : public IMemoryBlockObserver {
public:
    StaticPartitionMemoryBlock(MemoryBlockPtr pBlock, ptrdiff_t offset)
        : m_pBlock(std::move(pBlock)),
          m_offset(offset) {
        OPENVINO_ASSERT(m_pBlock, "Memory block is uninitialized");
    }

    [[nodiscard]] void* getRawPtr() const noexcept override {
        return static_cast<uint8_t*>(m_pBlock->getRawPtr()) + m_offset;
    }
    void setExtBuff(void* ptr, size_t size) override {
        OPENVINO_THROW("Unexpected setExtBuff call to StaticPartitionMemoryBlock");
    }
    bool resize(size_t size) override {
        // don't pass over as it's static memory
        return false;
    }
    [[nodiscard]] bool hasExtBuffer() const noexcept override {
        return m_pBlock->hasExtBuffer();
    }
    void registerMemory(Memory* memPtr) override {
        m_pBlock->registerMemory(memPtr);
    }
    void unregisterMemory(Memory* memPtr) override {
        m_pBlock->unregisterMemory(memPtr);
    }

private:
    MemoryBlockPtr m_pBlock;
    ptrdiff_t m_offset = 0;
};

class MemoryBlockWithRelease : public IMemoryBlockObserver {
public:
    MemoryBlockWithRelease() {
        auto pInternalMem = make_unique<MemoryBlockWithReuse>();
        m_pInternalMem = pInternalMem.get();
        m_pBlock = std::make_shared<DnnlMemoryBlock>(std::move(pInternalMem));
    }

    [[nodiscard]] void* getRawPtr() const noexcept override {
        return m_pBlock->getRawPtr();
    }
    void setExtBuff(void* ptr, size_t size) override {
        m_pBlock->setExtBuff(ptr, size);
    }
    bool resize(size_t size) override {
        return m_pBlock->resize(size);
    }
    [[nodiscard]] bool hasExtBuffer() const noexcept override {
        return m_pBlock->hasExtBuffer();
    }
    void registerMemory(Memory* memPtr) override {
        m_pBlock->registerMemory(memPtr);
    }
    void unregisterMemory(Memory* memPtr) override {
        m_pBlock->unregisterMemory(memPtr);
    }
    void free() {
        m_pInternalMem->free();
    }

private:
    MemoryBlockPtr m_pBlock;
    MemoryBlockWithReuse* m_pInternalMem;
};

class IMemoryManager {
public:
    virtual ~IMemoryManager() = default;
    virtual void insert(const MemoryRegion& reg, const std::vector<size_t>& syncInds) = 0;
    virtual const MemoryControl::MemorySolution& lastSolution() = 0;
    virtual void allocate() = 0;
    virtual void release() = 0;
};

using MemoryManagerPtr = std::shared_ptr<IMemoryManager>;

template <typename T, typename... Args>
std::shared_ptr<DnnlMemoryBlock> makeDnnlMemoryBlock(Args&&... args) {
    return std::make_shared<DnnlMemoryBlock>(make_unique<T>(std::forward<Args>(args)...));
}

class MemoryManagerIO : public IMemoryManager {
public:
    void insert(const MemoryRegion& reg, const std::vector<size_t>& syncInds) override {
        (void)syncInds;
        m_blocks.insert({reg.id, makeDnnlMemoryBlock<MemoryBlockWithReuse>()});
    }

    const MemoryControl::MemorySolution& lastSolution() override {
        return m_blocks;
    }

    void allocate() override {
        // nothing to do
    }
    void release() override {
        // nothing to do
    }

private:
    MemoryControl::MemorySolution m_blocks;
};

class MemoryManagerStatic : public IMemoryManager {
public:
    void insert(const MemoryRegion& reg, const std::vector<size_t>& syncInds) override {
        (void)syncInds;
        m_boxes.emplace_back(MemorySolver::Box{reg.start, reg.finish, reg.size, reg.id});
    }

    const MemoryControl::MemorySolution& lastSolution() override {
        if (!m_boxes.empty() && m_blocks.empty()) {
            solve();
        }
        return m_blocks;
    }

private:
    void solve() {
        constexpr size_t alignment = 32;
        std::for_each(m_boxes.begin(), m_boxes.end(), [=](MemorySolver::Box& box) {
            box.size = div_up(box.size, alignment);
        });

        ov::MemorySolver staticMemSolver(m_boxes);
        m_totalSize = static_cast<size_t>(staticMemSolver.solve()) * alignment;

        m_workspace = std::make_shared<MemoryBlockWithRelease>();

        for (const auto& box : m_boxes) {
            int64_t offset = staticMemSolver.get_offset(box.id);
            auto memoryBlock = std::make_shared<StaticPartitionMemoryBlock>(m_workspace, offset * alignment);
            m_blocks[box.id] = std::move(memoryBlock);
        }
        m_boxes.clear();
    }

    void allocate() override {
        if (m_workspace) {
            m_workspace->resize(m_totalSize);
        }
    }
    void release() override {
        if (m_workspace) {
            m_workspace->free();
        }
    }

private:
    MemoryControl::MemorySolution m_blocks;
    std::vector<MemorySolver::Box> m_boxes;
    std::shared_ptr<MemoryBlockWithRelease> m_workspace;
    size_t m_totalSize = 0;
};

class MemoryManageNonOverlapingSets : public IMemoryManager {
public:
    void insert(const MemoryRegion& reg, const std::vector<size_t>& syncInds) override {
        MemorySolver::Box box = {reg.start, reg.finish, reg.size, reg.id};
        if (-1 != reg.finish) {
            // We have to extend the lifespan of tensors that are crossing a sync point border in order to save
            // the intermediate computation results from possible loss due to the tensor resize
            auto itr_upper = std::upper_bound(syncInds.begin(), syncInds.end(), box.finish, [](int y, int x) {
                return y <= x;
            });
            auto itr_lower = std::lower_bound(syncInds.begin(), syncInds.end(), box.start);
            if (itr_lower != itr_upper) {  // across sections
                if (itr_upper == syncInds.end()) {
                    box.finish = -1;
                } else {
                    box.finish = *itr_upper;
                }
            }
        }
        m_boxes.emplace_back(box);
    }

    const MemoryControl::MemorySolution& lastSolution() override {
        if (!m_boxes.empty() && m_blocks.empty()) {
            solve();
            m_blocks = MemoryControl::MemorySolution{m_internalBlocks.begin(), m_internalBlocks.end()};
        }
        return m_blocks;
    }

private:
    void solve() {
        ov::MemorySolver::normalize_boxes(m_boxes);

        std::vector<std::vector<ov::MemorySolver::Box>> groups;  // groups of nonoverlapping boxes
        groups.push_back({m_boxes.front()});
        for (size_t i = 1; i < m_boxes.size(); ++i) {
            const auto& box = m_boxes[i];
            bool groupFound = false;
            for (auto& group : groups) {
                const auto& lastBox = group.back();
                if (lastBox.start > box.finish || lastBox.finish < box.start) {
                    group.push_back(box);
                    groupFound = true;
                    break;
                }
            }

            if (!groupFound) {
                groups.push_back({box});
            }
        }
        for (auto& group : groups) {
            auto grpMemBlock = std::make_shared<MemoryBlockWithRelease>();
            for (auto& box : group) {
                m_internalBlocks[box.id] = grpMemBlock;
            }
        }
        m_boxes.clear();
    }

    void allocate() override {
        // nothing to do
    }
    void release() override {
        for (auto&& item : m_internalBlocks) {
            item.second->free();
        }
    }

private:
    MemoryControl::MemorySolution m_blocks;
    std::unordered_map<MemoryControl::MemorySolution::key_type, std::shared_ptr<MemoryBlockWithRelease>>
        m_internalBlocks;
    std::vector<MemorySolver::Box> m_boxes;
};

}  // namespace

class MemoryControl::RegionHandler {
public:
    using Condition = std::function<bool(const MemoryRegion&)>;

public:
    RegionHandler(Condition cond, MemoryManagerPtr memManager)
        : m_cond(std::move(cond)),
          m_memManager(std::move(memManager)) {}

    bool insert(const MemoryRegion& reg, const std::vector<size_t>& syncInds) {
        if (!m_cond(reg)) {
            return false;
        }

        m_memManager->insert(reg, syncInds);
        return true;
    }

    [[nodiscard]] const MemoryControl::MemorySolution& lastSolution() const {
        return m_memManager->lastSolution();
    }

    void allocate() {
        m_memManager->allocate();
    }

    void release() {
        m_memManager->release();
    }

private:
    Condition m_cond;
    MemoryManagerPtr m_memManager;
};

namespace {

template <typename T, typename F, typename... Args>
MemoryControl::RegionHandlerPtr buildHandler(F&& f, Args&&... args) {
    return std::make_shared<MemoryControl::RegionHandler>(std::forward<F>(f),
                                                          std::make_shared<T>(std::forward<Args>(args)...));
}

}  // namespace

MemoryControl::MemoryControl() {
    // init handlers
    m_handlers.emplace_back(buildHandler<MemoryManagerStatic>([](const MemoryRegion& reg) {
        if (reg.size < 0 || MemoryRegion::RegionType::VARIABLE != reg.type ||
            MemoryRegion::AllocType::POD != reg.alloc_type) {
            return false;
        }
        return true;
    }));

    // handler for static tensors
    m_handlers.emplace_back(buildHandler<MemoryManageNonOverlapingSets>([](const MemoryRegion& reg) {
        if (reg.size >= 0 || MemoryRegion::RegionType::VARIABLE != reg.type ||
            MemoryRegion::AllocType::POD != reg.alloc_type) {
            return false;
        }
        return true;
    }));

    // handler for I/O tensors, so far simply individual blocks
    m_handlers.emplace_back(buildHandler<MemoryManagerIO>([](const MemoryRegion& reg) {
        if (MemoryRegion::RegionType::VARIABLE == reg.type || reg.alloc_type != MemoryRegion::AllocType::POD) {
            return false;
        }
        return true;
    }));
}

void MemoryControl::insert(const MemoryRegion& region, const std::vector<size_t>& syncInds) {
    for (auto&& handler : m_handlers) {
        if (handler->insert(region, syncInds)) {
            return;
        }
    }
    OPENVINO_THROW("No suitable hanlder was found for the given memory region");
}

void MemoryControl::insert(const std::vector<MemoryRegion>& regions, const std::vector<size_t>& syncInds) {
    for (auto&& region : regions) {
        insert(region, syncInds);
    }
}

MemoryControl::MemorySolution MemoryControl::solve() {
    MemoryControl::MemorySolution blocksMap;

    for (auto&& handler : m_handlers) {
        auto&& solution = handler->lastSolution();
        for (auto&& item : solution) {
            auto res = blocksMap.insert(item);
            OPENVINO_ASSERT(res.second, "Memory solutions has non unique entries");
        }
    }

    return blocksMap;
}

void MemoryControl::allocateMemory() {
    for (auto&& handler : m_handlers) {
        handler->allocate();
    }
    m_allocated = true;
}

void MemoryControl::releaseMemory() {
    for (auto&& handler : m_handlers) {
        handler->release();
    }
    m_allocated = false;
}

MemoryControl::Ptr NetworkMemoryControl::createMemoryControlUnit() {
    m_controlUnits.emplace_back(std::shared_ptr<MemoryControl>(new MemoryControl()));
    return m_controlUnits.back();
}

void NetworkMemoryControl::allocateMemory() {
    for (auto&& item : m_controlUnits) {
        item->allocateMemory();
    }
}

void NetworkMemoryControl::releaseMemory() {
    for (auto&& item : m_controlUnits) {
        item->releaseMemory();
    }
}

}  // namespace ov::intel_cpu
