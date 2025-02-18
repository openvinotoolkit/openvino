// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_control.hpp"

#include <cstddef>
#include <memory>
#include <ov_optional.hpp>
#include <queue>
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

    void* getRawPtr() const noexcept override {
        return static_cast<uint8_t*>(m_pBlock->getRawPtr()) + m_offset;
    }
    void setExtBuff(void* ptr, size_t size) override {
        OPENVINO_THROW("Unexpected setExtBuff call to StaticPartitionMemoryBlock");
    }
    bool resize(size_t size) override {
        // don't pass over as it's static memory
        return false;
    }
    bool hasExtBuffer() const noexcept override {
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

    void* getRawPtr() const noexcept override {
        return m_pBlock->getRawPtr();
    }
    void setExtBuff(void* ptr, size_t size) override {
        m_pBlock->setExtBuff(ptr, size);
    }
    bool resize(size_t size) override {
        return m_pBlock->resize(size);
    }
    bool hasExtBuffer() const noexcept override {
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

namespace {
class GreedyMemorySolver {
public:
    using Box = ov::MemorySolver::Box;
    using MemBlock = std::pair<size_t, size_t>;  // offset, size

public:
    // This storage allows to quickly search freee blocks by the start offset and by the size
    class FreeBlockStorage {
    public:
        using Map = std::map<size_t, size_t>;
        using MultiMap = std::multimap<size_t, size_t>;

    public:
        FreeBlockStorage() = default;

        void insert_free_block(MemBlock block) {
            // important property, freeing block may not overlap existing free blocks
            // fast track, try to insert before the free block
            const size_t end_offset = block.first + block.second;
            {
                // try to merge free blocks
                auto it = m_free_blocks.lower_bound(end_offset);
                if (it != m_free_blocks.end()) {
                    // merge blocks
                    if (it->first == end_offset) {
                        block.second += it->second;
                        it = m_free_blocks.erase(it);
                    }
                }
                if (it != m_free_blocks.begin()) {
                    std::advance(it, -1); // look at the previous block
                    if (it->first + it->second == block.first) {
                        // merge blocks
                        block.first = it->first;
                        block.second += it->second;
                        m_free_blocks.erase(it);
                    }
                }
            }

            //[todo] sanity checks
            m_free_blocks.insert(std::make_pair(block.first, block.second));
        }

        MemBlock get_suitable_slot(size_t size) {  // offset, size
            // search free block to reuse
            //  todo: there might be different strategy
            //  this one - use the lowest address as the adjacent block is the next to be retired
            for (auto it = m_free_blocks.begin(); it != m_free_blocks.end(); ++it) {
                const size_t block_size = it->second;
                if (block_size >= static_cast<size_t>(size)) {
                    return {it->first, block_size}; 
                }
            }
            return {0, 0};
        }

        void remove_slot(size_t offset) {
            auto it = m_free_blocks.find(offset);
            OPENVINO_ASSERT(it != m_free_blocks.end());
            m_free_blocks.erase(it);
        }

        MemBlock get_last_free_slot() {
            if (m_free_blocks.empty()) {
                return {0, 0};
            }
            auto it = m_free_blocks.rbegin();
            return {it->first, it->second};
        }

    public:
        Map m_free_blocks;     // offset -> size
    };

public:
    explicit GreedyMemorySolver(const std::vector<Box>& boxes)
        : m_active_boxes([](const Box& lhs, const Box& rhs) {
              return lhs.finish > rhs.finish;
          }),
          m_boxes(boxes) {
        MemorySolver::normalize_boxes(m_boxes);
        m_offsets.reserve(m_boxes.size());
    }

    int64_t solve() {
        for (auto&& box : m_boxes) {
            m_offsets.insert(std::make_pair(static_cast<size_t>(box.id), insert_box(box)));
            max_current_size = std::max(max_current_size, current_size);
        }
        return m_max_size;
    }

    size_t get_offset(size_t id) const {
        auto res = m_offsets.find(id);
        OPENVINO_ASSERT(res != m_offsets.end());
        return res->second;
    }

    size_t get_optimal_size() const {
        return max_current_size;
    }

private:
    using BoxCmp = std::function<bool(const Box&, const Box&)>;
    using BoxPriorityQueue = std::priority_queue<Box, std::vector<Box>, BoxCmp>;
    using VecBoxes = std::vector<Box>;
    // using MemBlockCmp = std::function<bool(const MemBlock&, const MemBlock&)>;

private:
    size_t insert_box(Box box) {                                           // return offset
        box.size = ((box.size + (m_alignment - 1)) & ~(m_alignment - 1));  // always allocate by aligned blocks
        current_size += box.size;
        // diagnostics
        max_box_size = std::max(max_box_size, static_cast<size_t>(box.size));

        OPENVINO_ASSERT(m_last_start <= box.start);  // the boxes mast be sorted by the start index
        m_last_start = box.start;
        while (!m_active_boxes.empty() && m_active_boxes.top().finish < box.start) {
            auto&& retire_box = m_active_boxes.top();
            m_free_slots.insert_free_block(
                std::make_pair(static_cast<size_t>(retire_box.id), static_cast<size_t>(retire_box.size)));
            current_size -= retire_box.size;
            m_active_boxes.pop();
        }

        // search free block to reuse
        auto slot = m_free_slots.get_suitable_slot(box.size);
        if (slot.second != 0) {
            OPENVINO_ASSERT(slot.second >= static_cast<size_t>(box.size));
            // block found, reuse block
            box.id = static_cast<int64_t>(slot.first);
            m_free_slots.remove_slot(slot.first);
            const size_t remaining_space = slot.second - box.size;
            if (remaining_space) {
                const size_t free_block_offset = slot.first + box.size;
                m_free_slots.insert_free_block(std::make_pair(free_block_offset, remaining_space));
            }
            m_active_boxes.emplace(std::move(box));
            return slot.first;
        }

        // no suitable free slots, extend memory
        size_t ret_offset = m_max_size;
        size_t size_to_extend = box.size;
        if (auto last_slot = m_free_slots.get_last_free_slot();
            last_slot.second != 0 && last_slot.first + last_slot.second == m_max_size) {
            // the last free slot is open, so we need to extend the memory only by the residual size
            ret_offset = last_slot.first;
            size_to_extend -= last_slot.second;
            m_free_slots.remove_slot(last_slot.first);
        }

        box.id = ret_offset;
        m_max_size += size_to_extend;
        m_active_boxes.emplace(std::move(box));
        return ret_offset;
    }

private:
    BoxPriorityQueue m_active_boxes;

    VecBoxes m_boxes;
    std::unordered_map<size_t, size_t> m_offsets;  // map box id to offset
    FreeBlockStorage m_free_slots;
    size_t m_max_size = 0lu;
    int64_t m_last_start = std::numeric_limits<int64_t>::min();
    // diagnostics
    size_t max_box_size = 0;
    size_t current_size = 0;
    size_t max_current_size = 0;
    // end diagnostics

    static constexpr size_t m_alignment = 1;  // 64lu;  // cache line size
};
}  // namespace

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

        std::cout << "Blocks count: " << m_boxes.size() << std::endl;

        {
            ov::MemorySolver staticMemSolver(m_boxes);
            auto start = std::chrono::steady_clock::now();
            m_totalSize = static_cast<size_t>(staticMemSolver.solve()) * alignment;
            auto end = std::chrono::steady_clock::now();
            std::cout << "DFF solve time, us: "
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                      << std::endl;
            std::cout << "DFF total size: " << m_totalSize << std::endl;
        }
        {
            GreedyMemorySolver staticMemSolver(m_boxes);
            auto start = std::chrono::steady_clock::now();
            m_totalSize = static_cast<size_t>(staticMemSolver.solve()) * alignment;
            auto end = std::chrono::steady_clock::now();
            std::cout << "BF solve time, us: "
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;
            std::cout << "BF Total size: " << m_totalSize << std::endl;
            std::cout << "Optimal size: " << staticMemSolver.get_optimal_size() * alignment << std::endl;
        }
        exit(0);

        // m_workspace = std::make_shared<MemoryBlockWithRelease>();

        // for (const auto& box : m_boxes) {
        //     int64_t offset = staticMemSolver.get_offset(box.id);
        //     auto memoryBlock = std::make_shared<StaticPartitionMemoryBlock>(m_workspace, offset * alignment);
        //     m_blocks[box.id] = std::move(memoryBlock);
        // }
        // m_boxes.clear();
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

    const MemoryControl::MemorySolution& lastSolution() const {
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
