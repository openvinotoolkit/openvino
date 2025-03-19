// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_control.hpp"

#include <cstddef>
#include <memory>
#include <queue>
#include <utility>

#include "openvino/runtime/memory_solver.hpp"
#include "utils/debug_capabilities.h"
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

    [[nodiscard]] size_t size() const {
        return m_pInternalMem->size();
    }

private:
    MemoryBlockPtr m_pBlock;
    MemoryBlockWithReuse* m_pInternalMem;
};

#ifdef CPU_DEBUG_CAPS
class IndividualMemoryBlockWithRelease : public IMemoryBlockObserver {
public:
    IndividualMemoryBlockWithRelease(std::shared_ptr<MemoryBlockWithRelease> pBlock) : m_pBlock(std::move(pBlock)) {}

    [[nodiscard]] void* getRawPtr() const noexcept override {
        return m_pBlock->getRawPtr();
    }
    void setExtBuff(void* ptr, size_t size) override {
        m_max_requested_size = std::max(m_max_requested_size, size);
        m_pBlock->setExtBuff(ptr, size);
    }
    bool resize(size_t size) override {
        m_max_requested_size = std::max(m_max_requested_size, size);
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
        m_max_requested_size = 0;
        if (m_pBlock->size() > 0) {
            m_pBlock->free();
        }
    }

    [[nodiscard]] size_t size() const {
        return m_max_requested_size;
    }

    [[nodiscard]] std::shared_ptr<const MemoryBlockWithRelease> getParentBlock() const {
        return m_pBlock;
    }

private:
    std::shared_ptr<MemoryBlockWithRelease> m_pBlock;
    size_t m_max_requested_size = 0;
};
#endif  // CPU_DEBUG_CAPS

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

template <typename T>
std::shared_ptr<DnnlMemoryBlock> makeDnnlMemoryBlock(std::unique_ptr<T> ptr) {
    return std::make_shared<DnnlMemoryBlock>(std::move(ptr));
}

class MemoryManagerIO : public IMemoryManager {
public:
    using BlockType = MemoryBlockWithReuse;

public:
    void insert(const MemoryRegion& reg, const std::vector<size_t>& syncInds) override {
        (void)syncInds;
        auto block = make_unique<BlockType>();
        CPU_DEBUG_CAP_ENABLE(m_blocks.emplace_back(*block);)
        m_solution.insert({reg.id, makeDnnlMemoryBlock(std::move(block))});
    }

    const MemoryControl::MemorySolution& lastSolution() override {
        return m_solution;
    }

    void allocate() override {
        // nothing to do
    }
    void release() override {
        // nothing to do
    }

private:
    static const char* getClassName() {
        return "MemoryManagerIO";
    }

private:
    MemoryControl::MemorySolution m_solution;
    CPU_DEBUG_CAP_ENABLE(std::vector<std::reference_wrapper<BlockType>> m_blocks;)
    CPU_DEBUG_CAP_ENABLE(friend MemoryStatisticsRecord dumpStatisticsImpl(const MemoryManagerIO& obj);)
};

class MemoryManagerStatic : public IMemoryManager {
public:
    void insert(const MemoryRegion& reg, const std::vector<size_t>& syncInds) override {
        (void)syncInds;
        OPENVINO_ASSERT(reg.size >= 0, getClassName(), ": got undefined block size");
        m_boxes.emplace_back(MemorySolver::Box{reg.start, reg.finish, reg.size, reg.id});
        reset_flag = true;
    }

    const MemoryControl::MemorySolution& lastSolution() override {
        if (reset_flag && !m_boxes.empty()) {
            solve();
            reset_flag = false;
        }
        return m_blocks;
    }

private:
    void solve() {
        auto boxes_to_process = m_boxes;
        constexpr size_t alignment = 32;
        std::for_each(boxes_to_process.begin(), boxes_to_process.end(), [=](MemorySolver::Box& box) {
            box.size = div_up(box.size, alignment);
        });

        ov::MemorySolver staticMemSolver(boxes_to_process);
        m_totalSize = static_cast<size_t>(staticMemSolver.solve()) * alignment;

        m_workspace = std::make_shared<MemoryBlockWithRelease>();

        for (const auto& box : boxes_to_process) {
            int64_t offset = staticMemSolver.get_offset(box.id);
            auto memoryBlock = std::make_shared<StaticPartitionMemoryBlock>(m_workspace, offset * alignment);
            m_blocks[box.id] = std::move(memoryBlock);
        }
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

    static const char* getClassName() {
        return "MemoryManagerStatic";
    }

private:
    MemoryControl::MemorySolution m_blocks;
    std::vector<MemorySolver::Box> m_boxes;
    std::shared_ptr<MemoryBlockWithRelease> m_workspace;
    size_t m_totalSize = 0;
    bool reset_flag = true;
    CPU_DEBUG_CAP_ENABLE(friend MemoryStatisticsRecord dumpStatisticsImpl(const MemoryManagerStatic& obj);)
};

class MemoryManagerNonOverlappingSets : public IMemoryManager {
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
        reset_flag = true;
    }

    const MemoryControl::MemorySolution& lastSolution() override {
        if (reset_flag && !m_boxes.empty()) {
            solve();
            m_blocks = MemoryControl::MemorySolution{m_internalBlocks.begin(), m_internalBlocks.end()};
            reset_flag = false;
        }
        return m_blocks;
    }

private:
#ifdef CPU_DEBUG_CAPS
    using InternalBlock = IndividualMemoryBlockWithRelease;
    std::shared_ptr<InternalBlock> internalBlock(const std::shared_ptr<MemoryBlockWithRelease>& block) {
        return std::make_shared<InternalBlock>(block);
    }
#else
    using InternalBlock = MemoryBlockWithRelease;
    std::shared_ptr<InternalBlock> internalBlock(const std::shared_ptr<MemoryBlockWithRelease>& block) {
        return block;
    }
#endif  // CPU_DEBUG_CAPS

private:
    void solve() {
        ov::MemorySolver::normalize_boxes(m_boxes);

        std::vector<std::vector<ov::MemorySolver::Box>> groups;  // groups of non overlapping boxes
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
            auto unique_block = std::make_shared<MemoryBlockWithRelease>();
            for (auto& box : group) {
                m_internalBlocks.insert({box.id, internalBlock(unique_block)});
            }
        }
    }

    void allocate() override {
        // nothing to do
    }
    void release() override {
        for (auto&& item : m_internalBlocks) {
            item.second->free();
        }
    }

    static const char* getClassName() {
        return "MemoryManagerNonOverlappingSets";
    }

private:
    MemoryControl::MemorySolution m_blocks;
    std::vector<MemorySolver::Box> m_boxes;
    std::unordered_map<MemoryControl::MemorySolution::key_type, std::shared_ptr<InternalBlock>> m_internalBlocks;
    bool reset_flag = true;
    CPU_DEBUG_CAP_ENABLE(friend MemoryStatisticsRecord dumpStatisticsImpl(const MemoryManagerNonOverlappingSets& obj);)
};

#ifdef CPU_DEBUG_CAPS
std::pair<int64_t, int64_t> calculateOptimalMemorySize(std::vector<MemorySolver::Box> boxes) {
    ov::MemorySolver::normalize_boxes(boxes);

    auto boxCmp = [](const MemorySolver::Box& l, const MemorySolver::Box& r) {
        return l.finish > r.finish;
    };
    std::priority_queue<MemorySolver::Box, std::vector<MemorySolver::Box>, decltype(boxCmp)> pq(boxCmp);

    int64_t current_size = 0;
    int64_t max_current_size = 0;
    int64_t max_box_size = 0;

    for (const auto& box : boxes) {
        max_box_size = std::max(max_box_size, box.size);
        current_size += box.size;
        while (!pq.empty() && pq.top().finish < box.start) {
            auto&& retire_box = pq.top();
            current_size -= retire_box.size;
            pq.pop();
        }
        pq.push(box);
        max_current_size = std::max(max_current_size, current_size);
    }

    return {max_current_size, max_box_size};
}

MemoryStatisticsRecord dumpStatisticsImpl(const MemoryManagerIO& obj) {
    MemoryStatisticsRecord retVal;
    retVal.id = MemoryManagerIO::getClassName();
    retVal.total_regions = obj.m_blocks.size();  // as the number of blocks ie equal to regions
    retVal.total_unique_blocks = obj.m_blocks.size();
    retVal.total_size = std::accumulate(obj.m_blocks.begin(),
                                        obj.m_blocks.end(),
                                        0,
                                        [](size_t acc, const MemoryManagerIO::BlockType& item) {
                                            return acc + item.size();
                                        });
    retVal.optimal_total_size = retVal.total_size;
    retVal.max_region_size = std::accumulate(obj.m_blocks.begin(),
                                             obj.m_blocks.end(),
                                             static_cast<size_t>(0),
                                             [](size_t acc, const MemoryManagerIO::BlockType& item) {
                                                 return std::max(acc, item.size());
                                             });
    return retVal;
}

MemoryStatisticsRecord dumpStatisticsImpl(const MemoryManagerStatic& obj) {
    MemoryStatisticsRecord retVal;
    retVal.id = MemoryManagerStatic::getClassName();
    retVal.total_regions = obj.m_boxes.size();
    retVal.total_unique_blocks = 1;  // in fact there is only one unique block
    retVal.total_size = obj.m_totalSize;

    {
        auto result = calculateOptimalMemorySize(obj.m_boxes);

        retVal.optimal_total_size = result.first;
        retVal.max_region_size = result.second;
    }
    return retVal;
}

MemoryStatisticsRecord dumpStatisticsImpl(const MemoryManagerNonOverlappingSets& obj) {
    static_assert(std::is_same_v<MemoryManagerNonOverlappingSets::InternalBlock, IndividualMemoryBlockWithRelease>,
                  "Unexpected block type");

    MemoryStatisticsRecord retVal;
    retVal.id = MemoryManagerNonOverlappingSets::getClassName();
    retVal.total_regions = obj.m_boxes.size();

    std::unordered_set<std::shared_ptr<const MemoryBlockWithRelease>> uniqueBlocks;
    for (auto&& item : obj.m_internalBlocks) {
        uniqueBlocks.insert(item.second->getParentBlock());
    }

    retVal.total_unique_blocks = uniqueBlocks.size();
    retVal.total_size = std::accumulate(uniqueBlocks.begin(),
                                        uniqueBlocks.end(),
                                        static_cast<size_t>(0),
                                        [](size_t acc, const auto& item) {
                                            return acc + item->size();
                                        });

    auto tmp_boxes = obj.m_boxes;
    for (auto&& box : tmp_boxes) {
        auto block = obj.m_internalBlocks.at(box.id);
        box.size = block->size();
    }

    auto result = calculateOptimalMemorySize(std::move(tmp_boxes));
    retVal.optimal_total_size = result.first;
    retVal.max_region_size = result.second;
    return retVal;
}
#endif

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

#ifdef CPU_DEBUG_CAPS
    [[nodiscard]] MemoryStatisticsRecord dumpStatistics() const {
        return m_statDumper(m_memManager);
    }

    using MemoryStatsDumper = std::function<MemoryStatisticsRecord(const MemoryManagerPtr&)>;

    void setDumper(MemoryStatsDumper dumper) {
        m_statDumper = std::move(dumper);
    }

private:
    MemoryStatsDumper m_statDumper;

#endif  // CPU_DEBUG_CAPS

private:
    Condition m_cond;
    MemoryManagerPtr m_memManager;
};

namespace {

template <typename T, typename F, typename... Args>
MemoryControl::RegionHandlerPtr buildHandler(F&& f, Args&&... args) {
    auto retVal = std::make_shared<MemoryControl::RegionHandler>(std::forward<F>(f),
                                                                 std::make_shared<T>(std::forward<Args>(args)...));
#ifdef CPU_DEBUG_CAPS
    retVal->setDumper([](const MemoryManagerPtr& ptr) {
        OPENVINO_ASSERT(ptr);
        return dumpStatisticsImpl(*static_cast<T*>(ptr.get()));
    });
#endif  // CPU_DEBUG_CAPS

    return retVal;
}

}  // namespace

MemoryControl::MemoryControl(std::string id) : m_id(std::move(id)) {
    // init handlers
    m_handlers.emplace_back(buildHandler<MemoryManagerStatic>([](const MemoryRegion& reg) {
        if (reg.size < 0 || MemoryRegion::RegionType::VARIABLE != reg.type ||
            MemoryRegion::AllocType::POD != reg.alloc_type) {
            return false;
        }
        return true;
    }));

    // handler for static tensors
    m_handlers.emplace_back(buildHandler<MemoryManagerNonOverlappingSets>([](const MemoryRegion& reg) {
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
    OPENVINO_THROW("No suitable handler was found for the given memory region");
}

void MemoryControl::insert(const MemoryRegions& regions, const std::vector<size_t>& syncInds) {
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

#ifdef CPU_DEBUG_CAPS
MemoryStatistics MemoryControl::dumpStatistics() const {
    MemoryStatistics profileData;
    for (auto&& handler : m_handlers) {
        profileData.push_back(handler->dumpStatistics());
    }
    return profileData;
}
#endif  // CPU_DEBUG_CAPS

MemoryControl::Ptr NetworkMemoryControl::createMemoryControlUnit(std::string id) {
    m_controlUnits.emplace_back(std::shared_ptr<MemoryControl>(new MemoryControl(std::move(id))));
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

std::vector<std::pair<std::string, MemoryStatistics>> NetworkMemoryControl::dumpStatistics() const {
#ifdef CPU_DEBUG_CAPS
    std::vector<std::pair<std::string, MemoryStatistics>> retVal;
    retVal.reserve(m_controlUnits.size());
    for (auto&& item : m_controlUnits) {
        retVal.emplace_back(item->getId(), item->dumpStatistics());
    }
    return retVal;
#else
    return {};
#endif  // CPU_DEBUG_CAPS
}

}  // namespace ov::intel_cpu
