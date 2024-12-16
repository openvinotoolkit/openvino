// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_control.hpp"

#include <ov_optional.hpp>
#include <queue>

#include "node.h"
#include "openvino/runtime/memory_solver.hpp"
#include "utils/general_utils.h"

namespace ov {
namespace intel_cpu {

namespace {

class StaticPartitionMemoryBlock : public IMemoryBlockObserver {
public:
    StaticPartitionMemoryBlock(MemoryBlockPtr pBlock, ptrdiff_t offset) : m_pBlock(pBlock), m_offset(offset) {
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

    size_t size() const {
        return m_pInternalMem->size();
    }

private:
    MemoryBlockPtr m_pBlock;
    MemoryBlockWithReuse* m_pInternalMem;
};

class IndividualMemoryBlockWithRelease : public IMemoryBlockObserver {
public:
    IndividualMemoryBlockWithRelease(const std::shared_ptr<MemoryBlockWithRelease>& pBlock) : m_pBlock(pBlock) {}

    void* getRawPtr() const noexcept override {
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
        m_max_requested_size = 0;
        if (m_pBlock->size() > 0) {
            m_pBlock->free();
        }
    }

    size_t size() const {
        return m_max_requested_size;
    }

private:
    std::shared_ptr<MemoryBlockWithRelease> m_pBlock;
    size_t m_max_requested_size = 0;
};

class IMemoryManager {
public:
    virtual ~IMemoryManager() = default;
    virtual void insert(const MemoryRegion& reg, const std::vector<size_t>& syncInds) = 0;
    virtual const MemoryControl::MemorySolution& lastSolution() = 0;
    virtual void allocate() = 0;
    virtual void release() = 0;
    virtual MemoryStatisticsRecord dumpStatistics() = 0;
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
        m_blocks.push_back(*block);
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

    MemoryStatisticsRecord dumpStatistics() override {
        MemoryStatisticsRecord retVal;
        retVal.id = getClassName();
        retVal.total_regions = m_blocks.size();  // as the number of blocks ie equal to regions
        retVal.total_unique_blocks = m_blocks.size();
        retVal.total_size = std::accumulate(m_blocks.begin(), m_blocks.end(), 0, [](size_t acc, const BlockType& item) {
            return acc + item.size();
        });
        retVal.optimal_total_size = retVal.total_size;
        // find max size memory block in m_blocks
        retVal.max_region_size =
            std::accumulate(m_blocks.begin(), m_blocks.end(), 0, [](size_t acc, const BlockType& item) {
                return std::max(acc, item.size());
            });
        return retVal;
    }

private:
    static const char* getClassName() {
        return "MemoryManagerIO";
    }

private:
    MemoryControl::MemorySolution m_solution;
    std::vector<std::reference_wrapper<BlockType>> m_blocks;
};

static std::pair<int64_t, int64_t> calculateOptimalMemorySize(std::vector<MemorySolver::Box> boxes) {
    ov::MemorySolver::normalize_boxes(boxes);

    auto boxCmp = [](const MemorySolver::Box& l, const MemorySolver::Box& r) {
        return l.finish > r.finish;
    };
    std::priority_queue<MemorySolver::Box, std::vector<MemorySolver::Box>, decltype(boxCmp)> pq(boxCmp);

    ptrdiff_t current_size = 0;
    ptrdiff_t max_current_size = 0;
    ptrdiff_t max_box_size = 0;

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

    MemoryStatisticsRecord dumpStatistics() override {
        MemoryStatisticsRecord retVal;
        retVal.id = getClassName();
        retVal.total_regions = m_boxes.size();
        retVal.total_unique_blocks = 1;  // in fact there is only one unique block
        retVal.total_size = m_totalSize;

        {
            auto result = calculateOptimalMemorySize(m_boxes);

            retVal.optimal_total_size = result.first;
            retVal.max_region_size = result.second;
        }
        return retVal;
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
        if (m_workspace)
            m_workspace->resize(m_totalSize);
    }
    void release() override {
        if (m_workspace)
            m_workspace->free();
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
        m_boxes.emplace_back(std::move(box));
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

    MemoryStatisticsRecord dumpStatistics() override {
        MemoryStatisticsRecord retVal;
        retVal.id = getClassName();
        retVal.total_regions = m_boxes.size();
        retVal.total_unique_blocks = m_unique_blocks.size();
        retVal.total_size = std::accumulate(m_unique_blocks.begin(),
                                            m_unique_blocks.end(),
                                            0,
                                            [](size_t acc, const decltype(m_unique_blocks)::value_type& item) {
                                                return acc + item->size();
                                            });

        auto tmp_boxes = m_boxes;
        for (auto&& box : tmp_boxes) {
            auto block = m_internalBlocks[box.id];
            box.size = block->size();
        }

        auto result = calculateOptimalMemorySize(std::move(tmp_boxes));
        retVal.optimal_total_size = result.first;
        retVal.max_region_size = result.second;

        // retVal.max_region_size = std::accumulate(m_internalBlocks.begin(),
        //                                         m_internalBlocks.end(),
        //                                         0,
        //                                         [](size_t acc, const decltype(m_internalBlocks)::value_type& item) {
        //                                             return std::max(acc, item.second->size());
        //                                         });
        return retVal;
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
            m_unique_blocks.push_back(std::make_shared<MemoryBlockWithRelease>());
            for (auto& box : group) {
                m_internalBlocks[box.id] = std::make_shared<IndividualMemoryBlockWithRelease>(m_unique_blocks.back());
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
    std::unordered_map<MemoryControl::MemorySolution::key_type, std::shared_ptr<IndividualMemoryBlockWithRelease>>
        m_internalBlocks;
    std::vector<MemorySolver::Box> m_boxes;
    std::vector<std::shared_ptr<MemoryBlockWithRelease>> m_unique_blocks;
    bool reset_flag = true;
};

}  // namespace

std::ostream& operator<<(std::ostream& os, const MemoryStatisticsRecord& record) {
    os << "Memory profile record: " << record.id << std::endl;
    os << "Total regions: " << record.total_regions << std::endl;
    os << "Total unique blocks: " << record.total_unique_blocks << std::endl;
    os << "Total size: " << record.total_size << " bytes" << std::endl;
    os << "Optimal total size: " << record.optimal_total_size << " bytes" << std::endl;
    os << "Max region size: " << record.max_region_size << " bytes" << std::endl;
    return os;
}

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

    MemoryStatisticsRecord dumpStatistics() const {
        return m_memManager->dumpStatistics();
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

MemoryStatistics MemoryControl::dumpStatistics() const {
    MemoryStatistics profileData;
    for (auto&& handler : m_handlers) {
        profileData.push_back(handler->dumpStatistics());
    }
    return profileData;
}

EdgeClusters MemoryControl::findEdgeClusters(const std::vector<EdgePtr>& graphEdges) {
    typedef std::unordered_map<EdgePtr, size_t> edge_cluster_idx_map_t;

    EdgeClusters edge_clusters;
    edge_cluster_idx_map_t edge_cluster_indices;

    for (auto& edge : graphEdges) {
        auto edge_it = edge_cluster_indices.find(edge);
        if (edge_it != edge_cluster_indices.end())
            continue;  // edge is visited

        size_t cluster_idx = edge_clusters.size();
        EdgePtr last_shared_edge = nullptr;

        // find cluster index
        for (auto shared_edge = edge->getSharedEdge(std::nothrow); shared_edge;
             shared_edge = shared_edge->getSharedEdge(std::nothrow)) {
            auto shared_edge_it = edge_cluster_indices.find(shared_edge);
            if (shared_edge_it != edge_cluster_indices.end()) {
                cluster_idx = shared_edge_it->second;
                last_shared_edge = shared_edge;
                break;
            }
        }

        // add shared edges to cluster
        edge_cluster_indices.emplace(edge, cluster_idx);

        if (cluster_idx == edge_clusters.size())
            edge_clusters.emplace_back(EdgeCluster{edge});
        else
            edge_clusters[cluster_idx].emplace(edge);

        for (auto shared_edge = edge->getSharedEdge(std::nothrow); shared_edge != last_shared_edge;
             shared_edge = shared_edge->getSharedEdge(std::nothrow)) {
            edge_cluster_indices.emplace(shared_edge, cluster_idx);
            edge_clusters[cluster_idx].emplace(shared_edge);
        }
    }

    return edge_clusters;
}

MemoryControl& NetworkMemoryControl::createMemoryControlUnit(std::string id) {
    m_controlUnits.emplace_back(std::unique_ptr<MemoryControl>(new MemoryControl(std::move(id))));
    return *(m_controlUnits.back());
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

std::unordered_map<std::string, MemoryStatistics> NetworkMemoryControl::dumpStatistics() const {
    std::unordered_map<std::string, MemoryStatistics> retVal;
    for (auto&& item : m_controlUnits) {
        retVal.insert({item->getId(), item->dumpStatistics()});
    }
    return retVal;
}

}  // namespace intel_cpu
}  // namespace ov
