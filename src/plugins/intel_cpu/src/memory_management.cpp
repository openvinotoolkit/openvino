// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_management.hpp"

#include <ov_optional.hpp>

#include "node.h"
#include "openvino/runtime/memory_solver.hpp"
#include "partitioned_mem_mgr.h"

namespace ov {
namespace intel_cpu {

namespace {

class IMemoryManager {
public:
    virtual ~IMemoryManager() = default;
    virtual void insert(const MemoryRegion& reg) = 0;
    virtual const MemoryControl::MemoryBlockMap& lastSolution() = 0;
};

using MemoryManagerPtr = std::shared_ptr<IMemoryManager>;

template<typename T, typename... Args>
std::shared_ptr<DnnlMemoryBlock> makeDnnlMemoryBlock(Args&&... args) {
    return std::make_shared<DnnlMemoryBlock>(make_unique<T>(std::forward<Args>(args)...));
}

class MemoryManagerIndividualBlocks : public IMemoryManager {
public:
    void insert(const MemoryRegion& reg) override {
        m_blocks.insert({reg.id, makeDnnlMemoryBlock<MemoryBlockWithReuse>()});
    }

    const MemoryControl::MemoryBlockMap& lastSolution() override {
        return m_blocks;
    }

private:
    MemoryControl::MemoryBlockMap m_blocks;
};

class MemoryManagerStaticSolver : public IMemoryManager {
public:
    void insert(const MemoryRegion& reg) override {
        m_boxes.emplace_back(MemorySolver::Box{reg.start, reg.finish, reg.size, reg.id});
    }

    const MemoryControl::MemoryBlockMap& lastSolution() override {
        if (!m_blocks.empty()) {
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
        size_t total_size = static_cast<size_t>(staticMemSolver.solve()) * alignment;

        m_workspace = makeDnnlMemoryBlock<MemoryBlockWithReuse>();
        m_workspace->resize(total_size);

        for (const auto& box : m_boxes) {
            int64_t offset = staticMemSolver.get_offset(box.id);
            auto memoryBlock = std::make_shared<PartitionedMemoryBlock>(m_workspace, total_size, offset, box.size * alignment);
            m_blocks[box.id] = std::move(memoryBlock);
        }
        // m_boxes.clear();
    }

private:
    MemoryControl::MemoryBlockMap m_blocks;
    std::vector<MemorySolver::Box> m_boxes;
    MemoryBlockPtr m_workspace;
};

class MemoryManageNonOverlapingSets : public IMemoryManager {
public:
    MemoryManageNonOverlapingSets(std::vector<size_t> syncInds) : m_syncInds(std::move(syncInds)) {}
    void insert(const MemoryRegion& reg) override {
        MemorySolver::Box box = {reg.start, reg.finish, reg.size, reg.id};
        if (-1 != reg.finish) {
            //We have to extend the lifespan of tensors that are crossing a sync point border in order to save
            //the intermediate computation results from possible loss due to the tensor resize
            auto itr_upper =
                std::upper_bound(m_syncInds.begin(), m_syncInds.end(), box.finish, [](int y, int x) {
                    return y <= x;
                });
            auto itr_lower = std::lower_bound(m_syncInds.begin(), m_syncInds.end(), box.start);
            if (itr_lower != itr_upper) { // across sections
                if (itr_upper == m_syncInds.end()) {
                    box.finish = -1;
                } else {
                    box.finish = *itr_upper;
                }
            }
        }
        m_boxes.emplace_back(std::move(box));
    }

    const MemoryControl::MemoryBlockMap& lastSolution() override {
        if (!m_blocks.empty()) {
            solve();
        }
        return m_blocks;
    }

private:
    void solve() {
        ov::MemorySolver::normalize_boxes(m_boxes);

        std::vector<std::vector<ov::MemorySolver::Box>> groups; //groups of nonoverlapping boxes
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
            auto grpMemBlock = makeDnnlMemoryBlock<MemoryBlockWithReuse>();
            for (auto& box : group) {
                m_blocks[box.id] = grpMemBlock;
            }
        }
        // m_boxes.clear();
    }

private:
    MemoryControl::MemoryBlockMap m_blocks;
    std::vector<MemorySolver::Box> m_boxes;
    std::vector<size_t> m_syncInds;
};

}  // namespace

class MemoryControl::RegionHandler {
public:
    using Condition = std::function<bool(const MemoryRegion&)>;

public:
    RegionHandler(Condition cond, MemoryManagerPtr memManager)
        : m_cond(std::move(cond)),
          m_memManager(std::move(memManager)) {}

    bool insert(const MemoryRegion& reg) {
        if (!m_cond(reg)) {
            return false;
        }

        m_memManager->insert(reg);
        return true;
    }

    const MemoryControl::MemoryBlockMap& lastSolution() const {
        return m_memManager->lastSolution();
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

MemoryControl::MemoryControl(std::vector<size_t> syncInds) {
    // init handlers

    // handler for dynamic tensors
    m_handlers.emplace_back(buildHandler<MemoryManagerStaticSolver>([](const MemoryRegion& reg) {
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
    }, std::move(syncInds)));

    //handler for I/O tensors, so far simply individual blocks
    m_handlers.emplace_back(buildHandler<MemoryManagerIndividualBlocks>([](const MemoryRegion& reg) {
        if (MemoryRegion::RegionType::VARIABLE == reg.type || reg.alloc_type != MemoryRegion::AllocType::POD) {
            return false;
        }
        return true;
    }));
}

void MemoryControl::insert(const MemoryRegion& region) {
    for (auto&& handler : m_handlers) {
        if (handler->insert(region)) {
            return;
        }
    }
    OPENVINO_THROW("No suitable hanlder was found for the given memory region");
}

MemoryControl::MemoryBlockMap MemoryControl::insert(const std::vector<MemoryRegion>& regions) {
    for (auto&& region : regions) {
        insert(region);
    }

    MemoryControl::MemoryBlockMap blocksMap;
    blocksMap.reserve(regions.size());

    for (auto&& handler : m_handlers) {
        auto&& solution = handler->lastSolution();
        for (auto&& item : solution) {
            auto res = blocksMap.insert(item);
            OPENVINO_ASSERT(res.second, "Memory solutions has non unique entries");
        }
    }

    return blocksMap;
}

edgeClusters MemoryControl::findEdgeClusters(const std::vector<EdgePtr>& graphEdges) {
    typedef std::unordered_map<EdgePtr, size_t> edge_cluster_idx_map_t;

    edgeClusters edge_clusters;
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
            edge_clusters.emplace_back(edgeCluster{edge});
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

}  // namespace intel_cpu
}  // namespace ov