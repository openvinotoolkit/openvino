// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "edge.h"

namespace ov::intel_cpu {

class IAuxiliaryMemoryControl {
public:
    virtual ~IAuxiliaryMemoryControl() = default;

    virtual MemoryPtr allocate(const dnnl::engine& engine, const MemoryDescPtr& desc) = 0;
    virtual void registerMemoryBlock(MemoryBlockPtr block) = 0;
    virtual void registerMemoryCallbacks(std::function<void()> release, std::function<void()> allocate) = 0;
    virtual void releaseMemory() = 0;
    virtual void allocateMemory() = 0;
};

using EdgeCluster = std::vector<EdgePtr>;
using EdgeClusters = std::vector<EdgeCluster>;

struct MemoryRegion {
    int start;     // Execution order index of first use.
    int finish;    // Execution order index of last use. -1 means inf
    int64_t size;  // size in bytes
    int64_t id;    // ID unique for each region

    enum class RegionType : uint8_t { VARIABLE, CONSTANT, INPUT, OUTPUT, IO } type;
    enum class AllocType : uint8_t { POD, STRING, UNKNOWN } alloc_type;
};

using MemoryRegions = std::vector<MemoryRegion>;
struct MemoryStatisticsRecord {
    const char* id;
    size_t total_regions;        // number of regions
    size_t total_unique_blocks;  // bytes
    size_t total_size;           // bytes
    size_t optimal_total_size;   // bytes
    size_t max_region_size;      // bytes
};

using MemoryStatistics = std::vector<MemoryStatisticsRecord>;

class MemoryControl {
public:
    class RegionHandler;

    using RegionHandlerPtr = std::shared_ptr<RegionHandler>;
    using MemorySolution = std::unordered_map<decltype(MemoryRegion::id), MemoryBlockPtr>;
    using Ptr = std::shared_ptr<MemoryControl>;
    using CPtr = std::shared_ptr<const MemoryControl>;

    void insert(const MemoryRegions& regions, const std::vector<size_t>& syncInds);

    MemorySolution solve();

    [[nodiscard]] bool allocated() const {
        return m_allocated;
    }

    void allocateMemory();
    void releaseMemory();

    [[nodiscard]] const std::string& getId() const {
        return m_id;
    }

private:
    explicit MemoryControl(std::string id);
    void insert(const MemoryRegion& region, const std::vector<size_t>& syncInds);
    [[nodiscard]] MemoryStatistics dumpStatistics() const;

    friend class NetworkMemoryControl;

    std::string m_id;
    std::vector<RegionHandlerPtr> m_handlers;
    bool m_allocated = false;
};

class NetworkMemoryControl : public IAuxiliaryMemoryControl {
public:
    NetworkMemoryControl() = default;
    MemoryControl::Ptr createMemoryControlUnit(std::string id);

    MemoryPtr allocate(const dnnl::engine& engine, const MemoryDescPtr& desc) override;
    void registerMemoryBlock(MemoryBlockPtr block) override;
    void registerMemoryCallbacks(std::function<void()> release, std::function<void()> allocate) override;

    void allocateMemory() override;
    void releaseMemory() override;

    [[nodiscard]] std::vector<std::pair<std::string, MemoryStatistics>> dumpStatistics() const;

    [[nodiscard]] const std::vector<MemoryControl::Ptr>& controlUnits() const {
        return m_controlUnits;
    }

private:
    std::vector<MemoryControl::Ptr> m_controlUnits;
    struct AuxiliaryMemory;
    std::vector<std::shared_ptr<AuxiliaryMemory>> m_auxiliaryMemory;
    std::vector<std::pair<std::function<void()>, std::function<void()>>> m_auxiliaryCallbacks;
};

static_assert(std::is_base_of_v<IAuxiliaryMemoryControl, NetworkMemoryControl>,
              "NetworkMemoryControl must implement IAuxiliaryMemoryControl");

}  // namespace ov::intel_cpu
