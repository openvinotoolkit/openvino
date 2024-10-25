// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "edge.h"
#include "graph.h"
#include "node.h"
#include "proxy_mem_blk.h"

namespace ov {
namespace intel_cpu {

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

class MemoryControl {
public:
    class RegionHandler;

    using RegionHandlerPtr = std::shared_ptr<RegionHandler>;
    using MemorySolution = std::unordered_map<decltype(MemoryRegion::id), MemoryBlockPtr>;

public:
    void insert(const MemoryRegions& regions,
                const std::vector<size_t>& syncInds);

    MemorySolution solve();

    bool allocated() const {
        return m_allocated;
    }

    void allocateMemory();
    void releaseMemory();

private:
    explicit MemoryControl();
    void insert(const MemoryRegion& region, const std::vector<size_t>& syncInds);

    friend class NetworkMemoryControl;

private:
    std::vector<RegionHandlerPtr> m_handlers;
    bool m_allocated = false;
};

class NetworkMemoryControl {
public:
    NetworkMemoryControl() = default;
    // @todo return std::reference_wrapper instead?
    MemoryControl* createMemoryControlUnit();

    void allocateMemory();
    void releaseMemory();

private:
    using value_type = std::unique_ptr<MemoryControl>;

private:
    std::vector<value_type> m_controlUnits;
};

}  // namespace intel_cpu
}  // namespace ov
