// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "edge.h"

namespace ov::intel_cpu {

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
    using Ptr = std::shared_ptr<MemoryControl>;
    using CPtr = std::shared_ptr<const MemoryControl>;

public:
    void insert(const MemoryRegions& regions, const std::vector<size_t>& syncInds);

    MemorySolution solve();

    [[nodiscard]] bool allocated() const {
        return m_allocated;
    }

    void allocateMemory();
    void releaseMemory();

private:
    MemoryControl();
    void insert(const MemoryRegion& region, const std::vector<size_t>& syncInds);

    friend class NetworkMemoryControl;

private:
    std::vector<RegionHandlerPtr> m_handlers;
    bool m_allocated = false;
};

class NetworkMemoryControl {
public:
    NetworkMemoryControl() = default;

    MemoryControl::Ptr createMemoryControlUnit();

    void allocateMemory();
    void releaseMemory();

    const std::vector<MemoryControl::Ptr>& controlUnits() const {
        return m_controlUnits;
    }

private:
    std::vector<MemoryControl::Ptr> m_controlUnits;
};

}  // namespace ov::intel_cpu
