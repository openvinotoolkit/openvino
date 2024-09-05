// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "edge.h"

namespace ov {
namespace intel_cpu {

using edgeCluster = std::unordered_set<EdgePtr>;
using edgeClusters = std::vector<edgeCluster>;

struct MemoryRegion {
    int start;     // Execution order index of first use.
    int finish;    // Execution order index of last use. -1 means inf
    int64_t size;  // size in bytes
    int64_t id;    // ID unique for each region

    enum class RegionType : uint8_t { VARIABLE, CONSTANT, INPUT, OUTPUT, IO } type;
    enum class AllocType : uint8_t { POD, STRING, UNKNOWN } alloc_type;
};

class MemoryControl {
public:
    class RegionHandler;

    using RegionHandlerPtr = std::shared_ptr<RegionHandler>;
    using MemoryBlockMap = std::unordered_map<decltype(MemoryRegion::id), MemoryBlockPtr>;

public:
    static edgeClusters findEdgeClusters(const std::vector<EdgePtr>& graphEdges);

    MemoryBlockMap insert(const std::vector<MemoryRegion>& regions);

    bool allocated() const {
        return m_allocated;
    }

    void allocateMemory();
    void releaseMemory();

private:
    explicit MemoryControl(std::vector<size_t> syncInds);
    void insert(const MemoryRegion& region);

    friend class NetworkMemoryControl;

private:
    std::vector<size_t> m_syncInds;
    std::vector<RegionHandlerPtr> m_handlers;
    bool m_allocated = false;
};

class NetworkMemoryControl {
public:
    NetworkMemoryControl() = default;
    MemoryControl& createMemoryControlUnit(std::vector<size_t> syncInds);

    void allocateMemory();
    void releaseMemory();

private:
    using value_type = std::unique_ptr<MemoryControl>;

private:
    std::vector<value_type> m_controlUnits;
};

}  // namespace intel_cpu
}  // namespace ov