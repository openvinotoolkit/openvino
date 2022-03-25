// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <list>
#include <utility>

#include "gna2-model-export-api.h"

#include "memory/gna_mem_regions.hpp"
#include "gna_lib_ver_selector.hpp"

struct GnaAllocation {
    void* ptr = nullptr;
    size_t sizeRequested = 0;
    size_t sizeGranted = 0;
    void SetTag(Gna2MemoryTag in) {
        isTagSet = true;
        tag = in;
    }
    bool isTag(Gna2MemoryTag in) {
        return isTagSet && in == tag;
    }
    std::string GetTagName() const {
        static const std::map< Gna2MemoryTag, std::string > tm = {
                { Gna2MemoryTagReadWrite, "Gna2MemoryTagReadWrite" },
                { Gna2MemoryTagInput, "Gna2MemoryTagInput" },
                { Gna2MemoryTagOutput, "Gna2MemoryTagOutput" },
                { Gna2MemoryTagReadOnly, "Gna2MemoryTagReadOnly" },
                { Gna2MemoryTagExternalBufferInput, "Gna2MemoryTagExternalBufferInput" },
                { Gna2MemoryTagExternalBufferOutput, "Gna2MemoryTagExternalBufferOutput" },
                { Gna2MemoryTagScratch, "Gna2MemoryTagScratch" },
                { Gna2MemoryTagState, "Gna2MemoryTagState" },
        };
        if (!isTagSet) {
            return "Gna2MemoryTag_NotSet_";
        }
        auto f = tm.find(tag);
        if (f != tm.end()) {
            return f->second;
        }
        return "Gna2MemoryTag_" + std::to_string(tag) + "_";
    }
    int GetRegionOrder() const {
        static const std::map<Gna2MemoryTag, int> tm = {
            {Gna2MemoryTagInput, 1},
            {Gna2MemoryTagOutput, 2},
            {Gna2MemoryTagReadOnly, 0},
            {Gna2MemoryTagScratch, 4},
            {Gna2MemoryTagState, 3},
        };
        if (!isTagSet) {
            return 10;
        }
        auto f = tm.find(tag);
        if (f != tm.end()) {
            return f->second;
        }
        return 1000;
    }
    std::pair<bool, size_t> getOffset(void* offset) const {
        std::pair<bool, size_t> v;
        v.first = offset >= ptr && offset < static_cast<uint8_t*>(ptr) + sizeGranted;
        v.second = v.first ? static_cast<uint8_t*>(offset) - static_cast<uint8_t*>(ptr) : 0;
        return v;
    }

    uint32_t sizeForExport() const {
        return ALIGN64(sizeRequested);
    }

private:
    Gna2MemoryTag tag = Gna2MemoryTagScratch;
    bool isTagSet = false;
};
typedef std::list<GnaAllocation> GnaAllAllocations;

namespace {
uint32_t getAllAllocationSize(const GnaAllAllocations& all) {
    uint32_t total = 0;
    for (auto& a : all) {
        total += a.sizeForExport();
    }
    return total;
}

GnaAllAllocations orderedAllocations(const GnaAllAllocations& all) {
    std::vector<GnaAllocation> allVector(all.begin(), all.end());
    std::sort(allVector.begin(), allVector.end(), [](const GnaAllocation& l, const GnaAllocation& r) {
        return l.GetRegionOrder() <= r.GetRegionOrder();
    });
    return GnaAllAllocations(allVector.begin(), allVector.end());
}

std::pair<bool, uint32_t> checkAndGetAllAllocationOffsetFromBase(const GnaAllAllocations& all, void* ptr) {
    uint32_t curOffset = 0;
    for (auto& r : orderedAllocations(all)) {
        auto ptrBegin = static_cast<uint8_t*>(r.ptr);
        const auto size = r.sizeForExport();
        if (ptr >= ptrBegin && ptr < ptrBegin + size) {
            curOffset += static_cast<uint8_t*>(ptr) - ptrBegin;
            return {true, curOffset};
        }
        curOffset += size;
    }
    return {false, 0};
}
}  // namespace
