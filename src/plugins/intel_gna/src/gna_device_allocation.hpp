// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "gna2-model-export-api.h"
#include "gna_lib_ver_selector.hpp"
#include "memory/gna_mem_regions.hpp"

using ov::intel_gna::memory::rRegion;

struct GnaAllocation {
    void* ptr = nullptr;
    size_t sizeRequested = 0;
    size_t sizeGranted = 0;
    void SetTag(Gna2MemoryTag in) {
        isTagSet = true;
        tag = in;
    }
    bool isTag(Gna2MemoryTag in) const {
        return isTagSet && in == tag;
    }
    std::string GetTagName() const {
        static const std::map<Gna2MemoryTag, std::string> tm = {
            {Gna2MemoryTagReadWrite, "Gna2MemoryTagReadWrite"},
            {Gna2MemoryTagInput, "Gna2MemoryTagInput"},
            {Gna2MemoryTagOutput, "Gna2MemoryTagOutput"},
            {Gna2MemoryTagReadOnly, "Gna2MemoryTagReadOnly"},
            {Gna2MemoryTagExternalBufferInput, "Gna2MemoryTagExternalBufferInput"},
            {Gna2MemoryTagExternalBufferOutput, "Gna2MemoryTagExternalBufferOutput"},
            {Gna2MemoryTagScratch, "Gna2MemoryTagScratch"},
            {Gna2MemoryTagState, "Gna2MemoryTagState"},
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

    static rRegion GetRegionForTag(Gna2MemoryTag tag) {
        static const std::map<Gna2MemoryTag, rRegion> tm = {
            {Gna2MemoryTagInput, rRegion::REGION_INPUTS},
            {Gna2MemoryTagOutput, rRegion::REGION_OUTPUTS},
            {Gna2MemoryTagReadOnly, rRegion::REGION_RO},
            {Gna2MemoryTagScratch, rRegion::REGION_SCRATCH},
            {Gna2MemoryTagState, rRegion::REGION_STATES},
            {Gna2MemoryTagExternalBufferInput, rRegion::REGION_INPUTS},
            {Gna2MemoryTagExternalBufferOutput, rRegion::REGION_OUTPUTS},
        };
        auto f = tm.find(tag);
        if (f != tm.end()) {
            return f->second;
        }
        return rRegion::REGION_AUTO;
    }

    bool operator<(const GnaAllocation& right) const {
        const auto region = GetRegionForTag(tag);
        const auto regionRight = GetRegionForTag(right.tag);
        return region < regionRight;
    }

    std::pair<bool, size_t> getOffset(void* offset) const {
        std::pair<bool, size_t> v;
        v.first = offset >= ptr && offset < static_cast<uint8_t*>(ptr) + sizeGranted;
        v.second = v.first ? static_cast<uint8_t*>(offset) - static_cast<uint8_t*>(ptr) : 0;
        return v;
    }

    uint32_t sizeForExport() const {
        return ALIGN64(static_cast<uint32_t>(sizeRequested));
    }

private:
    Gna2MemoryTag tag = Gna2MemoryTagScratch;
    bool isTagSet = false;
};

class GnaAllocations {
    std::list<GnaAllocation> allocations;

public:
    GnaAllocations() = default;
    template <class T>
    explicit GnaAllocations(T b, T e) : allocations(b, e) {}

    static uint32_t GetSizeForExport(const std::list<GnaAllocation>& allocations) {
        uint32_t total = 0;
        for (auto& a : allocations) {
            total += a.sizeForExport();
        }
        return total;
    }

    uint32_t GetSizeForExport() const {
        return GetSizeForExport(allocations);
    }

    std::list<GnaAllocation> GetAllocationsInExportOrder() const {
        std::vector<GnaAllocation> temp(allocations.begin(), allocations.end());
        std::stable_sort(temp.begin(), temp.end());
        return std::list<GnaAllocation>(temp.begin(), temp.end());
    }

    static std::pair<bool, uint64_t> GetOffsetForExport(const std::list<GnaAllocation>& orderedAllocations, void* ptr) {
        uint64_t curOffset = 0;
        for (auto& r : orderedAllocations) {
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

    bool SetTagFor(void* memPtr, Gna2MemoryTag memoryTag) {
        auto found = std::find_if(allocations.begin(), allocations.end(), [memPtr](const GnaAllocation& a) {
            return a.ptr == memPtr;
        });
        if (found != allocations.end()) {
            found->SetTag(memoryTag);
            return true;
        }
        return false;
    }

    bool Remove(void* memPtr) {
        auto found = std::find_if(allocations.begin(), allocations.end(), [memPtr](const GnaAllocation& a) {
            return a.ptr == memPtr;
        });
        if (found != allocations.end()) {
            allocations.erase(found);
            return true;
        }
        return false;
    }

    void Add(void* memPtr, uint32_t sizeRequested, uint32_t sizeGranted) {
        GnaAllocation newAllocation;
        newAllocation.ptr = memPtr;
        newAllocation.sizeRequested = sizeRequested;
        newAllocation.sizeGranted = sizeGranted;
        allocations.push_back(newAllocation);
    }

    const GnaAllocation* Get(const Gna2MemoryTag tag) const {
        for (auto&& a : allocations) {
            if (a.isTag(tag)) {
                return &a;
            }
        }
        return nullptr;
    }
};
