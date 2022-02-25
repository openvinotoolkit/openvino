// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gna2-model-export-api.h"
#include "gna2-common-api.h"
#include "gna2-model-suecreek-header.h"

#include <algorithm>
#include <cstdint>
#include <list>
#include <map>
#include <string>
#include <utility>
#include <vector>

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
    std::pair<bool, size_t> getOffset(void* offset) const {
        std::pair<bool, size_t> v;
        v.first = offset >= ptr && offset < static_cast<uint8_t*>(ptr) + sizeGranted;
        v.second = v.first ? static_cast<uint8_t*>(offset) - static_cast<uint8_t*>(ptr) : 0;
        return v;
    }

private:
    Gna2MemoryTag tag;
    bool isTagSet = false;
};
typedef std::list<GnaAllocation> GnaAllAllocations;

struct GnaEndpoint {
    std::string name;
    uint32_t byteSize = 0;
    uint32_t offset = 0;
    uint32_t numberOfBytesPerElement = 0;
    float scaleFactor = 0;
    void* gnaPointer = nullptr;

    template <class T>
    static uint32_t GetTotalByteSize(const T& container) {
        return std::accumulate(container.begin(), container.end(), 0, [](uint32_t cur, const GnaEndpoint& next) {
            return cur + next.byteSize;
        });
    }

    template <class T>
    static GnaEndpoint CreateFromDescriptor(const T& descriptor) {
        GnaEndpoint e;
        e.scaleFactor = descriptor.scale_factor;
        //  optionally descriptor.get_allocated_size()
        e.byteSize = descriptor.get_required_size();
        e.name = descriptor.name;
        e.numberOfBytesPerElement = static_cast<uint32_t>(descriptor.tensor_precision.size());
        if (!descriptor.ptrs.empty()) {
            e.gnaPointer = descriptor.ptrs.front();
        }
        return e;
    }

    template <class T>
    static std::vector<GnaEndpoint> CreateFromDescriptorContainer(const T& container) {
        std::vector<GnaEndpoint> result;
        for (const auto& e : container) {
            result.push_back(CreateFromDescriptor(e));
        }
        return result;
    }
};

void * ExportSueLegacyUsingGnaApi2(
    uint32_t modelId,
    uint32_t deviceIndex,
    Gna2ModelSueCreekHeader* modelHeader);

void ExportLdForDeviceVersion(
    uint32_t modelId,
    std::ostream & outStream,
    Gna2DeviceVersion deviceVersionToExport);

Gna2DeviceVersion getEmbeddedTargetFromCompileTarget(const std::string compileTarget);

void ExportTlvModel(uint32_t modelId,
    uint32_t deviceIndex,
    std::ostream& outStream,
    std::string compileTarget,
    const std::vector<GnaEndpoint>& inputs,
    const std::vector<GnaEndpoint>& outputs,
    const GnaAllAllocations& allAllocation);

void ExportGnaDescriptorPartiallyFilled(uint32_t numberOfLayers, std::ostream & outStream);
