// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstdint>
#include <list>
#include <map>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "common/gna_target.hpp"
#include "gna2-model-export-api.h"
#include "gna2-model-suecreek-header.h"
#include "gna_device_allocation.hpp"

namespace ov {
namespace intel_gna {

struct GnaEndpoint {
    std::string name;
    uint32_t byteSize = 0;
    uint32_t offset = 0;
    uint32_t numberOfBytesPerElement = 0;
    float scaleFactor = 0;
    void* gnaPointer = nullptr;

    template <class T>
    static uint32_t GetTotalByteSize(const T& container);

    template <class T>
    static GnaEndpoint CreateFromDescriptor(const T& descriptor);

    template <class T>
    static std::vector<GnaEndpoint> CreateFromDescriptorContainer(const T& container);
};

template <class T>
uint32_t GnaEndpoint::GetTotalByteSize(const T& container) {
    return std::accumulate(container.begin(), container.end(), 0, [](uint32_t cur, const GnaEndpoint& next) {
        return cur + next.byteSize;
    });
}

template <class T>
GnaEndpoint GnaEndpoint::CreateFromDescriptor(const T& descriptor) {
    GnaEndpoint e;
    e.scaleFactor = descriptor.scale_factor;
    e.byteSize = descriptor.get_required_size();
    e.name = descriptor.name;
    e.numberOfBytesPerElement = static_cast<uint32_t>(descriptor.tensor_precision.size());
    if (!descriptor.ptrs.empty()) {
        e.gnaPointer = descriptor.ptrs.front();
    }
    return e;
}

template <class T>
std::vector<GnaEndpoint> GnaEndpoint::CreateFromDescriptorContainer(const T& container) {
    std::vector<GnaEndpoint> result;
    for (const auto& e : container) {
        result.push_back(CreateFromDescriptor(e));
    }
    return result;
}

void* ExportSueLegacyUsingGnaApi2(uint32_t modelId, uint32_t deviceIndex, Gna2ModelSueCreekHeader* modelHeader);

void ExportTlvModel(uint32_t modelId,
                    uint32_t deviceIndex,
                    std::ostream& outStream,
                    const common::DeviceVersion& compileTarget,
                    const std::vector<GnaEndpoint>& inputs,
                    const std::vector<GnaEndpoint>& outputs,
                    const GnaAllocations& allAllocation);

}  // namespace intel_gna
}  // namespace ov
