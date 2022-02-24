// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gna2-common-api.h"
#include "gna2-model-suecreek-header.h"

#include <cstdint>
#include <string>
#include <vector>

struct GnaEndpoint {
    std::string name;
    uint32_t byteSize = 0;
    uint32_t offset = 0;
    uint32_t numberOfBytesPerElement = 0;
    float scaleFactor = 0;

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
        // optionally descriptor.get_allocated_size()
        e.byteSize = descriptor.get_required_size();
        e.name = descriptor.name;
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
    const std::vector<GnaEndpoint>& outputs);

void ExportGnaDescriptorPartiallyFilled(uint32_t numberOfLayers, std::ostream & outStream);
