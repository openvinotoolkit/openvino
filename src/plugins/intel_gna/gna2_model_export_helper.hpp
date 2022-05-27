// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gna2-common-api.h"
#include "gna2-model-suecreek-header.h"

#include <cstdint>
#include <string>

void * ExportSueLegacyUsingGnaApi2(
    uint32_t modelId,
    uint32_t deviceIndex,
    Gna2ModelSueCreekHeader* modelHeader);

void ExportLdForDeviceVersion(
    uint32_t modelId,
    std::ostream & outStream,
    Gna2DeviceVersion deviceVersionToExport);

void ExportTlvModel(uint32_t modelId,
    uint32_t deviceIndex,
    std::ostream& outStream,
    Gna2DeviceVersion deviceVersionToExport,
    uint32_t input_size,
    uint32_t output_size,
    float inputSF,
    float outputSF);

void ExportGnaDescriptorPartiallyFilled(uint32_t numberOfLayers, std::ostream & outStream);
