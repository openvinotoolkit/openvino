// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna2_model_export_helper.hpp"
#include "gna2_model_helper.hpp"
#include "gna_device.hpp"
#include "gna2-model-export-api.h"
#include "gna2-model-suecreek-header.h"
#include "gna_api_wrapper.hpp"
#include "gna2-device-api.h"

#include <cstdint>
#include <fstream>

void * ExportSueLegacyUsingGnaApi2(
    uint32_t modelId,
    uint32_t deviceIndex,
    Gna2ModelSueCreekHeader* modelHeader) {

    uint32_t exportConfig;
    auto status = Gna2ModelExportConfigCreate(gnaUserAllocatorAlignedPage, &exportConfig);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExportConfigCreate");

    status = Gna2ModelExportConfigSetSource(exportConfig, deviceIndex, modelId);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExportConfigSetSource");
    status = Gna2ModelExportConfigSetTarget(exportConfig, Gna2DeviceVersionEmbedded1_0);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExportConfigSetTarget");

    void * bufferSueCreekHeader = nullptr;
    uint32_t bufferSueCreekHeaderSize;

    status = Gna2ModelExport(exportConfig,
        Gna2ModelExportComponentLegacySueCreekHeader,
        &bufferSueCreekHeader, &bufferSueCreekHeaderSize);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExport(LegacySueCreekHeader)");

    (*modelHeader) = *(reinterpret_cast<Gna2ModelSueCreekHeader*>(bufferSueCreekHeader));

    void * bufferDump = nullptr;
    uint32_t bufferDumpSize;
    status = Gna2ModelExport(exportConfig,
        Gna2ModelExportComponentLegacySueCreekDump,
        &bufferDump,
        &bufferDumpSize);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExport(LegacySueCreekDump)");

    status = Gna2ModelExportConfigRelease(exportConfig);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExportConfigRelease");

    gnaUserFree(bufferSueCreekHeader);
    return bufferDump;
}


void ExportLdForDeviceVersion(
    uint32_t modelId,
    std::ostream & outStream,
    const Gna2DeviceVersion deviceVersionToExport) {

    uint32_t exportConfig;
    auto status = Gna2ModelExportConfigCreate(gnaUserAllocatorAlignedPage, &exportConfig);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExportConfigCreate");

    status = Gna2ModelExportConfigSetSource(exportConfig, 0, modelId);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExportConfigSetSource");
    status = Gna2ModelExportConfigSetTarget(exportConfig, deviceVersionToExport);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExportConfigSetTarget");

    void * ldDump;
    uint32_t ldDumpSize;

    status = Gna2ModelExport(exportConfig,
        Gna2ModelExportComponentLayerDescriptors,
        &ldDump, &ldDumpSize);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExport(LayerDescriptors)");

    outStream.write(static_cast<char*>(ldDump), ldDumpSize);

    status = Gna2ModelExportConfigRelease(exportConfig);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExportConfigRelease");

    gnaUserFree(ldDump);
}

void ExportGnaDescriptorPartiallyFilled(uint32_t number_of_layers, std::ostream& outStream) {
    const uint32_t scratchPadSize = 0x2000;
    const auto constScratchFill = static_cast<char>(-1);
    const uint32_t gnaDescSize = 32;
    char gd[gnaDescSize] = {};
    char gd2[gnaDescSize] = {};
    gd[0] = 1;
    *reinterpret_cast<uint32_t *>(gd + 4) = number_of_layers;
    *reinterpret_cast<uint32_t *>(gd + 8) = 0xffffffff;
    *reinterpret_cast<uint32_t *>(gd + 0xC) = 2 * sizeof(gd) + scratchPadSize;
    outStream.write(gd, sizeof(gd));
    outStream.write(gd2, sizeof(gd2));
    // TODO: GNA2: Scratchpad
    auto previousCharacter = outStream.fill(constScratchFill);
    auto previousWidth = outStream.width(scratchPadSize);
    outStream << constScratchFill;
    outStream.width(previousWidth);
    outStream.fill(previousCharacter);
}
