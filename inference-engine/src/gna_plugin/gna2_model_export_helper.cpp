// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if GNA_LIB_VER == 2
#include "gna2_model_export_helper.hpp"
#include "gna2_model_helper.hpp"
#include "gna_device.hpp"
#include "gna2-model-export-api.h"
#include "gna2-model-suecreek-header.h"
#include "gna_api_wrapper.hpp"

#include <cstdint>
#include <fstream>

void * ExportSueLegacyUsingGnaApi2(
    uint32_t modelId,
    Gna2ModelSueCreekHeader* modelHeader) {

    uint32_t exportConfig;
    auto status = Gna2ModelExportConfigCreate(gnaUserAllocatorAlignedPage, &exportConfig);
    GNADeviceHelper::checkGna2Status(status);

    status = Gna2ModelExportConfigSetSource(exportConfig, 0, modelId);
    GNADeviceHelper::checkGna2Status(status);
    status = Gna2ModelExportConfigSetTarget(exportConfig, Gna2DeviceVersionEmbedded1_0);
    GNADeviceHelper::checkGna2Status(status);

    void * bufferSueCreekHeader;
    uint32_t bufferSueCreekHeaderSize;

    status = Gna2ModelExport(exportConfig,
        Gna2ModelExportComponentLegacySueCreekHeader,
        &bufferSueCreekHeader, &bufferSueCreekHeaderSize);
    GNADeviceHelper::checkGna2Status(status);

    (*modelHeader) = *(reinterpret_cast<Gna2ModelSueCreekHeader*>(bufferSueCreekHeader));

    void * bufferDump;
    uint32_t bufferDumpSize;
    status = Gna2ModelExport(exportConfig,
        Gna2ModelExportComponentLegacySueCreekDump,
        &bufferDump,
        &bufferDumpSize);
    GNADeviceHelper::checkGna2Status(status);

    status = Gna2ModelExportConfigRelease(exportConfig);
    GNADeviceHelper::checkGna2Status(status);

    gnaUserFree(bufferSueCreekHeader);
    return bufferDump;
}


void ExportLdForNoMmu(uint32_t modelId, std::ostream & outStream) {
    uint32_t exportConfig;
    auto status = Gna2ModelExportConfigCreate(gnaUserAllocatorAlignedPage, &exportConfig);
    GNADeviceHelper::checkGna2Status(status);

    status = Gna2ModelExportConfigSetSource(exportConfig, 0, modelId);
    GNADeviceHelper::checkGna2Status(status);
    status = Gna2ModelExportConfigSetTarget(exportConfig, Gna2DeviceVersionEmbedded3_0);
    GNADeviceHelper::checkGna2Status(status);

    void * ldNoMmu;
    uint32_t ldNoMmuSize;

    status = Gna2ModelExport(exportConfig,
        Gna2ModelExportComponentLayerDescriptors,
        &ldNoMmu, &ldNoMmuSize);
    GNADeviceHelper::checkGna2Status(status);

    outStream.write(static_cast<char*>(ldNoMmu), ldNoMmuSize);

    status = Gna2ModelExportConfigRelease(exportConfig);
    GNADeviceHelper::checkGna2Status(status);

    gnaUserFree(ldNoMmu);
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
    outStream.fill(constScratchFill);
    outStream.width(scratchPadSize);
    outStream << constScratchFill;
}

#endif
