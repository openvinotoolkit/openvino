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

#include <gna2-tlv-writer.h>

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

#define Gna2TlvTypeOVInputScaleFactor GNA2_TLV_IMPL_CHAR_TO_TYPE("OVIS")
#define Gna2TlvTypeOVOutputScaleFactor GNA2_TLV_IMPL_CHAR_TO_TYPE("OVOS")
static_assert(std::numeric_limits<float>::is_iec559, "Float is not IEC 559 compatible");
typedef std::array<char, sizeof(Gna2TlvRecord) + sizeof(float)> TlvFloatRecord;

namespace {
TlvFloatRecord GetFloatInTLV(Gna2TlvType type, float value) {
    TlvFloatRecord r;
    reinterpret_cast<Gna2TlvRecord*>(r.data())->type = type;
    reinterpret_cast<Gna2TlvRecord*>(r.data())->length = sizeof(float);
    *reinterpret_cast<float*>(r.data() + sizeof(Gna2TlvRecord)) = value;
    return r;
}
}  // namespace

void ExportTlvModel(uint32_t modelId,
    uint32_t deviceIndex,
    std::ostream& outStream,
    Gna2DeviceVersion deviceVersionToExport,
    uint32_t input_size,
    uint32_t output_size,
    float inputSF,
    float outputSF) {

    uint32_t exportConfig;
    auto status = Gna2ModelExportConfigCreate(gnaUserAllocatorAlignedPage, &exportConfig);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExportConfigCreate");

    status = Gna2ModelExportConfigSetSource(exportConfig, deviceIndex, modelId);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExportConfigSetSource");
    status = Gna2ModelExportConfigSetTarget(exportConfig, deviceVersionToExport);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExportConfigSetTarget");

    // first descriptors
    void* bufferLayerDescriptors = nullptr;;
    uint32_t sizeOfLayerDescriptors;

    status = Gna2ModelExport(exportConfig,
        Gna2ModelExportComponentLayerDescriptors,
        &bufferLayerDescriptors, &sizeOfLayerDescriptors);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExport(Gna2ModelExportComponentLayerDescriptors)");

    // RO
    void* bufferROData = nullptr;;
    uint32_t sizeOfROData;

    status = Gna2ModelExport(exportConfig,
        Gna2ModelExportComponentReadOnlyDump,
        &bufferROData, &sizeOfROData);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExport(Gna2ModelExportComponentReadOnlyDump)");

    // RW - scratch
    void* bufferScratchRWData = nullptr;;
    uint32_t sizeOfScratchRWData;

    status = Gna2ModelExport(exportConfig,
        Gna2ModelExportComponentScratchDump,
        &bufferScratchRWData, &sizeOfScratchRWData);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExport(Gna2ModelExportComponentScratchDump)");

    //TODO: This must be first cover by model creation code
    void* bufferStateRWData = nullptr;
    uint32_t sizeOfStateRWData = 0;


    // RW - state
    status = Gna2ModelExport(exportConfig,
        Gna2ModelExportComponentStateDump,
        &bufferStateRWData, &sizeOfStateRWData);
    if (!Gna2StatusIsSuccessful(status)) {
        bufferStateRWData = nullptr;
        sizeOfStateRWData = 0;
    }

    // RW - external Input
    void* bufferInputRWData = nullptr;
    uint32_t sizeOfInputRWData;
    status = Gna2ModelExport(exportConfig,
        Gna2ModelExportComponentInputDump,
        &bufferInputRWData, &sizeOfInputRWData);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExport(Gna2ModelExportComponentInputDump)");

    // RW - external Output
    void* bufferOutputRWData = nullptr;
    uint32_t sizeOfOutputRWData;
    status = Gna2ModelExport(exportConfig,
        Gna2ModelExportComponentOutputDump,
        &bufferOutputRWData, &sizeOfOutputRWData);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExport(Gna2ModelExportComponentOutputDump)");

    char* outTlv = nullptr;

    const auto gnaLibraryVersion = GNADeviceHelper::GetGnaLibraryVersion();

#ifdef ENABLE_GNA_TLV_EXPORT
    const char* userData = nullptr;
    uint32_t userDataSize = 0;
    uint32_t outTlvSize = 0;
    auto tlv_status = Gna2ExportTlv(
        deviceVersionToExport,
        gnaUserAllocator,
        &outTlv,
        &outTlvSize,
        (const char*)bufferLayerDescriptors,
        sizeOfLayerDescriptors,
        (const char*)bufferROData,
        sizeOfROData,
        (const char*)bufferStateRWData,
        sizeOfStateRWData,
        sizeOfScratchRWData,
        input_size,
        output_size,
        gnaLibraryVersion.c_str(),
        userData,
        userDataSize);

    if (Gna2TlvStatusSuccess == tlv_status) {
        outStream.write(outTlv, outTlvSize);
        auto tlvInSF = GetFloatInTLV(Gna2TlvTypeOVInputScaleFactor, inputSF);
        auto tlvOutSF = GetFloatInTLV(Gna2TlvTypeOVOutputScaleFactor, outputSF);
        outStream.write(tlvInSF.data(), tlvInSF.size());
        outStream.write(tlvOutSF.data(), tlvOutSF.size());
    }
#endif
    gnaUserFree(outTlv);

    gnaUserFree(bufferLayerDescriptors);
    gnaUserFree(bufferROData);
    gnaUserFree(bufferScratchRWData);
    gnaUserFree(bufferStateRWData);

    gnaUserFree(bufferInputRWData);
    gnaUserFree(bufferOutputRWData);

    GNADeviceHelper::checkGna2Status((Gna2Status)status, "ExportTlvModel");
    status = Gna2ModelExportConfigRelease(exportConfig);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExportConfigRelease");
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

    void * ldDump = nullptr;
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
