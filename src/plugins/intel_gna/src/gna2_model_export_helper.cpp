// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna2_model_export_helper.hpp"

#include <cstdint>
#include <fstream>
#include <numeric>

#include "common/versioning.hpp"
#include "gna/gna_config.hpp"
#include "gna2-device-api.h"
#include "gna2-model-export-api.h"
#include "gna2-model-suecreek-header.h"
#include "gna2-tlv-writer.h"
#include "gna2_model_helper.hpp"
#include "gna_device.hpp"
#include "log/log.hpp"

namespace ov {
namespace intel_gna {
using namespace common;

#define Gna2TlvTypeOVInputScaleFactor  GNA2_TLV_IMPL_CHAR_TO_TYPE("OVIS")
#define Gna2TlvTypeOVOutputScaleFactor GNA2_TLV_IMPL_CHAR_TO_TYPE("OVOS")
#define Gna2TlvTypeOVString            GNA2_TLV_IMPL_CHAR_TO_TYPE("OVSS")
#define Gna2TlvTypeOVVersion           GNA2_TLV_IMPL_CHAR_TO_TYPE("OVVR")

static_assert(std::numeric_limits<float>::is_iec559, "Float is not IEC 559 compatible");
typedef std::array<char, sizeof(Gna2TlvRecord) + sizeof(float)> TlvFloatRecord;

static TlvFloatRecord GetFloatInTLV(Gna2TlvType type, float value) {
    TlvFloatRecord r;
    reinterpret_cast<Gna2TlvRecord*>(r.data())->type = type;
    reinterpret_cast<Gna2TlvRecord*>(r.data())->length = sizeof(float);
    *reinterpret_cast<float*>(r.data() + sizeof(Gna2TlvRecord)) = value;
    return r;
}

static std::vector<char> GetStringAsTlv(Gna2TlvType type, const std::string& s) {
    std::vector<char> record(sizeof(Gna2TlvRecord));
    reinterpret_cast<Gna2TlvRecord*>(record.data())->type = type;

    std::vector<char> vs(s.begin(), s.end());
    vs.resize(vs.size() + (4 - vs.size() % 4) % 4, 0);
    reinterpret_cast<Gna2TlvRecord*>(record.data())->length = vs.size();
    record.insert(record.end(), vs.begin(), vs.end());
    return record;
}

static std::string WriteAllEndpoints(std::ostream& outStream,
                                     const std::vector<GnaEndpoint>& allEndpoints,
                                     const Gna2TlvType sfTlvType,
                                     const GnaAllocation* allocation) {
    const std::string endPointType = sfTlvType == Gna2TlvTypeOVInputScaleFactor ? "Input" : "Output";

    if (allEndpoints.size() >= 1) {
        const auto scaleFactorTlv = GetFloatInTLV(sfTlvType, allEndpoints[0].scaleFactor);
        outStream.write(scaleFactorTlv.data(), scaleFactorTlv.size());
    }
    if (allEndpoints.size() != 1) {
        log::warning() << "Number of endpoints: " << allEndpoints.size() << " for " << endPointType << "\n";
    }

    std::stringstream stream;
    stream << "Endpoints for " << endPointType << ":\n";
    for (const auto& endpoint : allEndpoints) {
        stream << "name=[" << endpoint.name << "]\n";
        stream << "scaleFactor=[" << endpoint.scaleFactor << "]\n";
        stream << "byteSize=[" << endpoint.byteSize << "]\n";
        stream << "numberOfBytesPerElement=[" << endpoint.numberOfBytesPerElement << "]\n";
        if (allocation == nullptr) {
            stream << "allocation=[nullptr]\n";
        }
        if (endpoint.gnaPointer == nullptr) {
            stream << "gnaPointer=[nullptr]\n";
        }
        if (allocation != nullptr && endpoint.gnaPointer != nullptr) {
            const auto gnaOffset = allocation->getOffset(endpoint.gnaPointer);
            if (!gnaOffset.first) {
                stream << "offset=[invalid]\n";
            }
            stream << "offset=[" << gnaOffset.second << "]\n";
        }
    }
    return stream.str();
}

static void WriteStringToTlv(std::ostream& outStream, const Gna2TlvType tlvType, const std::string& value) {
    const auto& valueTlv = GetStringAsTlv(tlvType, value);
    outStream.write(valueTlv.data(), valueTlv.size());
}

void ExportTlvModel(uint32_t modelId,
                    uint32_t deviceIndex,
                    std::ostream& outStream,
                    const DeviceVersion& compile_target,
                    const std::vector<GnaEndpoint>& allInputs,
                    const std::vector<GnaEndpoint>& allOutputs,
                    const GnaAllocations& allAllocations) {
    if (compile_target == DeviceVersion::GNAEmbedded1_0) {
        THROW_GNA_EXCEPTION << "Unsupported compile target for TLV export: GNA Embedded 1.0" << std::endl;
    }

    uint32_t exportConfig;
    auto status = Gna2ModelExportConfigCreate(gnaUserAllocatorAlignedPage, &exportConfig);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExportConfigCreate");

    status = Gna2ModelExportConfigSetSource(exportConfig, deviceIndex, modelId);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExportConfigSetSource");
    status = Gna2ModelExportConfigSetTarget(exportConfig, DeviceToGna(compile_target));
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExportConfigSetTarget");

    // first descriptors
    void* bufferLayerDescriptors = nullptr;
    uint32_t sizeOfLayerDescriptors;

    status = Gna2ModelExport(exportConfig,
                             Gna2ModelExportComponentLayerDescriptors,
                             &bufferLayerDescriptors,
                             &sizeOfLayerDescriptors);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExport(Gna2ModelExportComponentLayerDescriptors)");

    // RO
    void* bufferROData = nullptr;
    uint32_t sizeOfROData;

    status = Gna2ModelExport(exportConfig, Gna2ModelExportComponentReadOnlyDump, &bufferROData, &sizeOfROData);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExport(Gna2ModelExportComponentReadOnlyDump)");

    // RW - scratch
    void* bufferScratchRWData = nullptr;
    uint32_t sizeOfScratchRWData;

    status =
        Gna2ModelExport(exportConfig, Gna2ModelExportComponentScratchDump, &bufferScratchRWData, &sizeOfScratchRWData);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExport(Gna2ModelExportComponentScratchDump)");

    // TODO: This must be first cover by model creation code
    void* bufferStateRWData = nullptr;
    uint32_t sizeOfStateRWData = 0;

    // RW - state
    status = Gna2ModelExport(exportConfig, Gna2ModelExportComponentStateDump, &bufferStateRWData, &sizeOfStateRWData);
    if (!Gna2StatusIsSuccessful(status)) {
        bufferStateRWData = nullptr;
        sizeOfStateRWData = 0;
    }

    // RW - external Input
    void* bufferInputRWData = nullptr;
    uint32_t sizeOfInputRWData;
    status = Gna2ModelExport(exportConfig, Gna2ModelExportComponentInputDump, &bufferInputRWData, &sizeOfInputRWData);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExport(Gna2ModelExportComponentInputDump)");

    // RW - external Output
    void* bufferOutputRWData = nullptr;
    uint32_t sizeOfOutputRWData;
    status =
        Gna2ModelExport(exportConfig, Gna2ModelExportComponentOutputDump, &bufferOutputRWData, &sizeOfOutputRWData);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExport(Gna2ModelExportComponentOutputDump)");

    char* outTlv = nullptr;

    const auto gnaLibraryVersion = GNADeviceHelper::GetGnaLibraryVersion();

    uint32_t outTlvSize = 0;

    auto tlv_status = Gna2ExportTlv(DeviceToGna(compile_target),
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
                                    GnaEndpoint::GetTotalByteSize(allInputs),
                                    GnaEndpoint::GetTotalByteSize(allOutputs),
                                    gnaLibraryVersion.c_str(),
                                    nullptr,
                                    0);

    if (Gna2TlvStatusSuccess == tlv_status) {
        outStream.write(outTlv, outTlvSize);
        auto metadata = WriteAllEndpoints(outStream,
                                          allInputs,
                                          Gna2TlvTypeOVInputScaleFactor,
                                          allAllocations.Get(Gna2MemoryTagInput));
        metadata += WriteAllEndpoints(outStream,
                                      allOutputs,
                                      Gna2TlvTypeOVOutputScaleFactor,
                                      allAllocations.Get(Gna2MemoryTagOutput));
        WriteStringToTlv(outStream, Gna2TlvTypeOVString, metadata);
        const auto& ovVersionString = ov::intel_gna::get_openvino_version_string();
        WriteStringToTlv(outStream, Gna2TlvTypeOVVersion, ovVersionString);
    }

    gnaUserFree(outTlv);

    gnaUserFree(bufferLayerDescriptors);
    gnaUserFree(bufferROData);
    gnaUserFree(bufferScratchRWData);
    gnaUserFree(bufferStateRWData);

    gnaUserFree(bufferInputRWData);
    gnaUserFree(bufferOutputRWData);

    status = Gna2ModelExportConfigRelease(exportConfig);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExportConfigRelease");
    if (Gna2TlvStatusSuccess != tlv_status) {
        THROW_GNA_EXCEPTION << "Not succesfull status returned: " << tlv_status << ", from Gna2ExportTlv() function\n";
    }
}

void* ExportSueLegacyUsingGnaApi2(uint32_t modelId, uint32_t deviceIndex, Gna2ModelSueCreekHeader* modelHeader) {
    uint32_t exportConfig;
    auto status = Gna2ModelExportConfigCreate(gnaUserAllocatorAlignedPage, &exportConfig);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExportConfigCreate");

    status = Gna2ModelExportConfigSetSource(exportConfig, deviceIndex, modelId);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExportConfigSetSource");
    status = Gna2ModelExportConfigSetTarget(exportConfig, DeviceToGna(DeviceVersion::GNAEmbedded1_0));
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExportConfigSetTarget");

    void* bufferSueCreekHeader = nullptr;
    uint32_t bufferSueCreekHeaderSize;

    status = Gna2ModelExport(exportConfig,
                             Gna2ModelExportComponentLegacySueCreekHeader,
                             &bufferSueCreekHeader,
                             &bufferSueCreekHeaderSize);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExport(LegacySueCreekHeader)");

    (*modelHeader) = *(reinterpret_cast<Gna2ModelSueCreekHeader*>(bufferSueCreekHeader));

    void* bufferDump = nullptr;
    uint32_t bufferDumpSize;
    status = Gna2ModelExport(exportConfig, Gna2ModelExportComponentLegacySueCreekDump, &bufferDump, &bufferDumpSize);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExport(LegacySueCreekDump)");

    status = Gna2ModelExportConfigRelease(exportConfig);
    GNADeviceHelper::checkGna2Status(status, "Gna2ModelExportConfigRelease");

    gnaUserFree(bufferSueCreekHeader);
    return bufferDump;
}

}  // namespace intel_gna
}  // namespace ov
