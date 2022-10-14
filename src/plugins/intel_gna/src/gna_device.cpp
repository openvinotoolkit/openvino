// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_device.hpp"

#include <map>
#include <string>
#include <cstring>
#include <mutex>
#include <vector>
#include <fstream>

#include "gna2-capability-api.h"
#include "gna2-device-api.h"
#include "gna2-inference-api.h"
#include "gna2-instrumentation-api.h"
#include "gna2-memory-api.h"
#include "gna2-model-export-api.h"
#include "gna2_model_export_helper.hpp"
#include "gna2_model_helper.hpp"
#include "gna2_model_debug_log.hpp"

#include "backend/am_intel_dnn.hpp"
#include "backend/gna_limitations.hpp"
#include "common/gna_target.hpp"
#include "gna/gna_config.hpp"
#include "gna_plugin_log.hpp"
#include "layers/gna_convolution_layer.hpp"
#include "memory/gna_mem_requests.hpp"


std::mutex GNADeviceHelper::acrossPluginsSync{};

GNADeviceHelper::GNADeviceHelper(std::string executionTargetIn,
                                 std::string compileTargetIn,
                                 bool swExactModeIn,
                                 bool isPerformanceMeasuring,
                                 bool deviceEmbedded)
    : swExactMode(swExactModeIn),
      executionTarget(executionTargetIn),
      compileTarget(compileTargetIn),
      isPerformanceMeasuring(isPerformanceMeasuring),
      nGnaDeviceIndex{selectGnaDevice()},
      useDeviceEmbeddedExport(deviceEmbedded) {
    open();
    initGnaPerfCounters();

    // check GNA Library version
    GetGnaLibraryVersion();

    maxLayersCount_ = retrieveMaxLayersCount();
}

GNADeviceHelper ::~GNADeviceHelper() {
    if (deviceOpened) {
        close();
    }
}


uint8_t* GNADeviceHelper::alloc(uint32_t size_requested, uint32_t *size_granted) {
    std::unique_lock<std::mutex> lockGnaCalls{ acrossPluginsSync };
    void * memPtr = nullptr;
    const auto status = Gna2MemoryAlloc(size_requested, size_granted, &memPtr);
    checkGna2Status(status, "Gna2MemoryAlloc");

    gnalog() << "Gna2MemoryAlloc(" << size_requested << ") -> " << *size_granted << ", " << memPtr << "\n";
    allAllocations.Add(memPtr, size_requested, *size_granted);
    if (memPtr == nullptr) {
        THROW_GNA_EXCEPTION << "GNAAlloc failed to allocate memory. Requested: " << size_requested << " Granted: " << *(size_granted);
    }

    dumpXNNROPtr = memPtr;
    dumpXNNROSize = *size_granted;
    return static_cast<uint8_t *>(memPtr);
}

void GNADeviceHelper::tagMemoryRegion(void* memPtr, const GNAPluginNS::memory::rRegion tag) {
    std::unique_lock<std::mutex> lockGnaCalls{ acrossPluginsSync };
    using GNAPluginNS::memory::rRegion;
    static const std::map<rRegion, Gna2MemoryTag> tagMap {
        {rRegion::REGION_INPUTS, Gna2MemoryTagInput},
        {rRegion::REGION_OUTPUTS, Gna2MemoryTagOutput},
        {rRegion::REGION_SCRATCH, Gna2MemoryTagScratch},
        {rRegion::REGION_RO, Gna2MemoryTagReadOnly},
        {rRegion::REGION_STATES, Gna2MemoryTagState},
        {rRegion::REGION_AUTO, Gna2MemoryTagState},
    };
    auto memoryTag = tagMap.at(tag);
    if (tag == rRegion::REGION_AUTO) {
        return;
    }
    const auto status = Gna2MemorySetTag(memPtr, memoryTag);
    checkGna2Status(status, "Gna2MemorySetTag");
    gnalog() << "Gna2MemorySetTag(" << memPtr << ", " << memoryTag << ")\n";
    const auto tagSuccess = allAllocations.SetTagFor(memPtr, memoryTag);
    if (!tagSuccess) {
        THROW_GNA_EXCEPTION << "Allocation not found when tagging memory\n";
    }
}

void GNADeviceHelper::free(void* ptr) {
    Gna2Status status;
    bool removeSuccess;
    std::string message;
    {
        std::unique_lock<std::mutex> lockGnaCalls{acrossPluginsSync};
        status = Gna2MemoryFree(ptr);
        message = checkGna2Status(status, "Gna2MemoryFree", true);
        removeSuccess = allAllocations.Remove(ptr);
    }
    if (!message.empty()) {
        gnawarn() << message;
    }
    if (!removeSuccess) {
        gnawarn() << "Allocation not found when freeing memory\n";
    }
}

std::string GNADeviceHelper::getGnaLibraryVersionPrivate() {
    char buffer[64] = {};
    const auto status = Gna2GetLibraryVersion(buffer, sizeof(buffer));
    if (status != Gna2StatusSuccess) {
        return "2.Gna2GetLibraryVersionReturned[" + std::to_string(status) + "]";
    }
    return buffer;
}

std::string GNADeviceHelper::GetGnaLibraryVersion() {
    static std::string gnaLibraryVersion{ getGnaLibraryVersionPrivate() };
    return gnaLibraryVersion;
}

void GNADeviceHelper::enableDiagnostics() {
    debugLogEnabled = true;
}

void GNADeviceHelper::dumpAllAllocations(const uint64_t idx, const std::string& infix) const {
    for (auto&& a : allAllocations.GetAllocationsInExportOrder()) {
        const auto& name = a.GetTagName();
        const auto filename = std::to_string(idx) + name + kDumpDelimiter + infix + kDumpExt;
        std::ofstream file(filename, std::ios::out | std::ios::binary);
        if (file) {
            file.write(static_cast<char*>(a.ptr), a.sizeGranted);
        } else {
            gnawarn() << "Can not dump memory region, file not created: '" << filename << "'\n";
        }
    }
}

uint32_t GNADeviceHelper::enqueueRequest(const uint32_t requestConfigID, Gna2AccelerationMode gna2AccelerationMode) {
    std::unique_lock<std::mutex> lockGnaCalls{ acrossPluginsSync };
    uint32_t reqId{};
    if ((gna2AccelerationMode == Gna2AccelerationModeHardware ||
         gna2AccelerationMode == Gna2AccelerationModeHardwareWithSoftwareFallback) &&
        detectedGnaDevVersion == Gna2DeviceVersionSoftwareEmulation) {
        gnawarn() << "GNA Device not detected, consider using other mode of acceleration";
    }

    const auto status1 = Gna2RequestConfigSetAccelerationMode(requestConfigID, gna2AccelerationMode);
    checkGna2Status(status1, "Gna2RequestConfigSetAccelerationMode");

    if (debugLogEnabled) {
        dumpAllAllocations(debugLogIndexRequestEnqueue, "BeforeGna2RequestEnqueue");
        debugLogIndexRequestEnqueue++;
    }

    const auto status2 = Gna2RequestEnqueue(requestConfigID, &reqId);
    checkGna2Status(status2, "Gna2RequestEnqueue");

    unwaitedRequestIds.insert(reqId);

    return reqId;
}

inline void enforceLegacyCnn(Gna2Operation& operation) {
    snprintf(
        const_cast<char*>(operation.Operands[1]->Layout),
        sizeof(operation.Operands[1]->Layout) / sizeof(char),
        "GNA1");
}

void GNADeviceHelper::enforceLegacyCnns(Gna2Model& gnaModel) {
    for (uint32_t i = 0; i < gnaModel.NumberOfOperations; i++) {
        if (gnaModel.Operations[i].Type == Gna2OperationTypeConvolution) {
            enforceLegacyCnn(gnaModel.Operations[i]);
        }
    }
}

void GNADeviceHelper::enforceLegacyCnnsWhenNeeded(Gna2Model& gnaModel) {
    for (uint32_t i = 0; i < gnaModel.NumberOfOperations; i++) {
        auto& op = gnaModel.Operations[i];
        if (GNAPluginNS::backend::AMIntelDNN::isOperationCnnLegacySpecific(op)) {
            enforceLegacyCnn(op);
        }
    }
}

uint32_t GNADeviceHelper::createModel(Gna2Model& gnaModel) const {
    std::unique_lock<std::mutex> lockGnaCalls{ acrossPluginsSync };
    uint32_t modelId = 0;
    const auto legacyExecTarget = enforceLegacyCnnNeeded();
    if (legacyExecTarget) {
        enforceLegacyCnns(gnaModel);
    }
    enforceLegacyCnnsWhenNeeded(gnaModel);

    GNAPluginNS::backend::AMIntelDNN::updateNumberOfOutputsIfPoolingEnabled(gnaModel, legacyExecTarget);

    if (debugLogEnabled) {
        std::string path =
#ifdef _WIN32
            ".\\";
#else
            "./";
#endif
        const std::string mode = useDeviceEmbeddedExport ? "_ee" : "";
        const auto fileSuffix = mode + "_devVersion_" + toHexString(detectedGnaDevVersion);
        DumpGna2Model(gnaModel, path, false, allAllocations, fileSuffix);
    }

    const auto status = Gna2ModelCreate(nGnaDeviceIndex, &gnaModel, &modelId);

    checkGna2Status(status, gnaModel);
    return modelId;
}

void GNADeviceHelper::releaseModel(const uint32_t model_id) {
    std::unique_lock<std::mutex> lockGnaCalls{ acrossPluginsSync };
    const auto status = Gna2ModelRelease(model_id);
    checkGna2Status(status, "Gna2ModelRelease");
}

bool GNADeviceHelper::enforceLegacyCnnNeeded() const {
    const auto execTargetDevice = getTargetDevice(true);
    return isUpTo20HwGnaDevice(execTargetDevice);
}

Gna2DeviceVersion GNADeviceHelper::parseTarget(const std::string& target) {
    static const std::map<std::string, Gna2DeviceVersion> targetMap {
        {GNAPluginNS::common::kGnaTarget2_0, Gna2DeviceVersion2_0},
        {GNAPluginNS::common::kGnaTarget3_0, Gna2DeviceVersion3_0},
        {GNAPluginNS::common::kGnaTargetUnspecified, Gna2DeviceVersionSoftwareEmulation},
    };
    const auto f = targetMap.find(target);
    if (f != targetMap.end()) {
        return f->second;
    }
    THROW_GNA_EXCEPTION << "Unsupported " << "GNA target: \"" << target << "\"\n";
}

Gna2DeviceVersion GNADeviceHelper::getDefaultTarget() const {
    if (detectedGnaDevVersion == Gna2DeviceVersionSoftwareEmulation)
        return Gna2DeviceVersion3_0;
    return detectedGnaDevVersion;
}

Gna2DeviceVersion GNADeviceHelper::getTargetDevice(const bool execTarget) const {
    const auto declared = execTarget ? executionTarget : compileTarget;
    if (declared == GNAPluginNS::common::kGnaTargetUnspecified) {
        return execTarget ? getDefaultTarget() : getTargetDevice(true);
    }
    return parseTarget(declared);
}

uint32_t GNADeviceHelper::createRequestConfig(const uint32_t modelID) const {
    std::unique_lock<std::mutex> lockGnaCalls{ acrossPluginsSync };
    uint32_t reqConfId = 0;
    auto status = Gna2RequestConfigCreate(modelID, &reqConfId);
    checkGna2Status(status, "Gna2RequestConfigCreate");

    status = Gna2InstrumentationConfigAssignToRequestConfig(instrumentationConfigId, reqConfId);
    checkGna2Status(status, "Gna2InstrumentationConfigAssignToRequestConfig");

    return reqConfId;
}

uint32_t GNADeviceHelper::getNumberOfGnaDevices() {
    std::unique_lock<std::mutex> lockGnaCalls{ acrossPluginsSync };
    uint32_t numberOfGnaDevices = 0;
    auto status = Gna2DeviceGetCount(&numberOfGnaDevices);
    checkGna2Status(status, "Gna2DeviceGetCount");
    return numberOfGnaDevices;
}

uint32_t GNADeviceHelper::selectGnaDevice() {
    const auto deviceCount = getNumberOfGnaDevices();
    if (deviceCount != 1) {
        THROW_GNA_EXCEPTION << "Unsupported number of GNA devices detected = " << deviceCount;
    }
    return 0;
}

void GNADeviceHelper::checkGna2Status(Gna2Status status, const Gna2Model& gnaModel) {
    if (!Gna2StatusIsSuccessful(status)) {
        std::vector<char> gna2StatusBuffer(1024);
        const auto s = Gna2StatusGetMessage(status, gna2StatusBuffer.data(), static_cast<uint32_t>(gna2StatusBuffer.size()));
        if (!Gna2StatusIsSuccessful(s))
            snprintf(gna2StatusBuffer.data(), gna2StatusBuffer.size(), "Gna2StatusGetMessage(%d) returned (%d)",
                static_cast<int>(status), static_cast<int>(s));
        if (status == Gna2StatusDeviceIngoingCommunicationError ||
            status == Gna2StatusDeviceOutgoingCommunicationError) {
            THROW_GNA_EXCEPTION << "Unsuccessful Gna2Status: (" << status << ") " <<
                gna2StatusBuffer.data() << ", consider updating the GNA driver" <<
                decoratedGnaLibVersion();
        }

        Gna2ModelError error{};
        auto getLastErrorStatus = Gna2ModelGetLastError(&error);
        checkGna2Status(getLastErrorStatus, "Gna2ModelGetLastError");

        std::stringstream ss;
        ss << "\n GNA Library Error:\n";
        const Gna2ItemType type = error.Source.Type;
        const std::string errorType = errorTypes.find(type) != errorTypes.end()
                                      ? errorTypes.at(type)
                                      : "Unknown Error Type";

        ss << "   Type (" << std::to_string(type) << "): " << errorType << "\n";

        if (error.Source.OperationIndex != GNA2_DISABLED) {
            const Gna2OperationType opTypeIndex = gnaModel.Operations[error.Source.OperationIndex].Type;
            const std::string operationType = operationTypes.find(opTypeIndex) != operationTypes.end()
                                              ? operationTypes.at(opTypeIndex)
                                              : "Unknown Operation Type";
            const std::string operandType = operandTypes.find({ opTypeIndex, error.Source.OperandIndex }) != operandTypes.end()
                                              ? operandTypes.at({ opTypeIndex, error.Source.OperandIndex })
                                              : "Unknown Operand Type";

            ss << "   OperationIndex (" << std::to_string(error.Source.OperationIndex) << "): "
                << operationType << "\n";
            ss << "   OperandIndex(" << std::to_string(error.Source.OperandIndex) << "): "
                << operandType << "\n";
            ss << "   ParamIndex (" << std::to_string(error.Source.ParameterIndex) << ")\n";
            ss << "   DimIndex (" << std::to_string(error.Source.ShapeDimensionIndex) << ")\n";
        }

        const Gna2ErrorType reason = error.Reason;
        const std::string errorReason = errorReasons.find(reason) != errorReasons.end()
                                        ? errorReasons.at(reason)
                                        : "Unknown Error Reason";
        ss << "   Reason (" << std::to_string(reason) << "): " << errorReason << "\n";
        ss << "   Value (0x" << std::hex << error.Value << ")";

        THROW_GNA_EXCEPTION << "\nUnsuccessful Gna2Status: (" << status << ") " <<
            gna2StatusBuffer.data() << ss.str() <<
            decoratedGnaLibVersion();
    }
}

std::string GNADeviceHelper::checkGna2Status(Gna2Status status,
                                             const std::string& from,
                                             bool returnInsteadThrow) {
    if (!Gna2StatusIsSuccessful(status)) {
        std::vector<char> gna2StatusBuffer(1024);
        const auto prefix = "Unsuccessful " + from + " call, Gna2Status: (";
        const auto s = Gna2StatusGetMessage(status, gna2StatusBuffer.data(), static_cast<uint32_t>(gna2StatusBuffer.size()));
        if (!Gna2StatusIsSuccessful(s))
            snprintf(gna2StatusBuffer.data(), gna2StatusBuffer.size(), "Gna2StatusGetMessage(%d) returned (%d)",
                static_cast<int>(status), static_cast<int>(s));
        std::string suffix;
        if (status == Gna2StatusDeviceIngoingCommunicationError ||
            status == Gna2StatusDeviceOutgoingCommunicationError) {
            suffix = ", consider updating the GNA driver";
        }
        std::ostringstream message;
        message << prefix << status << ") " << gna2StatusBuffer.data() << suffix <<
            decoratedGnaLibVersion();
        if (returnInsteadThrow) {
            return message.str();
        }
        THROW_GNA_EXCEPTION << message.str();
    }
    return {};
}

const std::map <Gna2ItemType, const std::string> GNADeviceHelper::errorTypes = {
            {Gna2ItemTypeNone, "Model context is not applicable or unnecessary"},
            {Gna2ItemTypeModelNumberOfOperations, "Gna2Model::NumberOfOperations"},
            {Gna2ItemTypeModelOperations, "Gna2Model::Operations array"},
            {Gna2ItemTypeOperationType, "Gna2Model::Operations[x]->Gna2Operation::Type"},
            {Gna2ItemTypeOperationOperands, "Gna2Model::Operations[x]->Gna2Operation::Operands array"},
            {Gna2ItemTypeOperationNumberOfOperands, "Gna2Model::Operations[x]->Gna2Operation::NumberOfOperands"},
            {Gna2ItemTypeOperationParameters, "Gna2Model::Operations[x]->Gna2Operation::Parameters array"},
            {Gna2ItemTypeOperationNumberOfParameters, "Gna2Model::Operations[x]->Gna2Operation::NumberOfParameters"},
            {Gna2ItemTypeOperandMode, "Gna2Model::Operations[x]->Gna2Operation::Operands[y]->Gna2Tensor::Mode"},
            {Gna2ItemTypeOperandLayout, "Gna2Model::Operations[x]->Gna2Operation::Operands[y]->Gna2Tensor::Layout"},
            {Gna2ItemTypeOperandType, "Gna2Model::Operations[x]->Gna2Operation::Operands[y]->Gna2Tensor::Type"},
            {Gna2ItemTypeOperandData, "Gna2Model::Operations[x]->Gna2Operation::Operands[y]->Gna2Tensor::Data"},
            {Gna2ItemTypeParameter, "Gna2Model::Operations[x]->Gna2Operation::Parameters[z]->Parameter, can be of type Gna2Shape, enumeration or integer"},
            {Gna2ItemTypeShapeNumberOfDimensions, "Gna2Model::Operations[x]->{Gna2Tensor}, Parameter}->Gna2Shape::NumberOfDimensions"},
            {Gna2ItemTypeShapeDimensions, "Gna2Model::Operations[x]->{Gna2Tensor}, Parameter}->Gna2Shape::Dimensions"},
            {Gna2ItemTypeInternal, "Internal model item, that is a derivative of other model parameters"}
};

const std::map <Gna2ErrorType, const std::string> GNADeviceHelper::errorReasons = {
            { Gna2ErrorTypeNone, "No error detected"},
            { Gna2ErrorTypeNotTrue, "Item value was expected to be true"},
            { Gna2ErrorTypeNotFalse, "Item value was expected to be false"},
            { Gna2ErrorTypeNullNotAllowed, "Item value was expected to be not null"},
            { Gna2ErrorTypeNullRequired, "Item value was expected to be null"},
            { Gna2ErrorTypeBelowRange, "Item value was below supported range"},
            { Gna2ErrorTypeAboveRange, "Item value was above supported range"},
            { Gna2ErrorTypeNotEqual, "Item value was not equal supported one"},
            { Gna2ErrorTypeNotGtZero, "Item value was below zero"},
            { Gna2ErrorTypeNotZero, "Item value was not equal zero"},
            { Gna2ErrorTypeNotOne, "Item value was not equal one"},
            { Gna2ErrorTypeNotInSet, "Item value was not in supported set of values"},
            { Gna2ErrorTypeNotMultiplicity, "Item value was not multiple of supported value"},
            { Gna2ErrorTypeNotSuccess, "Item value was invalid, no detailed information available"},
            { Gna2ErrorTypeNotAligned, "Item value was not aligned to supported value"},
            { Gna2ErrorTypeArgumentMissing, "Some operation argument was not provided"},
            { Gna2ErrorTypeArgumentInvalid, "Given operation argument was invalid or unexpected"},
            { Gna2ErrorTypeRuntime, "Runtime error occurred during model creation"},
            { Gna2ErrorTypeOther, "Unable to determine the root cause of the issue"}
};

const std::map <Gna2OperationType, const std::string> GNADeviceHelper::operationTypes = {
            { Gna2OperationTypeNone, "None"},
            { Gna2OperationTypeConvolution, "Convolution"},
            { Gna2OperationTypeCopy, "Copy"},
            { Gna2OperationTypeFullyConnectedAffine, "FullyConnectedAffine"},
            { Gna2OperationTypeElementWiseAffine, "ElementWiseAffine"},
            { Gna2OperationTypeGmm, "GMM"},
            { Gna2OperationTypeRecurrent, "Recurrent"},
            { Gna2OperationTypeTransposition, "Transpose"},
            { Gna2OperationTypeThreshold, "Threshold"}
};

const std::map <const std::pair<Gna2OperationType, int32_t>, const std::string> GNADeviceHelper::operandTypes = {
            {{Gna2OperationTypeConvolution, 0}, "Input"},
            {{Gna2OperationTypeConvolution, 1}, "Output"},
            {{Gna2OperationTypeConvolution, 2}, "Filters"},
            {{Gna2OperationTypeConvolution, 3}, "Biases"},
            {{Gna2OperationTypeConvolution, 4}, "Activation"},
            {{Gna2OperationTypeCopy, 0}, "Input"},
            {{Gna2OperationTypeCopy, 1}, "Output"},
            {{Gna2OperationTypeFullyConnectedAffine, 0}, "Input"},
            {{Gna2OperationTypeFullyConnectedAffine, 1}, "Output"},
            {{Gna2OperationTypeFullyConnectedAffine, 2}, "Weights"},
            {{Gna2OperationTypeFullyConnectedAffine, 3}, "Biases"},
            {{Gna2OperationTypeFullyConnectedAffine, 4}, "Activation"},
            {{Gna2OperationTypeFullyConnectedAffine, 5}, "WeightScaleFactors"},
            {{Gna2OperationTypeElementWiseAffine, 0}, "Input"},
            {{Gna2OperationTypeElementWiseAffine, 1}, "Output"},
            {{Gna2OperationTypeElementWiseAffine, 2}, "Weights"},
            {{Gna2OperationTypeElementWiseAffine, 3}, "Biases"},
            {{Gna2OperationTypeElementWiseAffine, 4}, "Activation"},
            {{Gna2OperationTypeGmm, 0}, "Input"},
            {{Gna2OperationTypeGmm, 1}, "Output"},
            {{Gna2OperationTypeGmm, 2}, "Means"},
            {{Gna2OperationTypeGmm, 3}, "InverseCovariances"},
            {{Gna2OperationTypeGmm, 4}, "Constants"},
            {{Gna2OperationTypeRecurrent, 0}, "Input"},
            {{Gna2OperationTypeRecurrent, 1}, "Output"},
            {{Gna2OperationTypeRecurrent, 2}, "Weights"},
            {{Gna2OperationTypeRecurrent, 3}, "Biases"},
            {{Gna2OperationTypeRecurrent, 4}, "Activation"},
            {{Gna2OperationTypeTransposition, 0}, "Input"},
            {{Gna2OperationTypeTransposition, 1}, "Output"},
            {{Gna2OperationTypeThreshold, 0}, "Input"},
            {{Gna2OperationTypeThreshold, 1}, "Output"}
};

GNAPluginNS::RequestStatus GNADeviceHelper::waitForRequest(uint32_t requestID, int64_t timeoutMilliseconds) {
    std::unique_lock<std::mutex> lockGnaCalls{ acrossPluginsSync };
    const auto status = Gna2RequestWait(requestID, static_cast<uint32_t>(timeoutMilliseconds));
    if (status == Gna2StatusWarningDeviceBusy) {
        return GNAPluginNS::RequestStatus::kPending;
    }
    unwaitedRequestIds.erase(requestID);
    if (status == Gna2StatusDriverQoSTimeoutExceeded) {
        return GNAPluginNS::RequestStatus::kAborted;
    }
    checkGna2Status(status, "Gna2RequestWait");

    if (debugLogEnabled) {
        dumpAllAllocations(debugLogIndexRequestWait, "AfterGna2RequestWait");
        debugLogIndexRequestWait++;
    }
    updateGnaPerfCounters();
    return GNAPluginNS::RequestStatus::kCompleted;
}

GNADeviceHelper::DumpResult GNADeviceHelper::dumpXnn(const uint32_t modelId) {
    DumpResult r;

    r.model.reset(
        ExportSueLegacyUsingGnaApi2(modelId, nGnaDeviceIndex, &r.header),
        gnaUserFree);

    if (r.model == nullptr) {
        THROW_GNA_EXCEPTION << "GNADumpXnn returned nullptr";
    }

    return r;
}

void GNADeviceHelper::dumpTLVForDeviceVersion(const uint32_t modelId,
                                              std::ostream& outStream,
                                              const std::vector<GnaEndpoint>& inputsContainer,
                                              const std::vector<GnaEndpoint>& outputsContainer) {
    const auto compileTarget = GetCompileTarget();
    ExportTlvModel(modelId,
                   nGnaDeviceIndex,
                   outStream,
                   compileTarget,
                   inputsContainer,
                   outputsContainer,
                   allAllocations);
}

void GNADeviceHelper::createVirtualDevice(Gna2DeviceVersion devVersion) {
    const auto status = Gna2DeviceCreateForExport(devVersion, &nGnaDeviceIndex);
    GNADeviceHelper::checkGna2Status(status, "Gna2DeviceCreateForExport(" + std::to_string(devVersion) + ")");
}

void GNADeviceHelper::updateGnaDeviceVersion() {
    const auto status = Gna2DeviceGetVersion(nGnaDeviceIndex, &detectedGnaDevVersion);
    checkGna2Status(status, "Gna2DeviceGetVersion");
}

void GNADeviceHelper::open() {
    std::unique_lock<std::mutex> lockGnaCalls{ acrossPluginsSync };
    updateGnaDeviceVersion();
    const auto gnaExecTarget = parseTarget(executionTarget);
    if (useDeviceEmbeddedExport) {
        const auto gnaCompileTarget = GetCompileTarget();
        const auto exportGeneration = getEmbeddedTargetFromCompileTarget(gnaCompileTarget);

        createVirtualDevice(exportGeneration);
        updateGnaDeviceVersion();
    } else if (!executionTarget.empty() && gnaExecTarget != detectedGnaDevVersion) {
        createVirtualDevice(gnaExecTarget);
        updateGnaDeviceVersion();
        if (detectedGnaDevVersion != gnaExecTarget) {
            THROW_GNA_EXCEPTION << "Wrong virtual GNA device version reported: " << detectedGnaDevVersion << " instead of: " << gnaExecTarget;
        }
    } else {
        const auto status = Gna2DeviceOpen(nGnaDeviceIndex);
        checkGna2Status(status, "Gna2DeviceOpen");
    }
    deviceOpened = true;
}

void GNADeviceHelper::close() {
    auto requestsToClose = unwaitedRequestIds;
    for (auto requestId : requestsToClose) {
        try {
            waitForRequest(requestId);
        } catch (...) {
            gnawarn() << "Request with Id " << requestId << " was not awaited successfully";
        }
    }
    std::unique_lock<std::mutex> lockGnaCalls{ acrossPluginsSync };
    const auto status = Gna2DeviceClose(nGnaDeviceIndex);
    const auto message = checkGna2Status(status, "Gna2DeviceClose", true);
    if (!message.empty()) {
        gnawarn() << "GNA Device was not successfully closed: " << message << std::endl;
    }
    deviceOpened = false;
}

void GNADeviceHelper::updateGnaPerfCounters() {
    if (!isPerformanceMeasuring)
        return;
    instrumentationTotal[0] = instrumentationResults[0];
    instrumentationTotal[1] = instrumentationResults[1];
    instrumentationResults[0] = 0;
    instrumentationResults[1] = 0;
}

void GNADeviceHelper::getGnaPerfCounters(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& retPerfCounters) {
    InferenceEngine::InferenceEngineProfileInfo info;
    info.status = InferenceEngine::InferenceEngineProfileInfo::EXECUTED;
    info.cpu_uSec = 0;
    info.execution_index = 0;
    info.realTime_uSec = 0;
    // Hardware
    info.realTime_uSec = instrumentationTotal[0];
    retPerfCounters["1.1 Total scoring time in HW"] = info;
    info.realTime_uSec = instrumentationTotal[1];
    retPerfCounters["1.2 Stall scoring time in HW"] = info;
}

std::string GNADeviceHelper::GetCompileTarget() const {
    static const std::map<Gna2DeviceVersion, std::string> targetMap = {
        {Gna2DeviceVersion2_0, GNAPluginNS::common::kGnaTarget2_0},
        {Gna2DeviceVersion3_0, GNAPluginNS::common::kGnaTarget3_0},
    };
    const auto target = getTargetDevice(false);
    auto found = targetMap.find(target);
    if (found == targetMap.end()) {
        THROW_GNA_EXCEPTION << "Unknown target Gna2DeviceVersion == " << target;
    }
    return found->second;
}

uint32_t GNADeviceHelper::maxLayersCount() const {
    return maxLayersCount_;
}

uint32_t GNADeviceHelper::retrieveMaxLayersCount() {
    using namespace GNAPluginNS::GNALimitations;

    switch (getTargetDevice(true)) {
    case Gna2DeviceVersion1_0:
        return kMaxLayersCountGNA1_0;
    case Gna2DeviceVersion2_0:
        return kMaxLayersCountGNA2_0;
    case Gna2DeviceVersion3_0:
    case Gna2DeviceVersion3_5:
    default:
        return kMaxLayersCountGNA3_X;
    }
}
