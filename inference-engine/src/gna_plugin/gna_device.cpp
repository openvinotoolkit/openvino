// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_device.hpp"

#include <map>
#include <string>
#include <cstring>
#include <mutex>
#include <vector>

#if GNA_LIB_VER == 2
#include "gna_api_wrapper.hpp"
#include "gna2-capability-api.h"
#include "gna2-device-api.h"
#include "gna2-inference-api.h"
#include "gna2-instrumentation-api.h"
#include "gna2-memory-api.h"
#include "gna2_model_export_helper.hpp"
#include "gna2_model_debug_log.hpp"
#else
#include "gna-api-status.h"
#include "gna-api.h"
#endif

#include "gna/gna_config.hpp"
#include "gna_plugin_log.hpp"

//#define MODEL_DUMP

std::mutex GNADeviceHelper::acrossPluginsSync{};

uint8_t* GNADeviceHelper::alloc(uint32_t size_requested, uint32_t *size_granted) {
    std::unique_lock<std::mutex> lockGnaCalls{ acrossPluginsSync };
    void * memPtr = nullptr;
#if GNA_LIB_VER == 1
    memPtr = GNAAlloc(nGNAHandle, size_requested, size_granted);
#else
    const auto status = Gna2MemoryAlloc(size_requested, size_granted, &memPtr);
    checkGna2Status(status, "Gna2MemoryAlloc");
#endif
    if (memPtr == nullptr) {
        THROW_GNA_EXCEPTION << "GNAAlloc failed to allocate memory. Requested: " << size_requested << " Granted: " << *(size_granted);
    }
    dumpXNNROPtr = memPtr;
    dumpXNNROSize = *size_granted;
    return static_cast<uint8_t *>(memPtr);
}

void GNADeviceHelper::free(void * ptr) {
    std::unique_lock<std::mutex> lockGnaCalls{ acrossPluginsSync };
#if GNA_LIB_VER == 1
    GNAFree(nGNAHandle);
#else
    const auto status = Gna2MemoryFree(ptr);
    checkGna2Status(status, "Gna2MemoryFree");
#endif
}

std::string GNADeviceHelper::getGnaLibraryVersionPrivate() {
#if GNA_LIB_VER == 1
    return "1.X";
#else
    char buffer[64] = {};
    const auto status = Gna2GetLibraryVersion(buffer, sizeof(buffer));
    if (status != Gna2StatusSuccess) {
        return "2.Gna2GetLibraryVersionReturned[" + std::to_string(status) + "]";
    }
    return buffer;
#endif
}

std::string GNADeviceHelper::GetGnaLibraryVersion() {
    static std::string gnaLibraryVersion{ getGnaLibraryVersionPrivate() };
    return gnaLibraryVersion;
}

#if GNA_LIB_VER == 1
uint32_t GNADeviceHelper::propagate(const intel_nnet_type_t *pNeuralNetwork,
                   const uint32_t *pActiveIndices,
                   uint32_t nActiveIndices,
                   intel_gna_proc_t nGNAProcType) {
    std::unique_lock<std::mutex> lockGnaCalls{ acrossPluginsSync };
    uint32_t reqId;

    nGNAStatus = GNAPropagateForward(nGNAHandle, pNeuralNetwork,
                                     pActiveIndices, nActiveIndices, &reqId, nGNAProcType);
    checkStatus();
    return reqId;
}
#else

void GNADeviceHelper::setUpActiveList(const uint32_t requestConfigId, uint32_t layerIndex, uint32_t* ptr_active_indices, uint32_t num_active_indices) {
    std::unique_lock<std::mutex> lockGnaCalls{ acrossPluginsSync };
    const auto status = Gna2RequestConfigEnableActiveList(requestConfigId, layerIndex, num_active_indices, ptr_active_indices);
    checkGna2Status(status, "Gna2RequestConfigEnableActiveList");
}

uint32_t GNADeviceHelper::propagate(const uint32_t requestConfigId, Gna2AccelerationMode gna2AccelerationMode) {
    std::unique_lock<std::mutex> lockGnaCalls{ acrossPluginsSync };
    uint32_t reqId{};
    if ((gna2AccelerationMode == Gna2AccelerationModeHardware ||
         gna2AccelerationMode == Gna2AccelerationModeHardwareWithSoftwareFallback) &&
        detectedGnaDevVersion == Gna2DeviceVersionSoftwareEmulation) {
        gnawarn() << "GNA Device not detected, consider using other mode of acceleration";
    }
    const auto status1 = Gna2RequestConfigSetAccelerationMode(requestConfigId, gna2AccelerationMode);
    checkGna2Status(status1, "Gna2RequestConfigSetAccelerationMode");
    const auto status2 = Gna2RequestEnqueue(requestConfigId, &reqId);
    checkGna2Status(status2, "Gna2RequestEnqueue");

    unwaitedRequestIds.insert(reqId);

    return reqId;
}

void GNADeviceHelper::enforceLegacyCnns(Gna2Model& gnaModel) {
    for (uint32_t i = 0; i < gnaModel.NumberOfOperations; i++) {
        if (gnaModel.Operations[i].Type == Gna2OperationTypeConvolution) {
            snprintf(
                const_cast<char*>(gnaModel.Operations[i].Operands[1]->Layout),
                sizeof(gnaModel.Operations[i].Operands[1]->Layout) / sizeof(char),
                "GNA1");
        }
    }
}

uint32_t GNADeviceHelper::createModel(Gna2Model& gnaModel) const {
    std::unique_lock<std::mutex> lockGnaCalls{ acrossPluginsSync };
    uint32_t modelId;
    if (enforceLegacyCnnNeeded()) {
        enforceLegacyCnns(gnaModel);
    }
#if GNA_LIB_VER == 2 && defined MODEL_DUMP
    std::string path =
#ifdef _WIN32
        ".\\";
#else
        "./";
#endif
    DumpGna2Model(gnaModel, path, false);
#endif
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
    const auto compileTargetDevice = getTargetDevice(false);
    return isGnaLibVersion2_1 && isUpTo20HwGnaDevice(compileTargetDevice);
}

namespace {
    const volatile auto Gna2DeviceVersion3_0 = static_cast<Gna2DeviceVersion>(0x30);
} // namespace

Gna2DeviceVersion GNADeviceHelper::parseDeclaredTarget(std::string target, const bool execTarget) const {
    auto parsed = Gna2DeviceVersion2_0;
    auto throwUnsupportedGnaTarget = [&](std::string extraSuffix) {
        auto key = execTarget ? InferenceEngine::GNAConfigParams::KEY_GNA_EXEC_TARGET : InferenceEngine::GNAConfigParams::KEY_GNA_COMPILE_TARGET;
        THROW_GNA_EXCEPTION << "Unsupported " << key << " = \"" << target << "\"" << extraSuffix;
    };
    if (target == InferenceEngine::GNAConfigParams::GNA_TARGET_3_0) {
        if (!isGnaLibVersion2_1)
            throwUnsupportedGnaTarget(", when GNA Library version is 2.0.X.Y");
        parsed = Gna2DeviceVersion3_0;
    } else if (target != InferenceEngine::GNAConfigParams::GNA_TARGET_2_0) {
        throwUnsupportedGnaTarget("");
    }
    return parsed;
}

Gna2DeviceVersion GNADeviceHelper::getDefaultTarget() const {
    if (detectedGnaDevVersion == Gna2DeviceVersionSoftwareEmulation)
        return isGnaLibVersion2_1 ? Gna2DeviceVersion3_0 : Gna2DeviceVersion2_0;
    return detectedGnaDevVersion;
}

Gna2DeviceVersion GNADeviceHelper::getTargetDevice(const bool execTarget) const {
    const auto declared = execTarget ? executionTarget : compileTarget;
    if (declared.empty()) {
        return execTarget ? getDefaultTarget() : getTargetDevice(true);
    }
    return parseDeclaredTarget(declared, execTarget);
}

uint32_t GNADeviceHelper::createRequestConfig(const uint32_t model_id) {
    std::unique_lock<std::mutex> lockGnaCalls{ acrossPluginsSync };
    uint32_t reqConfId;
    auto status = Gna2RequestConfigCreate(model_id, &reqConfId);
    checkGna2Status(status, "Gna2RequestConfigCreate");

    // When the GNA_SW_EXACT mode is chosen inference results should be computed exactly the same way
    // (bit exactly) as on the selected GNA execution target generation.
    // See the GNA Plugin's GNA_EXEC_TARGET config option description.
    if (swExactMode) {
        const auto consistentDevice = getTargetDevice(true);
        status = Gna2RequestConfigEnableHardwareConsistency(reqConfId, consistentDevice);
        checkGna2Status(status, "Gna2RequestConfigEnableHardwareConsistency(" + std::to_string(static_cast<long>(consistentDevice)) + ")");
    }
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
        const auto s = Gna2StatusGetMessage(status, gna2StatusBuffer.data(), gna2StatusBuffer.size());
        if (!Gna2StatusIsSuccessful(s))
            snprintf(gna2StatusBuffer.data(), gna2StatusBuffer.size(), "Gna2StatusGetMessage(%d) returned (%d)",
                static_cast<int>(status), static_cast<int>(s));
        if (status == Gna2StatusDeviceIngoingCommunicationError ||
            status == Gna2StatusDeviceOutgoingCommunicationError) {
            THROW_GNA_EXCEPTION << "Unsuccessful Gna2Status: (" << status << ") " <<
                gna2StatusBuffer.data() << ", consider updating the GNA driver" <<
                decoratedGnaLibVersion();
        }

        Gna2ModelError error;
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

void GNADeviceHelper::checkGna2Status(Gna2Status status, const std::string& from) {
    if (!Gna2StatusIsSuccessful(status)) {
        std::vector<char> gna2StatusBuffer(1024);
        const auto prefix = "Unsuccessful " + from + " call, Gna2Status: (";
        const auto s = Gna2StatusGetMessage(status, gna2StatusBuffer.data(), gna2StatusBuffer.size());
        if (!Gna2StatusIsSuccessful(s))
            snprintf(gna2StatusBuffer.data(), gna2StatusBuffer.size(), "Gna2StatusGetMessage(%d) returned (%d)",
                static_cast<int>(status), static_cast<int>(s));
        std::string suffix;
        if (status == Gna2StatusDeviceIngoingCommunicationError ||
            status == Gna2StatusDeviceOutgoingCommunicationError) {
            suffix = ", consider updating the GNA driver";
        }
        THROW_GNA_EXCEPTION << prefix << status << ") " << gna2StatusBuffer.data() << suffix <<
            decoratedGnaLibVersion();
    }
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
#endif

GnaWaitStatus GNADeviceHelper::wait(uint32_t reqId, int64_t millisTimeout) {
    std::unique_lock<std::mutex> lockGnaCalls{ acrossPluginsSync };
#if GNA_LIB_VER == 2
    const auto status = Gna2RequestWait(reqId, millisTimeout);
    if (status == Gna2StatusWarningDeviceBusy) {
        return GNA_REQUEST_PENDING;
    }
    unwaitedRequestIds.erase(reqId);
    if (status == Gna2StatusDriverQoSTimeoutExceeded) {
        return GNA_REQUEST_ABORTED;
    }
    checkGna2Status(status, "Gna2RequestWait");
#else
    if (isPerformanceMeasuring) {
        nGNAStatus = GNAWaitPerfRes(nGNAHandle, millisTimeout, reqId, &nGNAPerfResults);
    } else {
        nGNAStatus = GNAWait(nGNAHandle, millisTimeout, reqId);
    }
    checkStatus();
#endif
    updateGnaPerfCounters();
    return GNA_REQUEST_COMPLETED;
}

#if GNA_LIB_VER == 1
GNADeviceHelper::DumpResult GNADeviceHelper::dumpXnn(const intel_nnet_type_t *pNeuralNetwork,
                                    const uint32_t *pActiveIndices,
                                    uint32_t nActiveIndices) {
#else
GNADeviceHelper::DumpResult GNADeviceHelper::dumpXnn(const uint32_t modelId) {
#endif
    DumpResult r;

#if GNA_LIB_VER == 1
    if (!pNeuralNetwork) {
        THROW_GNA_EXCEPTION << "GNADumpXnn got invalid NeuralNetwork parameter \n";
    }
    r.model.reset(GNADumpXnn(pNeuralNetwork,
                             pActiveIndices,
                             nActiveIndices,
                             &r.header,
                             &nGNAStatus,
                             [](size_t count)-> void* {return new char[count]();}),
                             [](void * ptr) {::operator delete[](ptr);});
    checkStatus();
#else
    r.model.reset(
        ExportSueLegacyUsingGnaApi2(modelId, &r.header),
        gnaUserFree);
#endif

    if (r.model == nullptr) {
        THROW_GNA_EXCEPTION << "GNADumpXnn returned nullptr";
    }

    return r;
}

#if GNA_LIB_VER == 2

void GNADeviceHelper::dumpXnnForDeviceVersion(
    const uint32_t modelId,
    std::ostream & outStream,
    const Gna2DeviceVersion targetDeviceVersion) {

    Gna2ModelSueCreekHeader sueHeader;
    auto ptr = ExportSueLegacyUsingGnaApi2(modelId, &sueHeader);
    gnaUserFree(ptr);

    ExportGnaDescriptorPartiallyFilled(sueHeader.NumberOfLayers, outStream);

    ExportLdForDeviceVersion(modelId, outStream, targetDeviceVersion);
    if (dumpXNNROPtr == nullptr) {
        THROW_GNA_EXCEPTION << "Bad RO pointer (nullptr)";
    }
    outStream.write(static_cast<const char*>(dumpXNNROPtr), dumpXNNROSize);

    // TODO: GNA2: remove
    outStream.write("Gna2ModelSueCreekHeader", 24);
    outStream.write(reinterpret_cast<const char*>(&sueHeader), sizeof(sueHeader));
}
#endif

#if GNA_LIB_VER == 1
void GNADeviceHelper::checkStatus() const {
    if ((nGNAStatus != GNA_NOERROR) && (nGNAStatus != GNA_SSATURATE)) {
        THROW_GNA_EXCEPTION << "Bad GNA status " << nGNAStatus << ", " << GNAStatusName[nGNAStatus];
    }
}
#endif

void GNADeviceHelper::open(uint8_t n_threads) {
    std::unique_lock<std::mutex> lockGnaCalls{ acrossPluginsSync };
#if GNA_LIB_VER == 1
    nGNAHandle = GNADeviceOpenSetThreads(&nGNAStatus, n_threads);
    checkStatus();
#else
    auto status = Gna2DeviceGetVersion(nGnaDeviceIndex, &detectedGnaDevVersion);
    checkGna2Status(status, "Gna2DeviceGetVersion");
    status = Gna2DeviceOpen(nGnaDeviceIndex);
    checkGna2Status(status, "Gna2DeviceOpen");
    // TODO: GNA2: uncomment when scratchpad repaired
    // status = Gna2DeviceSetNumberOfThreads(nGnaDeviceIndex, n_threads);
    // checkGna2Status(status);
#endif
    deviceOpened = true;
}

void GNADeviceHelper::close() {
#if GNA_LIB_VER == 1
    std::unique_lock<std::mutex> lockGnaCalls{ acrossPluginsSync };
    GNADeviceClose(nGNAHandle);
    nGNAHandle = 0;
#else
    auto requestsToClose = unwaitedRequestIds;
    for (auto requestId : requestsToClose) {
        try {
            wait(requestId);
        } catch (...) {
            gnawarn() << "Request with Id " << requestId << " was not awaited successfully";
        }
    }
    {
        std::unique_lock<std::mutex> lockGnaCalls{ acrossPluginsSync };
        const auto status = Gna2DeviceClose(nGnaDeviceIndex);
        checkGna2Status(status, "Gna2DeviceClose");
    }
#endif
    deviceOpened = false;
}

void GNADeviceHelper::setOMPThreads(uint8_t const n_threads) {
    std::unique_lock<std::mutex> lockGnaCalls{ acrossPluginsSync };
#if GNA_LIB_VER == 1
    gmmSetThreads(n_threads);
#else
    const auto status = Gna2DeviceSetNumberOfThreads(nGnaDeviceIndex, n_threads);
    checkGna2Status(status, "Gna2DeviceSetNumberOfThreads");
#endif
}

void GNADeviceHelper::updateGnaPerfCounters() {
    if (!isPerformanceMeasuring)
        return;
#if GNA_LIB_VER == 2
    instrumentationTotal[0] = instrumentationResults[0];
    instrumentationTotal[1] = instrumentationResults[1];
    instrumentationResults[0] = 0;
    instrumentationResults[1] = 0;
#else
    nGNAPerfResultsTotal.hw.stall = nGNAPerfResults.hw.stall;
    nGNAPerfResultsTotal.hw.total = nGNAPerfResults.hw.total;
    nGNAPerfResultsTotal.lib.submit = nGNAPerfResults.lib.submit;
    nGNAPerfResultsTotal.lib.preprocess = nGNAPerfResults.lib.preprocess;
    nGNAPerfResultsTotal.lib.process = nGNAPerfResults.lib.process;
    nGNAPerfResultsTotal.lib.scoring = nGNAPerfResults.lib.scoring;
    nGNAPerfResultsTotal.lib.total = nGNAPerfResults.lib.total;
    nGNAPerfResultsTotal.lib.ioctlSubmit = nGNAPerfResults.lib.ioctlSubmit;
    nGNAPerfResultsTotal.lib.ioctlWaitOn = nGNAPerfResults.lib.ioctlWaitOn;

    nGNAPerfResultsTotal.total.start = nGNAPerfResults.total.start;
    nGNAPerfResultsTotal.total.stop = nGNAPerfResults.total.stop;

    nGNAPerfResultsTotal.drv.startHW = nGNAPerfResults.drv.startHW;
    nGNAPerfResultsTotal.drv.scoreHW = nGNAPerfResults.drv.scoreHW;
    nGNAPerfResultsTotal.drv.intProc = nGNAPerfResults.drv.intProc;
#endif
}

void GNADeviceHelper::getGnaPerfCounters(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& retPerfCounters) {
    InferenceEngine::InferenceEngineProfileInfo info;
    info.status = InferenceEngine::InferenceEngineProfileInfo::EXECUTED;
    info.cpu_uSec = 0;
    info.execution_index = 0;
    info.realTime_uSec = 0;
    // Hardware
#if GNA_LIB_VER == 1
    info.realTime_uSec = nGNAPerfResultsTotal.hw.total;
#else
    info.realTime_uSec = instrumentationTotal[0];
#endif
    retPerfCounters["1.1 Total scoring time in HW"] = info;
#if GNA_LIB_VER == 1
    info.realTime_uSec = nGNAPerfResultsTotal.hw.stall;
#else
    info.realTime_uSec = instrumentationTotal[1];
#endif
    retPerfCounters["1.2 Stall scoring time in HW"] = info;
}
