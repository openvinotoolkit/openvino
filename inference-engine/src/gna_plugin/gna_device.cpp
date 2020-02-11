// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_device.hpp"

#include <map>
#include <string>
#include <cstring>
#include <vector>

#if GNA_LIB_VER == 2
#include "gna_api_wrapper.hpp"
#include "gna2-device-api.h"
#include "gna2-inference-api.h"
#include "gna2-instrumentation-api.h"
#include "gna2-memory-api.h"
#include "gna2_model_export_helper.hpp"
#else
#include "gna-api-status.h"
#include "gna-api.h"
#endif

#include "details/ie_exception.hpp"
#include "gna_plugin_log.hpp"

uint8_t* GNADeviceHelper::alloc(uint32_t size_requested, uint32_t *size_granted) {
    void * memPtr;
#if GNA_LIB_VER == 1
    memPtr = GNAAlloc(nGNAHandle, size_requested, size_granted);
#else
    const auto status = Gna2MemoryAlloc(size_requested, size_granted, &memPtr);
    checkGna2Status(status);
#endif
    if (memPtr == nullptr) {
        THROW_GNA_EXCEPTION << "GNAAlloc failed to allocate memory. Requested: " << size_requested << " Granted: " << *(size_granted);
    }
    dumpXNNROPtr = memPtr;
    dumpXNNROSize = *size_granted;
    return static_cast<uint8_t *>(memPtr);
}

void GNADeviceHelper::free(void * ptr) {
#if GNA_LIB_VER == 1
    GNAFree(nGNAHandle);
#else
    const auto status = Gna2MemoryFree(ptr);
    checkGna2Status(status);
#endif
}

#if GNA_LIB_VER == 1
uint32_t GNADeviceHelper::propagate(const intel_nnet_type_t *pNeuralNetwork,
                   const uint32_t *pActiveIndices,
                   uint32_t nActiveIndices) {
    uint32_t reqId;

    nGNAStatus = GNAPropagateForward(nGNAHandle, pNeuralNetwork,
                                     pActiveIndices, nActiveIndices, &reqId, nGNAProcType);
    checkStatus();
    return reqId;
}
#else
void GNADeviceHelper::setUpActiveList(const uint32_t requestConfigId, uint32_t layerIndex, uint32_t* ptr_active_indices, uint32_t num_active_indices) {
    const auto status = Gna2RequestConfigEnableActiveList(requestConfigId, layerIndex, num_active_indices, ptr_active_indices);
    checkGna2Status(status);
}
void GNADeviceHelper::propagateSync(const uint32_t requestConfigId) {
    wait(propagate(requestConfigId));
}

uint32_t GNADeviceHelper::propagate(const uint32_t requestConfigId) {
    uint32_t reqId;
    const auto status = Gna2RequestEnqueue(requestConfigId, &reqId);
    checkGna2Status(status);
    return reqId;
}

uint32_t GNADeviceHelper::createModel(const Gna2Model& gnaModel) const {
    uint32_t modelId;
    const auto status = Gna2ModelCreate(nGnaDeviceIndex, &gnaModel, &modelId);
    checkGna2Status(status);
    return modelId;
}

void GNADeviceHelper::releseModel(const uint32_t model_id) {
    const auto status = Gna2ModelRelease(model_id);
    checkGna2Status(status);
}

uint32_t GNADeviceHelper::createRequestConfig(const uint32_t model_id) {
    uint32_t reqConfId;
    auto status = Gna2RequestConfigCreate(model_id, &reqConfId);
    checkGna2Status(status);
    status = Gna2RequestConfigSetAccelerationMode(reqConfId, gna2AccelerationMode);
    checkGna2Status(status);
    if (gna2HwConsistency != Gna2DeviceVersionSoftwareEmulation) {
        status = Gna2RequestConfigEnableHardwareConsistency(reqConfId, gna2HwConsistency);
        checkGna2Status(status);
    }
    status = Gna2InstrumentationConfigAssignToRequestConfig(instrumentationConfigId, reqConfId);
    checkGna2Status(status);

    return reqConfId;
}

void GNADeviceHelper::checkGna2Status(Gna2Status status) {
    if (!Gna2StatusIsSuccessful(status)) {
        std::vector<char> gna2StatusBuffer(1024);
        const auto s = Gna2StatusGetMessage(status, gna2StatusBuffer.data(), gna2StatusBuffer.size());
        if (!Gna2StatusIsSuccessful(s))
            snprintf(gna2StatusBuffer.data(), gna2StatusBuffer.size(), "Gna2StatusGetMessage(%d) returned (%d)",
                static_cast<int>(status), static_cast<int>(s));
        if (status == Gna2StatusDeviceIngoingCommunicationError ||
            status == Gna2StatusDeviceOutgoingCommunicationError) {
            THROW_GNA_EXCEPTION << "Unsuccessful Gna2Status: (" << status << ") " << gna2StatusBuffer.data() << ", consider updating the GNA driver";
        }
        THROW_GNA_EXCEPTION << "Unsuccessful Gna2Status: (" << status << ") " << gna2StatusBuffer.data();
    }
}
#endif

void GNADeviceHelper::wait(uint32_t reqId) {
#if GNA_LIB_VER == 2
    const auto status = Gna2RequestWait(reqId, GNA_TIMEOUT);
    checkGna2Status(status);
#else
    if (isPerformanceMeasuring) {
        nGNAStatus = GNAWaitPerfRes(nGNAHandle, GNA_TIMEOUT, reqId, &nGNAPerfResults);
    } else {
        nGNAStatus = GNAWait(nGNAHandle, GNA_TIMEOUT, reqId);
    }
    checkStatus();
#endif
    updateGnaPerfCounters();
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

void GNADeviceHelper::dumpXnnNoMmu(const uint32_t modelId, std::ostream & outStream) {
    Gna2ModelSueCreekHeader sueHeader;
    auto ptr = ExportSueLegacyUsingGnaApi2(modelId, &sueHeader);
    gnaUserFree(ptr);

    ExportGnaDescriptorPartiallyFilled(sueHeader.NumberOfLayers, outStream);

    ExportLdForNoMmu(modelId, outStream);
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
#if GNA_LIB_VER == 1
    nGNAHandle = GNADeviceOpenSetThreads(&nGNAStatus, n_threads);
    checkStatus();
#else
    auto status = Gna2DeviceGetVersion(nGnaDeviceIndex, &detectedGnaDevVersion);
    checkGna2Status(status);
    if (gna2AccelerationMode == Gna2AccelerationModeHardware &&
        detectedGnaDevVersion == Gna2DeviceVersionSoftwareEmulation) {
        gnalog() << "GNA Device not detected, consider using other mode of acceleration";
    }
    status = Gna2DeviceOpen(nGnaDeviceIndex);
    checkGna2Status(status);
    // TODO: GNA2: uncomment when scratchpad repaired
    // status = Gna2DeviceSetNumberOfThreads(nGnaDeviceIndex, n_threads);
    // checkGna2Status(status);
#endif
    deviceOpened = true;
}

void GNADeviceHelper::close() {
#if GNA_LIB_VER == 1
    GNADeviceClose(nGNAHandle);
    nGNAHandle = 0;
#else
    const auto status = Gna2DeviceClose(nGnaDeviceIndex);
    checkGna2Status(status);
#endif
    deviceOpened = false;
}

void GNADeviceHelper::setOMPThreads(uint8_t const n_threads) {
#if GNA_LIB_VER == 1
    gmmSetThreads(n_threads);
#else
    const auto status = Gna2DeviceSetNumberOfThreads(nGnaDeviceIndex, n_threads);
    checkGna2Status(status);
#endif
}

void GNADeviceHelper::updateGnaPerfCounters() {
    if (!isPerformanceMeasuring)
        return;
#if GNA_LIB_VER == 2
    instrumentationTotal[0] = instrumentationResults[0];
    instrumentationTotal[1] = instrumentationResults[1];
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
