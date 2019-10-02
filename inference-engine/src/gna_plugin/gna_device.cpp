// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_device.hpp"

#include <map>
#include <string>
#include <cstring>

#include "gna-api-status.h"
#include "gna-api.h"

#include "details/ie_exception.hpp"
#include "gna_plugin_log.hpp"
#include "gna/gna_config.hpp"

uint8_t* GNADeviceHelper::alloc(uint32_t size_requested, uint32_t *size_granted) {
    return reinterpret_cast<uint8_t *>(GNAAlloc(nGNAHandle, size_requested, size_granted));
}


uint32_t GNADeviceHelper::propagate(const intel_nnet_type_t *pNeuralNetwork,
                   const uint32_t *pActiveIndices,
                   uint32_t nActiveIndices) {
    uint32_t reqId;
    nGNAStatus = GNAPropagateForward(nGNAHandle, pNeuralNetwork,
                                     pActiveIndices, nActiveIndices, &reqId, nGNAProcType);
    checkStatus();
    return reqId;
}

void GNADeviceHelper::wait(uint32_t reqId) {
    if (isPerformanceMeasuring) {
        nGNAStatus = GNAWaitPerfRes(nGNAHandle, GNA_TIMEOUT, reqId, &nGNAPerfResults);
        updateGnaPerfCounters();
    } else {
        nGNAStatus = GNAWait(nGNAHandle, 1000000, reqId);
    }
    checkStatus();
}

GNADeviceHelper::DumpResult GNADeviceHelper::dumpXnn(const intel_nnet_type_t *pNeuralNetwork,
                                    const uint32_t *pActiveIndices,
                                    uint32_t nActiveIndices) {
    DumpResult r;
    intel_gna_status_t gna_status;

    if (!pNeuralNetwork) {
        THROW_GNA_EXCEPTION<< "GNADumpXnn got invalid NeuralNetwork parameter \n";
    }
    r.model.reset(GNADumpXnn(pNeuralNetwork,
                             pActiveIndices,
                             nActiveIndices,
                             &r.header,
                             &nGNAStatus,
                             [](size_t count)-> void* {return new char[count]();}),
                             [](void * ptr) {::operator delete[](ptr);});

    checkStatus();

    if (r.model == nullptr) {
        THROW_GNA_EXCEPTION << "GNADumpXnn returned nullptr";
    }

    return r;
}

void GNADeviceHelper::checkStatus() const {
    if ((nGNAStatus != GNA_NOERROR) && (nGNAStatus != GNA_SSATURATE)) {
        THROW_GNA_EXCEPTION << "Bad GNA status " << nGNAStatus << ", " << GNAStatusName[nGNAStatus];
    }
}

void GNADeviceHelper::open(uint8_t n_threads) {
    nGNAHandle = GNADeviceOpenSetThreads(&nGNAStatus, n_threads);

    checkStatus();
}

void GNADeviceHelper::close() {
    GNADeviceClose(nGNAHandle);
    nGNAHandle = 0;
}

void GNADeviceHelper::setOMPThreads(uint8_t const n_threads) {
    gmmSetThreads(n_threads);
}

void GNADeviceHelper::updateGnaPerfCounters() {
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
}

void GNADeviceHelper::getGnaPerfCounters(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& retPerfCounters) {
    InferenceEngine::InferenceEngineProfileInfo info;
    info.status = InferenceEngine::InferenceEngineProfileInfo::EXECUTED;
    info.cpu_uSec = 0;
    info.execution_index = 0;

    // Hardware
    info.realTime_uSec = nGNAPerfResultsTotal.hw.total;
    retPerfCounters["1.1 Total scoring time in HW"] = info;

    info.realTime_uSec = nGNAPerfResultsTotal.hw.stall;
    retPerfCounters["1.2 Stall scoring time in HW"] = info;
}
