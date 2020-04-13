// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <map>
#include <thread>

#include <ie_common.h>

#if GNA_LIB_VER == 2
#include "gna2-common-api.h"
#include "gna2-inference-api.h"
#include "gna2-instrumentation-api.h"

#include "gna2-memory-api.h"
#include "gna2-model-api.h"
#include "gna2-model-suecreek-header.h"
#else
#include <gna-api.h>
#include "gna-api-dumper.h"
#include "gna-api-instrumentation.h"
#endif


/**
 * holds gna - style handle in RAII way
 */
class GNADeviceHelper {
#if GNA_LIB_VER == 1
    intel_gna_status_t nGNAStatus = GNA_NOERROR;
    intel_gna_handle_t nGNAHandle = 0;
    intel_gna_proc_t nGNAProcType = GNA_AUTO;
    intel_gna_perf_t nGNAPerfResults;
    intel_gna_perf_t nGNAPerfResultsTotal;
#else
    uint32_t nGnaDeviceIndex = 0;
    Gna2AccelerationMode gna2AccelerationMode = Gna2AccelerationModeAuto;
    Gna2DeviceVersion gna2HwConsistency = Gna2DeviceVersionSoftwareEmulation;
    Gna2DeviceVersion detectedGnaDevVersion = Gna2DeviceVersionSoftwareEmulation;

    static const uint32_t TotalGna2InstrumentationPoints = 2;
    Gna2InstrumentationPoint gna2InstrumentationPoints[TotalGna2InstrumentationPoints] = {
        Gna2InstrumentationPointHwTotalCycles,
        Gna2InstrumentationPointHwStallCycles };

    uint64_t instrumentationResults[TotalGna2InstrumentationPoints] = {};
    uint64_t instrumentationTotal[TotalGna2InstrumentationPoints] = {};
    uint32_t instrumentationConfigId = 0;

#define MAX_TIMEOUT 500000
#endif
    const uint32_t GNA_TIMEOUT = MAX_TIMEOUT;
    bool isPerformanceMeasuring = false;
    bool deviceOpened = false;
public:
#if GNA_LIB_VER == 1
    explicit GNADeviceHelper(intel_gna_proc_t proc_type = GNA_AUTO,
                            uint8_t lib_async_n_threads = 1,
                            bool use_openmp = false,
                            bool isPerformanceMeasuring = false) :
                                    nGNAProcType(proc_type),
                                    isPerformanceMeasuring(isPerformanceMeasuring) {
#else
     explicit GNADeviceHelper(Gna2AccelerationMode gna2accMode = Gna2AccelerationModeAuto,
         Gna2DeviceVersion gna2HwConsistency = Gna2DeviceVersionSoftwareEmulation,
         uint8_t lib_async_n_threads = 1,
         bool use_openmp = false,
         bool isPerformanceMeasuring = false) :
         gna2AccelerationMode(gna2accMode),
         gna2HwConsistency(gna2HwConsistency),
         isPerformanceMeasuring(isPerformanceMeasuring) {
#endif
        open(lib_async_n_threads);
        initGnaPerfCounters();

        if (use_openmp) {
            uint8_t num_cores = std::thread::hardware_concurrency();
            setOMPThreads((num_cores != 0) ? num_cores : 1);
        }
    }

    GNADeviceHelper(const GNADeviceHelper&) = delete;
    GNADeviceHelper& operator= (const GNADeviceHelper&) = delete;
    ~GNADeviceHelper() {
        if (deviceOpened) {
            close();
        }
    }

    uint8_t *alloc(uint32_t size_requested, uint32_t *size_granted);

#if GNA_LIB_VER == 1
    void propagateSync(const intel_nnet_type_t *pNeuralNetwork,
                       const uint32_t *pActiveIndices,
                       uint32_t nActiveIndices);

    uint32_t propagate(const intel_nnet_type_t *pNeuralNetwork,
                       const uint32_t *pActiveIndices,
                       uint32_t nActiveIndices);
#else
    void setUpActiveList(unsigned req_config_id, uint32_t layerIndex, uint32_t* ptr_active_indices, uint32_t num_active_indices);
    void propagateSync(const uint32_t requestConfigId);
    uint32_t propagate(const uint32_t requestConfigId);
#if GNA_LIB_VER == 2
    uint32_t createModel(const Gna2Model& gnaModel) const;
#else
    uint32_t createModel(const intel_nnet_type_t& intel_nnet_type);
#endif
    void releseModel(const uint32_t model_id);
    uint32_t createRequestConfig(const uint32_t model_id);
    bool hasGnaHw() const {
        return Gna2DeviceVersionSoftwareEmulation != detectedGnaDevVersion;
    }
    static void checkGna2Status(Gna2Status status);
#endif
    void wait(uint32_t id);

    struct DumpResult {
#if GNA_LIB_VER == 2
        Gna2ModelSueCreekHeader header;
#else
        intel_gna_model_header header;
#endif
        std::shared_ptr<void> model;
    };

    const void * dumpXNNROPtr = nullptr;
    uint32_t dumpXNNROSize = 0;

#if GNA_LIB_VER == 1
    DumpResult dumpXnn(const intel_nnet_type_t *pNeuralNetwork,
                 const uint32_t *pActiveIndices,
                 uint32_t nActiveIndices);
    intel_gna_status_t getGNAStatus() const noexcept {
        return nGNAStatus;
    }
#else

    DumpResult dumpXnn(const uint32_t modelId);
    void dumpXnnNoMmu(const uint32_t modelId, std::ostream & outStream);
#endif
    void free(void * ptr);

    void updateGnaPerfCounters();
    void getGnaPerfCounters(std::map<std::string,
                        InferenceEngine::InferenceEngineProfileInfo>& retPerfCounters);
 private:
    void open(uint8_t const n_threads);

    void close();
#if GNA_LIB_VER == 1
    void checkStatus() const;
#endif
    void setOMPThreads(uint8_t const n_threads);

    void initGnaPerfCounters() {
#if GNA_LIB_VER == 1
        nGNAPerfResults = {{0, 0, 0, 0, 0, 0, 0}, {0, 0}, {0, 0, 0}, {0, 0}};
        nGNAPerfResultsTotal = {{0, 0, 0, 0, 0, 0, 0}, {0, 0}, {0, 0, 0}, {0, 0}};
#else
        const auto status = Gna2InstrumentationConfigCreate(TotalGna2InstrumentationPoints,
            gna2InstrumentationPoints,
            instrumentationResults,
            &instrumentationConfigId);
        checkGna2Status(status);
#endif
    }
};  // NOLINT
