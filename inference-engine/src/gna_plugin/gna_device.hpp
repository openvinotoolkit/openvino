// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gna-api.h"
#include "gna-api-dumper.h"
#include "gna-api-instrumentation.h"
#include "ie_common.h"
#include <memory>
#include <string>
#include <map>
#include <thread>

/**
 * holds gna - style handle in RAII way
 */
class GNADeviceHelper {
    intel_gna_status_t nGNAStatus = GNA_NOERROR;
    intel_gna_handle_t nGNAHandle = 0;
    intel_gna_proc_t nGNAProcType = GNA_AUTO;
    intel_gna_perf_t nGNAPerfResults;
    intel_gna_perf_t nGNAPerfResultsTotal;
    const uint32_t GNA_TIMEOUT = MAX_TIMEOUT;
    bool isPerformanceMeasuring;

 public:
    explicit GNADeviceHelper(intel_gna_proc_t proc_type = GNA_AUTO,
                            uint8_t lib_async_n_threads = 1,
                            bool use_openmp = false,
                            bool isPerformanceMeasuring = false) :
                                    nGNAProcType(proc_type),
                                    isPerformanceMeasuring(isPerformanceMeasuring) {
        initGnaPerfCounters();
        open(lib_async_n_threads);

        if (use_openmp) {
            uint8_t num_cores = std::thread::hardware_concurrency();
            setOMPThreads((num_cores != 0) ? num_cores : 1);
        }
    }

    ~GNADeviceHelper() {
        close();
    }

    uint8_t *alloc(uint32_t size_requested, uint32_t *size_granted);

    uint32_t propagate(const intel_nnet_type_t *pNeuralNetwork,
                       const uint32_t *pActiveIndices,
                       uint32_t nActiveIndices);

    void wait(uint32_t id);

    struct DumpResult {
        intel_gna_model_header header;
        std::shared_ptr<void> model;
    };

    DumpResult dumpXnn(const intel_nnet_type_t *pNeuralNetwork,
                 const uint32_t *pActiveIndices,
                 uint32_t nActiveIndices);


    void free() {
        GNAFree(nGNAHandle);
    }
    void updateGnaPerfCounters();
    void getGnaPerfCounters(std::map<std::string,
                        InferenceEngine::InferenceEngineProfileInfo>& retPerfCounters);

    intel_gna_status_t getGNAStatus() const noexcept {
        return nGNAStatus;
    }

 private:
    void open(uint8_t const n_threads);

    void close();

    void checkStatus() const;

    void setOMPThreads(uint8_t const n_threads);

    void initGnaPerfCounters() {
        nGNAPerfResults = {{0, 0, 0, 0, 0, 0, 0}, {0, 0}, {0, 0, 0}, {0, 0}};
        nGNAPerfResultsTotal = {{0, 0, 0, 0, 0, 0, 0}, {0, 0}, {0, 0, 0}, {0, 0}};
    }
};

