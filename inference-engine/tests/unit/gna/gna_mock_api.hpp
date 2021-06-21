// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <gmock/gmock.h>

#if defined(_WIN32)
    #ifdef libGNAStubs_EXPORTS
        #define GNA_STUBS_EXPORT __declspec(dllexport)
    #else
        #define GNA_STUBS_EXPORT __declspec(dllimport)
    #endif
#else
    #define GNA_STUBS_EXPORT
#endif

class GNACppApi {
public:
    GNA_STUBS_EXPORT GNACppApi();
    GNA_STUBS_EXPORT ~GNACppApi();
#if GNA_LIB_VER == 1
    MOCK_METHOD(intel_gna_status_t, GNAScoreGaussians, (
        // intel_gna_handle_t          nGNADevice,            // handle to GNA accelerator
        // const intel_feature_type_t* pFeatureType,
        const intel_feature_t*      pFeatureData,
        const intel_gmm_type_t*     pModelType,
        const intel_gmm_t*          pModelData,
        const uint32_t*             pActiveGMMIndices,
        uint32_t                    nActiveGMMIndices,
        uint32_t                    uMaximumScore,
        intel_gmm_mode_t            nGMMMode,
        uint32_t*                   pScores,
        uint32_t*                   pReqId,
        intel_gna_proc_t            nAccelerationType));


    MOCK_METHOD(intel_gna_status_t, GNAPropagateForward, (
        intel_gna_handle_t          nGNADevice,            // handle to GNA accelerator
        const intel_nnet_type_t*    pNeuralNetwork,
        const uint32_t*             pActiveIndices,
        uint32_t                    nActiveIndices,
        uint32_t*                   pReqId,
        intel_gna_proc_t            nAccelerationType));

    MOCK_METHOD(void *, GNAAlloc, (
        intel_gna_handle_t nGNADevice,   // handle to GNA accelerator
        uint32_t           sizeRequested,
        uint32_t*          sizeGranted));

    MOCK_METHOD(intel_gna_status_t, GNAFree, (intel_gna_handle_t nGNADevice));

    MOCK_METHOD(intel_gna_handle_t, GNADeviceOpen, (intel_gna_status_t* status));

    MOCK_METHOD(intel_gna_handle_t, GNADeviceOpenSetThreads, (intel_gna_status_t* status, uint8_t n_threads));
    MOCK_METHOD(intel_gna_status_t, GNADeviceClose, (intel_gna_handle_t nGNADevice));

    MOCK_METHOD(intel_gna_status_t, GNAWait, (
                 intel_gna_handle_t nGNADevice,            // handle to GNA accelerator
                 uint32_t           nTimeoutMilliseconds,
                 uint32_t           reqId));               // IN score request ID

    MOCK_METHOD(intel_gna_status_t, GNAWaitPerfRes, (
                 intel_gna_handle_t nGNADevice,            // handle to GNA accelerator
                 uint32_t           nTimeoutMilliseconds,
                 uint32_t           reqId,                 // IN score request ID
                 intel_gna_perf_t*  nGNAPerfResults));

    MOCK_METHOD(void*, GNADumpXnn, (
        const intel_nnet_type_t*    neuralNetwork,
        const uint32_t*             activeIndices,
        uint32_t                    activeIndicesCount,
        intel_gna_model_header*     modelHeader,
        intel_gna_status_t*         status,
        intel_gna_alloc_cb          customAlloc));

    MOCK_METHOD(intel_gna_handle_t, gmmSetThreads, (uint8_t num));
#else
    MOCK_METHOD(Gna2Status, Gna2MemoryAlloc, (
        uint32_t sizeRequested,
        uint32_t * sizeGranted,
        void ** memoryAddress));
    MOCK_METHOD(Gna2Status, Gna2RequestWait, (
        uint32_t requestId,
        uint32_t timeoutMilliseconds));
#endif
};
