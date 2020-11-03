// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <gmock/gmock-generated-function-mockers.h>
#if GNA_LIB_VER == 1
#include <gna-api.h>
#include <gna-api-instrumentation.h>
#include <gna-api-dumper.h>
#else
#include <gna2-instrumentation-api.h>
#include <gna2-inference-api.h>
#include <gna2-model-export-api.h>
#endif
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
    MOCK_METHOD10(GNAScoreGaussians, intel_gna_status_t(
        //intel_gna_handle_t          nGNADevice,            // handle to GNA accelerator
        //const intel_feature_type_t* pFeatureType,
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


    MOCK_METHOD6(GNAPropagateForward, intel_gna_status_t (
        intel_gna_handle_t          nGNADevice,            // handle to GNA accelerator
        const intel_nnet_type_t*    pNeuralNetwork,
        const uint32_t*             pActiveIndices,
        uint32_t                    nActiveIndices,
        uint32_t*                   pReqId,
        intel_gna_proc_t            nAccelerationType));

    MOCK_METHOD3(GNAAlloc, void *(
        intel_gna_handle_t nGNADevice,   // handle to GNA accelerator
        uint32_t           sizeRequested,
        uint32_t*          sizeGranted));

    MOCK_METHOD1(GNAFree, intel_gna_status_t (intel_gna_handle_t nGNADevice));

    MOCK_METHOD1(GNADeviceOpen, intel_gna_handle_t (intel_gna_status_t* status));

    MOCK_METHOD2(GNADeviceOpenSetThreads, intel_gna_handle_t (intel_gna_status_t* status, uint8_t n_threads));
    MOCK_METHOD1(GNADeviceClose, intel_gna_status_t (intel_gna_handle_t nGNADevice));

    MOCK_METHOD3(GNAWait, intel_gna_status_t(
                 intel_gna_handle_t nGNADevice,            // handle to GNA accelerator
                 uint32_t           nTimeoutMilliseconds,
                 uint32_t           reqId                  // IN score request ID);
    ));

    MOCK_METHOD4(GNAWaitPerfRes, intel_gna_status_t(
                 intel_gna_handle_t nGNADevice,            // handle to GNA accelerator
                 uint32_t           nTimeoutMilliseconds,
                 uint32_t           reqId,                 // IN score request ID);
                 intel_gna_perf_t*  nGNAPerfResults
    ));

    MOCK_METHOD6(GNADumpXnn, void* (
        const intel_nnet_type_t*    neuralNetwork,
        const uint32_t*             activeIndices,
        uint32_t                    activeIndicesCount,
        intel_gna_model_header*     modelHeader,
        intel_gna_status_t*         status,
        intel_gna_alloc_cb          customAlloc));

    MOCK_METHOD1(gmmSetThreads, intel_gna_handle_t (uint8_t num));
#else
    MOCK_METHOD3(Gna2MemoryAlloc, Gna2Status (
        uint32_t sizeRequested,
        uint32_t * sizeGranted,
        void ** memoryAddress));

    MOCK_METHOD1(Gna2DeviceOpen, Gna2Status (
        uint32_t deviceIndex));

    MOCK_METHOD2(Gna2DeviceSetNumberOfThreads, Gna2Status(
        uint32_t deviceIndex,
        uint32_t numberOfThreads));

    MOCK_METHOD1(Gna2DeviceClose, Gna2Status (
        uint32_t deviceIndex));

    MOCK_METHOD1(Gna2DeviceGetCount, Gna2Status (
        uint32_t * numberOfDevices));

    MOCK_METHOD1(Gna2MemoryFree, Gna2Status (
        void * memory));

    MOCK_METHOD3(Gna2StatusGetMessage, Gna2Status (
        enum Gna2Status status,
        char * messageBuffer,
        uint32_t messageBufferSize));

    MOCK_METHOD3(Gna2ModelCreate, Gna2Status (
        uint32_t deviceIndex,
        struct Gna2Model const * model,
        uint32_t * modelId));

    MOCK_METHOD1(Gna2ModelRelease, Gna2Status (
        uint32_t modelId));

    MOCK_METHOD1(Gna2ModelGetLastError, Gna2Status (
        struct Gna2ModelError* error));

    MOCK_METHOD2(Gna2RequestConfigCreate, Gna2Status (
        uint32_t modelId,
        uint32_t * requestConfigId));

    MOCK_METHOD4(Gna2RequestConfigEnableActiveList, Gna2Status (
        uint32_t requestConfigId,
        uint32_t operationIndex,
        uint32_t numberOfIndices,
        uint32_t const * indices));

    MOCK_METHOD2(Gna2RequestConfigEnableHardwareConsistency, Gna2Status (
        uint32_t requestConfigId,
        enum Gna2DeviceVersion deviceVersion));

    MOCK_METHOD2(Gna2RequestConfigSetAccelerationMode, Gna2Status (
        uint32_t requestConfigId,
        enum Gna2AccelerationMode accelerationMode));

    MOCK_METHOD2(Gna2RequestEnqueue, Gna2Status (
        uint32_t requestConfigId,
        uint32_t * requestId));

    MOCK_METHOD2(Gna2RequestWait, Gna2Status (
        uint32_t requestId,
        uint32_t timeoutMilliseconds));

    MOCK_METHOD2(Gna2ModelExportConfigCreate, Gna2Status (
        Gna2UserAllocator userAllocator,
        uint32_t * exportConfigId));

    MOCK_METHOD1(Gna2ModelExportConfigRelease, Gna2Status (
        uint32_t exportConfigId));

    MOCK_METHOD3(Gna2ModelExportConfigSetSource, Gna2Status (
        uint32_t exportConfigId,
        uint32_t sourceDeviceIndex,
        uint32_t sourceModelId));

    MOCK_METHOD2(Gna2ModelExportConfigSetTarget, Gna2Status (
        uint32_t exportConfigId,
        enum Gna2DeviceVersion targetDeviceVersion));

    MOCK_METHOD4(Gna2ModelExport, Gna2Status (
        uint32_t exportConfigId,
        enum Gna2ModelExportComponent componentType,
        void ** exportBuffer,
        uint32_t * exportBufferSize));

    MOCK_METHOD2(Gna2DeviceGetVersion, Gna2Status (
        uint32_t deviceIndex,
        enum Gna2DeviceVersion * deviceVersion));

    MOCK_METHOD4(Gna2InstrumentationConfigCreate, Gna2Status (
        uint32_t numberOfInstrumentationPoints,
        enum Gna2InstrumentationPoint* selectedInstrumentationPoints,
        uint64_t * results,
        uint32_t * instrumentationConfigId));

    MOCK_METHOD2(Gna2InstrumentationConfigAssignToRequestConfig, Gna2Status (
        uint32_t instrumentationConfigId,
        uint32_t requestConfigId));
#endif
};
