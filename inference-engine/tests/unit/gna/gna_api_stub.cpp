// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#define INTEL_GNA_DLLEXPORT 1

#if GNA_LIB_VER == 1
#include <gna-api.h>
#include <gna-api-instrumentation.h>
#include <gna-api-dumper.h>
#else
#include <gna2-instrumentation-api.h>
#include <gna2-inference-api.h>
#include <gna2-model-export-api.h>
#endif

#include "gna_mock_api.hpp"

static GNACppApi *current = nullptr;

GNACppApi::GNACppApi() {
    current = this;
}

GNACppApi::~GNACppApi() {
    current = nullptr;
}

#ifdef __cplusplus
extern "C" {  // API uses C linkage so that it can be used by C and C++ applications
#endif
#if GNA_LIB_VER == 2

GNA2_API enum Gna2Status Gna2MemoryAlloc(
    uint32_t sizeRequested,
    uint32_t *sizeGranted,
    void **memoryAddress) {
    if (current != nullptr) {
        return current->Gna2MemoryAlloc(sizeRequested, sizeGranted, memoryAddress);
    }
    if (sizeGranted != nullptr) {
        *sizeGranted = sizeRequested;
    }
    *memoryAddress = reinterpret_cast<void*>(1);
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2DeviceOpen(
    uint32_t deviceIndex) {
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2DeviceSetNumberOfThreads(
    uint32_t deviceIndex,
    uint32_t numberOfThreads) {
    return Gna2StatusSuccess;
}

GNA2_API Gna2Status Gna2DeviceClose(
    uint32_t deviceIndex) {
    return Gna2StatusSuccess;
}

GNA2_API Gna2Status Gna2DeviceGetCount(
    uint32_t* numberOfDevices) {
    if (numberOfDevices != nullptr) {
        *numberOfDevices = 1;
    }
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2MemoryFree(
    void * memory) {
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2StatusGetMessage(enum Gna2Status status,
    char * messageBuffer, uint32_t messageBufferSize) {
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2ModelCreate(
    uint32_t deviceIndex,
    struct Gna2Model const * model,
    uint32_t * modelId) {
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2ModelRelease(
    uint32_t modelId) {
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2ModelGetLastError(struct Gna2ModelError* error) {
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2RequestConfigCreate(
    uint32_t modelId,
    uint32_t * requestConfigId) {
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2RequestConfigEnableActiveList(
    uint32_t requestConfigId,
    uint32_t operationIndex,
    uint32_t numberOfIndices,
    uint32_t const * indices) {
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2RequestConfigEnableHardwareConsistency(
    uint32_t requestConfigId,
    enum Gna2DeviceVersion deviceVersion) {
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2RequestConfigSetAccelerationMode(
    uint32_t requestConfigId,
    enum Gna2AccelerationMode accelerationMode) {
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2RequestEnqueue(
    uint32_t requestConfigId,
    uint32_t * requestId) {
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2RequestWait(
    uint32_t requestId,
    uint32_t timeoutMilliseconds) {
    if (current != nullptr) {
        return current->Gna2RequestWait(requestId, timeoutMilliseconds);
    }
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2ModelExportConfigCreate(
    Gna2UserAllocator userAllocator,
    uint32_t * exportConfigId) {
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2ModelExportConfigRelease(
    uint32_t exportConfigId) {
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2ModelExportConfigSetSource(
    uint32_t exportConfigId,
    uint32_t sourceDeviceIndex,
    uint32_t sourceModelId) {
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2ModelExportConfigSetTarget(
    uint32_t exportConfigId,
    enum Gna2DeviceVersion targetDeviceVersion) {
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2ModelExport(
    uint32_t exportConfigId,
    enum Gna2ModelExportComponent componentType,
    void ** exportBuffer,
    uint32_t * exportBufferSize) {
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2DeviceGetVersion(
    uint32_t deviceIndex,
    enum Gna2DeviceVersion * deviceVersion) {
    *deviceVersion = Gna2DeviceVersionSoftwareEmulation;
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2InstrumentationConfigCreate(
    uint32_t numberOfInstrumentationPoints,
    enum Gna2InstrumentationPoint* selectedInstrumentationPoints,
    uint64_t * results,
    uint32_t * instrumentationConfigId) {
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2InstrumentationConfigAssignToRequestConfig(
    uint32_t instrumentationConfigId,
    uint32_t requestConfigId) {
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2GetLibraryVersion(
    char* versionBuffer,
    uint32_t versionBufferSize) {
    if (versionBuffer != nullptr && versionBufferSize > 0) {
        versionBuffer[0] = '\0';
        return Gna2StatusSuccess;
    }
    return Gna2StatusNullArgumentNotAllowed;
}

#elif GNA_LIB_VER == 1

/**
 * intel_gna_status_t members printable descriptions
 *   Size: NUMGNASTATUS + 1
 */
DLLDECL const char *GNAStatusName[] = {"status"};

/**
 * intel_gmm_mode_t members printable descriptions
 *   Size: NUMGMMMODES + 1
 */
DLLDECL const char *GMMModeName[] = {"model"};

/**
 * // TODO: fill
 */
DLLDECL intel_gna_status_t GNAScoreGaussians(
        intel_gna_handle_t handle,
        const intel_feature_type_t *pFeatureType,
        const intel_feature_t *pFeatureData,
        const intel_gmm_type_t *pModelType,
        const intel_gmm_t *pModelData,
        const uint32_t *pActiveGMMIndices,
        uint32_t nActiveGMMIndices,
        uint32_t uMaximumScore,
        intel_gmm_mode_t nGMMMode,
        uint32_t *pScores,
        uint32_t *pReqId,
        intel_gna_proc_t nAccelerationType
) {
    if (current != nullptr) {
        return current->GNAScoreGaussians(
                // handle,
                // pFeatureType,
                pFeatureData,
                pModelType,
                pModelData,
                pActiveGMMIndices,
                nActiveGMMIndices,
                uMaximumScore,
                nGMMMode,
                pScores,
                pReqId,
                nAccelerationType);
    }
    return GNA_NOERROR;
}

DLLDECL intel_gna_status_t GNAPropagateForward(
        intel_gna_handle_t handle,
        const intel_nnet_type_t *pNeuralNetwork,
        const uint32_t *pActiveIndices,
        uint32_t nActiveIndices,
        uint32_t *pReqId,
        intel_gna_proc_t nAccelerationType
) {
    if (current != nullptr) {
        return current->GNAPropagateForward(
                handle,
                pNeuralNetwork,
                pActiveIndices,
                nActiveIndices,
                pReqId,
                nAccelerationType);
    }
    return GNA_NOERROR;
}

// TODO: add output status
/**
 * // TODO: fill
 */
DLLDECL void *GNAAlloc(
        intel_gna_handle_t nGNADevice,   // handle to GNA accelerator
        uint32_t sizeRequested,
        uint32_t *sizeGranted
) {
    if (current != nullptr) {
        return current->GNAAlloc(nGNADevice, sizeRequested, sizeGranted);
    }
    if (sizeGranted != nullptr) {
        *sizeGranted = sizeRequested;
    }
    return reinterpret_cast<void *>(1);
}

/**
 * // TODO: fill
 */
DLLDECL intel_gna_status_t GNAFree(
        intel_gna_handle_t nGNADevice   // handle to GNA accelerator
) {
    if (current != nullptr) {
        return current->GNAFree(nGNADevice);
    }
    return GNA_NOERROR;
}

/**
 * // TODO: fill
 */
DLLDECL intel_gna_handle_t GNADeviceOpen(
        intel_gna_status_t *status            // Status of the call
) {
    if (current != nullptr) {
        return current->GNADeviceOpen(status);
    }
    return 0;
}

/**
* // TODO: fill
*/
DLLDECL intel_gna_handle_t GNADeviceOpenSetThreads(
        intel_gna_status_t *status,            // Status of the call
        uint8_t n_threads                // Number of worker threads
) {
    if (current != nullptr) {
        return current->GNADeviceOpenSetThreads(status, n_threads);
    }
    return GNA_NOERROR;
}

/**
 * // TODO: fill
 */
DLLDECL intel_gna_status_t GNADeviceClose(
        intel_gna_handle_t nGNADevice  // handle to GNA accelerator
) {
    if (current != nullptr) {
        return current->GNADeviceClose(nGNADevice);
    }
    return GNA_NOERROR;
}

/**
 * // TODO: fill
 */
DLLDECL intel_gna_status_t GNAWait(
        intel_gna_handle_t nGNADevice,            // handle to GNA accelerator
        uint32_t nTimeoutMilliseconds,
        uint32_t reqId                  // IN score request ID
) {
    if (current != nullptr) {
        return current->GNAWait(nGNADevice, nTimeoutMilliseconds, reqId);
    }
    return GNA_NOERROR;
}

DLLDECL intel_gna_status_t GNAWaitPerfRes(
        intel_gna_handle_t nGNADevice,            // handle to GNA accelerator
        uint32_t nTimeoutMilliseconds,
        uint32_t reqId,                 // IN score request ID
        intel_gna_perf_t *nGNAPerfResults
) {
    if (current != nullptr) {
        return current->GNAWaitPerfRes(nGNADevice,
                                       nTimeoutMilliseconds,
                                       reqId,
                                       nGNAPerfResults);
    }
    return GNA_NOERROR;
}

DLLDECL void *GNADumpXnn(
        const intel_nnet_type_t *neuralNetwork,
        const uint32_t *activeIndices,
        uint32_t activeIndicesCount,
        intel_gna_model_header *modelHeader,
        intel_gna_status_t *status,
        intel_gna_alloc_cb customAlloc) {
    if (current != nullptr) {
        return current->GNADumpXnn(neuralNetwork,
                                   activeIndices,
                                   activeIndicesCount,
                                   modelHeader,
                                   status,
                                   customAlloc);
    }
    return nullptr;
}

DLLDECL void gmmSetThreads(
        int num
) {
    current->gmmSetThreads((num != 0) ? num : 1);
}
#endif  // GNA_LIB_VER == 1

#ifdef __cplusplus
}
#endif
