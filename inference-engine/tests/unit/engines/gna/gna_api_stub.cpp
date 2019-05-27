// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define INTEL_GNA_DLLEXPORT 1
#include <gna-api.h>
#include <gna-api-dumper.h>
#include <gna-api-instrumentation.h>
#include "gna_mock_api.hpp"

static GNACppApi * current = nullptr;

GNACppApi :: GNACppApi() {
    current = this;
}

GNACppApi :: ~GNACppApi() {
    current = nullptr;
}

#ifdef __cplusplus
extern "C" {  // API uses C linkage so that it can be used by C and C++ applications
#endif


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
    intel_gna_handle_t          handle,
    const intel_feature_type_t* pFeatureType,
    const intel_feature_t*      pFeatureData,
    const intel_gmm_type_t*     pModelType,
    const intel_gmm_t*          pModelData,
    const uint32_t*             pActiveGMMIndices,
    uint32_t                    nActiveGMMIndices,
    uint32_t                    uMaximumScore,
    intel_gmm_mode_t            nGMMMode,
    uint32_t*                   pScores,
    uint32_t*                   pReqId,
    intel_gna_proc_t            nAccelerationType
) {
    if (current != nullptr) {
        return current->GNAScoreGaussians(
            //handle,
            //pFeatureType,
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
    intel_gna_handle_t          handle,
    const intel_nnet_type_t*    pNeuralNetwork,
    const uint32_t*             pActiveIndices,
    uint32_t                    nActiveIndices,
    uint32_t*                   pReqId,
    intel_gna_proc_t            nAccelerationType
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
    uint32_t           sizeRequested,
    uint32_t*          sizeGranted
) {
    if (current != nullptr) {
        return current->GNAAlloc(nGNADevice, sizeRequested, sizeGranted);
    }
    if (sizeGranted != nullptr) {
        *sizeGranted = sizeRequested;
    }
    return (void*)1;
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
    intel_gna_status_t* status	        // Status of the call
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
    intel_gna_status_t* status,	        // Status of the call
    uint8_t n_threads				// Number of worker threads
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
    intel_gna_handle_t nGNADevice // handle to GNA accelerator
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
    uint32_t           nTimeoutMilliseconds,
    uint32_t           reqId                  // IN score request ID
) {
    if (current != nullptr) {
        return current->GNAWait(nGNADevice, nTimeoutMilliseconds, reqId);
    }
    return GNA_NOERROR;
}

DLLDECL intel_gna_status_t GNAWaitPerfRes(
    intel_gna_handle_t nGNADevice,            // handle to GNA accelerator
    uint32_t           nTimeoutMilliseconds,
    uint32_t           reqId,                 // IN score request ID
    intel_gna_perf_t*  nGNAPerfResults
) {
    if (current != nullptr) {
        return current->GNAWaitPerfRes(nGNADevice,
                                       nTimeoutMilliseconds,
                                       reqId,
                                       nGNAPerfResults);
    }
    return GNA_NOERROR;
}

DLLDECL void* GNADumpXnn(
    const intel_nnet_type_t*    neuralNetwork,
    const uint32_t*             activeIndices,
    uint32_t                    activeIndicesCount,
    intel_gna_model_header*     modelHeader,
    intel_gna_status_t*         status,
    intel_gna_alloc_cb          customAlloc) {
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
#ifdef __cplusplus
}
#endif

