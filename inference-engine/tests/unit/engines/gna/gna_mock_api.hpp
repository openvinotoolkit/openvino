//
// Copyright 2017-2018 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once
#include <gmock/gmock-generated-function-mockers.h>

class GNACppApi {

 public:
    GNACppApi();
    ~GNACppApi();
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
};
