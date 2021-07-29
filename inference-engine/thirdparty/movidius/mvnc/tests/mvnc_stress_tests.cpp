// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "mvnc.h"
#include "ncPrivateTypes.h"
#include "mvnc_stress_test_cases.h"

//------------------------------------------------------------------------------
//      MvncStressTests Tests
//------------------------------------------------------------------------------
/**
* @brief Open and close device for 1001 times
*/
TEST_P(MvncStressTests, OpenClose1001) {
    const int iterations = 1001;
    ncDeviceHandle_t *deviceHandle = nullptr;
    ncDeviceDescr_t deviceDesc = {};
    deviceDesc.protocol = _deviceProtocol;
    deviceDesc.platform = NC_ANY_PLATFORM;

    for (int i = 0; i < iterations; ++i) {
        printf("Iteration %d of %d\n", i, iterations);
        ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, deviceDesc, m_ncDeviceOpenParams));
        ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle, m_watchdogHndl));
        deviceHandle = nullptr;
    }
}

/**
* @brief Allocate and deallocate graph on device for 1001 times
*/
TEST_P(MvncStressTests, AllocateDeallocateGraph1001) {
    const int iterations = 1001;
    ncDeviceDescr_t deviceDesc = {};
    deviceDesc.protocol = _deviceProtocol;
    deviceDesc.platform = NC_ANY_PLATFORM;

    // Load graph
    const std::string blobPath = "bvlc_googlenet_fp16.blob";
    std::vector<char> _blob;

    if (!readBINFile(blobPath, _blob)) GTEST_SKIP_("Blob not found\n");

    // Open device
    ncDeviceHandle_t *deviceHandle = nullptr;
    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, deviceDesc, m_ncDeviceOpenParams));

    for (int i = 0; i < iterations; ++i) {
        printf("Iteration %d of %d\n", i, iterations);

        // Create graph handlers
        ncGraphHandle_t*  graphHandle = nullptr;
        std::string graphName = "graph";

        ASSERT_NO_ERROR(ncGraphCreate(graphName.c_str(), &graphHandle));
        ASSERT_TRUE(graphHandle != nullptr);

        // Allocate graph
        ASSERT_NO_ERROR(ncGraphAllocate(deviceHandle, graphHandle,
                                        _blob.data(), _blob.size(),     // Blob
                                        _blob.data(), sizeof(ElfN_Ehdr) + sizeof(blob_header_v2)) );   // Header

        // Destroy graph
        ASSERT_NO_ERROR(ncGraphDestroy(&graphHandle));
    }
    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle, m_watchdogHndl));
}


/**
* @brief Run the full cycle of inference 101 times.
* It includes opening device, allocating graph and fifos, inference,
 * destroying graph and fifos, closing device
*/
TEST_P(MvncStressTests, FullCycleOfWork101Times) {
    const int iterations = 101;
    ncDeviceDescr_t deviceDesc = {};
    deviceDesc.protocol = _deviceProtocol;
    deviceDesc.platform = NC_ANY_PLATFORM;

    const std::string blobPath = "bvlc_googlenet_fp16.blob";
    std::vector<char> blob;
    if (!readBINFile(blobPath, blob)) GTEST_SKIP_("Blob not found\n");

    for (int i = 0; i < iterations; i++) {
        ncDeviceHandle_t *deviceHandle = nullptr;
        ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, deviceDesc, m_ncDeviceOpenParams));

        ncGraphHandle_t*  graphHandle = nullptr;
        std::string graphName = "graph";
        ASSERT_NO_ERROR(ncGraphCreate(graphName.c_str(), &graphHandle));
        ASSERT_TRUE(graphHandle != nullptr);

        ASSERT_NO_ERROR(ncGraphAllocate(deviceHandle, graphHandle,
                                        blob.data(), blob.size(),     // Blob
                                        blob.data(), sizeof(ElfN_Ehdr) + sizeof(blob_header_v2) ));


        unsigned int dataLength = sizeof(int);

        int numInputs = 0;
        ASSERT_NO_ERROR(ncGraphGetOption(graphHandle, NC_RO_GRAPH_INPUT_COUNT, &numInputs, &dataLength));

        int numOutputs = 0;
        ASSERT_NO_ERROR(ncGraphGetOption(graphHandle, NC_RO_GRAPH_OUTPUT_COUNT, &numOutputs, &dataLength));

        dataLength = sizeof(ncTensorDescriptor_t);

        ncTensorDescriptor_t inputDesc = {};
        ASSERT_NO_ERROR(ncGraphGetOption(graphHandle, NC_RO_GRAPH_INPUT_TENSOR_DESCRIPTORS, &inputDesc,
                                         &dataLength));


        ncTensorDescriptor_t outputDesc = {};
        ASSERT_NO_ERROR(ncGraphGetOption(graphHandle, NC_RO_GRAPH_OUTPUT_TENSOR_DESCRIPTORS, &outputDesc,
                                         &dataLength));

        unsigned int fifo_elements = 4;

        ncFifoHandle_t *inputFifoHandle = nullptr;
        ASSERT_NO_ERROR(ncFifoCreate("input", NC_FIFO_HOST_WO, &inputFifoHandle));

        ASSERT_NO_ERROR(ncFifoAllocate(inputFifoHandle, deviceHandle, &inputDesc, fifo_elements));

        ncFifoHandle_t *outputFifoHandle = nullptr;
        ASSERT_NO_ERROR(ncFifoCreate("output", NC_FIFO_HOST_RO, &outputFifoHandle));

        ASSERT_NO_ERROR(ncFifoAllocate(outputFifoHandle, deviceHandle, &outputDesc, fifo_elements));

        uint8_t *input_data = new uint8_t[inputDesc.totalSize];
        uint8_t *result_data = new uint8_t[outputDesc.totalSize];
        ASSERT_NO_ERROR(ncGraphQueueInferenceWithFifoElem(graphHandle,
                                                          inputFifoHandle, outputFifoHandle,
                                                          input_data, &inputDesc.totalSize, nullptr));

        void *userParam = nullptr;
        ASSERT_NO_ERROR(ncFifoReadElem(outputFifoHandle, result_data, &outputDesc.totalSize, &userParam));

        delete[] input_data;
        delete[] result_data;
        ASSERT_NO_ERROR(ncFifoDestroy(&inputFifoHandle));
        ASSERT_NO_ERROR(ncFifoDestroy(&outputFifoHandle));

        ASSERT_NO_ERROR(ncGraphDestroy(&graphHandle));

        ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle, m_watchdogHndl));
    }

}

INSTANTIATE_TEST_SUITE_P(MvncTestsCommon,
                        MvncStressTests,
                        ::testing::ValuesIn(myriadProtocols),
                        PrintToStringParamName());
