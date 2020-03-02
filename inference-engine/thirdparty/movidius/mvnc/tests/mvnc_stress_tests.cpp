// Copyright (C) 2018-2020 Intel Corporation
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

INSTANTIATE_TEST_CASE_P(MvncTestsCommon,
                        MvncStressTests,
                        ::testing::ValuesIn(myriadProtocols),
                        PrintToStringParamName());
