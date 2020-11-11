// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include "multi/multi_helpers.hpp"
#include "common_test_utils/test_constants.hpp"

#define MULTI CommonTestUtils::DEVICE_MULTI
#define CPU CommonTestUtils::DEVICE_CPU
#define GPU CommonTestUtils::DEVICE_GPU
#define MYRIAD CommonTestUtils::DEVICE_MYRIAD

const std::vector<DevicesNamesAndSupportPair> device_names_and_support_for_remote_blobs {
#if ENABLE_MKL_DNN
        {{CPU}, false}, // CPU via MULTI
    #if ENABLE_CLDNN
        {{GPU, CPU}, true}, // GPU+CPU
        {{CPU, GPU}, true}, // CPU+GPU
    #endif
    #if ENABLE_MYRIAD
        {{CPU, MYRIAD}, false},  // CPU+MYX
    #endif
#endif
#if ENABLE_CLDNN
        {{GPU}, true}, // GPU via MULTI,
#endif
#if ENABLE_MYRIAD
        {{MYRIAD}, false}, // MYX via MULTI
#endif
};

INSTANTIATE_TEST_CASE_P(smoke_RemoteBlobMulti, MultiDevice_Test,
        ::testing::ValuesIn(device_names_and_support_for_remote_blobs), MultiDevice_Test::getTestCaseName);
