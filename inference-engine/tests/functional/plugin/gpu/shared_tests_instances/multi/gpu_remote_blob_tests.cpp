// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include "multi/multi_remote_blob_tests.hpp"
#include "common_test_utils/test_constants.hpp"

const std::vector<DevicesNamesAndSupportPair> device_names_and_support_for_remote_blobs {
        {{GPU}, true}, // GPU via MULTI,
#if ENABLE_MKL_DNN
        {{GPU, CPU}, true}, // GPU+CPU
        {{CPU, GPU}, true}, // CPU+GPU
#endif
};

INSTANTIATE_TEST_CASE_P(smoke_RemoteBlobMultiGPU, MultiDevice_SupportTest,
                        ::testing::ValuesIn(device_names_and_support_for_remote_blobs), MultiDevice_SupportTest::getTestCaseName);