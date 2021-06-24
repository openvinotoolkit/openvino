// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include "multi/multi_remote_blob_tests.hpp"
#include "common_test_utils/test_constants.hpp"

const std::vector<DevicesNamesAndSupportPair> device_names_and_support_for_remote_blobs {
        {{MYRIAD}, false}, // MYX via MULTI
#ifdef ENABLE_MKL_DNN
        {{CPU, MYRIAD}, false},  // CPU+MYX
#endif
};

INSTANTIATE_TEST_SUITE_P(smoke_RemoteBlobMultiMyriad, MultiDevice_SupportTest,
                        ::testing::ValuesIn(device_names_and_support_for_remote_blobs), MultiDevice_SupportTest::getTestCaseName);