// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include "multi/multi_remote_blob_tests.hpp"
#include "common_test_utils/test_constants.hpp"

const std::vector<DevicesNamesAndSupportPair> device_names_and_support_for_remote_blobs {
        {{CPU}, false}, // CPU via MULTI
};

INSTANTIATE_TEST_CASE_P(smoke_RemoteBlobMultiCPU, MultiDevice_SupportTest,
        ::testing::ValuesIn(device_names_and_support_for_remote_blobs), MultiDevice_SupportTest::getTestCaseName);
