// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cpp/ie_infer_request.hpp>
#include <openvino/runtime/remote_tensor.hpp>

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

TEST(RemoteTensorOVTests, throwsOnGetParams) {
    ov::runtime::RemoteTensor tensor;
    ASSERT_THROW(tensor.get_params(), ov::Exception);
}

TEST(RemoteTensorOVTests, throwsOnGetDeviceName) {
    ov::runtime::RemoteTensor tensor;
    ASSERT_THROW(tensor.get_device_name(), ov::Exception);
}
