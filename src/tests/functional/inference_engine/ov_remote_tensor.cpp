// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cpp/ie_infer_request.hpp>
#include <openvino/runtime/remote_tensor.hpp>

using namespace ::testing;
using namespace std;

TEST(RemoteTensorOVTests, throwsOnGetParams) {
    ov::runtime::RemoteTensor tensor;
    ASSERT_THROW(tensor.get_params(), ov::Exception);
}

TEST(RemoteTensorOVTests, throwsOnGetDeviceName) {
    ov::runtime::RemoteTensor tensor;
    ASSERT_THROW(tensor.get_device_name(), ov::Exception);
}

TEST(RemoteTensorOVTests, remoteTensorFromEmptyTensorThrow) {
    ov::runtime::Tensor empty_tensor;
    ov::runtime::RemoteTensor remote_tensor;
    ASSERT_FALSE(empty_tensor.is<ov::runtime::RemoteTensor>());
    ASSERT_THROW(empty_tensor.as<ov::runtime::RemoteTensor>(), ov::Exception);
}

TEST(RemoteTensorOVTests, remoteTensorConvertToRemoteThrow) {
    ov::runtime::Tensor tensor{ov::element::f32, {1, 2, 3, 4}};
    ov::runtime::RemoteTensor remote_tensor;
    ASSERT_FALSE(tensor.is<ov::runtime::RemoteTensor>());
    ASSERT_THROW(tensor.as<ov::runtime::RemoteTensor>(), ov::Exception);
}
