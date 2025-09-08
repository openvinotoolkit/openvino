// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/runtime/remote_tensor.hpp>

using namespace ::testing;
using namespace std;

TEST(RemoteTensorOVTests, throwsOnGetParams) {
    ov::RemoteTensor tensor;
    ASSERT_THROW(tensor.get_params(), ov::Exception);
}

TEST(RemoteTensorOVTests, throwsOnGetDeviceName) {
    ov::RemoteTensor tensor;
    ASSERT_THROW(tensor.get_device_name(), ov::Exception);
}

TEST(RemoteTensorOVTests, remoteTensorFromEmptyTensorThrow) {
    ov::Tensor empty_tensor;
    ov::RemoteTensor remote_tensor;
    ASSERT_FALSE(empty_tensor.is<ov::RemoteTensor>());
    ASSERT_THROW(empty_tensor.as<ov::RemoteTensor>(), ov::Exception);
}

TEST(RemoteTensorOVTests, remoteTensorConvertToRemoteThrow) {
    ov::Tensor tensor{ov::element::f32, {1, 2, 3, 4}};
    ov::RemoteTensor remote_tensor;
    ASSERT_FALSE(tensor.is<ov::RemoteTensor>());
    ASSERT_THROW(tensor.as<ov::RemoteTensor>(), ov::Exception);
}
