// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cpp/ie_infer_request.hpp>
#include <openvino/core/tensor.hpp>

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;


TEST(TensorOVTests, throwsOnUninitializedData) {
    ov::Tensor tensor;
    ASSERT_THROW(tensor.data(), ov::Exception);
}

TEST(TensorOVTests, throwsOnUninitializedAs) {
    ov::Tensor tensor;
    ASSERT_THROW(tensor.data<float>(), ov::Exception);
}

TEST(TensorOVTests, throwsOnAsWithIncorrectElementType) {
    ov::Tensor tensor{ov::element::f32, {1, 1, 2, 2}};
    ASSERT_THROW(tensor.data<std::int32_t>(), ov::Exception);
}

TEST(TensorOVTests, throwsOnUninitializedGetElementType) {
    ov::Tensor tensor;
    ASSERT_THROW(tensor.get_element_type(), ov::Exception);
}

TEST(TensorOVTests, throwsOnUninitializedSetShape) {
    ov::Tensor tensor;
    ASSERT_THROW(tensor.set_shape({}), ov::Exception);
}

TEST(TensorOVTests, throwsOnUninitializedGetShape) {
    ov::Tensor tensor;
    ASSERT_THROW(tensor.get_shape(), ov::Exception);
}

TEST(TensorOVTests, throwsOnUninitializedGetSize) {
    ov::Tensor tensor;
    ASSERT_THROW(tensor.get_size(), ov::Exception);
}

TEST(TensorOVTests, throwsOnUninitializedGetByteSize) {
    ov::Tensor tensor;
    ASSERT_THROW(tensor.get_byte_size(), ov::Exception);
}

TEST(TensorOVTests, throwsOnUninitializedGetStrides) {
    ov::Tensor tensor;
    ASSERT_THROW(tensor.get_strides(), ov::Exception);
}
