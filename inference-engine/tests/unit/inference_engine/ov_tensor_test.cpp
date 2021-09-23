// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_blob.h>
#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>

#include "unit_test_utils/mocks/mock_allocator.hpp"
#include "openvino/core/tensor.hpp"

class OVTensorTest: public ::testing::Test {};

TEST_F(OVTensorTest, canCreateTensor) {
    ov::Shape shape = { 1, 1, 1 };
    ov::Tensor t{ov::element::f32, shape};
    ASSERT_NE(t.get_size(), 0);
    ASSERT_NE(t.data(), nullptr);
}

TEST_F(OVTensorTest, canCreateTensorUsingAllocator) {
    ov::Shape shape = {1, 2, 3};
    auto allocator = std::make_shared<OVMockAllocator>();

    EXPECT_CALL(*allocator, allocate(::testing::_, ::testing::_)).WillRepeatedly(testing::Return(reinterpret_cast<void*>(1)));
    EXPECT_CALL(*allocator, deallocate(::testing::_, ::testing::_, ::testing::_)).Times(1);

    {
        ov::Tensor t{ov::element::f32, shape, ov::Allocator{allocator}};
    }
}

TEST_F(OVTensorTest, canAccessExternalData) {
    ov::Shape shape = {1, 1, 3};
    float data[] = {5.f, 6.f, 7.f};
    ov::Tensor t{ov::element::f32, shape, data, 3};
    {
        float *ptr = t.data<float>();
        ASSERT_EQ(ptr[2] , 7);
    }
}

TEST_F(OVTensorTest, canAccessExternalDataWithStrides) {
    ov::Shape shape = {2, 3};
    float data[] = {5.f, 6.f, 7.f, 0.f, 1.f, 42.f, 3.f, 0.f};
    ov::Tensor t{ov::element::f32, shape, data, 8, {4, 1}};
    {
        ASSERT_EQ((ov::Shape{2, 3}), t.get_shape());
        float *ptr = t.data<float>();
        ASSERT_EQ(ptr[5] , 42);
    }
}

TEST_F(OVTensorTest, saveDimsAndSizeAfterMove) {
    ov::Shape shape = {1, 2, 3};
    ov::Tensor t{ov::element::f32, shape};

    ov::Tensor new_tensor(std::move(t));

    ASSERT_EQ(new_tensor.get_shape(), (ov::Shape{1, 2, 3}));
    ASSERT_EQ(new_tensor.get_element_type(), ov::element::f32);
}

// SetShape
TEST_F(OVTensorTest, canSetShape) {
    ov::Tensor t{ov::element::f32, {1, 2, 3}};
    ASSERT_EQ(t.get_shape(), (ov::Shape{1, 2, 3}));
    ASSERT_NO_THROW(t.set_shape({4, 5, 6}));
    ASSERT_EQ(t.get_shape(), (ov::Shape{4, 5, 6}));
}

TEST_F(OVTensorTest, makeRangeRoiTensor) {
    ov::Tensor t{ov::element::i8, {1, 3, 6, 5}}; // RGB picture of size (WxH) = 5x6
    ov::Tensor roi_tensor{t, {0, 0, 1, 2}, {1, 3, 5, 4}};
    ov::Shape ref_shape = {1, 3, 4, 2};
    ptrdiff_t ref_offset = 7;
    ov::Strides ref_strides = {90, 30, 5, 1};
    ASSERT_EQ(roi_tensor.get_shape(), ref_shape);
    ASSERT_EQ(roi_tensor.data<int8_t>() - t.data<int8_t>(), ref_offset);
    ASSERT_EQ(roi_tensor.get_strides(), ref_strides);
}

TEST_F(OVTensorTest, makeRangeRoiTensorInt4) {
    ov::Tensor t{ov::element::i4, {1, 3, 6, 5}}; // RGB picture of size (WxH) = 5x6
    ov::Tensor roi_tensor{t, {0, 0, 1, 2}, {1, 3, 5, 4}};
    ov::Shape ref_shape = {1, 3, 4, 2};
    ptrdiff_t ref_offset = 7;
    ov::Strides ref_strides = {90, 30, 5, 1};
    ASSERT_EQ(roi_tensor.get_shape(), ref_shape);
    ASSERT_EQ(roi_tensor.data<int8_t>() - t.data<int8_t>(), ref_offset);
    ASSERT_EQ(roi_tensor.get_strides(), ref_strides);
}

TEST_F(OVTensorTest, makeRangeRoiBlobWrongSize) {
    ov::Tensor t{ov::element::f32, {1, 3, 4, 4}};
    ASSERT_THROW((ov::Tensor{t, {0, 0, 1, 1}, {1, 3, 5, 5}}), ov::Exception);
}

TEST_F(OVTensorTest, readRangeRoiBlob) {
    ov::Tensor t{ov::element::i32, {1, 3, 4, 8}};
    {
        const auto origPtr = t.data<int32_t>();
        ASSERT_NE(nullptr, origPtr);
        for (size_t i = 0; i < t.get_size(); ++i) {
            origPtr[i] = i;
        }
    }
    ov::Tensor roi_tensor{t, {0, 0, 2, 4}, {1, 3, 4, 8}};
    ASSERT_NE(false, static_cast<bool>(roi_tensor));
    {
        auto roi = roi_tensor.data<int32_t>();
        ASSERT_NE(nullptr, roi);
        auto strides = roi_tensor.get_strides();
        for (size_t n = 0; n < roi_tensor.get_shape()[0]; ++n) {
        for (size_t c = 0; c < roi_tensor.get_shape()[1]; ++c) {
        for (size_t h = 0; h < roi_tensor.get_shape()[2]; ++h) {
        for (size_t w = 0; w < roi_tensor.get_shape()[3]; ++w) {
            auto actual = roi[w*strides[3] + h*strides[2] + c*strides[1] + n*strides[0]];
            auto expected = t.data<int32_t>()[(w + 4)*strides[3] + (h + 2)*strides[2] + (c + 0)*strides[1] + (n + 0)*strides[0]];
            ASSERT_EQ(expected, actual) << ov::Shape{n, c, h , w};
        }}}}
    }
}
