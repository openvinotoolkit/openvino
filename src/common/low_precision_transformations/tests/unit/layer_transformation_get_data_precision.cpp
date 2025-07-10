// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <gtest/gtest.h>
#include "low_precision/layer_transformation.hpp"
#include "low_precision/network_helper.hpp"

using namespace ov;

TEST(smoke_LPT_LayerTransformation, getDataPrecision_reqU8_U8_to_U8) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 299, 299});
    const auto low = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{0.f});
    const auto high = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{2.55f});
    const auto fakeQuantize = std::make_shared<ov::op::v0::FakeQuantize>(input, low, high, low, high, 256);

    auto const dequantization = ov::pass::low_precision::QuantizationDetails::getDetails(fakeQuantize);

    auto const precisionDetails = ov::pass::low_precision::LayerTransformation::getDataPrecision(fakeQuantize, dequantization, {element::u8});
    ASSERT_EQ(element::u8, precisionDetails.precision);
    ASSERT_EQ(0.f, precisionDetails.min);
    ASSERT_EQ(255.f, precisionDetails.max);
    ASSERT_EQ(false, precisionDetails.hasZeroPoint);
    ASSERT_EQ(false, precisionDetails.empty());
}

TEST(smoke_LPT_LayerTransformation, getDataPrecision_reqU8_65535_to_U8) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 299, 299});
    const auto low = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{0.f});
    const auto high = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{2.55f});
    const auto fakeQuantize = std::make_shared<ov::op::v0::FakeQuantize>(input, low, high, low, high, 65535);

    auto const dequantization = ov::pass::low_precision::QuantizationDetails::getDetails(fakeQuantize);

    auto const precisionDetails = ov::pass::low_precision::LayerTransformation::getDataPrecision(fakeQuantize, dequantization, {element::u8});
    ASSERT_TRUE(precisionDetails.empty());
}

TEST(smoke_LPT_LayerTransformation, getDataPrecision_reqI8_I8_to_I8) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 299, 299});
    const auto low = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{-1.28f});
    const auto high = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{1.27f});
    const auto fakeQuantize = std::make_shared<ov::op::v0::FakeQuantize>(input, low, high, low, high, 256);

    auto const dequantization = ov::pass::low_precision::QuantizationDetails::getDetails(fakeQuantize);

    auto const precisionDetails =
        ov::pass::low_precision::LayerTransformation::getDataPrecision(fakeQuantize, dequantization, {element::i8});
    ASSERT_EQ(element::i8, precisionDetails.precision);
    ASSERT_EQ(-128.f, precisionDetails.min);
    ASSERT_EQ(127.f, precisionDetails.max);
    ASSERT_EQ(false, precisionDetails.hasZeroPoint);
    ASSERT_EQ(false, precisionDetails.empty());
}

TEST(smoke_LPT_LayerTransformation, getDataPrecision_reqU8_I8_to_U8zp) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 299, 299});
    const auto low = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{-1.28f});
    const auto high = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{1.27f});
    const auto fakeQuantize = std::make_shared<ov::op::v0::FakeQuantize>(input, low, high, low, high, 256);

    auto const dequantization = ov::pass::low_precision::QuantizationDetails::getDetails(fakeQuantize);

    auto const precisionDetails = ov::pass::low_precision::LayerTransformation::getDataPrecision(fakeQuantize, dequantization, {element::u8});
    ASSERT_EQ(element::u8, precisionDetails.precision);
    ASSERT_EQ(0.f, precisionDetails.min);
    ASSERT_EQ(255.f, precisionDetails.max);
    ASSERT_EQ(true, precisionDetails.hasZeroPoint);
    ASSERT_EQ(false, precisionDetails.empty());
}

TEST(smoke_LPT_LayerTransformation, getDataPrecision_reqI8_U8_to_I8zp) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 299, 299});
    const auto low = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{0.f});
    const auto high = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{2.55f});
    const auto fakeQuantize = std::make_shared<ov::op::v0::FakeQuantize>(input, low, high, low, high, 256);

    auto const dequantization = ov::pass::low_precision::QuantizationDetails::getDetails(fakeQuantize);

    auto const precisionDetails = ov::pass::low_precision::LayerTransformation::getDataPrecision(fakeQuantize, dequantization, {element::i8});
    ASSERT_EQ(element::i8, precisionDetails.precision);
    ASSERT_EQ(-128.f, precisionDetails.min);
    ASSERT_EQ(127.f, precisionDetails.max);
    ASSERT_EQ(true, precisionDetails.hasZeroPoint);
    ASSERT_EQ(false, precisionDetails.empty());
}

TEST(smoke_LPT_LayerTransformation, getDataPrecision_reqU8_I8zp_to_U8zp) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 299, 299});
    const auto low = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{-0.875227511f});
    const auto high = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{0.882119000f});
    const auto fakeQuantize = std::make_shared<ov::op::v0::FakeQuantize>(input, low, high, low, high, 256);

    auto const dequantization = ov::pass::low_precision::QuantizationDetails::getDetails(fakeQuantize);

    auto const precisionDetails = ov::pass::low_precision::LayerTransformation::getDataPrecision(fakeQuantize, dequantization, {element::u8});
    ASSERT_EQ(element::u8, precisionDetails.precision);
    ASSERT_EQ(0.f, precisionDetails.min);
    ASSERT_EQ(255.f, precisionDetails.max);
    ASSERT_EQ(true, precisionDetails.hasZeroPoint);
    ASSERT_EQ(false, precisionDetails.empty());
}

TEST(smoke_LPT_LayerTransformation, getDataPrecision_reqI8_U8zp_to_I8zp) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 299, 299});
    const auto low = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{0.875227511f});
    const auto high = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{0.882119000f});
    const auto fakeQuantize = std::make_shared<ov::op::v0::FakeQuantize>(input, low, high, low, high, 256);

    auto const dequantization = ov::pass::low_precision::QuantizationDetails::getDetails(fakeQuantize);

    auto const precisionDetails = ov::pass::low_precision::LayerTransformation::getDataPrecision(fakeQuantize, dequantization, {element::i8});
    ASSERT_EQ(element::i8, precisionDetails.precision);
    ASSERT_EQ(-128.f, precisionDetails.min);
    ASSERT_EQ(127.f, precisionDetails.max);
    ASSERT_EQ(true, precisionDetails.hasZeroPoint);
    ASSERT_EQ(false, precisionDetails.empty());
}

TEST(smoke_LPT_LayerTransformation, getDataPrecision_reqNone_I8zp_to_undefzp) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 299, 299});
    const auto low = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{-0.875227511f});
    const auto high = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{0.882119000f});
    const auto fakeQuantize = std::make_shared<ov::op::v0::FakeQuantize>(input, low, high, low, high, 256);

    auto const dequantization = ov::pass::low_precision::QuantizationDetails::getDetails(fakeQuantize);

    auto const precisionDetails = ov::pass::low_precision::LayerTransformation::getDataPrecision(fakeQuantize, dequantization, {});
    ASSERT_EQ(element::dynamic, precisionDetails.precision);
    ASSERT_EQ(0.f, precisionDetails.min);
    ASSERT_EQ(0.f, precisionDetails.max);
    ASSERT_EQ(false, precisionDetails.hasZeroPoint);
    ASSERT_EQ(true, precisionDetails.empty());
}

TEST(smoke_LPT_LayerTransformation, getDataPrecision_reqNone_U8zp_to_undefzp) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 299, 299});
    const auto low = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{0.875227511f});
    const auto high = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{0.882119000f});
    const auto fakeQuantize = std::make_shared<ov::op::v0::FakeQuantize>(input, low, high, low, high, 256);

    auto const dequantization = ov::pass::low_precision::QuantizationDetails::getDetails(fakeQuantize);

    auto const precisionDetails = ov::pass::low_precision::LayerTransformation::getDataPrecision(fakeQuantize, dequantization, {});
    ASSERT_EQ(element::dynamic, precisionDetails.precision);
    ASSERT_EQ(0.f, precisionDetails.min);
    ASSERT_EQ(0.f, precisionDetails.max);
    ASSERT_EQ(false, precisionDetails.hasZeroPoint);
    ASSERT_EQ(true, precisionDetails.empty());
}
