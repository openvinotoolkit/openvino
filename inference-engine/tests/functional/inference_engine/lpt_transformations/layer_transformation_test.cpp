// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include "low_precision_transformations/layer_transformation.hpp"
#include "low_precision_transformations/fake_quantize.hpp"

#include <ie_data.h>

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

class LayerTransformationTests : public ::testing::Test {
protected:
    const QuantizationDetails i8levels255WithoutZeroPoint = QuantizationDetails(255ul, { -1.27f }, { 1.27f }, { -1.27f }, { 1.27f }, 1ul, 1ul, 1ul);
    const QuantizationDetails i8levels255WithZeroPoint = QuantizationDetails(255ul, { -1.27f / 2.f }, { 1.27f }, { -1.27f / 2.f }, { 1.27f }, 1ul, 1ul, 1ul);
    const QuantizationDetails i8levels256WithoutZeroPoint = QuantizationDetails(256ul, { -1.28f }, { 1.27f }, { -1.28f }, { 1.27f }, 1ul, 1ul, 1ul);
    const QuantizationDetails u8levels256WithoutZeroPoint = QuantizationDetails(256ul, { 0.f }, { 1.23f }, { 0.f }, { 1.23f }, 1ul, 1ul, 1ul);
    const QuantizationDetails u8levels256WithZeroPoint = QuantizationDetails(256ul, { 0.12f }, { 1.23f }, { 0.12f }, { 1.23f }, 1ul, 1ul, 1ul);
};

TEST_F(LayerTransformationTests, getPrecisionDetailsI8levels255WithoutZeroPoint) {
    LayerTransformation::Params params = LayerTransformation::Params();
    FakeQuantizeTransformation fakeQuantizeTransformation(params);
    const LayerTransformation::PrecisionDetails precisionDetails = fakeQuantizeTransformation.getPrecisionDetails(i8levels255WithoutZeroPoint);
    ASSERT_EQ(Precision::I8, precisionDetails.precision);
    ASSERT_TRUE(precisionDetails.hasNegativeOutput);
    ASSERT_FALSE(precisionDetails.hasZeroPoint);
}

TEST_F(LayerTransformationTests, getPrecisionDetailsI8levels255WithZeroPoint) {
    LayerTransformation::Params params = LayerTransformation::Params();
    FakeQuantizeTransformation fakeQuantizeTransformation(params);
    const LayerTransformation::PrecisionDetails precisionDetails = fakeQuantizeTransformation.getPrecisionDetails(i8levels255WithZeroPoint);
    ASSERT_EQ(Precision::UNSPECIFIED, precisionDetails.precision);
    ASSERT_TRUE(precisionDetails.hasNegativeOutput);
    ASSERT_TRUE(precisionDetails.hasZeroPoint);
}

TEST_F(LayerTransformationTests, getPrecisionDetailsI8levels256WithoutZeroPoint) {
    LayerTransformation::Params params = LayerTransformation::Params();
    FakeQuantizeTransformation fakeQuantizeTransformation(params);
    const LayerTransformation::PrecisionDetails precisionDetails = fakeQuantizeTransformation.getPrecisionDetails(i8levels256WithoutZeroPoint);
    ASSERT_EQ(Precision::I8, precisionDetails.precision);
    ASSERT_TRUE(precisionDetails.hasNegativeOutput);
    ASSERT_FALSE(precisionDetails.hasZeroPoint);
}

TEST_F(LayerTransformationTests, getPrecisionDetailsU8levels256WithoutZeroPoint) {
    LayerTransformation::Params params = LayerTransformation::Params();
    FakeQuantizeTransformation fakeQuantizeTransformation(params);
    const LayerTransformation::PrecisionDetails precisionDetails = fakeQuantizeTransformation.getPrecisionDetails(u8levels256WithoutZeroPoint);
    ASSERT_EQ(Precision::U8, precisionDetails.precision);
    ASSERT_FALSE(precisionDetails.hasNegativeOutput);
    ASSERT_FALSE(precisionDetails.hasZeroPoint);
}

TEST_F(LayerTransformationTests, getPrecisionDetailsU8levels256WithZeroPoint) {
    LayerTransformation::Params params = LayerTransformation::Params();
    FakeQuantizeTransformation fakeQuantizeTransformation(params);
    const LayerTransformation::PrecisionDetails precisionDetails = fakeQuantizeTransformation.getPrecisionDetails(u8levels256WithZeroPoint);
    ASSERT_EQ(Precision::UNSPECIFIED, precisionDetails.precision);
    ASSERT_FALSE(precisionDetails.hasNegativeOutput);
    ASSERT_TRUE(precisionDetails.hasZeroPoint);
}
