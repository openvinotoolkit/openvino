// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "low_precision/transformer.hpp"
#include "low_precision/multiply_to_group_convolution.hpp"

using namespace ::testing;
using namespace ngraph::pass::low_precision;

class smoke_LPTLowPrecisionTransformationsTests : public Test {};

TEST_F(smoke_LPTLowPrecisionTransformationsTests, remove) {
    LowPrecisionTransformations transformations = LowPrecisionTransformer::getAllTransformations(LayerTransformation::Params());
    auto transformation = transformations.find("Convolution");
    ASSERT_NE(0, transformation.size());

    transformations.remove("Convolution");
    transformation = transformations.find("Convolution");
    ASSERT_EQ(0, transformation.size());
}

TEST_F(smoke_LPTLowPrecisionTransformationsTests, removeBranchSpecificTransformations) {
    LowPrecisionTransformations transformations = LowPrecisionTransformer::getAllTransformations(LayerTransformation::Params());
    auto transformation = transformations.find("Concat");
    ASSERT_NE(0, transformation.size());

    transformations.removeBranchSpecificTransformations("Concat");
    transformation = transformations.find("Concat");
    ASSERT_EQ(0, transformation.size());
}

TEST_F(smoke_LPTLowPrecisionTransformationsTests, removeTransformations) {
    LowPrecisionTransformations transformations = LowPrecisionTransformer::getAllTransformations(LayerTransformation::Params());
    auto transformation = transformations.find("MatMul");
    ASSERT_NE(0, transformation.size());

    transformations.removeTransformations("MatMul");
    transformation = transformations.find("MatMul");
    ASSERT_EQ(0, transformation.size());
}

TEST_F(smoke_LPTLowPrecisionTransformationsTests, removeCleanupTransformations) {
    LowPrecisionTransformations transformations = LowPrecisionTransformer::getAllTransformations(LayerTransformation::Params());
    auto transformation = transformations.find("Multiply");
    ASSERT_NE(0, transformation.size());
    const size_t originalSize = transformation.size();

    transformations.removeCleanupTransformations("Multiply");
    transformation = transformations.find("Multiply");
    ASSERT_EQ(originalSize - 1, transformation.size());
}

TEST_F(smoke_LPTLowPrecisionTransformationsTests, removeStandaloneCleanupTransformations) {
    LowPrecisionTransformations transformations = LowPrecisionTransformer::getAllTransformations(LayerTransformation::Params());
    auto transformation = transformations.find("MultiplyToGroupConvolutionTransformation");
    ASSERT_NE(1, transformation.size());

    transformations.removeStandaloneCleanup<MultiplyToGroupConvolutionTransformation, ngraph::opset1::Multiply>();
    transformation = transformations.find("MultiplyToGroupConvolutionTransformation");
    ASSERT_EQ(0, transformation.size());
}
