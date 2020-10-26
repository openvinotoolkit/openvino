// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "low_precision/transformer.hpp"

using namespace ::testing;
using namespace ngraph::pass::low_precision;

class LowPrecisionTransformationsTests : public Test {};

TEST_F(LowPrecisionTransformationsTests, remove) {
    LowPrecisionTransformations transformations = LowPrecisionTransformer::getAllTransformations(LayerTransformation::Params());
    auto transformation = transformations.find("Convolution");
    ASSERT_NE(0, transformation.size());

    transformations.remove("Convolution");
    transformation = transformations.find("Convolution");
    ASSERT_EQ(0, transformation.size());
}

TEST_F(LowPrecisionTransformationsTests, removeBranchSpecificTransformations) {
    LowPrecisionTransformations transformations = LowPrecisionTransformer::getAllTransformations(LayerTransformation::Params());
    auto transformation = transformations.find("Concat");
    ASSERT_NE(0, transformation.size());

    transformations.removeBranchSpecificTransformations("Concat");
    transformation = transformations.find("Concat");
    ASSERT_EQ(0, transformation.size());
}

TEST_F(LowPrecisionTransformationsTests, removeTransformations) {
    LowPrecisionTransformations transformations = LowPrecisionTransformer::getAllTransformations(LayerTransformation::Params());
    auto transformation = transformations.find("MatMul");
    ASSERT_NE(0, transformation.size());

    transformations.removeTransformations("MatMul");
    transformation = transformations.find("MatMul");
    ASSERT_EQ(0, transformation.size());
}

TEST_F(LowPrecisionTransformationsTests, removeCleanupTransformations) {
    LowPrecisionTransformations transformations = LowPrecisionTransformer::getAllTransformations(LayerTransformation::Params());
    auto transformation = transformations.find("Multiply");
    ASSERT_NE(0, transformation.size());
    const size_t originalSize = transformation.size();

    transformations.removeCleanupTransformations("Multiply");
    transformation = transformations.find("Multiply");
    ASSERT_EQ(originalSize - 1, transformation.size());
}
