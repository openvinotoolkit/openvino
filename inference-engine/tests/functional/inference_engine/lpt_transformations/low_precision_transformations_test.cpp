// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "low_precision_transformations/transformer.hpp"

using namespace ::testing;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

class LowPrecisionTransformationsTests : public Test {};

TEST_F(LowPrecisionTransformationsTests, remove) {
    LowPrecisionTransformations transformations = LowPrecisionTransformer::getAllTransformations(LayerTransformation::Params());
    LayerTransformationPtr transformation = transformations.find("ScaleShift");
    ASSERT_NE(nullptr, transformation);

    transformations.remove("ScaleShift");
    transformation = transformations.find("ScaleShift");
    ASSERT_EQ(nullptr, transformation);
}

TEST_F(LowPrecisionTransformationsTests, removeBranchSpecificTransformations) {
    LowPrecisionTransformations transformations = LowPrecisionTransformer::getAllTransformations(LayerTransformation::Params());
    LayerTransformationPtr transformation = transformations.find("Concat");
    ASSERT_NE(nullptr, transformation);

    transformations.removeBranchSpecificTransformations("Concat");
    transformation = transformations.find("Concat");
    ASSERT_EQ(nullptr, transformation);
}

TEST_F(LowPrecisionTransformationsTests, removeTransformations) {
    LowPrecisionTransformations transformations = LowPrecisionTransformer::getAllTransformations(LayerTransformation::Params());
    LayerTransformationPtr transformation = transformations.find("FullyConnected");
    ASSERT_NE(nullptr, transformation);

    transformations.removeTransformations("FullyConnected");
    transformation = transformations.find("FullyConnected");
    ASSERT_EQ(nullptr, transformation);
}

TEST_F(LowPrecisionTransformationsTests, removeCleanupTransformations) {
    LowPrecisionTransformations transformations = LowPrecisionTransformer::getAllTransformations(LayerTransformation::Params());
    LayerTransformationPtr transformation = transformations.find("ScaleShift");
    ASSERT_NE(nullptr, transformation);

    transformations.removeCleanupTransformations("ScaleShift");
    transformation = transformations.find("ScaleShift");
    ASSERT_EQ(nullptr, transformation);
}
