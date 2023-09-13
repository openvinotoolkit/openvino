// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "low_precision/concat.hpp"
#include "low_precision/convolution.hpp"
#include "low_precision/mat_mul.hpp"
#include "low_precision/fuse_convert.hpp"

using namespace ::testing;
using namespace ov::pass::low_precision;

class smoke_LPT_LowPrecisionTransformationsTests : public Test {};

// TODO: LPT: not implemented
TEST_F(smoke_LPT_LowPrecisionTransformationsTests, DISABLED_removeAll) {
    //TODO: FIXME
    ASSERT_EQ(1, 0);
    //LowPrecisionTransformations transformations = LowPrecisionTransformer::getAllTransformations(LayerTransformation::Params());
    //auto transformation = transformations.find("Convolution");
    //ASSERT_NE(0, transformation.size());

    //transformations.removeAll<ov::pass::low_precision::ConvolutionTransformation, ov::opset1::Convolution>();
    //transformation = transformations.find("Convolution");
    //ASSERT_EQ(0, transformation.size());
}
//
//TEST_F(LowPrecisionTransformationsTests, removeBranchSpecific) {
//    LowPrecisionTransformations transformations = LowPrecisionTransformer::getAllTransformations(LayerTransformation::Params());
//    auto transformation = transformations.find("Concat");
//    ASSERT_NE(0, transformation.size());
//
//    transformations.removeBranchSpecific<ov::pass::low_precision::ConcatMultiChannelsTransformation, ov::opset1::Concat>();
//    transformation = transformations.find("Concat");
//    ASSERT_EQ(0, transformation.size());
//}
//
//TEST_F(LowPrecisionTransformationsTests, remove) {
//    LowPrecisionTransformations transformations = LowPrecisionTransformer::getAllTransformations(LayerTransformation::Params());
//    auto transformation = transformations.find("MatMul");
//    ASSERT_NE(0, transformation.size());
//
//    transformations.remove<ov::pass::low_precision::MatMulTransformation, ov::opset1::MatMul>();
//    transformation = transformations.find("MatMul");
//    ASSERT_EQ(0, transformation.size());
//}
//
//TEST_F(LowPrecisionTransformationsTests, removeCleanup) {
//    LowPrecisionTransformations transformations = LowPrecisionTransformer::getAllTransformations(LayerTransformation::Params());
//    auto transformation = transformations.find("Multiply");
//    ASSERT_NE(0, transformation.size());
//    const size_t originalSize = transformation.size();
//
//    transformations.removeCleanup<ov::pass::low_precision::FuseConvertTransformation, ov::op::v1::Multiply>();
//    transformation = transformations.find("Multiply");
//    ASSERT_EQ(originalSize - 1, transformation.size());
//}
//
//TEST_F(LowPrecisionTransformationsTests, removeStandaloneCleanup) {
//    LowPrecisionTransformations transformations = LowPrecisionTransformer::getAllTransformations(LayerTransformation::Params());
//    auto transformation = transformations.find("Multiply");
//    ASSERT_NE(0, transformation.size());
//    const size_t originalSize = transformation.size();
//
//    transformations.removeStandaloneCleanup<ov::pass::low_precision::SubtractMultiplyToMultiplyAddTransformation, ov::op::v1::Multiply>();
//    transformation = transformations.find("Multiply");
//    ASSERT_EQ(originalSize - 1, transformation.size());
//}
