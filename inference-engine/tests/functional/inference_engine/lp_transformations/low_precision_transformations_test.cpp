// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "low_precision/concat.hpp"
#include "low_precision/convolution.hpp"
#include "low_precision/mat_mul.hpp"
#include "low_precision/fuse_convert.hpp"
#include "low_precision/subtract_multiply_to_multiply_add.hpp"

using namespace ::testing;
using namespace ngraph::pass::low_precision;

class smoke_LPT_LowPrecisionTransformationsTests : public Test {};

// TODO: LPT: not implemented
TEST_F(smoke_LPT_LowPrecisionTransformationsTests, DISABLED_removeAll) {
    //TODO: FIXME
    ASSERT_EQ(1, 0);
    //LowPrecisionTransformations transformations = LowPrecisionTransformer::getAllTransformations(LayerTransformation::Params());
    //auto transformation = transformations.find("Convolution");
    //ASSERT_NE(0, transformation.size());

    //transformations.removeAll<ngraph::pass::low_precision::ConvolutionTransformation, ngraph::opset1::Convolution>();
    //transformation = transformations.find("Convolution");
    //ASSERT_EQ(0, transformation.size());
}
//
//TEST_F(LowPrecisionTransformationsTests, removeBranchSpecific) {
//    LowPrecisionTransformations transformations = LowPrecisionTransformer::getAllTransformations(LayerTransformation::Params());
//    auto transformation = transformations.find("Concat");
//    ASSERT_NE(0, transformation.size());
//
//    transformations.removeBranchSpecific<ngraph::pass::low_precision::ConcatMultiChannelsTransformation, ngraph::opset1::Concat>();
//    transformation = transformations.find("Concat");
//    ASSERT_EQ(0, transformation.size());
//}
//
//TEST_F(LowPrecisionTransformationsTests, remove) {
//    LowPrecisionTransformations transformations = LowPrecisionTransformer::getAllTransformations(LayerTransformation::Params());
//    auto transformation = transformations.find("MatMul");
//    ASSERT_NE(0, transformation.size());
//
//    transformations.remove<ngraph::pass::low_precision::MatMulTransformation, ngraph::opset1::MatMul>();
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
//    transformations.removeCleanup<ngraph::pass::low_precision::FuseConvertTransformation, ngraph::opset1::Multiply>();
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
//    transformations.removeStandaloneCleanup<ngraph::pass::low_precision::SubtractMultiplyToMultiplyAddTransformation, ngraph::opset1::Multiply>();
//    transformation = transformations.find("Multiply");
//    ASSERT_EQ(originalSize - 1, transformation.size());
//}
