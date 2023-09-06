// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include "transformations/utils/utils.hpp"

#include "common_test_utils/ov_test_utils.hpp"

using namespace testing;
using namespace ov;
using namespace ov::pass;

// TODO: LPT: not implemented
TEST(DISABLED_LPT, isQuantizedTransformation) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{ 1, 3, 16, 16 });
    const auto mulConst = op::v0::Constant::create(element::f32, Shape{}, { 1.f });
    const auto mul = std::make_shared<ov::op::v1::Multiply>(input, mulConst);
    const auto shapeConst = op::v0::Constant::create(ov::element::i64, ov::Shape{ 4 }, { 1, 3, 16, 16 });
    const auto layer = std::make_shared<ov::op::v1::Reshape>(mul, shapeConst, true);

    // TODO: FIXME
    EXPECT_EQ(1, 0);

    //const auto transformations = ov::pass::low_precision::LowPrecisionTransformer::getAllTransformations();

    //for (const auto& transformation : transformations.transformations) {
    //    ASSERT_NO_THROW(transformation.second->isQuantized(layer));
    //}
}
