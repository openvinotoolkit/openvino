// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/smart_reshape/rematerialized_batch.hpp"

#include <gtest/gtest.h>

#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"

using namespace ov;

namespace {

Output<Node> leading_dimension_of(const std::shared_ptr<op::v0::Parameter>& parameter) {
    const auto shape_of = std::make_shared<op::v3::ShapeOf>(parameter);
    return std::make_shared<op::v8::Gather>(shape_of,
                                            op::v0::Constant::create(element::i64, Shape{1}, {0}),
                                            op::v0::Constant::create(element::i64, Shape{}, {0}));
}

std::shared_ptr<op::v1::Reshape> reshape_to(const Output<Node>& data, const OutputVector& target) {
    return std::make_shared<op::v1::Reshape>(data, std::make_shared<op::v0::Concat>(target, 0), false);
}

}  // namespace

TEST(RematerializedBatchTests, ReportsBatchPinnedBeforeItIsRebuilt) {
    const auto param = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4});
    const auto live = leading_dimension_of(param);
    const auto inferred = op::v0::Constant::create(element::i64, Shape{1}, {-1});
    const auto pinned = op::v0::Constant::create(element::i64, Shape{1}, {1});

    const auto kept = reshape_to(param, {live, op::v0::Constant::create(element::i64, Shape{1}, {4})});
    const auto lost = reshape_to(kept, {pinned, inferred});
    const auto rebuilt = reshape_to(lost, {live, inferred});
    const auto model = std::make_shared<Model>(OutputVector{rebuilt}, ParameterVector{param});

    EXPECT_EQ(find_rematerialized_batch(*model, 0), 1);
}

TEST(RematerializedBatchTests, ReportsNothingWhenBatchReachesTheOutput) {
    const auto param = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4});
    const auto live = leading_dimension_of(param);
    const auto inferred = op::v0::Constant::create(element::i64, Shape{1}, {-1});

    const auto model =
        std::make_shared<Model>(OutputVector{reshape_to(param, {live, inferred})}, ParameterVector{param});

    EXPECT_FALSE(find_rematerialized_batch(*model, 0));
}

TEST(RematerializedBatchTests, ReportsNothingWhenBatchIsDroppedForGood) {
    const auto param = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4});
    const auto inferred = op::v0::Constant::create(element::i64, Shape{1}, {-1});
    const auto pinned = op::v0::Constant::create(element::i64, Shape{1}, {1});

    const auto model =
        std::make_shared<Model>(OutputVector{reshape_to(param, {pinned, inferred})}, ParameterVector{param});

    EXPECT_FALSE(find_rematerialized_batch(*model, 0));
}

TEST(RematerializedBatchTests, ReportsNothingForModelWithoutReshape) {
    const auto param = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4});
    const auto model =
        std::make_shared<Model>(OutputVector{std::make_shared<op::v0::Relu>(param)}, ParameterVector{param});

    EXPECT_FALSE(find_rematerialized_batch(*model, 0));
}

TEST(RematerializedBatchTests, ReportsNothingForUnknownInput) {
    const auto param = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4});
    const auto model =
        std::make_shared<Model>(OutputVector{std::make_shared<op::v0::Relu>(param)}, ParameterVector{param});

    EXPECT_FALSE(find_rematerialized_batch(*model, 1));
}
