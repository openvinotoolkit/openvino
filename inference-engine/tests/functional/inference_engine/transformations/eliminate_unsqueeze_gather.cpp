// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transformations/common_optimizations/eliminate_unsqueeze_gather.hpp>

#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <common_test_utils/test_common.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pass/manager.hpp>

namespace {

using TensorType  = ngraph::element::Type_t;
using TensorShape = ngraph::Shape;

class EliminateUnsqueezeGatherTest : public CommonTestUtils::TestsCommon,
                                     public testing::WithParamInterface<std::tuple<TensorType, TensorShape, size_t>> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& inType = std::get<0>(parameters);
        const auto& inShape = std::get<1>(parameters);
        const auto& axis = std::get<2>(parameters);

        ngraph::helpers::CompareFunctions(*transform(inShape, inType, axis),
                                          *reference(inShape, inType, axis));
    }

protected:
    std::shared_ptr<const ngraph::Function> transform(
            const TensorShape& inShape,
            const TensorType& inType,
            size_t axis) {
        const auto parameter = std::make_shared<ngraph::opset6::Parameter>(inType, inShape);

        const auto unsqueeze = std::make_shared<ngraph::opset6::Unsqueeze>(
                parameter,
                ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {axis}));

        const auto gather = std::make_shared<ngraph::opset6::Gather>(
                unsqueeze,
                ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0}),
                ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {axis}));

        const auto function = std::make_shared<ngraph::Function>(
                ngraph::NodeVector{gather},
                ngraph::ParameterVector{parameter},
                "Actual");

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::EliminateUnsqueezeGather>();
        manager.run_passes(function);

        return function;
    }

    std::shared_ptr<const ngraph::Function> reference(
            const TensorShape& inShape,
            const TensorType& inType,
            size_t axis) {
        const auto parameter = std::make_shared<ngraph::opset6::Parameter>(inType, inShape);

        return std::make_shared<ngraph::Function>(
                ngraph::NodeVector{parameter},
                ngraph::ParameterVector{parameter},
                "Reference");
    }
};

TEST_P(EliminateUnsqueezeGatherTest, CompareFunctions) {
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, EliminateUnsqueezeGatherTest, testing::Combine(
        testing::Values(
                ngraph::element::f16,
                ngraph::element::f32,
                ngraph::element::i32,
                ngraph::element::i64,
                ngraph::element::u8),
        testing::Values(
                TensorShape{3, 128, 256}),
        testing::Values(0, 1, 2, 3)
));

} // namespace
