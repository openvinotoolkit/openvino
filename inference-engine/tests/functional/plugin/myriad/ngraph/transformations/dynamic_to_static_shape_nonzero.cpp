// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_nonzero.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape.hpp"
#include "vpu/ngraph/operations/static_shape_nonzero.hpp"
#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"

#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset3.hpp>

#include <common_test_utils/test_common.hpp>
#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <map>
#include <vector>

namespace {

using TensorType  = ngraph::element::Type_t;
using TensorShape = ngraph::Shape;

typedef std::tuple<
        TensorType, // input type
        TensorShape, // input shape
        TensorType // output type
> dynamicToStaticShapeNonZeroTestParams;

class DynamicToStaticShapeNonZeroTests : public CommonTestUtils::TestsCommon,
                                         public testing::WithParamInterface<dynamicToStaticShapeNonZeroTestParams> {
public:
    void prepareFunctions() {
        const auto& parameters = GetParam();
        const auto& inputType = std::get<0>(parameters);
        const auto& inputShape = std::get<1>(parameters);
        const auto& resultType = std::get<2>(parameters);

        // Create a function with only op::NonZero
        // And then run conversion pass
        {
            const auto input = std::make_shared<ngraph::opset3::Parameter>(inputType, inputShape);

            const auto nonZero = std::make_shared<ngraph::opset3::NonZero>(input, resultType);
            nonZero->set_friendly_name(s_FriendlyName);

            actual = std::make_shared<ngraph::Function>(ngraph::NodeVector{nonZero}, ngraph::ParameterVector{input});
            const auto transformation = vpu::Transformations{{ngraph::opset3::NonZero::type_info, vpu::dynamicToStaticShapeNonZero}};
            vpu::DynamicToStaticShape(transformation).run_on_function(actual);
        }

        // Create a reference function
        {
            const auto input = std::make_shared<ngraph::opset1::Parameter>(inputType, inputShape);

            const auto staticShapeNonZero = std::make_shared<ngraph::vpu::op::StaticShapeNonZero>(input, resultType);
            const auto dynamicShapeResolver = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(
                    staticShapeNonZero->output(0), staticShapeNonZero->output(1));
            dynamicShapeResolver->set_friendly_name(std::string(s_FriendlyName));

            expected = std::make_shared<ngraph::Function>(ngraph::NodeVector{dynamicShapeResolver}, ngraph::ParameterVector{input});
        }
    }

    void compareFunctions() {
        ASSERT_NO_THROW(ngraph::helpers::CompareFunctions(*actual, *expected));

        const auto actualResultNode = actual->get_output_op(0);
        const auto actualResolverNode = actualResultNode->input(0).get_source_output().get_node_shared_ptr();

        const auto expectedResultNode = expected->get_output_op(0);
        const auto expectedResolverNode = expectedResultNode->input(0).get_source_output().get_node_shared_ptr();

        EXPECT_EQ(actualResolverNode->get_friendly_name(), expectedResolverNode->get_friendly_name());
    }

protected:
    std::shared_ptr<ngraph::Function> actual;
    std::shared_ptr<ngraph::Function> expected;

    static const char s_FriendlyName[];
};

const char DynamicToStaticShapeNonZeroTests::s_FriendlyName[] = "NonZero";

TEST_P(DynamicToStaticShapeNonZeroTests, CompareFunctions) {
    prepareFunctions();
    compareFunctions();
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, DynamicToStaticShapeNonZeroTests, testing::Combine(
    testing::Values(
        ngraph::element::f16,
        ngraph::element::f32,
        ngraph::element::i32,
        ngraph::element::i64,
        ngraph::element::u8),
    testing::Values(
        TensorShape{1000},
        TensorShape{4, 1000},
        TensorShape{3, 128, 256},
        TensorShape{2, 3, 128, 256}),
    testing::Values(
        ngraph::element::i32,
        ngraph::element::i64)
));

}  // namespace
