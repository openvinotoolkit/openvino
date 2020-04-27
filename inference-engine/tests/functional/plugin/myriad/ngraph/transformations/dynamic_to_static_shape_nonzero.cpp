// Copyright (C) 2020 Intel Corporation
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

class DynamicToStaticShapeNonZeroTests : public CommonTestUtils::TestsCommon, public testing::WithParamInterface<std::tuple<TensorType, TensorShape>> {
public:
    void prepareFunctions() {
        const auto& parameters = GetParam();
        const auto& tensorType = std::get<0>(parameters);
        const auto& tensorShape = std::get<1>(parameters);

        // Create a function with only op::NonZero
        // And then run conversion pass
        {
            const auto input = std::make_shared<ngraph::opset3::Parameter>(tensorType, tensorShape);

            const auto nonZero = std::make_shared<ngraph::opset3::NonZero>(input);
            nonZero->set_friendly_name(s_FriendlyName);

            actual = std::make_shared<ngraph::Function>(ngraph::NodeVector{nonZero}, ngraph::ParameterVector{input});
            const auto transformation = vpu::Transformations{{ngraph::opset3::NonZero::type_info, vpu::dynamicToStaticShapeNonZero}};
            vpu::DynamicToStaticShape(transformation).transform(*actual);
        }

        // Create a reference function
        {
            const auto input = std::make_shared<ngraph::opset1::Parameter>(tensorType, tensorShape);

            const auto staticShapeNonZero = std::make_shared<ngraph::vpu::op::StaticShapeNonZero>(input);
            staticShapeNonZero->set_friendly_name(std::string(s_FriendlyName) + "/static_shape");
            const auto dynamicShapeResolver = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(
                    staticShapeNonZero->output(0), staticShapeNonZero->output(1));
            dynamicShapeResolver->set_friendly_name(std::string(s_FriendlyName) + "/resolve_shape");

            expected = std::make_shared<ngraph::Function>(ngraph::NodeVector{dynamicShapeResolver}, ngraph::ParameterVector{input});
        }
    }

    void compareFunctions() {
        ASSERT_NO_THROW(ngraph::helpers::CompareFunctions(*actual, *expected));

        auto actualResultNode = actual->get_output_op(0);
        auto actualResolverNode = actualResultNode->input(0).get_source_output().get_node_shared_ptr();
        auto actualNonZeroNode = actualResolverNode->input(0).get_source_output().get_node_shared_ptr();

        auto expectedResultNode = expected->get_output_op(0);
        auto expectedResolverNode = expectedResultNode->input(0).get_source_output().get_node_shared_ptr();
        auto expectedNonZeroNode = expectedResolverNode->input(0).get_source_output().get_node_shared_ptr();

        EXPECT_EQ(actualResolverNode->get_friendly_name(), expectedResolverNode->get_friendly_name());
        EXPECT_EQ(actualNonZeroNode->get_friendly_name(), expectedNonZeroNode->get_friendly_name());
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

INSTANTIATE_TEST_CASE_P(NGraph, DynamicToStaticShapeNonZeroTests, testing::Combine(
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
        TensorShape{2, 3, 128, 256})
));

}  // namespace
