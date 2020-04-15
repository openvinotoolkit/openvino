// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_nonzero.hpp"
#include "vpu/ngraph/operations/static_shape_nonzero.hpp"
#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"

#include "../utils/ngraph_utils.h"

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>

#include <cpp/ie_cnn_network.h>

#include <common_test_utils/test_common.hpp>
#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <map>
#include <vector>

namespace {

using TensorType  = ngraph::element::Type_t;
using TensorShape = ngraph::Shape;

class DynamicToStaticShapeNonZeroTests
        : public CommonTestUtils::TestsCommon,
          public testing::WithParamInterface<std::tuple<TensorType, TensorShape>> {
public:
    void prepareFunctions() {
        const auto& parameters = GetParam();
        const auto& tensorType = std::get<0>(parameters);
        const auto& tensorShape = std::get<1>(parameters);

        // Create a function with only opset3::NonZero
        // And then run conversion pass
        {
            const auto input = std::make_shared<ngraph::op::Parameter>(tensorType, tensorShape);

            const auto nonZero = std::make_shared<ngraph::opset3::NonZero>(input);
            nonZero->set_friendly_name(s_FriendlyName);

            m_resfunction = std::make_shared<ngraph::Function>(
                    ngraph::NodeVector{nonZero}, ngraph::ParameterVector{input});
            ngraph::pass::DynamicToStaticShapeNonZero().run_on_function(m_resfunction);
        }

        // Create a reference function
        {
            const auto input = std::make_shared<ngraph::opset1::Parameter>(tensorType, tensorShape);

            const auto staticShapeNonZero = std::make_shared<ngraph::op::StaticShapeNonZero>(input);
            staticShapeNonZero->set_friendly_name(s_FriendlyName + "/static_shape");
            const auto dynamicShapeResolver = std::make_shared<ngraph::op::DynamicShapeResolver>(
                    staticShapeNonZero->output(0), staticShapeNonZero->output(1));
            dynamicShapeResolver->set_friendly_name(s_FriendlyName + "/resolve_shape");

            m_refFunction = std::make_shared<ngraph::Function>(
                    ngraph::NodeVector{dynamicShapeResolver}, ngraph::ParameterVector{input});
        }
    }

    void compareFunctions() {
        FuncTestUtils::CompareFunctions(m_resfunction, m_refFunction);

        auto actualResultNode = m_resfunction->get_output_op(0);
        auto actualResolverNode = actualResultNode->input(0).get_source_output().get_node_shared_ptr();
        auto actualNonZeroNode = actualResolverNode->input(0).get_source_output().get_node_shared_ptr();

        auto expectedResultNode = m_refFunction->get_output_op(0);
        auto expectedResolverNode = expectedResultNode->input(0).get_source_output().get_node_shared_ptr();
        auto expectedNonZeroNode = expectedResolverNode->input(0).get_source_output().get_node_shared_ptr();

        EXPECT_EQ(actualResolverNode->get_friendly_name(), expectedResolverNode->get_friendly_name());
        EXPECT_EQ(actualNonZeroNode->get_friendly_name(), expectedNonZeroNode->get_friendly_name());
    }

protected:
    std::shared_ptr<ngraph::Function> m_resfunction;
    std::shared_ptr<ngraph::Function> m_refFunction;

    static const std::string s_FriendlyName;
};

const std::string DynamicToStaticShapeNonZeroTests::s_FriendlyName = "non_zero";

TEST_P(DynamicToStaticShapeNonZeroTests, inferAndValidate) {
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
