// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_common.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <gtest/gtest.h>

#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset4.hpp"

#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"
#include "vpu/ngraph/transformations/merge_subsequent_dsr_operations.hpp"

namespace {

TEST(MergeSubsequentDSROperations, smoke_SingleDSRFunction) {
    // shape
    //      \
    //        dsr
    //      /
    // data

    const auto inputType  = ngraph::element::f16;
    const auto inputShape = ngraph::Shape{1};

    const auto data = std::make_shared<ngraph::opset4::Parameter>(inputType, inputShape);
    const auto shape = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::i64, ngraph::Shape{inputShape.size()});

    const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, shape);
    const auto reference = std::make_shared<const ngraph::Function>(
        ngraph::NodeVector{dsr},
        ngraph::ParameterVector{data, shape},
        "SingleDSRFunction");
    auto actual = ngraph::clone_function(*reference);

    vpu::MergeSubsequentDSROperations().run_on_function(actual);

    ASSERT_NO_THROW(ngraph::helpers::CompareFunctions(*reference, *actual));
}

TEST(MergeSubsequentDSROperations, smoke_DSR_ReLU_DSR_ReLU_DSR) {
    //          one_1
    //               \
    //   one_0        sum_1 - - - - - - - - - - dsr_2
    //        \     /                          /
    // shape - sum_0 - - - - - dsr_1 - relu_1 -
    //        \               /
    //         dsr_0 - relu_0
    //        /
    // data -
    //

    const auto inputType  = ngraph::element::f16;
    const auto inputShape = ngraph::Shape{1};

    const auto data  = std::make_shared<ngraph::opset4::Parameter>(inputType, inputShape);
    const auto shape = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::i64, ngraph::Shape{inputShape.size()});
    const auto dsr_0 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, shape);

    const auto relu_0 = std::make_shared<ngraph::opset4::Relu>(dsr_0);

    // emulates shape subgraph for operation ReLU
    const auto one_0 = std::make_shared<ngraph::opset4::Constant>(ngraph::element::i64, ngraph::Shape{1}, std::vector<std::int64_t>{1});
    const auto sum_0 = std::make_shared<ngraph::opset4::Add>(shape, one_0);

    const auto dsr_1 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(relu_0, sum_0);

    const auto relu_1 = std::make_shared<ngraph::opset4::Relu>(dsr_1);

    // emulates shape subgraph for operation ReLU
    const auto one_1 = std::make_shared<ngraph::opset4::Constant>(ngraph::element::i64, ngraph::Shape{1}, std::vector<std::int64_t>{1});
    const auto sum_1 = std::make_shared<ngraph::opset4::Add>(sum_0, one_1);

    const auto dsr_2 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(relu_1, sum_1);

    const auto reference = std::make_shared<const ngraph::Function>(
        ngraph::NodeVector{dsr_2},
        ngraph::ParameterVector{data, shape},
        "DSR_ReLU_DSR_ReLU_DSR");
    auto actual = ngraph::clone_function(*reference);

    vpu::MergeSubsequentDSROperations().run_on_function(actual);

    ASSERT_NO_THROW(ngraph::helpers::CompareFunctions(*reference, *actual));
}

TEST(MergeSubsequentDSROperations, smoke_DSR_ReLU_DSR_DSR) {
    // Before:
    //          one_1
    //               \
    //   one_0        sum_1 - - - - - - dsr_2
    //        \     /                  /
    // shape - sum_0 - - - - - dsr_1 -
    //        \               /
    //         dsr_0 - relu_0
    //        /
    // data -
    //
    // After:
    //          one_1
    //               \
    //   one_0        sum_1 - - - - - - dsr_2
    //        \     /                  /
    // shape - sum_0                  /
    //        \                      /
    //         dsr_0 - relu_0 - - - -
    //        /
    // data -

    const auto inputType  = ngraph::element::f16;
    const auto inputShape = ngraph::Shape{1};

    std::shared_ptr<ngraph::Function> actual;
    {
        const auto data  = std::make_shared<ngraph::opset4::Parameter>(inputType, inputShape);
        const auto shape = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::i64, ngraph::Shape{inputShape.size()});
        const auto dsr_0 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, shape);

        const auto relu_0 = std::make_shared<ngraph::opset4::Relu>(dsr_0);

        // emulates shape subgraph for operation ReLU
        const auto one_0 = std::make_shared<ngraph::opset4::Constant>(ngraph::element::i64, ngraph::Shape{1}, std::vector<std::int64_t>{1});
        const auto sum_0 = std::make_shared<ngraph::opset4::Add>(shape, one_0);

        const auto dsr_1 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(relu_0, sum_0);

        // emulates shape subgraph for operation ReLU
        const auto one_1 = std::make_shared<ngraph::opset4::Constant>(ngraph::element::i64, ngraph::Shape{1}, std::vector<std::int64_t>{1});
        const auto sum_1 = std::make_shared<ngraph::opset4::Add>(sum_0, one_1);

        const auto dsr_2 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(dsr_1, sum_1);

        actual = std::make_shared<ngraph::Function>(
            ngraph::NodeVector{dsr_2},
            ngraph::ParameterVector{data, shape},
            "DSR_ReLU_DSR_DSR");
    }

    std::shared_ptr<const ngraph::Function> reference;
    {
        const auto data  = std::make_shared<ngraph::opset4::Parameter>(inputType, inputShape);
        const auto shape = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::i64, ngraph::Shape{inputShape.size()});
        const auto dsr_0 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, shape);

        const auto relu_0 = std::make_shared<ngraph::opset4::Relu>(dsr_0);

        // emulates shape subgraph for operation ReLU
        const auto one_0 = std::make_shared<ngraph::opset4::Constant>(ngraph::element::i64, ngraph::Shape{1}, std::vector<std::int64_t>{1});
        const auto sum_0 = std::make_shared<ngraph::opset4::Add>(shape, one_0);

        // emulates shape subgraph for operation ReLU
        const auto one_1 = std::make_shared<ngraph::opset4::Constant>(ngraph::element::i64, ngraph::Shape{1}, std::vector<std::int64_t>{1});
        const auto sum_1 = std::make_shared<ngraph::opset4::Add>(sum_0, one_1);

        const auto dsr_2 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(relu_0, sum_1);

        reference = std::make_shared<const ngraph::Function>(
            ngraph::NodeVector{dsr_2},
            ngraph::ParameterVector{data, shape},
            "DSR_ReLU_DSR_DSR");
    }

    vpu::MergeSubsequentDSROperations().run_on_function(actual);

    ASSERT_NO_THROW(ngraph::helpers::CompareFunctions(*reference, *actual));
}

} //namespace
