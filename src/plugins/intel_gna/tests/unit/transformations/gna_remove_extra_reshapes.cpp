// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "transformations/remove_extra_reshapes.hpp"

namespace testing {

TEST(TransformationTests, RemoveExtraReshapesTestReshapeNotEqualInputOutput) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);
    const ngraph::Shape data_shape{1, 3, 64, 64};

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, data_shape);
        auto new_shape = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {1, 3, 64 * 64});
        auto reshape_operation = std::make_shared<ngraph::opset7::Reshape>(input_params, new_shape, true);
        auto max_pool_operation = std::make_shared<ngraph::opset7::MaxPool>(reshape_operation,
                                                                            ngraph::Strides{1},
                                                                            ngraph::Shape{0},
                                                                            ngraph::Shape{0},
                                                                            ngraph::Shape{3});
        auto result = std::make_shared<ngraph::opset7::Result>(max_pool_operation);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});

        reference_func = ngraph::clone_function(*func);

        ngraph::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::intel_gna::pass::RemoveExtraReshapes>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, RemoveExtraReshapesTestReshapeEqualInputOutput) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);
    const ngraph::Shape data_shape{1, 3, 64, 64};

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, data_shape);
        auto new_shape = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 3, 64, 64});
        auto reshape_operation = std::make_shared<ngraph::opset7::Reshape>(input_params, new_shape, true);
        auto max_pool_operation = std::make_shared<ngraph::opset7::MaxPool>(reshape_operation,
                                                                            ngraph::Strides{1, 1},
                                                                            ngraph::Shape{0, 0},
                                                                            ngraph::Shape{0, 0},
                                                                            ngraph::Shape{3, 3});
        auto result = std::make_shared<ngraph::opset7::Result>(max_pool_operation);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});

        ngraph::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::intel_gna::pass::RemoveExtraReshapes>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, data_shape);
        auto max_pool_operation = std::make_shared<ngraph::opset7::MaxPool>(input_params,
                                                                            ngraph::Strides{1, 1},
                                                                            ngraph::Shape{0, 0},
                                                                            ngraph::Shape{1, 1},
                                                                            ngraph::Shape{4, 4});
        auto result = std::make_shared<ngraph::opset7::Result>(max_pool_operation);
        reference_func =
            std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default();
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

}  // namespace testing
