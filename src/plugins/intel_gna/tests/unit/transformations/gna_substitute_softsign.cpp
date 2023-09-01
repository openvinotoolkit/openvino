// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/opsets/opset9.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "transformations/substitute_softsign.hpp"
namespace testing {

namespace {

std::shared_ptr<ngraph::Function> createSoftSignFunction() {
    auto input_params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1, 1, 64});

    auto softsign = std::make_shared<ov::op::v9::SoftSign>(input_params);

    ngraph::ResultVector results{std::make_shared<ngraph::op::Result>(softsign)};

    return std::make_shared<ngraph::Function>(ngraph::ResultVector{results}, ngraph::ParameterVector{input_params});
}

}  // namespace

TEST(TransformationTests, SubstituteSoftSignMulPower) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        auto input_params =
            std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1, 1, 64});

        auto abs = std::make_shared<ngraph::op::Abs>(input_params);

        auto const_1 = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {1});
        auto const_neg_1 = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {-1});

        auto add = std::make_shared<ngraph::opset8::Add>(abs, const_1);
        auto power = std::make_shared<ngraph::opset8::Power>(add, const_neg_1);

        auto mul = std::make_shared<ngraph::opset8::Multiply>(power, input_params);
        ngraph::ResultVector results{std::make_shared<ngraph::op::Result>(mul)};

        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{results}, ngraph::ParameterVector{input_params});
        ngraph::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::intel_gna::pass::SubstituteSoftsign>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    reference_func = createSoftSignFunction();

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, SubstituteSoftSignDivide) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        auto input_params =
            std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1, 1, 64});

        auto abs = std::make_shared<ngraph::opset8::Abs>(input_params);

        auto const_1 = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {1});
        auto add = std::make_shared<ngraph::opset8::Add>(abs, const_1);

        auto divide = std::make_shared<ngraph::opset8::Divide>(input_params, add);
        ngraph::ResultVector results{std::make_shared<ngraph::opset8::Result>(divide)};

        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{results}, ngraph::ParameterVector{input_params});
        ngraph::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::intel_gna::pass::SubstituteSoftsign>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    reference_func = createSoftSignFunction();

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, SubstituteSoftSignMulPowerInvalidAddConst) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        auto input_params =
            std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1, 1, 64});

        auto abs = std::make_shared<ngraph::op::Abs>(input_params);

        auto const_1 = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {1.1});
        auto const_neg_1 = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {-1});

        auto add = std::make_shared<ngraph::opset8::Add>(abs, const_1);
        auto power = std::make_shared<ngraph::opset8::Power>(add, const_neg_1);

        auto mul = std::make_shared<ngraph::opset8::Multiply>(power, input_params);
        ngraph::ResultVector results{std::make_shared<ngraph::op::Result>(mul)};

        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{results}, ngraph::ParameterVector{input_params});
        ngraph::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::intel_gna::pass::SubstituteSoftsign>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    reference_func = ngraph::clone_function(*func);

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, SubstituteSoftSignMulPowerInvalidPowerConst) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        auto input_params =
            std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1, 1, 64});

        auto abs = std::make_shared<ngraph::op::Abs>(input_params);

        auto const_1 = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {1});
        auto const_neg_1 = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {-1.1});

        auto add = std::make_shared<ngraph::opset8::Add>(abs, const_1);
        auto power = std::make_shared<ngraph::opset8::Power>(add, const_neg_1);

        auto mul = std::make_shared<ngraph::opset8::Multiply>(power, input_params);
        ngraph::ResultVector results{std::make_shared<ngraph::op::Result>(mul)};

        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{results}, ngraph::ParameterVector{input_params});
        ngraph::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::intel_gna::pass::SubstituteSoftsign>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    reference_func = ngraph::clone_function(*func);

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

}  // namespace testing
