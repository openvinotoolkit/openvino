// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/gather_sinking_matmul.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "gather_sinking_test_utils.hpp"
#include "gtest/gtest.h"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace ov;
using namespace ov::opset12;

TEST(GatherSinkingMatMul, Forward) {
    std::shared_ptr<Model> function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{20, 20});

        auto gather = make_gather(input_params, gather_forward, /* axis */ 1);

        auto input_const1 = Constant::create(ov::element::f32, {20, 20}, {1});
        auto matmul1 = std::make_shared<MatMul>(gather, input_const1);

        auto input_const2 = Constant::create(ov::element::f32, {20, 20}, {1});
        auto matmul2 = std::make_shared<MatMul>(input_const2, matmul1);

        const auto result = std::make_shared<Result>(matmul2);
        function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    std::shared_ptr<Model> orig_function = function->clone();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_gna::pass::GatherSinkingMatmulForward>();
    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    std::shared_ptr<Model> reference_function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{20, 20});
        auto input_const = Constant::create(ov::element::f32, {20, 20}, {1});

        auto gather = make_gather(input_const, gather_backward, /* axis */ 0);

        auto input_const1 = Constant::create(ov::element::f32, {20, 20}, {1});
        auto matmul1 = std::make_shared<MatMul>(input_params, gather);

        auto input_const2 = Constant::create(ov::element::f32, {20, 20}, {1});
        auto matmul2 = std::make_shared<MatMul>(input_const2, matmul1);

        const auto result = std::make_shared<Result>(matmul2);
        reference_function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(GatherSinkingMatMul, Backward) {
    std::shared_ptr<Model> function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{20, 20});

        auto input_const1 = Constant::create(ov::element::f32, {20, 20}, {1});
        auto matmul1 = std::make_shared<MatMul>(input_params, input_const1);

        auto input_const2 = Constant::create(ov::element::f32, {20, 20}, {1});
        auto matmul2 = std::make_shared<MatMul>(input_const2, matmul1);

        auto gather = make_gather(matmul2, gather_forward, /* axis */ 1);

        const auto result = std::make_shared<Result>(gather);
        function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    std::shared_ptr<Model> orig_function = function->clone();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_gna::pass::GatherSinkingMatmulBackward>();
    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    std::shared_ptr<Model> reference_function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{20, 20});
        auto input_const = Constant::create(ov::element::f32, {20, 20}, {1});

        auto gather = make_gather(input_const, gather_forward, /* axis */ 1);

        auto input_const1 = Constant::create(ov::element::f32, {20, 20}, {1});
        auto matmul1 = std::make_shared<MatMul>(input_params, gather);

        auto input_const2 = Constant::create(ov::element::f32, {20, 20}, {1});
        auto matmul2 = std::make_shared<MatMul>(input_const2, matmul1);

        const auto result = std::make_shared<Result>(matmul2);
        reference_function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid) << result.message;
}
