// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/core/function.hpp>
#include <openvino/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/remove_multi_subgraph_op_dangling_params.hpp>
#include <transformations/common_optimizations/remove_concat_zero_dim_input.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ov;
using namespace ov::opset8;

TEST(TransformationTests, RemoveLoopDanglingParameters) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    auto trip_count = std::make_shared<Constant>(element::i64, Shape{}, 10);
    auto condition = std::make_shared<Constant>(element::boolean, Shape{}, true);

    auto a = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto ai = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto b = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto bi = std::make_shared<Parameter>(element::f32, Shape{2, 2});

    auto mul = std::make_shared<Multiply>(bi, bi);
    auto abs = std::make_shared<Abs>(mul);
    {
        auto body = std::make_shared<Function>(OutputVector{condition, abs}, ParameterVector{ai, bi});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        loop->set_invariant_input(ai, a);
        loop->set_invariant_input(bi, b);

        auto loop_res = std::make_shared<Result>(loop->get_iter_value(abs));
        f = std::make_shared<Function>(OutputVector{loop_res}, ParameterVector{a, b});

        ov::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<pass::RemoveMultiSubGraphOpDanglingParams>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto body = std::make_shared<Function>(OutputVector{condition, abs}, ParameterVector{bi});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        loop->set_invariant_input(bi, b);

        auto loop_res = std::make_shared<Result>(loop->get_iter_value(abs));
        f_ref = std::make_shared<Function>(OutputVector{loop_res}, ParameterVector{a, b});
    }

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, Remove2LoopDanglingParameters) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    auto trip_count = std::make_shared<Constant>(element::i64, Shape{}, 10);
    auto condition = std::make_shared<Constant>(element::boolean, Shape{}, true);

    auto a = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto ai = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto b = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto bi = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto c = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto ci = std::make_shared<Parameter>(element::f32, Shape{2, 2});

    auto mul = std::make_shared<Multiply>(bi, bi);
    auto abs = std::make_shared<Abs>(mul);
    {
        auto body = std::make_shared<Function>(OutputVector{condition, abs}, ParameterVector{ai, bi, ci});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        loop->set_invariant_input(ai, a);
        loop->set_invariant_input(bi, b);
        loop->set_invariant_input(ci, c);

        auto loop_res = std::make_shared<Result>(loop->get_iter_value(abs));
        f = std::make_shared<Function>(OutputVector{loop_res}, ParameterVector{a, b, c});

        pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<pass::RemoveMultiSubGraphOpDanglingParams>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto body = std::make_shared<Function>(OutputVector{condition, abs}, ParameterVector{bi});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        loop->set_invariant_input(bi, b);

        auto loop_res = std::make_shared<Result>(loop->get_iter_value(abs));
        f_ref = std::make_shared<Function>(OutputVector{loop_res}, ParameterVector{a, b, c});
    }

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, RemoveLoopDanglingParametersIfConcatEmptyTensor) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    auto trip_count = std::make_shared<Constant>(element::i64, Shape{}, 10);
    auto condition = std::make_shared<Constant>(element::boolean, Shape{}, true);

    auto a = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto ai = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto b = std::make_shared<Parameter>(element::f32, Shape{0, 2}); // empty tensor
    auto bi = std::make_shared<Parameter>(element::f32, Shape{0, 2});
    bi->set_friendly_name("to_remove");
    {
        auto concat = std::make_shared<Concat>(NodeVector{ai, bi}, 0);
        auto body = std::make_shared<Function>(OutputVector{condition, concat}, ParameterVector{ai, bi});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        loop->set_invariant_input(ai, a);
        loop->set_invariant_input(bi, b);

        auto loop_res = std::make_shared<Result>(loop->get_iter_value(concat));
        f = std::make_shared<Function>(OutputVector{loop_res}, ParameterVector{a, b});

        pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<pass::RemoveConcatZeroDimInput>();
        manager.register_pass<pass::RemoveMultiSubGraphOpDanglingParams>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto concat = std::make_shared<Concat>(NodeVector{ai}, 0);
        auto body = std::make_shared<Function>(OutputVector{condition, concat}, ParameterVector{ai});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        loop->set_invariant_input(ai, a);

        auto loop_res = std::make_shared<Result>(loop->get_iter_value(concat));
        f_ref = std::make_shared<Function>(OutputVector{loop_res}, ParameterVector{a, b});
    }

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}
