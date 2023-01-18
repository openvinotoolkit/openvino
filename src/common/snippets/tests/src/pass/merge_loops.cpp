// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/pass/manager.hpp>

#include <snippets/snippets_isa.hpp>
#include <snippets/pass/loop_fusion.hpp>

#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST(TransformationTests, UnaryEltwisesLoops) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    auto shape = Shape{2, 3, 240};
    const size_t vector_size = 16;
    const std::vector<int64_t> inner_ptr_increments(2, vector_size);
    const std::vector<int64_t> inner_finalization_offsets(2, 0);
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, shape);

        auto outer_loop_begin_up = std::make_shared<snippets::op::LoopBegin>(OutputVector{data});
        auto inner_loop_begin_up = std::make_shared<snippets::op::LoopBegin>(OutputVector{outer_loop_begin_up});
        auto load_up = std::make_shared<snippets::op::Load>(inner_loop_begin_up->output(0));
        auto relu = std::make_shared<op::v0::Relu>(load_up);
        auto store_up = std::make_shared<snippets::op::Store>(relu);
        auto inner_loop_end_up = std::make_shared<snippets::op::LoopEnd>(
                OutputVector{store_up, inner_loop_begin_up->output(1)}, shape[shape.size() - 1], vector_size,
                inner_ptr_increments, inner_finalization_offsets);
        auto outer_loop_end_up = std::make_shared<snippets::op::LoopEnd>(
                OutputVector{inner_loop_end_up->output(0), outer_loop_begin_up->output(1)}, shape[shape.size() - 2], 1,
                std::vector<int64_t>{0, 0}, std::vector<int64_t>{0, 0});

        auto buffer = std::make_shared<snippets::op::Buffer>(outer_loop_end_up);

        auto outer_loop_begin_down = std::make_shared<snippets::op::LoopBegin>(OutputVector{buffer});
        auto inner_loop_begin_down = std::make_shared<snippets::op::LoopBegin>(OutputVector{outer_loop_begin_down});
        auto load_down = std::make_shared<snippets::op::Load>(inner_loop_begin_down->output(0));
        auto hswish = std::make_shared<op::v4::HSwish>(load_down);
        auto store_down = std::make_shared<snippets::op::Store>(hswish);
        auto inner_loop_end_down = std::make_shared<snippets::op::LoopEnd>(
                OutputVector{store_down, inner_loop_begin_down->output(1)}, shape[shape.size() - 1], vector_size,
                inner_ptr_increments, inner_finalization_offsets);
        auto outer_loop_end_down = std::make_shared<snippets::op::LoopEnd>(
                OutputVector{inner_loop_end_down->output(0), outer_loop_begin_down->output(1)}, shape[shape.size() - 2], 1,
                std::vector<int64_t>{0, 0}, std::vector<int64_t>{0, 0});

        f = std::make_shared<Function>(OutputVector{outer_loop_end_down->output(0)}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<snippets::pass::LoopFusion>();
        m.run_passes(f);
    }
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, shape);

        auto outer_loop_begin = std::make_shared<snippets::op::LoopBegin>(OutputVector{data});
        auto inner_loop_begin = std::make_shared<snippets::op::LoopBegin>(OutputVector{outer_loop_begin});
        auto load = std::make_shared<snippets::op::Load>(inner_loop_begin->output(0));
        auto relu = std::make_shared<op::v0::Relu>(load);
        auto hswish = std::make_shared<op::v4::HSwish>(relu);
        auto store = std::make_shared<snippets::op::Store>(hswish);
        auto inner_loop_end = std::make_shared<snippets::op::LoopEnd>(
                OutputVector{store, inner_loop_begin->output(1)}, shape[shape.size() - 1], vector_size,
                inner_ptr_increments, inner_finalization_offsets);
        auto outer_loop_end = std::make_shared<snippets::op::LoopEnd>(
                OutputVector{inner_loop_end->output(0), outer_loop_begin->output(1)}, shape[shape.size() - 2], 1,
                std::vector<int64_t>{0, 0}, std::vector<int64_t>{0, 0});

        f_ref = std::make_shared<Function>(OutputVector{outer_loop_end->output(0)}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, BinaryEltwisesLoops) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    auto shape = Shape{2, 3, 240};
    const size_t vector_size = 16;
    {
        const std::vector<int64_t> inner_ptr_increments(3, vector_size);
        const std::vector<int64_t> inner_finalization_offsets(3, 0);

        auto data0 = std::make_shared<opset1::Parameter>(element::f32, shape);
        auto data1 = std::make_shared<opset1::Parameter>(element::f32, shape);

        auto outer_loop_begin_up = std::make_shared<snippets::op::LoopBegin>(OutputVector{data0, data1});
        auto inner_loop_begin_up = std::make_shared<snippets::op::LoopBegin>(OutputVector{outer_loop_begin_up->output(0),
                                                                                          outer_loop_begin_up->output(1)});
        auto load0_up = std::make_shared<snippets::op::Load>(inner_loop_begin_up->output(0));
        auto load1_up = std::make_shared<snippets::op::Load>(inner_loop_begin_up->output(1));
        auto add = std::make_shared<op::v1::Add>(load0_up, load1_up);
        auto relu = std::make_shared<op::v0::Relu>(add);
        auto store_up = std::make_shared<snippets::op::Store>(relu);
        auto inner_loop_end_up = std::make_shared<snippets::op::LoopEnd>(
                OutputVector{store_up, inner_loop_begin_up->output(2)}, shape[shape.size() - 1], vector_size,
                inner_ptr_increments, inner_finalization_offsets);
        auto outer_loop_end_up = std::make_shared<snippets::op::LoopEnd>(
                OutputVector{inner_loop_end_up->output(0), outer_loop_begin_up->output(2)}, shape[shape.size() - 2], 1,
                std::vector<int64_t>{0, 0, 0}, std::vector<int64_t>{0, 0, 0});

        auto buffer = std::make_shared<snippets::op::Buffer>(outer_loop_end_up);

        auto data2 = std::make_shared<opset1::Parameter>(element::f32, shape);

        auto outer_loop_begin_down = std::make_shared<snippets::op::LoopBegin>(OutputVector{buffer, data2});
        auto inner_loop_begin_down = std::make_shared<snippets::op::LoopBegin>(OutputVector{outer_loop_begin_down->output(0),
                                                                                            outer_loop_begin_down->output(1)});
        auto load0_down = std::make_shared<snippets::op::Load>(inner_loop_begin_down->output(0));
        auto load1_down = std::make_shared<snippets::op::Load>(inner_loop_begin_down->output(1));
        auto mul = std::make_shared<op::v1::Multiply>(load0_down, load1_down);
        auto hswish = std::make_shared<op::v4::HSwish>(mul);
        auto store_down = std::make_shared<snippets::op::Store>(hswish);
        auto inner_loop_end_down = std::make_shared<snippets::op::LoopEnd>(
                OutputVector{store_down, inner_loop_begin_down->output(2)}, shape[shape.size() - 1], vector_size,
                inner_ptr_increments, inner_finalization_offsets);
        auto outer_loop_end_down = std::make_shared<snippets::op::LoopEnd>(
                OutputVector{inner_loop_end_down->output(0), outer_loop_begin_down->output(2)}, shape[shape.size() - 2], 1,
                std::vector<int64_t>{0, 0, 0}, std::vector<int64_t>{0, 0, 0});

        f = std::make_shared<Function>(OutputVector{outer_loop_end_down->output(0)}, ParameterVector{data0, data1, data2});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<snippets::pass::LoopFusion>();
        m.run_passes(f);
    }
    {
        const std::vector<int64_t> inner_ptr_increments(4, vector_size);
        const std::vector<int64_t> inner_finalization_offsets(4, 0);

        auto data0 = std::make_shared<opset1::Parameter>(element::f32, shape);
        auto data1 = std::make_shared<opset1::Parameter>(element::f32, shape);
        auto data2 = std::make_shared<opset1::Parameter>(element::f32, shape);

        auto outer_loop_begin = std::make_shared<snippets::op::LoopBegin>(OutputVector{data0, data1, data2});
        auto inner_loop_begin = std::make_shared<snippets::op::LoopBegin>(OutputVector{outer_loop_begin->output(0),
                                                                                       outer_loop_begin->output(1),
                                                                                       outer_loop_begin->output(2)});
        auto load0 = std::make_shared<snippets::op::Load>(inner_loop_begin->output(0));
        auto load1 = std::make_shared<snippets::op::Load>(inner_loop_begin->output(1));
        auto load2 = std::make_shared<snippets::op::Load>(inner_loop_begin->output(2));
        auto add = std::make_shared<op::v1::Add>(load0, load1);
        auto relu = std::make_shared<op::v0::Relu>(add);
        auto mul = std::make_shared<op::v1::Multiply>(relu, load2);
        auto hswish = std::make_shared<op::v4::HSwish>(mul);
        auto store = std::make_shared<snippets::op::Store>(hswish);
        auto inner_loop_end = std::make_shared<snippets::op::LoopEnd>(
                OutputVector{store, inner_loop_begin->output(3)}, shape[shape.size() - 1], vector_size,
                inner_ptr_increments, inner_finalization_offsets);
        auto outer_loop_end = std::make_shared<snippets::op::LoopEnd>(
                OutputVector{inner_loop_end->output(0), outer_loop_begin->output(3)}, shape[shape.size() - 2], 1,
                std::vector<int64_t>{0, 0, 0, 0}, std::vector<int64_t>{0, 0, 0, 0});

        f_ref = std::make_shared<Function>(OutputVector{outer_loop_end->output(0)}, ParameterVector{data0, data1, data2});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
