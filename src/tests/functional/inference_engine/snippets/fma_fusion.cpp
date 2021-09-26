// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/pass/manager.hpp>
#include <snippets/snippets_isa.hpp>
#include <snippets/pass/mul_add_to_fma.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST(TransformationTests, MulAddToFMAFusionTest1) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto input_1 = std::make_shared<opset1::Parameter>(element::f32, PartialShape{1, 3, 2, 2});
        auto input_2 = std::make_shared<opset1::Parameter>(element::f32, PartialShape{1, 3, 2, 2});
        auto load_1 = std::make_shared<snippets::isa::Load>(input_1);
        auto load_2 = std::make_shared<snippets::isa::Load>(input_2);
        auto mul = std::make_shared<opset1::Multiply>(load_1, load_2);

        auto input_3 = std::make_shared<opset1::Parameter>(element::f32, PartialShape{ 1, 3, 2, 2 });
        auto load_3 = std::make_shared<snippets::isa::Load>(input_3);
        auto add = std::make_shared<opset1::Add>(mul, load_3);

        auto store = std::make_shared<snippets::isa::Store>(add);
        f = std::make_shared<Function>(NodeVector{ store }, ParameterVector{ input_1, input_2, input_3 });

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<snippets::pass::MulAddToFMA>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto input_1 = std::make_shared<opset1::Parameter>(element::f32, PartialShape{ 1, 3, 2, 2 });
        auto input_2 = std::make_shared<opset1::Parameter>(element::f32, PartialShape{ 1, 3, 2, 2 });
        auto load_1 = std::make_shared<snippets::isa::Load>(input_1);
        auto load_2 = std::make_shared<snippets::isa::Load>(input_2);
        auto input_3 = std::make_shared<opset1::Parameter>(element::f32, PartialShape{ 1, 3, 2, 2 });
        auto load_3 = std::make_shared<snippets::isa::Load>(input_3);

        auto fma = std::make_shared<snippets::op::FMA>(load_1, load_2, load_3);

        auto store = std::make_shared<snippets::isa::Store>(fma);
        f_ref = std::make_shared<Function>(NodeVector{ store }, ParameterVector{ input_1, input_2, input_3 });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, MulAddToFMAFusionTest2) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto input_1 = std::make_shared<opset1::Parameter>(element::f32, PartialShape{ 1, 3, 2, 2 });
        auto input_2 = std::make_shared<opset1::Parameter>(element::f32, PartialShape{ 1, 3, 2, 2 });
        auto load_1 = std::make_shared<snippets::isa::Load>(input_1);
        auto load_2 = std::make_shared<snippets::isa::Load>(input_2);
        auto mul = std::make_shared<opset1::Multiply>(load_1, load_2);

        auto input_3 = std::make_shared<opset1::Parameter>(element::f32, PartialShape{ 1, 3, 2, 2 });
        auto load_3 = std::make_shared<snippets::isa::Load>(input_3);
        auto add = std::make_shared<opset1::Add>(load_3, mul);

        auto store = std::make_shared<snippets::isa::Store>(add);
        f = std::make_shared<Function>(NodeVector{ store }, ParameterVector{ input_1, input_2, input_3 });

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<snippets::pass::MulAddToFMA>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto input_1 = std::make_shared<opset1::Parameter>(element::f32, PartialShape{ 1, 3, 2, 2 });
        auto input_2 = std::make_shared<opset1::Parameter>(element::f32, PartialShape{ 1, 3, 2, 2 });
        auto load_1 = std::make_shared<snippets::isa::Load>(input_1);
        auto load_2 = std::make_shared<snippets::isa::Load>(input_2);
        auto input_3 = std::make_shared<opset1::Parameter>(element::f32, PartialShape{ 1, 3, 2, 2 });
        auto load_3 = std::make_shared<snippets::isa::Load>(input_3);

        auto fma = std::make_shared<snippets::op::FMA>(load_1, load_2, load_3);

        auto store = std::make_shared<snippets::isa::Store>(fma);
        f_ref = std::make_shared<Function>(NodeVector{ store }, ParameterVector{ input_1, input_2, input_3 });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, MulAddToFMAFusionTest3) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    auto get_f = []() {
        auto input_1 = std::make_shared<opset1::Parameter>(element::f32, PartialShape{ 1, 3, 2, 2 });
        auto input_2 = std::make_shared<opset1::Parameter>(element::f32, PartialShape{ 1, 3, 2, 2 });
        auto load_1 = std::make_shared<snippets::isa::Load>(input_1);
        auto load_2 = std::make_shared<snippets::isa::Load>(input_2);
        auto mul = std::make_shared<opset1::Multiply>(load_1, load_2);

        auto additional_consumer = std::make_shared<ngraph::opset1::Relu>(mul);
        auto store_1 = std::make_shared<snippets::isa::Store>(additional_consumer);

        auto input_3 = std::make_shared<opset1::Parameter>(element::f32, PartialShape{ 1, 3, 2, 2 });
        auto load_3 = std::make_shared<snippets::isa::Load>(input_3);
        auto add = std::make_shared<opset1::Add>(load_3, mul);

        auto store_2 = std::make_shared<snippets::isa::Store>(add);
        return std::make_shared<Function>(NodeVector{ store_1, store_2 }, ParameterVector{ input_1, input_2, input_3 });
    };

    f = get_f();
    pass::Manager m;
    m.register_pass<pass::InitNodeInfo>();
    m.register_pass<snippets::pass::MulAddToFMA>();
    m.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));

    f_ref = get_f();

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, MulAddToFMAFusionTest4) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto input_1 = std::make_shared<opset1::Parameter>(element::f32, PartialShape{1, 1, 1, 1});
        input_1->set_friendly_name("input_1");
        auto input_2 = std::make_shared<opset1::Parameter>(element::f32, PartialShape{1, 1, 2, 2});
        input_2->set_friendly_name("input_2");
        auto load_1 = std::make_shared<snippets::isa::Load>(input_1);
        auto load_2 = std::make_shared<snippets::isa::Load>(input_2);
        auto mul = std::make_shared<opset1::Multiply>(load_1, load_2);

        auto input_3 = std::make_shared<opset1::Parameter>(element::f32, PartialShape{1, 3, 1, 1});
        input_3->set_friendly_name("input_3");
        auto load_3 = std::make_shared<snippets::isa::Load>(input_3);
        auto add = std::make_shared<opset1::Add>(mul, load_3);

        auto store = std::make_shared<snippets::isa::Store>(add);
        f = std::make_shared<Function>(NodeVector{store}, ParameterVector{input_1, input_2, input_3});


        const auto result_shape_before = f->get_output_partial_shape(0);
        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<snippets::pass::MulAddToFMA>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
        EXPECT_EQ(result_shape_before, f->get_output_partial_shape(0));
    }
    {
        auto input_1 = std::make_shared<opset1::Parameter>(element::f32, PartialShape{1, 1, 1, 1});
        input_1->set_friendly_name("input_1");
        auto input_2 = std::make_shared<opset1::Parameter>(element::f32, PartialShape{1, 1, 2, 2});
        input_2->set_friendly_name("input_2");
        auto load_1 = std::make_shared<snippets::isa::Load>(input_1);
        auto load_2 = std::make_shared<snippets::isa::Load>(input_2);
        auto input_3 = std::make_shared<opset1::Parameter>(element::f32, PartialShape{1, 3, 1, 1});
        input_3->set_friendly_name("input_3");
        auto load_3 = std::make_shared<snippets::isa::Load>(input_3);

        auto fma = std::make_shared<snippets::op::FMA>(load_1, load_2, load_3);

        auto store = std::make_shared<snippets::isa::Store>(fma);
        f_ref = std::make_shared<Function>(NodeVector{store}, ParameterVector{input_1, input_2, input_3});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}