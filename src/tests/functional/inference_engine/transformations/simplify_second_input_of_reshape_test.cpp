// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <transformations/common_optimizations/simplify_shape_of_sub_graph.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"


using namespace testing;
using namespace ngraph;

namespace {
auto gather = [](const std::shared_ptr<Node> input, std::vector<int64_t> indices) -> Output<Node> {
    std::shared_ptr<Node> indices_node = opset7::Constant::create(element::i64, {indices.size()}, indices);
    std::shared_ptr<Node> axis_node = opset7::Constant::create(element::i64, {}, { 0 });
    return std::make_shared<opset7::Gather>(input, indices_node, axis_node);
};
}
auto fake_quantize = [](const std::shared_ptr<Node> input) -> Output<Node> {
    auto il = opset7::Constant::create(element::f32, Shape{}, { 0.f });
    auto ih = opset7::Constant::create(element::f32, Shape{}, { 25.5f });
    auto ol = opset7::Constant::create(element::f32, Shape{}, { 0.f });
    auto oh = opset7::Constant::create(element::f32, Shape{}, { 25.5f });
    return std::make_shared<opset7::FakeQuantize>(input, il, ih, ol, oh, 256);
};

TEST(TransformationTests, SimplifySecondInputOfReshapeTest1) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    PartialShape data_shape{1, 128, 12, 64};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto gather_op = gather(shape_of, std::vector<int64_t>{0, 1});
        auto constant = opset7::Constant::create(element::i64, Shape{1}, {768});
        auto concat = std::make_shared<opset7::Concat>(OutputVector{ gather_op, constant }, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        f = std::make_shared<Function>(NodeVector{reshape}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::SimplifySecondInputOfReshape>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ASSERT_EQ(reshape->get_output_partial_shape(0), PartialShape({ 1, 128, 768 }));
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto reshape_pattern = opset7::Constant::create(element::i64, Shape{ 3 }, { 0, 0, 768 });
        auto reshape = std::make_shared<opset7::Reshape>(data, reshape_pattern, true);
        f_ref = std::make_shared<Function>(NodeVector{reshape}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SimplifySecondInputOfReshapeTest2) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    PartialShape data_shape{ 1, 128, 12, 64 };
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto fq = fake_quantize(data);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto gather_op = gather(shape_of, std::vector<int64_t>{0, 1});
        auto constant = opset7::Constant::create(element::i64, Shape{ 1 }, { 768 });
        auto concat = std::make_shared<opset7::Concat>(OutputVector{ gather_op, constant }, 0);

        auto reshape = std::make_shared<opset7::Reshape>(fq, concat, true);
        f = std::make_shared<Function>(NodeVector{ reshape }, ParameterVector{ data });

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::SimplifySecondInputOfReshape>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ASSERT_EQ(reshape->get_output_partial_shape(0), PartialShape({ 1, 128, 768 }));
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto fq = fake_quantize(data);
        auto reshape_pattern = opset7::Constant::create(element::i64, Shape{ 3 }, { 0, 0, 768 });
        auto reshape = std::make_shared<opset7::Reshape>(fq, reshape_pattern, true);
        f_ref = std::make_shared<Function>(NodeVector{ reshape }, ParameterVector{ data });
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SimplifySecondInputOfReshapeTest3) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    PartialShape data_shape{ 1, 128, 768 };
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto gather_op = gather(shape_of, std::vector<int64_t>{0, 1});
        auto constant_1 = opset7::Constant::create(element::i64, Shape{ 1 }, { 12 });
        auto constant_2 = opset7::Constant::create(element::i64, Shape{ 1 }, { 64 });
        auto concat = std::make_shared<opset7::Concat>(OutputVector{ gather_op, constant_1, constant_2 }, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        f = std::make_shared<Function>(NodeVector{ reshape }, ParameterVector{ data });

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::SimplifySecondInputOfReshape>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ASSERT_EQ(reshape->get_output_partial_shape(0), PartialShape({ 1, 128, 12, 64 }));
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto reshape_pattern = opset7::Constant::create(element::i64, Shape{ 4 }, { 0, 0, 12, 64 });
        auto reshape = std::make_shared<opset7::Reshape>(data, reshape_pattern, true);
        f_ref = std::make_shared<Function>(NodeVector{ reshape }, ParameterVector{ data });
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SimplifySecondInputOfReshapeTest4) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    PartialShape data_shape{ 1, 128, 768 };
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto fq = fake_quantize(data);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto gather_op = gather(shape_of, std::vector<int64_t>{0, 1});
        auto constant_1 = opset7::Constant::create(element::i64, Shape{ 1 }, { 12 });
        auto constant_2 = opset7::Constant::create(element::i64, Shape{ 1 }, { 64 });
        auto concat = std::make_shared<opset7::Concat>(OutputVector{ gather_op, constant_1, constant_2 }, 0);

        auto reshape = std::make_shared<opset7::Reshape>(fq, concat, true);
        f = std::make_shared<Function>(NodeVector{ reshape }, ParameterVector{ data });

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::SimplifySecondInputOfReshape>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ASSERT_EQ(reshape->get_output_partial_shape(0), PartialShape({ 1, 128, 12, 64 }));
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto fq = fake_quantize(data);
        auto reshape_pattern = opset7::Constant::create(element::i64, Shape{ 4 }, { 0, 0, 12, 64 });
        auto reshape = std::make_shared<opset7::Reshape>(fq, reshape_pattern, true);
        f_ref = std::make_shared<Function>(NodeVector{ reshape }, ParameterVector{ data });
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SimplifySecondInputOfReshapeTest5) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    PartialShape data_shape = PartialShape::dynamic(3);
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto gather_op = gather(shape_of, std::vector<int64_t>{0, 1});
        auto constant = opset7::Constant::create(element::i64, Shape{ 1 }, { -1 });
        auto concat = std::make_shared<opset7::Concat>(OutputVector{ gather_op, constant }, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        f = std::make_shared<Function>(NodeVector{ reshape }, ParameterVector{ data });

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::SimplifySecondInputOfReshape>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ASSERT_EQ(reshape->get_output_partial_shape(0), PartialShape::dynamic(3));
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto reshape_pattern = opset7::Constant::create(element::i64, Shape{ 3 }, { 0, 0, -1 });
        auto reshape = std::make_shared<opset7::Reshape>(data, reshape_pattern, true);
        f_ref = std::make_shared<Function>(NodeVector{ reshape }, ParameterVector{ data });
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SimplifySecondInputOfReshapeTest6) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    PartialShape data_shape = PartialShape::dynamic();
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto gather_op = gather(shape_of, std::vector<int64_t>{0, 1});
        auto constant = opset7::Constant::create(element::i64, Shape{ 1 }, { -1 });
        auto concat = std::make_shared<opset7::Concat>(OutputVector{ gather_op, constant }, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        f = std::make_shared<Function>(NodeVector{ reshape }, ParameterVector{ data });

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::SimplifySecondInputOfReshape>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ASSERT_EQ(reshape->get_output_partial_shape(0), PartialShape::dynamic(3));
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto reshape_pattern = opset7::Constant::create(element::i64, Shape{ 3 }, { 0, 0, -1 });
        auto reshape = std::make_shared<opset7::Reshape>(data, reshape_pattern, true);
        f_ref = std::make_shared<Function>(NodeVector{ reshape }, ParameterVector{ data });
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SimplifySecondInputOfReshapeTest7) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    PartialShape data_shape{ 1, 128, 12, 64 };
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto gather_op = gather(shape_of, std::vector<int64_t>{2, 3});
        auto constant_1 = opset7::Constant::create(element::i64, Shape{ 1 }, { 64 });
        auto constant_2 = opset7::Constant::create(element::i64, Shape{ 1 }, { 2 });
        auto concat = std::make_shared<opset7::Concat>(OutputVector{ constant_1, constant_2, gather_op }, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        f = std::make_shared<Function>(NodeVector{ reshape }, ParameterVector{ data });

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::SimplifySecondInputOfReshape>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ASSERT_EQ(reshape->get_output_partial_shape(0), PartialShape({ 64, 2, 12, 64 }));
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto reshape_pattern = opset7::Constant::create(element::i64, Shape{ 4 }, { 64, 2, 0, 0 });
        auto reshape = std::make_shared<opset7::Reshape>(data, reshape_pattern, true);
        f_ref = std::make_shared<Function>(NodeVector{ reshape }, ParameterVector{ data });
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SimplifySecondInputOfReshapeTest8) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    PartialShape data_shape{ 1, 128, 12, 64 };
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto gather_op = gather(shape_of, std::vector<int64_t>{2});
        auto constant_1 = opset7::Constant::create(element::i64, Shape{ 1 }, { 64 });
        auto constant_2 = opset7::Constant::create(element::i64, Shape{ 1 }, { 2 });
        auto constant_3 = opset7::Constant::create(element::i64, Shape{ 1 }, { 64 });
        auto concat = std::make_shared<opset7::Concat>(OutputVector{ constant_1, constant_2, gather_op, constant_3 }, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        f = std::make_shared<Function>(NodeVector{ reshape }, ParameterVector{ data });

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::SimplifySecondInputOfReshape>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ASSERT_EQ(reshape->get_output_partial_shape(0), PartialShape({ 64, 2, 12, 64 }));
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto reshape_pattern = opset7::Constant::create(element::i64, Shape{ 4 }, { 64, 2, 0, 64 });
        auto reshape = std::make_shared<opset7::Reshape>(data, reshape_pattern, true);
        f_ref = std::make_shared<Function>(NodeVector{ reshape }, ParameterVector{ data });
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SimplifySecondInputOfReshapeTest9) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    PartialShape data_shape{ 1, 128, 12, 64 };
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto gather_op = gather(shape_of, std::vector<int64_t>{0, 2});
        auto constant = opset7::Constant::create(element::i64, Shape{ 1 }, { -1 });
        auto concat = std::make_shared<opset7::Concat>(OutputVector{ gather_op, constant }, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        f = std::make_shared<Function>(NodeVector{ reshape }, ParameterVector{ data });
        f_ref = f;

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::SimplifySecondInputOfReshape>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ASSERT_EQ(reshape->get_output_partial_shape(0), PartialShape({ 1, 12, 8192 }));
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SimplifySecondInputOfReshapeTest10) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    PartialShape data_shape{ 1, 128, 12, 64 };
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of_1 = std::make_shared<opset7::ShapeOf>(data);
        auto shape_of_2 = std::make_shared<opset7::ShapeOf>(data);
        auto gather_op_1 = gather(shape_of_1, std::vector<int64_t>{0, 1});
        auto gather_op_2 = gather(shape_of_2, std::vector<int64_t>{3});
        auto gather_op_3 = gather(shape_of_2, std::vector<int64_t>{2});
        auto concat = std::make_shared<opset7::Concat>(OutputVector{ gather_op_1, gather_op_2, gather_op_3 }, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        f = std::make_shared<Function>(NodeVector{ reshape }, ParameterVector{ data });

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::SimplifySecondInputOfReshape>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ASSERT_EQ(reshape->get_output_partial_shape(0), PartialShape({ 1, 128, 64, 12 }));
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto constant = opset7::Constant::create(element::i64, Shape{ 2 }, { 0, 0 });
        auto gather_op_2 = gather(shape_of, std::vector<int64_t>{3});
        auto gather_op_3 = gather(shape_of, std::vector<int64_t>{2});
        auto concat = std::make_shared<opset7::Concat>(OutputVector{ constant, gather_op_2, gather_op_3 }, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        f_ref = std::make_shared<Function>(NodeVector{ reshape }, ParameterVector{ data });
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SimplifySecondInputOfReshapeTest11) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    PartialShape data_shape{ 1, 128, 12, 64 };
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of_1 = std::make_shared<opset7::ShapeOf>(data);
        auto shape_of_2 = std::make_shared<opset7::ShapeOf>(data);
        auto concat_input_0 = gather(shape_of_1, std::vector<int64_t>{0});
        auto concat_input_1 = ngraph::opset7::Constant::create(ngraph::element::i64, {1}, { 64 });
        auto concat_input_2 = gather(shape_of_2, std::vector<int64_t>{2});
        auto concat_input_3 = ngraph::opset7::Constant::create(ngraph::element::i64, {1}, { 128 });
        auto concat = std::make_shared<opset7::Concat>(OutputVector{ concat_input_0, concat_input_1, concat_input_2, concat_input_3 }, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        f = std::make_shared<Function>(NodeVector{ reshape }, ParameterVector{ data });

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::SimplifySecondInputOfReshape>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ASSERT_EQ(reshape->get_output_partial_shape(0), PartialShape({ 1, 64, 12, 128 }));
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto constant = opset7::Constant::create(element::i64, Shape{ 4 }, { 0, 64, 0, 128 });
        auto reshape = std::make_shared<opset7::Reshape>(data, constant, true);
        f_ref = std::make_shared<Function>(NodeVector{ reshape }, ParameterVector{ data });
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SimplifySecondInputOfReshapeTest12) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    PartialShape data_shape{ 1, 128, 768 };
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto gelu = std::make_shared<opset7::Gelu>(data);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto gather_op = gather(shape_of, std::vector<int64_t>{0, 1});
        auto constant_1 = opset7::Constant::create(element::i64, Shape{ 1 }, { 12 });
        auto constant_2 = opset7::Constant::create(element::i64, Shape{ 1 }, { 64 });
        auto concat = std::make_shared<opset7::Concat>(OutputVector{ gather_op, constant_1, constant_2 }, 0);

        auto reshape = std::make_shared<opset7::Reshape>(gelu, concat, true);
        f = std::make_shared<Function>(NodeVector{ reshape }, ParameterVector{ data });

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::SimplifySecondInputOfReshape>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ASSERT_EQ(reshape->get_output_partial_shape(0), PartialShape({ 1, 128, 12, 64 }));
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto gelu = std::make_shared<opset7::Gelu>(data);
        auto reshape_pattern = opset7::Constant::create(element::i64, Shape{ 4 }, { 0, 0, 12, 64 });
        auto reshape = std::make_shared<opset7::Reshape>(gelu, reshape_pattern, true);
        f_ref = std::make_shared<Function>(NodeVector{ reshape }, ParameterVector{ data });
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SimplifySecondInputOfReshapeTest13) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    PartialShape data_shape{ 1, 128, 12, 64 };
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data, element::i32);
        auto gather_op = gather(shape_of, std::vector<int64_t>{0, 1});
        auto constant = opset7::Constant::create(element::i32, Shape{ 1 }, { 768 });
        auto concat = std::make_shared<opset7::Concat>(OutputVector{ gather_op, constant }, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        f = std::make_shared<Function>(NodeVector{ reshape }, ParameterVector{ data });

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::SimplifySecondInputOfReshape>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ASSERT_EQ(reshape->get_output_partial_shape(0), PartialShape({ 1, 128, 768 }));
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto reshape_pattern = opset7::Constant::create(element::i32, Shape{ 3 }, { 0, 0, 768 });
        auto reshape = std::make_shared<opset7::Reshape>(data, reshape_pattern, true);
        f_ref = std::make_shared<Function>(NodeVector{ reshape }, ParameterVector{ data });
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SimplifySecondInputOfReshapeTest14) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    PartialShape data_shape{ 1, 128, 12, 64 };
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of = std::make_shared<opset1::ShapeOf>(data);
        auto gather_op = gather(shape_of, std::vector<int64_t>{0, 1});
        auto constant = opset7::Constant::create(element::i64, Shape{ 1 }, { 768 });
        auto concat = std::make_shared<opset7::Concat>(OutputVector{ gather_op, constant }, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        f = std::make_shared<Function>(NodeVector{ reshape }, ParameterVector{ data });

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::SimplifySecondInputOfReshape>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ASSERT_EQ(reshape->get_output_partial_shape(0), PartialShape({ 1, 128, 768 }));
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto reshape_pattern = opset7::Constant::create(element::i64, Shape{ 3 }, { 0, 0, 768 });
        auto reshape = std::make_shared<opset7::Reshape>(data, reshape_pattern, true);
        f_ref = std::make_shared<Function>(NodeVector{ reshape }, ParameterVector{ data });
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SimplifySecondInputOfReshapeTest15) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    PartialShape data_shape{ 1, 128, 768 };
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto gelu = std::make_shared<opset7::Gelu>(data);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto gather_op = gather(shape_of, std::vector<int64_t>{0, 1});
        auto constant = opset7::Constant::create(element::i64, Shape{ 2 }, { 12, 64 });
        auto concat = std::make_shared<opset7::Concat>(OutputVector{ gather_op, constant }, 0);

        auto reshape = std::make_shared<opset7::Reshape>(gelu, concat, true);
        f = std::make_shared<Function>(NodeVector{ reshape }, ParameterVector{ data });

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::SimplifySecondInputOfReshape>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ASSERT_EQ(reshape->get_output_partial_shape(0), PartialShape({ 1, 128, 12, 64 }));
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto gelu = std::make_shared<opset7::Gelu>(data);
        auto reshape_pattern = opset7::Constant::create(element::i64, Shape{ 4 }, { 0, 0, 12, 64 });
        auto reshape = std::make_shared<opset7::Reshape>(gelu, reshape_pattern, true);
        f_ref = std::make_shared<Function>(NodeVector{ reshape }, ParameterVector{ data });
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SimplifySecondInputOfReshapeTest16) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    PartialShape data_shape{ 1, 128, 12, 64 };
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto gather_op_1 = gather(shape_of, std::vector<int64_t>{0});
        auto gather_op_2 = gather(shape_of, std::vector<int64_t>{1});
        auto constant = opset7::Constant::create(element::i64, Shape{ 1 }, { 768 });
        auto concat = std::make_shared<opset7::Concat>(OutputVector{ gather_op_1, gather_op_2, constant }, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        f = std::make_shared<Function>(NodeVector{ reshape }, ParameterVector{ data });

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::SimplifySecondInputOfReshape>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ASSERT_EQ(reshape->get_output_partial_shape(0), PartialShape({ 1, 128, 768 }));
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto reshape_pattern = opset7::Constant::create(element::i64, Shape{ 3 }, { 0, 0, 768 });
        auto reshape = std::make_shared<opset7::Reshape>(data, reshape_pattern, true);
        f_ref = std::make_shared<Function>(NodeVector{ reshape }, ParameterVector{ data });
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SimplifySecondInputOfReshapeTest17) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    PartialShape data_shape{-1, 256, -1};
    {
        auto data_1 = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto data_2 = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of_1 = std::make_shared<opset7::ShapeOf>(data_1);
        auto gather_op_1 = gather(shape_of_1, std::vector<int64_t>{0});

        auto constant_1 = opset7::Constant::create(element::i64, Shape{1}, {4});
        auto constant_2 = opset7::Constant::create(element::i64, Shape{1}, {64});

        auto shape_of_2 = std::make_shared<opset7::ShapeOf>(data_2);
        auto gather_op_2 = gather(shape_of_2, std::vector<int64_t>{2});
        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather_op_1, constant_1, constant_2, gather_op_2}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data_2, concat, true);
        f = std::make_shared<Function>(NodeVector{reshape}, ParameterVector{data_1, data_2});
        f_ref = ngraph::clone_function(*f);

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::SimplifySecondInputOfReshape>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ASSERT_EQ(reshape->get_output_partial_shape(0), PartialShape({-1, 4, 64, -1}));
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}
