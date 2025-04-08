// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset7.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/simplify_shape_of_sub_graph.hpp"
#include "transformations/init_node_info.hpp"

using namespace testing;
using namespace ov;

namespace {
auto gather = [](const std::shared_ptr<Node> input, std::vector<int64_t> indices) -> Output<Node> {
    std::shared_ptr<Node> indices_node = opset7::Constant::create(element::i64, {indices.size()}, indices);
    std::shared_ptr<Node> axis_node = opset7::Constant::create(element::i64, {}, {0});
    return std::make_shared<opset7::Gather>(input, indices_node, axis_node);
};

auto fake_quantize = [](const std::shared_ptr<Node> input) -> Output<Node> {
    auto il = opset7::Constant::create(element::f32, Shape{}, {0.f});
    auto ih = opset7::Constant::create(element::f32, Shape{}, {25.5f});
    auto ol = opset7::Constant::create(element::f32, Shape{}, {0.f});
    auto oh = opset7::Constant::create(element::f32, Shape{}, {25.5f});
    return std::make_shared<opset7::FakeQuantize>(input, il, ih, ol, oh, 256);
};
}  // namespace

TEST_F(TransformationTestsF, SimplifySecondInputOfReshapeTest1) {
    PartialShape data_shape{1, 128, 12, 64};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto gather_op = gather(shape_of, std::vector<int64_t>{0, 1});
        auto constant = opset7::Constant::create(element::i64, Shape{1}, {768});
        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather_op, constant}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});

        manager.register_pass<ov::pass::SimplifySecondInputOfReshape>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto reshape_pattern = opset7::Constant::create(element::i64, Shape{3}, {0, 0, 768});
        auto reshape = std::make_shared<opset7::Reshape>(data, reshape_pattern, true);
        model_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, SimplifySecondInputOfReshapeTest2) {
    PartialShape data_shape{1, 128, 12, 64};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto fq = fake_quantize(data);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto gather_op = gather(shape_of, std::vector<int64_t>{0, 1});
        auto constant = opset7::Constant::create(element::i64, Shape{1}, {768});
        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather_op, constant}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(fq, concat, true);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});

        manager.register_pass<ov::pass::SimplifySecondInputOfReshape>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto fq = fake_quantize(data);
        auto reshape_pattern = opset7::Constant::create(element::i64, Shape{3}, {0, 0, 768});
        auto reshape = std::make_shared<opset7::Reshape>(fq, reshape_pattern, true);
        model_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, SimplifySecondInputOfReshapeTest3) {
    PartialShape data_shape{1, 128, 768};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto gather_op = gather(shape_of, std::vector<int64_t>{0, 1});
        auto constant_1 = opset7::Constant::create(element::i64, Shape{1}, {12});
        auto constant_2 = opset7::Constant::create(element::i64, Shape{1}, {64});
        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather_op, constant_1, constant_2}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});

        manager.register_pass<ov::pass::SimplifySecondInputOfReshape>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto reshape_pattern = opset7::Constant::create(element::i64, Shape{4}, {0, 0, 12, 64});
        auto reshape = std::make_shared<opset7::Reshape>(data, reshape_pattern, true);
        model_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, SimplifySecondInputOfReshapeTest4) {
    PartialShape data_shape{1, 128, 768};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto fq = fake_quantize(data);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto gather_op = gather(shape_of, std::vector<int64_t>{0, 1});
        auto constant_1 = opset7::Constant::create(element::i64, Shape{1}, {12});
        auto constant_2 = opset7::Constant::create(element::i64, Shape{1}, {64});
        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather_op, constant_1, constant_2}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(fq, concat, true);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});

        manager.register_pass<ov::pass::SimplifySecondInputOfReshape>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto fq = fake_quantize(data);
        auto reshape_pattern = opset7::Constant::create(element::i64, Shape{4}, {0, 0, 12, 64});
        auto reshape = std::make_shared<opset7::Reshape>(fq, reshape_pattern, true);
        model_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, SimplifySecondInputOfReshapeTest5) {
    PartialShape data_shape = PartialShape::dynamic(3);
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto gather_op = gather(shape_of, std::vector<int64_t>{0, 1});
        auto constant = opset7::Constant::create(element::i64, Shape{1}, {-1});
        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather_op, constant}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});

        manager.register_pass<ov::pass::SimplifySecondInputOfReshape>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto reshape_pattern = opset7::Constant::create(element::i64, Shape{3}, {0, 0, -1});
        auto reshape = std::make_shared<opset7::Reshape>(data, reshape_pattern, true);
        model_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, SimplifySecondInputOfReshapeTest6) {
    PartialShape data_shape = PartialShape::dynamic();
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto gather_op = gather(shape_of, std::vector<int64_t>{0, 1});
        auto constant = opset7::Constant::create(element::i64, Shape{1}, {-1});
        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather_op, constant}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});

        manager.register_pass<ov::pass::SimplifySecondInputOfReshape>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto reshape_pattern = opset7::Constant::create(element::i64, Shape{3}, {0, 0, -1});
        auto reshape = std::make_shared<opset7::Reshape>(data, reshape_pattern, true);
        model_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, SimplifySecondInputOfReshapeTest7) {
    PartialShape data_shape{1, 128, 12, 64};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto gather_op = gather(shape_of, std::vector<int64_t>{2, 3});
        auto constant_1 = opset7::Constant::create(element::i64, Shape{1}, {64});
        auto constant_2 = opset7::Constant::create(element::i64, Shape{1}, {2});
        auto concat = std::make_shared<opset7::Concat>(OutputVector{constant_1, constant_2, gather_op}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});

        manager.register_pass<ov::pass::SimplifySecondInputOfReshape>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto reshape_pattern = opset7::Constant::create(element::i64, Shape{4}, {64, 2, 0, 0});
        auto reshape = std::make_shared<opset7::Reshape>(data, reshape_pattern, true);
        model_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, SimplifySecondInputOfReshapeTest8) {
    PartialShape data_shape{1, 128, 12, 64};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto gather_op = gather(shape_of, std::vector<int64_t>{2});
        auto constant_1 = opset7::Constant::create(element::i64, Shape{1}, {64});
        auto constant_2 = opset7::Constant::create(element::i64, Shape{1}, {2});
        auto constant_3 = opset7::Constant::create(element::i64, Shape{1}, {64});
        auto concat = std::make_shared<opset7::Concat>(OutputVector{constant_1, constant_2, gather_op, constant_3}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});

        manager.register_pass<ov::pass::SimplifySecondInputOfReshape>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto reshape_pattern = opset7::Constant::create(element::i64, Shape{4}, {64, 2, 0, 64});
        auto reshape = std::make_shared<opset7::Reshape>(data, reshape_pattern, true);
        model_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, SimplifySecondInputOfReshapeTest9) {
    PartialShape data_shape{1, 128, 12, 64};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto gather_op = gather(shape_of, std::vector<int64_t>{0, 2});
        auto constant = opset7::Constant::create(element::i64, Shape{1}, {-1});
        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather_op, constant}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});

        manager.register_pass<ov::pass::SimplifySecondInputOfReshape>();
    }
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, SimplifySecondInputOfReshapeTest10) {
    PartialShape data_shape{1, 128, 12, 64};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of_1 = std::make_shared<opset7::ShapeOf>(data);
        auto shape_of_2 = std::make_shared<opset7::ShapeOf>(data);
        auto gather_op_1 = gather(shape_of_1, std::vector<int64_t>{0, 1});
        auto gather_op_2 = gather(shape_of_2, std::vector<int64_t>{3});
        auto gather_op_3 = gather(shape_of_2, std::vector<int64_t>{2});
        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather_op_1, gather_op_2, gather_op_3}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});

        manager.register_pass<ov::pass::SimplifySecondInputOfReshape>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto constant = opset7::Constant::create(element::i64, Shape{2}, {0, 0});
        auto gather_op_2 = gather(shape_of, std::vector<int64_t>{3});
        auto gather_op_3 = gather(shape_of, std::vector<int64_t>{2});
        auto concat = std::make_shared<opset7::Concat>(OutputVector{constant, gather_op_2, gather_op_3}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        model_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, SimplifySecondInputOfReshapeTest11) {
    PartialShape data_shape{1, 128, 12, 64};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of_1 = std::make_shared<opset7::ShapeOf>(data);
        auto shape_of_2 = std::make_shared<opset7::ShapeOf>(data);
        auto concat_input_0 = gather(shape_of_1, std::vector<int64_t>{0});
        auto concat_input_1 = opset7::Constant::create(element::i64, {1}, {64});
        auto concat_input_2 = gather(shape_of_2, std::vector<int64_t>{2});
        auto concat_input_3 = opset7::Constant::create(element::i64, {1}, {128});
        auto concat = std::make_shared<opset7::Concat>(
            OutputVector{concat_input_0, concat_input_1, concat_input_2, concat_input_3},
            0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});

        manager.register_pass<ov::pass::SimplifySecondInputOfReshape>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto constant = opset7::Constant::create(element::i64, Shape{4}, {0, 64, 0, 128});
        auto reshape = std::make_shared<opset7::Reshape>(data, constant, true);
        model_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, SimplifySecondInputOfReshapeTest12) {
    PartialShape data_shape{1, 128, 768};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto gelu = std::make_shared<opset7::Gelu>(data);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto gather_op = gather(shape_of, std::vector<int64_t>{0, 1});
        auto constant_1 = opset7::Constant::create(element::i64, Shape{1}, {12});
        auto constant_2 = opset7::Constant::create(element::i64, Shape{1}, {64});
        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather_op, constant_1, constant_2}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(gelu, concat, true);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});

        manager.register_pass<ov::pass::SimplifySecondInputOfReshape>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto gelu = std::make_shared<opset7::Gelu>(data);
        auto reshape_pattern = opset7::Constant::create(element::i64, Shape{4}, {0, 0, 12, 64});
        auto reshape = std::make_shared<opset7::Reshape>(gelu, reshape_pattern, true);
        model_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, SimplifySecondInputOfReshapeTest13) {
    PartialShape data_shape{1, 128, 12, 64};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data, element::i32);
        auto gather_op = gather(shape_of, std::vector<int64_t>{0, 1});
        auto constant = opset7::Constant::create(element::i32, Shape{1}, {768});
        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather_op, constant}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});

        manager.register_pass<ov::pass::SimplifySecondInputOfReshape>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto reshape_pattern = opset7::Constant::create(element::i32, Shape{3}, {0, 0, 768});
        auto reshape = std::make_shared<opset7::Reshape>(data, reshape_pattern, true);
        model_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, SimplifySecondInputOfReshapeTest14) {
    PartialShape data_shape{1, 128, 12, 64};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of = std::make_shared<ov::op::v0::ShapeOf>(data);
        auto gather_op = gather(shape_of, std::vector<int64_t>{0, 1});
        auto constant = opset7::Constant::create(element::i64, Shape{1}, {768});
        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather_op, constant}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});

        manager.register_pass<ov::pass::SimplifySecondInputOfReshape>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto reshape_pattern = opset7::Constant::create(element::i64, Shape{3}, {0, 0, 768});
        auto reshape = std::make_shared<opset7::Reshape>(data, reshape_pattern, true);
        model_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, SimplifySecondInputOfReshapeTest15) {
    PartialShape data_shape{1, 128, 768};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto gelu = std::make_shared<opset7::Gelu>(data);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto gather_op = gather(shape_of, std::vector<int64_t>{0, 1});
        auto constant = opset7::Constant::create(element::i64, Shape{2}, {12, 64});
        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather_op, constant}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(gelu, concat, true);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});

        manager.register_pass<ov::pass::SimplifySecondInputOfReshape>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto gelu = std::make_shared<opset7::Gelu>(data);
        auto reshape_pattern = opset7::Constant::create(element::i64, Shape{4}, {0, 0, 12, 64});
        auto reshape = std::make_shared<opset7::Reshape>(gelu, reshape_pattern, true);
        model_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, SimplifySecondInputOfReshapeTest16) {
    PartialShape data_shape{1, 128, 12, 64};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto gather_op_1 = gather(shape_of, std::vector<int64_t>{0});
        auto gather_op_2 = gather(shape_of, std::vector<int64_t>{1});
        auto constant = opset7::Constant::create(element::i64, Shape{1}, {768});
        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather_op_1, gather_op_2, constant}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});

        manager.register_pass<ov::pass::SimplifySecondInputOfReshape>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto reshape_pattern = opset7::Constant::create(element::i64, Shape{3}, {0, 0, 768});
        auto reshape = std::make_shared<opset7::Reshape>(data, reshape_pattern, true);
        model_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, SimplifySecondInputOfReshapeTest17) {
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
        auto concat =
            std::make_shared<opset7::Concat>(OutputVector{gather_op_1, constant_1, constant_2, gather_op_2}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data_2, concat, true);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data_1, data_2});

        manager.register_pass<ov::pass::SimplifySecondInputOfReshape>();
    }
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, SimplifySecondInputOfReshapeTest18) {
    /*
     * InputShape [2, 6, 6] ---> Gather[0] ---------> Concat ---> OutputShape [2, 2, 3, 6]
     *                     \     Constant([2, 3]) ---' /
     *                      `--> Gather[2] -----------'
     */
    PartialShape data_shape{2, 6, 6};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto gather_1 = gather(shape_of, {0});
        auto constant = opset7::Constant::create(element::i64, Shape{2}, {2, 3});
        auto gather_2 = gather(shape_of, std::vector<int64_t>{2});
        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather_1, constant, gather_2}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});

        manager.register_pass<ov::pass::SimplifySecondInputOfReshape>();
    }

    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto gather_1 = opset7::Constant::create(element::i64, Shape{1}, {0});
        auto constant = opset7::Constant::create(element::i64, Shape{2}, {2, 3});
        auto gather_2 = gather(shape_of, std::vector<int64_t>{2});
        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather_1, constant, gather_2}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        model_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, SimplifySecondInputOfReshapeTest19) {
    /*
     * InputShape [2, 4, 2, 6] ---> Gather[0] -----> Concat ---> OutputShape [2, 8, 6]
     *                        \     Constant([8]) ---' /
     *                         `--> Gather[3] --------'
     */
    PartialShape data_shape{2, 4, 2, 6};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto gather_1 = gather(shape_of, {0});
        auto constant = opset7::Constant::create(element::i64, Shape{1}, {8});
        auto gather_2 = gather(shape_of, std::vector<int64_t>{3});
        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather_1, constant, gather_2}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});

        manager.register_pass<ov::pass::SimplifySecondInputOfReshape>();
    }

    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto gather_1 = opset7::Constant::create(element::i64, Shape{1}, {0});
        auto constant = opset7::Constant::create(element::i64, Shape{1}, {8});
        auto gather_2 = gather(shape_of, std::vector<int64_t>{3});
        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather_1, constant, gather_2}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        model_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, SimplifySecondInputOfReshapeTest20) {
    /*
     * InputShape [2, 4, 2, 2, 6] ---> Gather[0] -----> Concat ---> OutputShape [2, 2, 2, 2, 2, 6]
     *                      | |       Constant([2]) ---' / | |
     *               -----X | \ X---> Gather[2, 3] -----' / /
     *                      \  `----> Constant[2] -------' /
     *                       `------> Gather[4] ----------'
     */
    PartialShape data_shape{2, 4, 2, 2, 6};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto data_copy = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto shape_of_copy = std::make_shared<opset7::ShapeOf>(data_copy);

        auto gather_1 = gather(shape_of, {0});
        auto constant_1 = opset7::Constant::create(element::i64, Shape{1}, {2});
        auto gather_2 = gather(shape_of_copy, {2, 3});
        auto constant_2 = opset7::Constant::create(element::i64, Shape{1}, {2});
        auto gather_3 = gather(shape_of, {4});
        auto concat =
            std::make_shared<opset7::Concat>(OutputVector{gather_1, constant_1, gather_2, constant_2, gather_3}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data, data_copy});

        manager.register_pass<ov::pass::SimplifySecondInputOfReshape>();
    }

    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto data_copy = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto shape_of_copy = std::make_shared<opset7::ShapeOf>(data_copy);

        auto gather_1 = opset7::Constant::create(element::i64, Shape{1}, {0});
        auto constant_1 = opset7::Constant::create(element::i64, Shape{1}, {2});
        auto gather_2 = gather(shape_of_copy, {2, 3});
        auto constant_2 = opset7::Constant::create(element::i64, Shape{1}, {2});
        auto gather_3 = gather(shape_of, {4});
        auto concat =
            std::make_shared<opset7::Concat>(OutputVector{gather_1, constant_1, gather_2, constant_2, gather_3}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        model_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data, data_copy});
    }
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, SimplifySecondInputOfReshapeTest21) {
    PartialShape data_shape{1, 128, 12, 64};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto gather_op = gather(shape_of, std::vector<int64_t>{0, 1});
        auto constant = opset7::Constant::create(element::i64, Shape{1}, {768});
        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather_op, constant}, -1);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});

        manager.register_pass<ov::pass::SimplifySecondInputOfReshape>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto reshape_pattern = opset7::Constant::create(element::i64, Shape{3}, {0, 0, 768});
        auto reshape = std::make_shared<opset7::Reshape>(data, reshape_pattern, true);
        model_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, SimplifySecondInputOfReshapeTestFalseSpecialZero) {
    PartialShape data_shape{1, 128, 12, 64};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto gather_op = gather(shape_of, std::vector<int64_t>{0, 1});
        auto constant = opset7::Constant::create(element::i64, Shape{1}, {768});
        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather_op, constant}, -1);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, false);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});

        manager.register_pass<ov::pass::SimplifySecondInputOfReshape>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto reshape_pattern = opset7::Constant::create(element::i64, Shape{3}, {0, 0, 768});
        auto reshape = std::make_shared<opset7::Reshape>(data, reshape_pattern, true);
        model_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, SimplifySecondInputOfReshapeTestFalseSpecialZeroZeroDim) {
    PartialShape data_shape{1, 0, 12, 64};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_of = std::make_shared<opset7::ShapeOf>(data);
        auto gather_op = gather(shape_of, std::vector<int64_t>{0, 1});
        auto constant = opset7::Constant::create(element::i64, Shape{1}, {768});
        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather_op, constant}, -1);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, false);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});

        manager.register_pass<ov::pass::SimplifySecondInputOfReshape>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto reshape_pattern = opset7::Constant::create(element::i64, Shape{3}, {0, 0, 768});
        auto reshape = std::make_shared<opset7::Reshape>(data, reshape_pattern, true);
        model_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CONST_VALUES);
}
