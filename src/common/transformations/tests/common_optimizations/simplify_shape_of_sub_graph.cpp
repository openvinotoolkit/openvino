// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/simplify_shape_of_sub_graph.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/opsets/opset7_decl.hpp"
#include "openvino/opsets/opset8_decl.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace testing;
using namespace ov;

auto gatherv7 =
    [](const std::shared_ptr<Node> input, std::vector<int64_t> indices, bool scalar = false) -> Output<Node> {
    std::shared_ptr<Node> indices_node;
    if (scalar)
        indices_node = opset7::Constant::create(element::i64, {}, indices);
    else
        indices_node = opset7::Constant::create(element::i64, {indices.size()}, indices);
    return std::make_shared<opset7::Gather>(input, indices_node, opset7::Constant::create(element::i64, {}, {0}));
};

auto gatherv8 =
    [](const std::shared_ptr<Node> input, std::vector<int64_t> indices, bool scalar = false) -> Output<Node> {
    std::shared_ptr<Node> indices_node;
    if (scalar)
        indices_node = opset7::Constant::create(element::i64, {}, indices);
    else
        indices_node = opset7::Constant::create(element::i64, {indices.size()}, indices);
    return std::make_shared<opset8::Gather>(input, indices_node, opset7::Constant::create(element::i64, {}, {0}));
};

TEST_F(TransformationTestsF, ShapeSubGraphTestGatherv7) {
    Shape data_shape{1, 2, 3, 4};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_op_1 = std::make_shared<opset7::ShapeOf>(data);
        auto gather_1 = gatherv7(shape_op_1, {1}, true);
        auto unsqueeze_1 =
            std::make_shared<opset7::Unsqueeze>(gather_1, opset7::Constant::create(element::i64, {1}, {0}));

        auto shape_op_2 = std::make_shared<opset7::ShapeOf>(data);
        auto gather_2 = gatherv7(shape_op_2, {2}, true);
        auto unsqueeze_2 =
            std::make_shared<opset7::Unsqueeze>(gather_2, opset7::Constant::create(element::i64, {1}, {0}));

        auto const_1 = opset7::Constant::create(element::i64, Shape{1}, {2});
        auto const_2 = opset7::Constant::create(element::i64, Shape{1}, {2});

        auto concat = std::make_shared<opset7::Concat>(OutputVector{unsqueeze_1, unsqueeze_2, const_1, const_2}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, false);
        model = std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data});
        manager.register_pass<ov::pass::SimplifyShapeOfSubGraph>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_op_1 = std::make_shared<opset7::ShapeOf>(data);
        auto gather_1 = gatherv7(shape_op_1, {1, 2});

        auto const_1 = opset7::Constant::create(element::i64, Shape{1}, {2});
        auto const_2 = opset7::Constant::create(element::i64, Shape{1}, {2});

        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather_1, const_1, const_2}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, false);
        model_ref = std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ShapeSubGraphTestGatherv8) {
    Shape data_shape{1, 2, 3, 4};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_op_1 = std::make_shared<opset7::ShapeOf>(data);
        auto gather_1 = gatherv8(shape_op_1, {1}, true);
        auto unsqueeze_1 =
            std::make_shared<opset7::Unsqueeze>(gather_1, opset7::Constant::create(element::i64, {1}, {0}));

        auto shape_op_2 = std::make_shared<opset7::ShapeOf>(data);
        auto gather_2 = gatherv8(shape_op_2, {2}, true);
        auto unsqueeze_2 =
            std::make_shared<opset7::Unsqueeze>(gather_2, opset7::Constant::create(element::i64, {1}, {0}));

        auto const_1 = opset7::Constant::create(element::i64, Shape{1}, {2});
        auto const_2 = opset7::Constant::create(element::i64, Shape{1}, {2});

        auto concat = std::make_shared<opset7::Concat>(OutputVector{unsqueeze_1, unsqueeze_2, const_1, const_2}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, false);
        model = std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data});
        manager.register_pass<ov::pass::SimplifyShapeOfSubGraph>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_op_1 = std::make_shared<opset7::ShapeOf>(data);
        auto gather_1 = gatherv8(shape_op_1, {1, 2});

        auto const_1 = opset7::Constant::create(element::i64, Shape{1}, {2});
        auto const_2 = opset7::Constant::create(element::i64, Shape{1}, {2});

        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather_1, const_1, const_2}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, false);
        model_ref = std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ShapeNopSubGraphTestGatherv7) {
    PartialShape data_shape{-1, -1};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_op_1 = std::make_shared<opset7::ShapeOf>(data);
        auto gather_1 = gatherv7(shape_op_1, {0}, true);
        auto unsqueeze_1 =
            std::make_shared<opset7::Unsqueeze>(gather_1, opset7::Constant::create(element::i64, {1}, {0}));

        auto shape_op_2 = std::make_shared<opset7::ShapeOf>(data);
        auto gather_2 = gatherv7(shape_op_2, {1}, true);
        auto unsqueeze_2 =
            std::make_shared<opset7::Unsqueeze>(gather_2, opset7::Constant::create(element::i64, {1}, {0}));

        auto concat = std::make_shared<opset7::Concat>(OutputVector{unsqueeze_1, unsqueeze_2}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, false);
        model = std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data});
        manager.register_pass<ov::pass::SimplifyShapeOfSubGraph>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto shape_op_1 = std::make_shared<opset7::ShapeOf>(data);
        auto reshape = std::make_shared<opset7::Reshape>(data, shape_op_1, false);
        model_ref = std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ShapeNopSubGraphTestGatherv8) {
    PartialShape data_shape{-1, -1};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_op_1 = std::make_shared<opset7::ShapeOf>(data);
        auto gather_1 = gatherv8(shape_op_1, {0}, true);
        auto unsqueeze_1 =
            std::make_shared<opset7::Unsqueeze>(gather_1, opset7::Constant::create(element::i64, {1}, {0}));

        auto shape_op_2 = std::make_shared<opset7::ShapeOf>(data);
        auto gather_2 = gatherv8(shape_op_2, {1}, true);
        auto unsqueeze_2 =
            std::make_shared<opset7::Unsqueeze>(gather_2, opset7::Constant::create(element::i64, {1}, {0}));

        auto concat = std::make_shared<opset7::Concat>(OutputVector{unsqueeze_1, unsqueeze_2}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, false);
        model = std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data});
        manager.register_pass<ov::pass::SimplifyShapeOfSubGraph>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto shape_op_1 = std::make_shared<opset7::ShapeOf>(data);
        auto reshape = std::make_shared<opset7::Reshape>(data, shape_op_1, false);
        model_ref = std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GroupedGatherEliminationNegative) {
    PartialShape data_shape{2, 128};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);

        auto shape_op = std::make_shared<opset7::ShapeOf>(data);
        auto gather = gatherv8(shape_op, {1}, true);
        auto unsqueeze = std::make_shared<opset7::Unsqueeze>(gather, opset7::Constant::create(element::i64, {1}, {0}));

        auto constant_1 = opset7::Constant::create(element::i64, {1}, {0});
        auto constant_2 = opset7::Constant::create(element::i64, {1}, {1});
        auto concat = std::make_shared<opset7::Concat>(OutputVector{constant_1, constant_2, unsqueeze}, 0);

        auto reshape = std::make_shared<opset7::Reshape>(data, concat, true);
        model = std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data});
        manager.register_pass<ov::pass::GroupedGatherElimination>();
    }
}

TEST_F(TransformationTestsF, GroupedGatherEliminationNotCompatibleIndiciesSameSign) {
    PartialShape data_shape{2, 128};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto shape_op = std::make_shared<opset7::ShapeOf>(data);

        auto indices_1 = opset7::Constant::create(element::i32, {1}, {1});
        auto indices_2 = opset7::Constant::create(element::i64, {1}, {1});
        auto axis = opset7::Constant::create(element::i64, {}, {0});
        auto gather_1 = std::make_shared<opset8::Gather>(shape_op, indices_1, axis);
        auto gather_2 = std::make_shared<opset8::Gather>(shape_op, indices_2, axis);

        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather_1, gather_2}, 0);

        model = std::make_shared<Model>(OutputVector{concat}, ParameterVector{data});
        manager.register_pass<pass::GroupedGatherElimination>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto shape_op = std::make_shared<opset7::ShapeOf>(data);

        auto indices_1 = opset7::Constant::create(element::i32, {1}, {1});
        auto indices_2 = opset7::Constant::create(element::i64, {1}, {1});
        auto convert_1 = std::make_shared<opset7::Convert>(indices_1, element::i64);
        auto joint_indices = std::make_shared<opset7::Concat>(OutputVector{convert_1, indices_2}, 0);
        auto axis = opset7::Constant::create(element::i64, {}, {0});

        auto gather = std::make_shared<opset8::Gather>(shape_op, joint_indices, axis);

        model_ref = std::make_shared<Model>(OutputVector{gather}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GroupedGatherEliminationNotCompatibleIndiciesCanConvert_1) {
    PartialShape data_shape{2, 128};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto shape_op = std::make_shared<opset7::ShapeOf>(data);

        auto indices_1 = opset7::Constant::create(element::u32, {1}, {1});
        auto indices_2 = opset7::Constant::create(element::i64, {1}, {1});
        auto axis = opset7::Constant::create(element::i64, {}, {0});
        auto gather_1 = std::make_shared<opset8::Gather>(shape_op, indices_1, axis);
        auto gather_2 = std::make_shared<opset8::Gather>(shape_op, indices_2, axis);

        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather_1, gather_2}, 0);

        model = std::make_shared<Model>(OutputVector{concat}, ParameterVector{data});
        manager.register_pass<pass::GroupedGatherElimination>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto shape_op = std::make_shared<opset7::ShapeOf>(data);

        auto indices_1 = opset7::Constant::create(element::u32, {1}, {1});
        auto indices_2 = opset7::Constant::create(element::i64, {1}, {1});
        auto convert_1 = std::make_shared<opset7::Convert>(indices_1, element::i64);
        auto joint_indices = std::make_shared<opset7::Concat>(OutputVector{convert_1, indices_2}, 0);
        auto axis = opset7::Constant::create(element::i64, {}, {0});

        auto gather = std::make_shared<opset8::Gather>(shape_op, joint_indices, axis);

        model_ref = std::make_shared<Model>(OutputVector{gather}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GroupedGatherEliminationNotCompatibleIndiciesCanConvert_2) {
    PartialShape data_shape{2, 128};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto shape_op = std::make_shared<opset7::ShapeOf>(data);

        auto indices_1 = opset7::Constant::create(element::u32, {1}, {1});
        auto indices_2 = opset7::Constant::create(element::i32, {1}, {1});
        auto axis = opset7::Constant::create(element::i64, {}, {0});
        auto gather_1 = std::make_shared<opset8::Gather>(shape_op, indices_1, axis);
        auto gather_2 = std::make_shared<opset8::Gather>(shape_op, indices_2, axis);

        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather_1, gather_2}, 0);

        model = std::make_shared<Model>(OutputVector{concat}, ParameterVector{data});
        manager.register_pass<pass::GroupedGatherElimination>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto shape_op = std::make_shared<opset7::ShapeOf>(data);
        auto indices_1 = opset7::Constant::create(element::u32, {1}, {1});
        auto indices_2 = opset7::Constant::create(element::i32, {1}, {1});
        auto convert_1 = std::make_shared<opset7::Convert>(indices_1, element::i64);
        auto convert_2 = std::make_shared<opset7::Convert>(indices_2, element::i64);
        auto joint_indices = std::make_shared<opset7::Concat>(OutputVector{convert_1, convert_2}, 0);
        auto axis = opset7::Constant::create(element::i64, {}, {0});

        auto gather = std::make_shared<opset8::Gather>(shape_op, joint_indices, axis);

        model_ref = std::make_shared<Model>(OutputVector{gather}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GroupedGatherEliminationCompatibleIndicies_1) {
    PartialShape data_shape{2, 128};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto shape_op = std::make_shared<opset7::ShapeOf>(data);

        auto indices_1 = opset7::Constant::create(element::i64, {1}, {1});
        auto indices_2 = opset7::Constant::create(element::i64, {1}, {1});
        auto axis = opset7::Constant::create(element::i64, {}, {0});
        auto gather_1 = std::make_shared<opset8::Gather>(shape_op, indices_1, axis);
        auto gather_2 = std::make_shared<opset8::Gather>(shape_op, indices_2, axis);

        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather_1, gather_2}, 0);

        model = std::make_shared<Model>(OutputVector{concat}, ParameterVector{data});
        manager.register_pass<pass::GroupedGatherElimination>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto shape_op = std::make_shared<opset7::ShapeOf>(data);
        auto joint_indices = opset7::Constant::create(element::i64, {2}, {1, 1});
        auto axis = opset7::Constant::create(element::i64, {}, {0});

        auto gather = std::make_shared<opset8::Gather>(shape_op, joint_indices, axis);

        model_ref = std::make_shared<Model>(OutputVector{gather}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GroupedGatherEliminationCompatibleIndicies_2) {
    PartialShape data_shape{2, 128};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto shape_op = std::make_shared<opset7::ShapeOf>(data);

        auto indices_1 = opset7::Constant::create(element::u64, {1}, {1});
        auto indices_2 = opset7::Constant::create(element::u64, {1}, {1});
        auto axis = opset7::Constant::create(element::i64, {}, {0});
        auto gather_1 = std::make_shared<opset8::Gather>(shape_op, indices_1, axis);
        auto gather_2 = std::make_shared<opset8::Gather>(shape_op, indices_2, axis);

        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather_1, gather_2}, 0);

        model = std::make_shared<Model>(OutputVector{concat}, ParameterVector{data});
        manager.register_pass<pass::GroupedGatherElimination>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto shape_op = std::make_shared<opset7::ShapeOf>(data);
        auto joint_indices = opset7::Constant::create(element::u64, {2}, {1, 1});
        auto axis = opset7::Constant::create(element::i64, {}, {0});

        auto gather = std::make_shared<opset8::Gather>(shape_op, joint_indices, axis);

        model_ref = std::make_shared<Model>(OutputVector{gather}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GroupedGatherEliminationNotCompatibleIndiciesCannotConvert) {
    PartialShape data_shape{2, 128};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto shape_op = std::make_shared<opset7::ShapeOf>(data);

        auto indices_1 = opset7::Constant::create(element::u64, {1}, {1});
        auto indices_2 = opset7::Constant::create(element::i64, {1}, {1});
        auto axis = opset7::Constant::create(element::i64, {}, {0});
        auto gather_1 = std::make_shared<opset8::Gather>(shape_op, indices_1, axis);
        auto gather_2 = std::make_shared<opset8::Gather>(shape_op, indices_2, axis);

        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather_1, gather_2}, 0);

        model = std::make_shared<Model>(OutputVector{concat}, ParameterVector{data});
        manager.register_pass<pass::GroupedGatherElimination>();
    }
}

TEST_F(TransformationTestsF, ConcatAbsCombo) {
    PartialShape shape = PartialShape::dynamic(4);
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, shape);
        auto shape_op = std::make_shared<opset7::ShapeOf>(data);
        auto gather = gatherv8(shape_op, {0});

        auto one = opset7::Constant::create(element::i64, {1}, {1});
        auto minus_one = opset7::Constant::create(element::i64, {1}, {-1});
        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather, minus_one, one, minus_one}, 0);
        auto abs = std::make_shared<opset7::Abs>(concat);

        model = std::make_shared<Model>(OutputVector{abs}, ParameterVector{data});
        manager.register_pass<pass::AbsSinking>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, shape);
        auto shape_op = std::make_shared<opset7::ShapeOf>(data);
        auto gather = gatherv8(shape_op, {0});

        auto one0 = opset7::Constant::create(element::i64, {1}, {1});
        auto one1 = opset7::Constant::create(element::i64, {1}, {1});
        auto one2 = opset7::Constant::create(element::i64, {1}, {1});
        auto concat = std::make_shared<opset7::Concat>(OutputVector{gather, one0, one1, one2}, 0);

        model_ref = std::make_shared<Model>(OutputVector{concat}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, SingleAbsOnShape) {
    PartialShape shape = PartialShape::dynamic(4);
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, shape);
        auto shape_op = std::make_shared<opset7::ShapeOf>(data);
        auto abs = std::make_shared<opset7::Abs>(shape_op);

        model = std::make_shared<Model>(OutputVector{abs}, ParameterVector{data});
        manager.register_pass<pass::AbsSinking>();
    }
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, shape);
        auto shape_op = std::make_shared<opset7::ShapeOf>(data);

        model_ref = std::make_shared<Model>(OutputVector{shape_op}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, AbsInTheUnknown) {
    PartialShape data_shape{2, 128};
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, data_shape);
        auto abs = std::make_shared<opset7::Abs>(data);

        model = std::make_shared<Model>(OutputVector{abs}, ParameterVector{data});
        manager.register_pass<pass::AbsSinking>();
    }
}

TEST_F(TransformationTestsF, AbsSinkingDynamicDimensionPreservation) {
    // Test that AbsSinking doesn't remove Abs operations when input contains -1 (dynamic dimensions)
    // This test reproduces the issue with PyTorch expand operation
    {
        // Create constant with -1 values (dynamic dimensions in PyTorch/ONNX)
        auto const_with_neg_ones = opset7::Constant::create(element::i64, {2}, {-1, -1});
        auto abs = std::make_shared<opset7::Abs>(const_with_neg_ones);

        model = std::make_shared<Model>(OutputVector{abs}, ParameterVector{});
        manager.register_pass<pass::AbsSinking>();
    }
    {
        // Expected: AbsSinking should NOT remove Abs when input contains -1
        auto const_with_neg_ones = opset7::Constant::create(element::i64, {2}, {-1, -1});
        auto abs = std::make_shared<opset7::Abs>(const_with_neg_ones);

        model_ref = std::make_shared<Model>(OutputVector{abs}, ParameterVector{});
    }
}

TEST_F(TransformationTestsF, AbsSinkingPositiveValuesOptimization) {
    // Test that AbsSinking correctly removes Abs when input contains only positive values
    {
        auto const_with_pos_values = opset7::Constant::create(element::i64, {2}, {1, 2});
        auto abs = std::make_shared<opset7::Abs>(const_with_pos_values);

        model = std::make_shared<Model>(OutputVector{abs}, ParameterVector{});
        manager.register_pass<pass::AbsSinking>();
    }
    {
        // Expected: AbsSinking should remove Abs when input contains only positive values
        auto const_with_pos_values = opset7::Constant::create(element::i64, {2}, {1, 2});

        model_ref = std::make_shared<Model>(OutputVector{const_with_pos_values}, ParameterVector{});
    }
}
