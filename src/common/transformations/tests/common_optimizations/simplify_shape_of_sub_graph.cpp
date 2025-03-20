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
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace testing;
using namespace ov;

auto gatherv7 =
    [](const std::shared_ptr<Node> input, std::vector<int64_t> indices, bool scalar = false) -> Output<Node> {
    std::shared_ptr<Node> indices_node;
    if (scalar)
        indices_node = op::v0::Constant::create(element::i64, {}, indices);
    else
        indices_node = op::v0::Constant::create(element::i64, {indices.size()}, indices);
    return std::make_shared<op::v7::Gather>(input, indices_node, op::v0::Constant::create(element::i64, {}, {0}));
};

auto gatherv8 =
    [](const std::shared_ptr<Node> input, std::vector<int64_t> indices, bool scalar = false) -> Output<Node> {
    std::shared_ptr<Node> indices_node;
    if (scalar)
        indices_node = op::v0::Constant::create(element::i64, {}, indices);
    else
        indices_node = op::v0::Constant::create(element::i64, {indices.size()}, indices);
    return std::make_shared<op::v8::Gather>(input, indices_node, op::v0::Constant::create(element::i64, {}, {0}));
};

TEST_F(TransformationTestsF, ShapeSubGraphTestGatherv7) {
    Shape data_shape{1, 2, 3, 4};
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);

        auto shape_op_1 = std::make_shared<op::v3::ShapeOf>(data);
        auto gather_1 = gatherv7(shape_op_1, {1}, true);
        auto unsqueeze_1 =
            std::make_shared<op::v0::Unsqueeze>(gather_1, op::v0::Constant::create(element::i64, {1}, {0}));

        auto shape_op_2 = std::make_shared<op::v3::ShapeOf>(data);
        auto gather_2 = gatherv7(shape_op_2, {2}, true);
        auto unsqueeze_2 =
            std::make_shared<op::v0::Unsqueeze>(gather_2, op::v0::Constant::create(element::i64, {1}, {0}));

        auto const_1 = op::v0::Constant::create(element::i64, Shape{1}, {2});
        auto const_2 = op::v0::Constant::create(element::i64, Shape{1}, {2});

        auto concat = std::make_shared<op::v0::Concat>(OutputVector{unsqueeze_1, unsqueeze_2, const_1, const_2}, 0);

        auto reshape = std::make_shared<op::v1::Reshape>(data, concat, false);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
        manager.register_pass<ov::pass::SimplifyShapeOfSubGraph>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);

        auto shape_op_1 = std::make_shared<op::v3::ShapeOf>(data);
        auto gather_1 = gatherv7(shape_op_1, {1, 2});

        auto const_1 = op::v0::Constant::create(element::i64, Shape{1}, {2});
        auto const_2 = op::v0::Constant::create(element::i64, Shape{1}, {2});

        auto concat = std::make_shared<op::v0::Concat>(OutputVector{gather_1, const_1, const_2}, 0);

        auto reshape = std::make_shared<op::v1::Reshape>(data, concat, false);
        model_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ShapeSubGraphTestGatherv8) {
    Shape data_shape{1, 2, 3, 4};
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);

        auto shape_op_1 = std::make_shared<op::v3::ShapeOf>(data);
        auto gather_1 = gatherv8(shape_op_1, {1}, true);
        auto unsqueeze_1 =
            std::make_shared<op::v0::Unsqueeze>(gather_1, op::v0::Constant::create(element::i64, {1}, {0}));

        auto shape_op_2 = std::make_shared<op::v3::ShapeOf>(data);
        auto gather_2 = gatherv8(shape_op_2, {2}, true);
        auto unsqueeze_2 =
            std::make_shared<op::v0::Unsqueeze>(gather_2, op::v0::Constant::create(element::i64, {1}, {0}));

        auto const_1 = op::v0::Constant::create(element::i64, Shape{1}, {2});
        auto const_2 = op::v0::Constant::create(element::i64, Shape{1}, {2});

        auto concat = std::make_shared<op::v0::Concat>(OutputVector{unsqueeze_1, unsqueeze_2, const_1, const_2}, 0);

        auto reshape = std::make_shared<op::v1::Reshape>(data, concat, false);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
        manager.register_pass<ov::pass::SimplifyShapeOfSubGraph>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);

        auto shape_op_1 = std::make_shared<op::v3::ShapeOf>(data);
        auto gather_1 = gatherv8(shape_op_1, {1, 2});

        auto const_1 = op::v0::Constant::create(element::i64, Shape{1}, {2});
        auto const_2 = op::v0::Constant::create(element::i64, Shape{1}, {2});

        auto concat = std::make_shared<op::v0::Concat>(OutputVector{gather_1, const_1, const_2}, 0);

        auto reshape = std::make_shared<op::v1::Reshape>(data, concat, false);
        model_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ShapeNopSubGraphTestGatherv7) {
    PartialShape data_shape{-1, -1};
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);

        auto shape_op_1 = std::make_shared<op::v3::ShapeOf>(data);
        auto gather_1 = gatherv7(shape_op_1, {0}, true);
        auto unsqueeze_1 =
            std::make_shared<op::v0::Unsqueeze>(gather_1, op::v0::Constant::create(element::i64, {1}, {0}));

        auto shape_op_2 = std::make_shared<op::v3::ShapeOf>(data);
        auto gather_2 = gatherv7(shape_op_2, {1}, true);
        auto unsqueeze_2 =
            std::make_shared<op::v0::Unsqueeze>(gather_2, op::v0::Constant::create(element::i64, {1}, {0}));

        auto concat = std::make_shared<op::v0::Concat>(OutputVector{unsqueeze_1, unsqueeze_2}, 0);

        auto reshape = std::make_shared<op::v1::Reshape>(data, concat, false);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
        manager.register_pass<ov::pass::SimplifyShapeOfSubGraph>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto shape_op_1 = std::make_shared<op::v3::ShapeOf>(data);
        auto reshape = std::make_shared<op::v1::Reshape>(data, shape_op_1, false);
        model_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ShapeNopSubGraphTestGatherv8) {
    PartialShape data_shape{-1, -1};
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);

        auto shape_op_1 = std::make_shared<op::v3::ShapeOf>(data);
        auto gather_1 = gatherv8(shape_op_1, {0}, true);
        auto unsqueeze_1 =
            std::make_shared<op::v0::Unsqueeze>(gather_1, op::v0::Constant::create(element::i64, {1}, {0}));

        auto shape_op_2 = std::make_shared<op::v3::ShapeOf>(data);
        auto gather_2 = gatherv8(shape_op_2, {1}, true);
        auto unsqueeze_2 =
            std::make_shared<op::v0::Unsqueeze>(gather_2, op::v0::Constant::create(element::i64, {1}, {0}));

        auto concat = std::make_shared<op::v0::Concat>(OutputVector{unsqueeze_1, unsqueeze_2}, 0);

        auto reshape = std::make_shared<op::v1::Reshape>(data, concat, false);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
        manager.register_pass<ov::pass::SimplifyShapeOfSubGraph>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto shape_op_1 = std::make_shared<op::v3::ShapeOf>(data);
        auto reshape = std::make_shared<op::v1::Reshape>(data, shape_op_1, false);
        model_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GroupedGatherEliminationNegative) {
    PartialShape data_shape{2, 128};
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);

        auto shape_op = std::make_shared<op::v3::ShapeOf>(data);
        auto gather = gatherv8(shape_op, {1}, true);
        auto unsqueeze = std::make_shared<op::v0::Unsqueeze>(gather, op::v0::Constant::create(element::i64, {1}, {0}));

        auto constant_1 = op::v0::Constant::create(element::i64, {1}, {0});
        auto constant_2 = op::v0::Constant::create(element::i64, {1}, {1});
        auto concat = std::make_shared<op::v0::Concat>(OutputVector{constant_1, constant_2, unsqueeze}, 0);

        auto reshape = std::make_shared<op::v1::Reshape>(data, concat, true);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
        manager.register_pass<ov::pass::GroupedGatherElimination>();
    }
}

TEST_F(TransformationTestsF, GroupedGatherEliminationNotCompatibleIndiciesSameSign) {
    PartialShape data_shape{2, 128};
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto shape_op = std::make_shared<op::v3::ShapeOf>(data);

        auto indices_1 = op::v0::Constant::create(element::i32, {1}, {1});
        auto indices_2 = op::v0::Constant::create(element::i64, {1}, {1});
        auto axis = op::v0::Constant::create(element::i64, {}, {0});
        auto gather_1 = std::make_shared<op::v8::Gather>(shape_op, indices_1, axis);
        auto gather_2 = std::make_shared<op::v8::Gather>(shape_op, indices_2, axis);

        auto concat = std::make_shared<op::v0::Concat>(OutputVector{gather_1, gather_2}, 0);

        model = std::make_shared<Model>(NodeVector{concat}, ParameterVector{data});
        manager.register_pass<pass::GroupedGatherElimination>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto shape_op = std::make_shared<op::v3::ShapeOf>(data);

        auto indices_1 = op::v0::Constant::create(element::i32, {1}, {1});
        auto indices_2 = op::v0::Constant::create(element::i64, {1}, {1});
        auto convert_1 = std::make_shared<op::v0::Convert>(indices_1, element::i64);
        auto joint_indices = std::make_shared<op::v0::Concat>(OutputVector{convert_1, indices_2}, 0);
        auto axis = op::v0::Constant::create(element::i64, {}, {0});

        auto gather = std::make_shared<op::v8::Gather>(shape_op, joint_indices, axis);

        model_ref = std::make_shared<Model>(NodeVector{gather}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GroupedGatherEliminationNotCompatibleIndiciesCanConvert_1) {
    PartialShape data_shape{2, 128};
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto shape_op = std::make_shared<op::v3::ShapeOf>(data);

        auto indices_1 = op::v0::Constant::create(element::u32, {1}, {1});
        auto indices_2 = op::v0::Constant::create(element::i64, {1}, {1});
        auto axis = op::v0::Constant::create(element::i64, {}, {0});
        auto gather_1 = std::make_shared<op::v8::Gather>(shape_op, indices_1, axis);
        auto gather_2 = std::make_shared<op::v8::Gather>(shape_op, indices_2, axis);

        auto concat = std::make_shared<op::v0::Concat>(OutputVector{gather_1, gather_2}, 0);

        model = std::make_shared<Model>(NodeVector{concat}, ParameterVector{data});
        manager.register_pass<pass::GroupedGatherElimination>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto shape_op = std::make_shared<op::v3::ShapeOf>(data);

        auto indices_1 = op::v0::Constant::create(element::u32, {1}, {1});
        auto indices_2 = op::v0::Constant::create(element::i64, {1}, {1});
        auto convert_1 = std::make_shared<op::v0::Convert>(indices_1, element::i64);
        auto joint_indices = std::make_shared<op::v0::Concat>(OutputVector{convert_1, indices_2}, 0);
        auto axis = op::v0::Constant::create(element::i64, {}, {0});

        auto gather = std::make_shared<op::v8::Gather>(shape_op, joint_indices, axis);

        model_ref = std::make_shared<Model>(NodeVector{gather}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GroupedGatherEliminationNotCompatibleIndiciesCanConvert_2) {
    PartialShape data_shape{2, 128};
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto shape_op = std::make_shared<op::v3::ShapeOf>(data);

        auto indices_1 = op::v0::Constant::create(element::u32, {1}, {1});
        auto indices_2 = op::v0::Constant::create(element::i32, {1}, {1});
        auto axis = op::v0::Constant::create(element::i64, {}, {0});
        auto gather_1 = std::make_shared<op::v8::Gather>(shape_op, indices_1, axis);
        auto gather_2 = std::make_shared<op::v8::Gather>(shape_op, indices_2, axis);

        auto concat = std::make_shared<op::v0::Concat>(OutputVector{gather_1, gather_2}, 0);

        model = std::make_shared<Model>(NodeVector{concat}, ParameterVector{data});
        manager.register_pass<pass::GroupedGatherElimination>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto shape_op = std::make_shared<op::v3::ShapeOf>(data);
        auto indices_1 = op::v0::Constant::create(element::u32, {1}, {1});
        auto indices_2 = op::v0::Constant::create(element::i32, {1}, {1});
        auto convert_1 = std::make_shared<op::v0::Convert>(indices_1, element::i64);
        auto convert_2 = std::make_shared<op::v0::Convert>(indices_2, element::i64);
        auto joint_indices = std::make_shared<op::v0::Concat>(OutputVector{convert_1, convert_2}, 0);
        auto axis = op::v0::Constant::create(element::i64, {}, {0});

        auto gather = std::make_shared<op::v8::Gather>(shape_op, joint_indices, axis);

        model_ref = std::make_shared<Model>(NodeVector{gather}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GroupedGatherEliminationCompatibleIndicies_1) {
    PartialShape data_shape{2, 128};
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto shape_op = std::make_shared<op::v3::ShapeOf>(data);

        auto indices_1 = op::v0::Constant::create(element::i64, {1}, {1});
        auto indices_2 = op::v0::Constant::create(element::i64, {1}, {1});
        auto axis = op::v0::Constant::create(element::i64, {}, {0});
        auto gather_1 = std::make_shared<op::v8::Gather>(shape_op, indices_1, axis);
        auto gather_2 = std::make_shared<op::v8::Gather>(shape_op, indices_2, axis);

        auto concat = std::make_shared<op::v0::Concat>(OutputVector{gather_1, gather_2}, 0);

        model = std::make_shared<Model>(NodeVector{concat}, ParameterVector{data});
        manager.register_pass<pass::GroupedGatherElimination>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto shape_op = std::make_shared<op::v3::ShapeOf>(data);
        auto joint_indices = op::v0::Constant::create(element::i64, {2}, {1, 1});
        auto axis = op::v0::Constant::create(element::i64, {}, {0});

        auto gather = std::make_shared<op::v8::Gather>(shape_op, joint_indices, axis);

        model_ref = std::make_shared<Model>(NodeVector{gather}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GroupedGatherEliminationCompatibleIndicies_2) {
    PartialShape data_shape{2, 128};
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto shape_op = std::make_shared<op::v3::ShapeOf>(data);

        auto indices_1 = op::v0::Constant::create(element::u64, {1}, {1});
        auto indices_2 = op::v0::Constant::create(element::u64, {1}, {1});
        auto axis = op::v0::Constant::create(element::i64, {}, {0});
        auto gather_1 = std::make_shared<op::v8::Gather>(shape_op, indices_1, axis);
        auto gather_2 = std::make_shared<op::v8::Gather>(shape_op, indices_2, axis);

        auto concat = std::make_shared<op::v0::Concat>(OutputVector{gather_1, gather_2}, 0);

        model = std::make_shared<Model>(NodeVector{concat}, ParameterVector{data});
        manager.register_pass<pass::GroupedGatherElimination>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto shape_op = std::make_shared<op::v3::ShapeOf>(data);
        auto joint_indices = op::v0::Constant::create(element::u64, {2}, {1, 1});
        auto axis = op::v0::Constant::create(element::i64, {}, {0});

        auto gather = std::make_shared<op::v8::Gather>(shape_op, joint_indices, axis);

        model_ref = std::make_shared<Model>(NodeVector{gather}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GroupedGatherEliminationNotCompatibleIndiciesCannotConvert) {
    PartialShape data_shape{2, 128};
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto shape_op = std::make_shared<op::v3::ShapeOf>(data);

        auto indices_1 = op::v0::Constant::create(element::u64, {1}, {1});
        auto indices_2 = op::v0::Constant::create(element::i64, {1}, {1});
        auto axis = op::v0::Constant::create(element::i64, {}, {0});
        auto gather_1 = std::make_shared<op::v8::Gather>(shape_op, indices_1, axis);
        auto gather_2 = std::make_shared<op::v8::Gather>(shape_op, indices_2, axis);

        auto concat = std::make_shared<op::v0::Concat>(OutputVector{gather_1, gather_2}, 0);

        model = std::make_shared<Model>(NodeVector{concat}, ParameterVector{data});
        manager.register_pass<pass::GroupedGatherElimination>();
    }
}

TEST_F(TransformationTestsF, ConcatAbsCombo) {
    PartialShape shape = PartialShape::dynamic(4);
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, shape);
        auto shape_op = std::make_shared<op::v3::ShapeOf>(data);
        auto gather = gatherv8(shape_op, {0});

        auto one = op::v0::Constant::create(element::i64, {1}, {1});
        auto minus_one = op::v0::Constant::create(element::i64, {1}, {-1});
        auto concat = std::make_shared<op::v0::Concat>(OutputVector{gather, minus_one, one, minus_one}, 0);
        auto abs = std::make_shared<op::v0::Abs>(concat);

        model = std::make_shared<Model>(NodeVector{abs}, ParameterVector{data});
        manager.register_pass<pass::AbsSinking>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, shape);
        auto shape_op = std::make_shared<op::v3::ShapeOf>(data);
        auto gather = gatherv8(shape_op, {0});

        auto one0 = op::v0::Constant::create(element::i64, {1}, {1});
        auto one1 = op::v0::Constant::create(element::i64, {1}, {1});
        auto one2 = op::v0::Constant::create(element::i64, {1}, {1});
        auto concat = std::make_shared<op::v0::Concat>(OutputVector{gather, one0, one1, one2}, 0);

        model_ref = std::make_shared<Model>(NodeVector{concat}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, SingleAbsOnShape) {
    PartialShape shape = PartialShape::dynamic(4);
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, shape);
        auto shape_op = std::make_shared<op::v3::ShapeOf>(data);
        auto abs = std::make_shared<op::v0::Abs>(shape_op);

        model = std::make_shared<Model>(NodeVector{abs}, ParameterVector{data});
        manager.register_pass<pass::AbsSinking>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, shape);
        auto shape_op = std::make_shared<op::v3::ShapeOf>(data);

        model_ref = std::make_shared<Model>(OutputVector{shape_op}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, AbsInTheUnknown) {
    PartialShape data_shape{2, 128};
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto abs = std::make_shared<op::v0::Abs>(data);

        model = std::make_shared<Model>(NodeVector{abs}, ParameterVector{data});
        manager.register_pass<pass::AbsSinking>();
    }
}
