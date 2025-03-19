// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "transformations/common_optimizations/shared_ops_optimization.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/ceiling.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace ov;
using namespace ov::op;

class SharedTransformationTestsF : public TransformationTestsF {
public:
    void TearDown() override {
        TransformationTestsF::TearDown();
        size_t op_count = model->get_ops().size(), op_count_ref = model_ref->get_ops().size();
        EXPECT_EQ(op_count, op_count_ref) << "Number of operations differ between models: model op count = " << op_count
                                          << " ref_model op count = " << op_count_ref;
    };

    static Output<Node> make_slice(const Output<Node>& out,
                                   const int64_t& start,
                                   const int64_t& stop,
                                   const int64_t& step,
                                   const int64_t& axis) {
        return std::make_shared<v8::Slice>(out,
                                           v0::Constant::create(element::i64, Shape{1}, {start}),
                                           v0::Constant::create(element::i64, Shape{1}, {stop}),
                                           v0::Constant::create(element::i64, Shape{1}, {step}),
                                           v0::Constant::create(element::i64, Shape{1}, {axis}));
    }

    static Output<Node> make_tile(const Output<Node>& out, const std::vector<int64_t>& repeats) {
        return std::make_shared<v0::Tile>(out, v0::Constant::create(element::i64, Shape{repeats.size()}, repeats));
    }

    static Output<Node> make_reshape(const Output<Node>& out, const std::vector<int64_t>& order) {
        return std::make_shared<v1::Reshape>(out, v0::Constant::create(element::i64, Shape{order.size()}, order), true);
    }
};

TEST_F(SharedTransformationTestsF, SharedSlice) {
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

        auto slice_0 = make_slice(data, 1, 2, 3, 3);
        auto slice_1 = make_slice(data, 1, 2, 3, 3);
        auto slice_2 = make_slice(data, 1, 3, 3, 3);
        auto slice_3 = make_slice(data, 1, 2, 3, 3);
        auto slice_4 = make_slice(data, 1, 2, 3, 3);

        auto concat = std::make_shared<v0::Concat>(OutputVector{slice_0, slice_1, slice_2, slice_3, slice_4}, 0);
        model = std::make_shared<ov::Model>(OutputVector{concat}, ParameterVector{data});
        manager.register_pass<ov::pass::SharedOpOptimization>();
    }
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

        auto slice_0 = make_slice(data, 1, 2, 3, 3);
        auto slice_2 = make_slice(data, 1, 3, 3, 3);

        auto concat = std::make_shared<v0::Concat>(OutputVector{slice_0, slice_0, slice_2, slice_0, slice_0}, 0);
        model_ref = std::make_shared<ov::Model>(OutputVector{concat}, ParameterVector{data});
    }
}

TEST_F(SharedTransformationTestsF, SharedRecursively) {
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

        auto slice_0 = make_slice(data, 1, 2, 3, 3);
        auto slice_1 = make_slice(data, 1, 2, 3, 3);
        auto slice_2 = make_slice(data, 1, 3, 3, 3);

        auto tile_0_0 = make_tile(slice_0, {1, 2, 3, 4});
        auto transpose_0_0 = make_reshape(slice_0, {0, 0, 0, -1});
        auto tile_0_1 = make_tile(slice_0, {1, 2, 3, 4});
        auto transpose_0_1 = make_reshape(slice_0, {0, 0, 0, -1});
        auto tile_0_2 = make_tile(slice_0, {1, 2, 3, 4});
        auto transpose_0_2 = make_reshape(slice_0, {0, 0, 0, -1});

        auto tile_1_0 = make_tile(slice_1, {1, 2, 3, 4});
        auto transpose_1_0 = make_reshape(slice_1, {0, 0, 0, -1});
        auto tile_1_1 = make_tile(slice_1, {1, 2, 3, 4});
        auto transpose_1_1 = make_reshape(slice_1, {0, 0, 0, -1});
        auto tile_1_2 = make_tile(slice_1, {1, 2, 3, 4});
        auto transpose_1_2 = make_reshape(slice_1, {0, 0, 0, -1});

        auto tile_2_0 = make_tile(slice_2, {1, 2, 3, 4});
        auto transpose_2_0 = make_reshape(slice_2, {0, 0, 0, -1});
        auto tile_2_1 = make_tile(slice_2, {1, 2, 3, 4});
        auto transpose_2_1 = make_reshape(slice_2, {0, 0, 0, -1});
        auto tile_2_2 = make_tile(slice_2, {1, 2, 3, 4});
        auto transpose_2_2 = make_reshape(slice_2, {0, 0, 0, -1});

        auto concat = std::make_shared<v0::Concat>(
            OutputVector{// source from slice 0
                         tile_0_0,
                         transpose_0_0,
                         tile_0_1,
                         transpose_0_1,
                         tile_0_2,
                         transpose_0_2,
                         // source from slice 1
                         tile_1_0,
                         transpose_1_0,
                         tile_1_1,
                         transpose_1_1,
                         tile_1_2,
                         transpose_1_2,
                         // source from slice 2
                         tile_2_0,
                         transpose_2_0,
                         tile_2_1,
                         transpose_2_1,
                         tile_2_2,
                         transpose_2_2},
            0);

        model = std::make_shared<ov::Model>(OutputVector{concat}, ParameterVector{data});
        manager.register_pass<ov::pass::SharedOpOptimization>();
    }
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

        auto slice_0 = make_slice(data, 1, 2, 3, 3);
        auto slice_2 = make_slice(data, 1, 3, 3, 3);

        auto tile_0_0 = make_tile(slice_0, {1, 2, 3, 4});
        auto transpose_0_0 = make_reshape(slice_0, {0, 0, 0, -1});

        auto tile_2_0 = make_tile(slice_2, {1, 2, 3, 4});
        auto transpose_2_0 = make_reshape(slice_2, {0, 0, 0, -1});

        auto concat = std::make_shared<v0::Concat>(
            OutputVector{// source from slice 0
                         tile_0_0,
                         transpose_0_0,
                         tile_0_0,
                         transpose_0_0,
                         tile_0_0,
                         transpose_0_0,
                         // source from slice 0
                         tile_0_0,
                         transpose_0_0,
                         tile_0_0,
                         transpose_0_0,
                         tile_0_0,
                         transpose_0_0,
                         // source from slice 2
                         tile_2_0,
                         transpose_2_0,
                         tile_2_0,
                         transpose_2_0,
                         tile_2_0,
                         transpose_2_0},
            0);

        model_ref = std::make_shared<ov::Model>(OutputVector{concat}, ParameterVector{data});
    }
}

TEST_F(SharedTransformationTestsF, SharedConcat) {
    {
        auto pre_constant_0 = v0::Constant::create(element::f32, Shape{4}, std::vector<float>{3.14f, 42.f, 0.f, 14.f});
        auto pre_constant_1 = v0::Constant::create(element::f32, Shape{4}, std::vector<float>{3.14f, 42.f, 0.f, 14.f});
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1});
        auto post_constant = v0::Constant::create(element::f32, Shape{1}, std::vector<float>{3.14f});

        auto concat_0 = std::make_shared<v0::Concat>(OutputVector{pre_constant_0, data, post_constant}, 0);
        auto concat_1 = std::make_shared<v0::Concat>(OutputVector{pre_constant_1, data, post_constant}, 0);

        auto concat = std::make_shared<v0::Concat>(OutputVector{concat_0, concat_1}, 0);
        model = std::make_shared<ov::Model>(OutputVector{concat}, ParameterVector{data});
        manager.register_pass<ov::pass::SharedOpOptimization>();
    }
    {
        auto pre_constant_0 = v0::Constant::create(element::f32, Shape{4}, std::vector<float>{3.14f, 42.f, 0.f, 14.f});
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1});
        auto post_constant = v0::Constant::create(element::f32, Shape{1}, std::vector<float>{3.14f});

        auto concat_0 = std::make_shared<v0::Concat>(OutputVector{pre_constant_0, data, post_constant}, 0);

        auto concat = std::make_shared<v0::Concat>(OutputVector{concat_0, concat_0}, 0);
        model_ref = std::make_shared<ov::Model>(OutputVector{concat}, ParameterVector{data});
    }
}

TEST_F(SharedTransformationTestsF, SharedSliceInThreeGroups) {
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape::dynamic(10));

        auto slice_0_0 = make_slice(data, 1, 2, 3, 4);
        auto slice_1_0 = make_slice(data, 2, 3, 4, 5);
        auto slice_2_0 = make_slice(data, 3, 4, 5, 6);

        auto slice_0_1 = make_slice(data, 1, 2, 3, 4);
        auto slice_1_1 = make_slice(data, 2, 3, 4, 5);
        auto slice_2_1 = make_slice(data, 3, 4, 5, 6);

        auto slice_0_2 = make_slice(data, 1, 2, 3, 4);
        auto slice_1_2 = make_slice(data, 2, 3, 4, 5);
        auto slice_2_2 = make_slice(data, 3, 4, 5, 6);

        auto concat = std::make_shared<v0::Concat>(OutputVector{slice_0_0,
                                                                slice_1_0,
                                                                slice_2_0,
                                                                slice_0_1,
                                                                slice_1_1,
                                                                slice_2_1,
                                                                slice_0_2,
                                                                slice_1_2,
                                                                slice_2_2},
                                                   0);

        model = std::make_shared<ov::Model>(OutputVector{concat}, ParameterVector{data});
        manager.register_pass<ov::pass::SharedOpOptimization>();
    }
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape::dynamic(10));

        auto slice_0_0 = make_slice(data, 1, 2, 3, 4);
        auto slice_1_0 = make_slice(data, 2, 3, 4, 5);
        auto slice_2_0 = make_slice(data, 3, 4, 5, 6);

        auto concat = std::make_shared<v0::Concat>(OutputVector{slice_0_0,
                                                                slice_1_0,
                                                                slice_2_0,
                                                                slice_0_0,
                                                                slice_1_0,
                                                                slice_2_0,
                                                                slice_0_0,
                                                                slice_1_0,
                                                                slice_2_0},
                                                   0);

        model_ref = std::make_shared<ov::Model>(OutputVector{concat}, ParameterVector{data});
    }
}

TEST_F(SharedTransformationTestsF, SharedConcatCheckOpWithResultIsntReplaced) {
    {
        auto pre_constant_0 = v0::Constant::create(element::f32, Shape{4}, std::vector<float>{3.14f, 42.f, 0.f, 14.f});
        auto pre_constant_1 = v0::Constant::create(element::f32, Shape{4}, std::vector<float>{3.14f, 42.f, 0.f, 14.f});
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1});
        auto post_constant = v0::Constant::create(element::f32, Shape{1}, std::vector<float>{3.14f});

        auto concat_0 = std::make_shared<v0::Concat>(OutputVector{pre_constant_0, data, post_constant}, 0);
        auto concat_1 = std::make_shared<v0::Concat>(OutputVector{pre_constant_1, data, post_constant}, 0);

        model = std::make_shared<ov::Model>(OutputVector{concat_0, concat_1}, ParameterVector{data});
        manager.register_pass<ov::pass::SharedOpOptimization>();
    }
}

TEST_F(SharedTransformationTestsF, SharedShapeOfTest) {
    Shape input_shape{120, 4};
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, input_shape);

        auto shapeof1_i32 = std::make_shared<v3::ShapeOf>(input, element::i32);
        auto shapeof2_i64 = std::make_shared<v3::ShapeOf>(input, element::i64);
        auto shapeof3_i32 = std::make_shared<v3::ShapeOf>(input, element::i32);
        auto shapeof4_i32 = std::make_shared<v3::ShapeOf>(input, element::i32);
        auto shapeof5_i64 = std::make_shared<v3::ShapeOf>(input, element::i64);
        auto shapeof6_i32 = std::make_shared<v3::ShapeOf>(input, element::i32);
        auto shapeof7_i32 = std::make_shared<v3::ShapeOf>(input, element::i32);

        auto shapeof1_i32_convert = std::make_shared<v0::Convert>(shapeof1_i32, element::i64);
        auto shapeof3_i32_convert = std::make_shared<v0::Convert>(shapeof3_i32, element::i64);
        auto shapeof4_i32_convert = std::make_shared<v0::Convert>(shapeof4_i32, element::i64);
        auto shapeof6_i32_convert = std::make_shared<v0::Convert>(shapeof6_i32, element::i64);
        auto shapeof7_i32_convert = std::make_shared<v0::Convert>(shapeof7_i32, element::i64);

        OutputVector inputs_of_concat{shapeof1_i32_convert,
                                      shapeof2_i64,
                                      shapeof3_i32_convert,
                                      shapeof4_i32_convert,
                                      shapeof5_i64,
                                      shapeof6_i32_convert,
                                      shapeof7_i32_convert};

        auto concat = std::make_shared<v0::Concat>(inputs_of_concat, 0);
        model = std::make_shared<Model>(NodeVector{concat}, ParameterVector{input});
        manager.register_pass<pass::SharedOpOptimization>();
    }
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, input_shape);

        auto shapeof1_i32 = std::make_shared<v3::ShapeOf>(input, element::i32);
        auto shapeof2_i64 = std::make_shared<v3::ShapeOf>(input, element::i64);

        auto shapeof1_i32_convert = std::make_shared<v0::Convert>(shapeof1_i32, element::i64);

        OutputVector inputs_of_concat{shapeof1_i32_convert,
                                      shapeof2_i64,
                                      shapeof1_i32_convert,
                                      shapeof1_i32_convert,
                                      shapeof2_i64,
                                      shapeof1_i32_convert,
                                      shapeof1_i32_convert};

        auto concat = std::make_shared<v0::Concat>(inputs_of_concat, 0);
        model_ref = std::make_shared<Model>(NodeVector{concat}, ParameterVector{input});
    }
}

TEST_F(SharedTransformationTestsF, SharedShapeOfTestI64Only) {
    Shape input_shape{120, 4};
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, input_shape);

        auto shapeof1_i64 = std::make_shared<v3::ShapeOf>(input, element::i64);
        auto shapeof2_i64 = std::make_shared<v3::ShapeOf>(input, element::i64);
        auto shapeof3_i64 = std::make_shared<v3::ShapeOf>(input, element::i64);

        OutputVector inputs_of_concat{shapeof1_i64, shapeof2_i64, shapeof3_i64};

        auto concat = std::make_shared<v0::Concat>(inputs_of_concat, 0);
        model = std::make_shared<Model>(NodeVector{concat}, ParameterVector{input});
        manager.register_pass<pass::SharedOpOptimization>();
    }
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, input_shape);
        auto shapeof1_i64 = std::make_shared<v3::ShapeOf>(input, element::i64);

        OutputVector inputs_of_concat{shapeof1_i64, shapeof1_i64, shapeof1_i64};

        auto concat = std::make_shared<v0::Concat>(inputs_of_concat, 0);
        model_ref = std::make_shared<Model>(NodeVector{concat}, ParameterVector{input});
    }
}

TEST_F(SharedTransformationTestsF, Sharedv1Broadcasts) {
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, PartialShape::dynamic());
        auto target_shape = std::make_shared<v0::Parameter>(element::i64, PartialShape::dynamic());
        auto broadcast_v1_0 = std::make_shared<v1::Broadcast>(input, target_shape);
        auto broadcast_v1_1 = std::make_shared<v1::Broadcast>(input, target_shape, AutoBroadcastType::PDPD);
        auto broadcast_v1_2 = std::make_shared<v1::Broadcast>(input, target_shape);
        auto concat = std::make_shared<v0::Concat>(OutputVector{broadcast_v1_0, broadcast_v1_1, broadcast_v1_2}, 0);
        model = std::make_shared<Model>(NodeVector{concat}, ParameterVector{input, target_shape});
        manager.register_pass<pass::SharedOpOptimization>();
    }
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, PartialShape::dynamic());
        auto target_shape = std::make_shared<v0::Parameter>(element::i64, PartialShape::dynamic());
        auto broadcast_v1_0 = std::make_shared<v1::Broadcast>(input, target_shape);
        auto broadcast_v1_1 = std::make_shared<v1::Broadcast>(input, target_shape, AutoBroadcastType::PDPD);
        auto concat = std::make_shared<v0::Concat>(OutputVector{broadcast_v1_0, broadcast_v1_1, broadcast_v1_0}, 0);
        model_ref = std::make_shared<Model>(NodeVector{concat}, ParameterVector{input, target_shape});
    }
}

TEST_F(SharedTransformationTestsF, Sharedv3Broadcasts) {
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, PartialShape::dynamic());
        auto target_shape = std::make_shared<v0::Parameter>(element::i64, PartialShape::dynamic());
        auto broadcast_v1_0 = std::make_shared<v3::Broadcast>(input, target_shape);
        auto broadcast_v1_1 = std::make_shared<v3::Broadcast>(input, target_shape, BroadcastType::BIDIRECTIONAL);
        auto broadcast_v1_2 = std::make_shared<v3::Broadcast>(input, target_shape);
        auto concat = std::make_shared<v0::Concat>(OutputVector{broadcast_v1_0, broadcast_v1_1, broadcast_v1_2}, 0);
        model = std::make_shared<Model>(NodeVector{concat}, ParameterVector{input, target_shape});
        manager.register_pass<pass::SharedOpOptimization>();
    }
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, PartialShape::dynamic());
        auto target_shape = std::make_shared<v0::Parameter>(element::i64, PartialShape::dynamic());
        auto broadcast_v1_0 = std::make_shared<v3::Broadcast>(input, target_shape);
        auto broadcast_v1_1 = std::make_shared<v3::Broadcast>(input, target_shape, BroadcastType::BIDIRECTIONAL);
        auto concat = std::make_shared<v0::Concat>(OutputVector{broadcast_v1_0, broadcast_v1_1, broadcast_v1_0}, 0);
        model_ref = std::make_shared<Model>(NodeVector{concat}, ParameterVector{input, target_shape});
    }
}

TEST_F(SharedTransformationTestsF, SharedShapeOfTestI32Only) {
    Shape input_shape{120, 4};
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, input_shape);

        auto shapeof1_i32 = std::make_shared<v3::ShapeOf>(input, element::i32);
        auto shapeof2_i32 = std::make_shared<v3::ShapeOf>(input, element::i32);
        auto shapeof3_i32 = std::make_shared<v3::ShapeOf>(input, element::i32);
        auto shapeof4_i32 = std::make_shared<v3::ShapeOf>(input, element::i32);
        auto shapeof5_i32 = std::make_shared<v3::ShapeOf>(input, element::i32);

        auto shapeof1_i32_convert = std::make_shared<v0::Convert>(shapeof1_i32, element::i64);
        auto shapeof2_i32_convert = std::make_shared<v0::Convert>(shapeof2_i32, element::i64);
        auto shapeof3_i32_convert = std::make_shared<v0::Convert>(shapeof3_i32, element::i64);
        auto shapeof4_i32_convert = std::make_shared<v0::Convert>(shapeof4_i32, element::i64);
        auto shapeof5_i32_convert = std::make_shared<v0::Convert>(shapeof5_i32, element::i64);

        OutputVector inputs_of_concat{shapeof1_i32_convert,
                                      shapeof2_i32_convert,
                                      shapeof3_i32_convert,
                                      shapeof4_i32_convert,
                                      shapeof5_i32_convert};

        auto concat = std::make_shared<v0::Concat>(inputs_of_concat, 0);
        model = std::make_shared<Model>(NodeVector{concat}, ParameterVector{input});
        manager.register_pass<pass::SharedOpOptimization>();
    }
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, input_shape);

        auto shapeof1_i32 = std::make_shared<v3::ShapeOf>(input, element::i32);

        auto shapeof1_i32_convert = std::make_shared<v0::Convert>(shapeof1_i32, element::i64);

        OutputVector inputs_of_concat{shapeof1_i32_convert,
                                      shapeof1_i32_convert,
                                      shapeof1_i32_convert,
                                      shapeof1_i32_convert,
                                      shapeof1_i32_convert};

        auto concat = std::make_shared<v0::Concat>(inputs_of_concat, 0);
        model_ref = std::make_shared<Model>(NodeVector{concat}, ParameterVector{input});
    }
}

TEST_F(SharedTransformationTestsF, SharedShapeOfTestMixed) {
    Shape input_shape{120, 4};
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, input_shape);

        auto shapeof1 = std::make_shared<v0::ShapeOf>(input);
        auto shapeof2_i64 = std::make_shared<v3::ShapeOf>(input, element::i64);
        auto shapeof3_i32 = std::make_shared<v3::ShapeOf>(input, element::i32);
        auto shapeof4 = std::make_shared<v0::ShapeOf>(input);
        auto shapeof5_i64 = std::make_shared<v3::ShapeOf>(input, element::i64);
        auto shapeof6_i32 = std::make_shared<v3::ShapeOf>(input, element::i32);
        auto shapeof7_i32 = std::make_shared<v3::ShapeOf>(input, element::i32);

        auto shapeof3_i32_convert = std::make_shared<v0::Convert>(shapeof3_i32, element::i64);
        auto shapeof6_i32_convert = std::make_shared<v0::Convert>(shapeof6_i32, element::i64);
        auto shapeof7_i32_convert = std::make_shared<v0::Convert>(shapeof7_i32, element::i64);

        OutputVector inputs_of_concat{shapeof1,
                                      shapeof2_i64,
                                      shapeof3_i32_convert,
                                      shapeof4,
                                      shapeof5_i64,
                                      shapeof6_i32_convert,
                                      shapeof7_i32_convert};

        auto concat = std::make_shared<v0::Concat>(inputs_of_concat, 0);
        model = std::make_shared<Model>(NodeVector{concat}, ParameterVector{input});
        manager.register_pass<pass::SharedOpOptimization>();
    }
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, input_shape);

        auto shapeof1 = std::make_shared<v3::ShapeOf>(input, element::i64);
        auto shapeof2_i32 = std::make_shared<v3::ShapeOf>(input, element::i32);

        auto shapeof3_i32_convert = std::make_shared<v0::Convert>(shapeof2_i32, element::i64);

        OutputVector inputs_of_concat{shapeof1,
                                      shapeof1,
                                      shapeof3_i32_convert,
                                      shapeof1,
                                      shapeof1,
                                      shapeof3_i32_convert,
                                      shapeof3_i32_convert};

        auto concat = std::make_shared<v0::Concat>(inputs_of_concat, 0);
        model_ref = std::make_shared<Model>(NodeVector{concat}, ParameterVector{input});
    }
}

namespace {
OutputVector createShapeNodesInMemory(const std::vector<size_t>& node_order_in_memory,
                                      std::shared_ptr<void>& memory,
                                      const std::string& node_name_prefix,
                                      const std::shared_ptr<Node>& input,
                                      element::Type output_type) {
    OutputVector outputs;
    memory.reset(::malloc(node_order_in_memory.size() * sizeof(v3::ShapeOf)), ::free);
    for (size_t i = 0; i < node_order_in_memory.size(); ++i) {
        v3::ShapeOf* node_addr = static_cast<v3::ShapeOf*>(memory.get()) + node_order_in_memory[i];
        auto node_ptr =
            std::shared_ptr<v3::ShapeOf>(new (node_addr) v3::ShapeOf(input, output_type), [](v3::ShapeOf* node) {
                node->v3::ShapeOf::~ShapeOf();
            });
        std::stringstream ss;
        ss << node_name_prefix << i;
        node_ptr->set_friendly_name(ss.str());
        outputs.push_back(node_ptr->output(0));
    }

    return outputs;
}

std::shared_ptr<Model> createModelWithShapes(const Shape& input_shape,
                                             const std::vector<size_t>& node_order_in_memory,
                                             const std::string& node_name_prefix,
                                             std::shared_ptr<void>& buffer) {
    auto input = std::make_shared<v0::Parameter>(element::f32, input_shape);
    auto shape_nodes = createShapeNodesInMemory(node_order_in_memory, buffer, node_name_prefix, input, element::i64);

    NodeVector inputs_of_concat;
    for (const auto& shape_node : shape_nodes) {
        auto node = std::make_shared<v0::Convert>(shape_node, element::i64);
        inputs_of_concat.push_back(node);
    }

    auto concat = std::make_shared<v0::Concat>(inputs_of_concat, 0);
    return std::make_shared<Model>(NodeVector{concat}, ParameterVector{input});
}
}  // namespace

/**
 * @brief Check that node address is not influenced on the transformation result
 */
TEST(TransformationTests, SharedShapeOfTestRandomOrder) {
    Shape input_shape{120, 4};
    std::shared_ptr<void> buffer;
    // nodes are placed into pre-allocated memory in order that is specified in next variable
    std::vector<std::vector<size_t>> node_orders_in_memory = {{0, 1}, {1, 0}};

    std::vector<std::shared_ptr<Model>> models;
    for (const auto& node_order_in_memory : node_orders_in_memory) {
        auto model = createModelWithShapes(input_shape, node_order_in_memory, "Shape_", buffer);

        ov::pass::Manager manager;
        manager.register_pass<pass::SharedOpOptimization>();
        manager.run_passes(model);

        const auto model_ops = model->get_ops();
        const auto op_it = std::find_if(model_ops.begin(), model_ops.end(), [](const std::shared_ptr<Node>& node) {
            return node->get_friendly_name() == "Shape_0";
        });
        ASSERT_TRUE(op_it != model_ops.end()) << "node Shape_0 is not found in model";
        // we need to clone while memory will be reused on the next iteration for the new model
        models.push_back(model->clone());
    }

    FunctionsComparator comparator = FunctionsComparator::with_default();
    comparator.compare(models[0], models[1]);
}

TEST_F(SharedTransformationTestsF, SharedCeiling) {
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

        auto ceiling_1 = std::make_shared<v0::Ceiling>(data);
        auto ceiling_2 = std::make_shared<v0::Ceiling>(data);

        auto concat = std::make_shared<v0::Concat>(OutputVector{ceiling_1, ceiling_2}, 0);
        model = std::make_shared<ov::Model>(OutputVector{concat}, ParameterVector{data});
        manager.register_pass<ov::pass::SharedOpOptimization>();
    }
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

        auto ceiling_1 = std::make_shared<v0::Ceiling>(data);

        auto concat = std::make_shared<v0::Concat>(OutputVector{ceiling_1, ceiling_1}, 0);
        model_ref = std::make_shared<ov::Model>(OutputVector{concat}, ParameterVector{data});
    }
}

TEST_F(SharedTransformationTestsF, SharedFloor) {
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

        auto op_1 = std::make_shared<v0::Floor>(data);
        auto op_2 = std::make_shared<v0::Floor>(data);

        auto concat = std::make_shared<v0::Concat>(OutputVector{op_1, op_2}, 0);
        model = std::make_shared<ov::Model>(OutputVector{concat}, ParameterVector{data});
        manager.register_pass<ov::pass::SharedOpOptimization>();
    }
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

        auto op_1 = std::make_shared<v0::Floor>(data);

        auto concat = std::make_shared<v0::Concat>(OutputVector{op_1, op_1}, 0);
        model_ref = std::make_shared<ov::Model>(OutputVector{concat}, ParameterVector{data});
    }
}

TEST_F(SharedTransformationTestsF, SharedRelu) {
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

        auto op_1 = std::make_shared<v0::Relu>(data);
        auto op_2 = std::make_shared<v0::Relu>(data);

        auto concat = std::make_shared<v0::Concat>(OutputVector{op_1, op_2}, 0);
        model = std::make_shared<ov::Model>(OutputVector{concat}, ParameterVector{data});
        manager.register_pass<ov::pass::SharedOpOptimization>();
    }
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

        auto op_1 = std::make_shared<v0::Relu>(data);

        auto concat = std::make_shared<v0::Concat>(OutputVector{op_1, op_1}, 0);
        model_ref = std::make_shared<ov::Model>(OutputVector{concat}, ParameterVector{data});
    }
}

TEST_F(SharedTransformationTestsF, SharedMultiply) {
    {
        auto data_0 = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
        auto data_1 = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

        auto op_0 = std::make_shared<v1::Multiply>(data_0, data_1);
        auto op_1 = std::make_shared<v1::Multiply>(data_0, data_1);

        auto concat = std::make_shared<v0::Concat>(OutputVector{op_0, op_1}, 0);
        model = std::make_shared<ov::Model>(OutputVector{concat}, ParameterVector{data_0, data_1});
        manager.register_pass<ov::pass::SharedOpOptimization>();
    }
    {
        auto data_0 = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
        auto data_1 = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

        auto op_0 = std::make_shared<v1::Multiply>(data_0, data_1);

        auto concat = std::make_shared<v0::Concat>(OutputVector{op_0, op_0}, 0);
        model_ref = std::make_shared<ov::Model>(OutputVector{concat}, ParameterVector{data_0, data_1});
    }
}

TEST_F(SharedTransformationTestsF, SharedDivide) {
    {
        auto data_0 = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
        auto data_1 = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

        auto op_0 = std::make_shared<v1::Divide>(data_0, data_1);
        auto op_1 = std::make_shared<v1::Divide>(data_0, data_1);

        auto concat = std::make_shared<v0::Concat>(OutputVector{op_0, op_1}, 0);
        model = std::make_shared<ov::Model>(OutputVector{concat}, ParameterVector{data_0, data_1});
        manager.register_pass<ov::pass::SharedOpOptimization>();
    }
    {
        auto data_0 = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
        auto data_1 = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

        auto op_0 = std::make_shared<v1::Divide>(data_0, data_1);

        auto concat = std::make_shared<v0::Concat>(OutputVector{op_0, op_0}, 0);
        model_ref = std::make_shared<ov::Model>(OutputVector{concat}, ParameterVector{data_0, data_1});
    }
}

TEST_F(SharedTransformationTestsF, SharedPad) {
    {
        auto data_0 = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
        auto data_1 = std::make_shared<v0::Parameter>(element::i32, PartialShape{4});
        auto data_2 = std::make_shared<v0::Parameter>(element::i32, PartialShape{4});

        auto op_1 = std::make_shared<v12::Pad>(data_0, data_1, data_2, PadMode::REFLECT);
        auto op_2 = std::make_shared<v12::Pad>(data_0, data_1, data_2, PadMode::REFLECT);

        auto concat = std::make_shared<v0::Concat>(OutputVector{op_1, op_2}, 0);
        model = std::make_shared<ov::Model>(OutputVector{concat}, ParameterVector{data_0, data_1, data_2});
        manager.register_pass<ov::pass::SharedOpOptimization>();
    }
    {
        auto data_0 = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
        auto data_1 = std::make_shared<v0::Parameter>(element::i32, PartialShape{4});
        auto data_2 = std::make_shared<v0::Parameter>(element::i32, PartialShape{4});

        auto op_1 = std::make_shared<v12::Pad>(data_0, data_1, data_2, PadMode::REFLECT);

        auto concat = std::make_shared<v0::Concat>(OutputVector{op_1, op_1}, 0);
        model_ref = std::make_shared<ov::Model>(OutputVector{concat}, ParameterVector{data_0, data_1, data_2});
    }
}

TEST_F(SharedTransformationTestsF, SharedMaxPool) {
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

        auto op_1 =
            std::make_shared<v8::MaxPool>(data, Strides{1, 1}, Strides{1, 1}, Shape{1, 1}, Shape{1, 1}, Shape{3, 3});
        auto op_2 =
            std::make_shared<v8::MaxPool>(data, Strides{1, 1}, Strides{1, 1}, Shape{1, 1}, Shape{1, 1}, Shape{3, 3});

        auto concat = std::make_shared<v0::Concat>(OutputVector{op_1, op_2}, 0);
        model = std::make_shared<ov::Model>(OutputVector{concat}, ParameterVector{data});
        manager.register_pass<ov::pass::SharedOpOptimization>();
    }
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

        auto op_1 =
            std::make_shared<v8::MaxPool>(data, Strides{1, 1}, Strides{1, 1}, Shape{1, 1}, Shape{1, 1}, Shape{3, 3});

        auto concat = std::make_shared<v0::Concat>(OutputVector{op_1, op_1}, 0);
        model_ref = std::make_shared<ov::Model>(OutputVector{concat}, ParameterVector{data});
    }
}

TEST_F(SharedTransformationTestsF, TopologicalOrder) {
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

        auto shape_of = std::make_shared<v3::ShapeOf>(data);

        auto gather_0 = std::make_shared<v8::Gather>(shape_of,
                                                     v0::Constant::create(element::i32, {1}, {0}),
                                                     v0::Constant::create(element::i32, {}, {0}));

        auto gather_1 = std::make_shared<v8::Gather>(shape_of,
                                                     v0::Constant::create(element::i32, {1}, {0}),
                                                     v0::Constant::create(element::i32, {}, {0}));

        auto gather_2 = std::make_shared<v8::Gather>(shape_of,
                                                     v0::Constant::create(element::i32, {1}, {0}),
                                                     v0::Constant::create(element::i32, {}, {0}));

        auto add_0 = std::make_shared<v1::Add>(gather_0, gather_0);
        auto add_1 = std::make_shared<v1::Add>(gather_1, gather_1);
        auto add_2 = std::make_shared<v1::Add>(gather_2, gather_2);

        auto concat_0 =
            std::make_shared<v0::Concat>(OutputVector{gather_0, add_0, v0::Constant::create(element::i64, {1}, {0})},
                                         0);
        auto concat_1 =
            std::make_shared<v0::Concat>(OutputVector{gather_1, add_1, v0::Constant::create(element::i64, {1}, {0})},
                                         0);
        auto concat_2 =
            std::make_shared<v0::Concat>(OutputVector{gather_2, add_2, v0::Constant::create(element::i64, {1}, {0})},
                                         0);

        auto concat = std::make_shared<v0::Concat>(OutputVector{concat_0, concat_1}, 0);
        auto output = std::make_shared<v0::Concat>(OutputVector{concat, concat_2}, 0);

        model = std::make_shared<ov::Model>(OutputVector{output}, ParameterVector{data});
        manager.register_pass<ov::pass::SharedOpOptimization>();
    }
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
        auto shape_of = std::make_shared<v3::ShapeOf>(data);
        auto gather_0 = std::make_shared<v8::Gather>(shape_of,
                                                     v0::Constant::create(element::i32, {1}, {0}),
                                                     v0::Constant::create(element::i32, {}, {0}));
        auto add_0 = std::make_shared<v1::Add>(gather_0, gather_0);
        auto concat_0 =
            std::make_shared<v0::Concat>(OutputVector{gather_0, add_0, v0::Constant::create(element::i64, {1}, {0})},
                                         0);
        auto concat = std::make_shared<v0::Concat>(OutputVector{concat_0, concat_0}, 0);
        auto output = std::make_shared<v0::Concat>(OutputVector{concat, concat_0}, 0);
        model_ref = std::make_shared<ov::Model>(OutputVector{output}, ParameterVector{data});
    }
}
