// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>

#include <transformations/common_optimizations/shared_ops_optimization.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/tile.hpp"

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
