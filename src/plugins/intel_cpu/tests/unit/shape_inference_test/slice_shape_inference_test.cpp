// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/slice.hpp"
#include "slice_shape_inference.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

class SliceStaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v8::Slice> {
protected:
    void SetUp() override {
        output_shapes.resize(num_of_outputs);
    }

    size_t num_of_outputs = 1;
    StaticDimension::value_type max_d = std::numeric_limits<StaticDimension::value_type>::max();
};

TEST_F(SliceStaticShapeInferenceTest, reverse_steps_start_stop_outside_dimension_default_axes) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto start =
        op::v0::Constant::create(element::i64, ov::Shape{5}, std::vector<int64_t>{100, 5, -1, INT64_MAX, 5});
    const auto stop =
        op::v0::Constant::create(element::i64, ov::Shape{5}, std::vector<int64_t>{-100, INT64_MIN, -6, 5, -10});
    const auto steps = op::v0::Constant::create(element::i64, ov::Shape{5}, {-1, -2, -1, -1, -2});

    const auto op = make_op(data, start, stop, steps);

    input_shapes.push_back({3, 4, 5, max_d, max_d});
    input_shapes.resize(4, start->get_shape());

    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), num_of_outputs);
    EXPECT_EQ(output_shapes.front(), StaticShape({3, 2, 5, max_d, 3}));
}

TEST_F(SliceStaticShapeInferenceTest, reverse_step_on_signle_axis_but_start_stop_steps_in_const_map) {
    constexpr auto et = element::i64;

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto start = std::make_shared<op::v0::Parameter>(et, PartialShape::dynamic());
    const auto stop = std::make_shared<op::v0::Parameter>(et, PartialShape::dynamic());
    const auto steps = std::make_shared<op::v0::Parameter>(et, PartialShape::dynamic());
    const auto axes = op::v0::Constant::create(element::i64, ov::Shape{1}, {-1});

    auto start_buff = std::vector<int64_t>{100};
    auto stop_buff = std::vector<int64_t>{2};
    auto steps_buff = std::vector<int64_t>{-2};

    const auto start_tensor = ov::Tensor(element::i64, ov::Shape{1}, static_cast<void*>(start_buff.data()));
    const auto stop_tensor = ov::Tensor(element::i64, ov::Shape{1}, static_cast<void*>(stop_buff.data()));
    const auto steps_tensor = ov::Tensor(element::i64, ov::Shape{1}, static_cast<void*>(steps_buff.data()));

    const auto op = make_op(data, start, stop, steps, axes);

    input_shapes = StaticShapeVector{{3, 4, 10}, {1}, {1}, {1}, axes->get_shape()};

    const std::unordered_map<size_t, ov::Tensor>& constant_data = {{1, start_tensor},
                                                                   {2, stop_tensor},
                                                                   {3, steps_tensor}};

    output_shapes = shape_inference(op.get(), input_shapes, constant_data);

    EXPECT_EQ(output_shapes.size(), num_of_outputs);
    EXPECT_EQ(output_shapes.front(), StaticShape({3, 4, 4}));
}

TEST_F(SliceStaticShapeInferenceTest, forward_step_all_data_in_const_map) {
    constexpr auto et = element::i64;

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto start = std::make_shared<op::v0::Parameter>(et, PartialShape::dynamic());
    const auto stop = std::make_shared<op::v0::Parameter>(et, PartialShape::dynamic());
    const auto steps = std::make_shared<op::v0::Parameter>(et, PartialShape::dynamic());

    auto start_buff = std::vector<int64_t>{0, 2, 10, 3, 3, INT64_MIN, INT64_MIN};
    auto stop_buff = std::vector<int64_t>{10, 8, 12, 15, INT64_MAX, -5, -5};
    auto steps_buff = std::vector<int64_t>{1, 2, 1, 3, 4, 2, 2};
    auto axes_buff = std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6};

    const auto common_shape = ov::Shape{start_buff.size()};

    const auto start_tensor = ov::Tensor(element::i64, common_shape, static_cast<void*>(start_buff.data()));
    const auto stop_tensor = ov::Tensor(element::i64, common_shape, static_cast<void*>(stop_buff.data()));
    const auto steps_tensor = ov::Tensor(element::i64, common_shape, static_cast<void*>(steps_buff.data()));
    const auto axes_tensor = ov::Tensor(element::i64, common_shape, static_cast<void*>(axes_buff.data()));

    const auto op = make_op(data, start, stop, steps);

    input_shapes.push_back({10, 10, 8, max_d, max_d, max_d, 10});
    input_shapes.resize(5, common_shape);

    const std::unordered_map<size_t, ov::Tensor>& constant_data = {{1, start_tensor},
                                                                   {2, stop_tensor},
                                                                   {3, steps_tensor},
                                                                   {4, axes_tensor}};

    output_shapes = shape_inference(op.get(), input_shapes, constant_data);

    EXPECT_EQ(output_shapes.size(), num_of_outputs);
    EXPECT_EQ(output_shapes.front(), StaticShape({10, 3, 0, 4, max_d, max_d, 3}));
}

TEST_F(SliceStaticShapeInferenceTest, identity_and_step_2_on_unbounded_dims_in_const_map) {
    constexpr auto et = element::i64;

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto start = std::make_shared<op::v0::Parameter>(et, PartialShape::dynamic());
    const auto stop = std::make_shared<op::v0::Parameter>(et, PartialShape::dynamic());
    const auto steps = std::make_shared<op::v0::Parameter>(et, PartialShape::dynamic());
    const auto axes = op::v0::Constant::create(element::i64, ov::Shape{4}, {0, 1, 2, 3});

    auto start_buff = std::vector<int64_t>{0, 0, 0, INT64_MAX};
    auto stop_buff = std::vector<int64_t>{INT64_MAX, INT64_MAX, 10, INT64_MIN};
    auto steps_buff = std::vector<int64_t>{1, 2, 1, -1};

    const auto start_tensor = ov::Tensor(element::i64, ov::Shape{4}, static_cast<void*>(start_buff.data()));
    const auto stop_tensor = ov::Tensor(element::i64, ov::Shape{4}, static_cast<void*>(stop_buff.data()));
    const auto steps_tensor = ov::Tensor(element::i64, ov::Shape{4}, static_cast<void*>(steps_buff.data()));
    const auto const_data =
        std::unordered_map<size_t, ov::Tensor>{{1, start_tensor}, {2, stop_tensor}, {3, steps_tensor}};

    const auto op = make_op(data, start, stop, steps, axes);

    input_shapes = StaticShapeVector{{max_d, max_d, 10, 4}, {4}, {4}, {4}, {4}};
    output_shapes = shape_inference(op.get(), input_shapes, const_data);

    EXPECT_EQ(output_shapes.size(), num_of_outputs);
    EXPECT_EQ(output_shapes.front(), StaticShape({max_d, max_d, 10, 4}));
}

TEST(SliceShapeInferenceUtils, is_identity_slice_with_static_dimension) {
    // The production call sites of the size-preservation predicate are compiled for ov::Dimension only (they sit
    // inside `if constexpr`), so this is the instantiation that keeps the helper free of the ov::Dimension symbol API.
    using ov::op::slice::Bounds;
    constexpr auto i64_max = std::numeric_limits<int64_t>::max();
    constexpr auto i64_min = std::numeric_limits<int64_t>::min();
    const auto max_d = std::numeric_limits<StaticDimension::value_type>::max();

    EXPECT_TRUE(ov::op::slice::is_identity_slice(StaticDimension(10), Bounds{0, 0}, Bounds{i64_max, i64_max}, 1));
    EXPECT_FALSE(ov::op::slice::is_identity_slice(StaticDimension(10), Bounds{0, 0}, Bounds{i64_max, i64_max}, 2));
    EXPECT_TRUE(ov::op::slice::is_identity_slice(StaticDimension(10), Bounds{-1, -1}, Bounds{i64_min, i64_min}, -1));
    EXPECT_TRUE(ov::op::slice::is_identity_slice(StaticDimension(max_d), Bounds{0, 0}, Bounds{i64_max, i64_max}, 1));
    EXPECT_FALSE(ov::op::slice::is_identity_slice(StaticDimension(max_d), Bounds{0, 0}, Bounds{i64_max, i64_max}, 2));
}
