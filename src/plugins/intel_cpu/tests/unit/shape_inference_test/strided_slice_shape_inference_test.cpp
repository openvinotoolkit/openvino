// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "strided_slice_shape_inference.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

class StridedSliceStaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v1::StridedSlice> {
protected:
    void SetUp() override {
        output_shapes.resize(1);
    }
};


TEST_F(StridedSliceStaticShapeInferenceTest, reverse_stride_begin_end_clip_to_dimension) {
    const auto mask = std::vector<int64_t>(4, 0);

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());
    const auto begin = op::v0::Constant::create(element::i64, ov::Shape{3}, {100});
    const auto end = op::v0::Constant::create(element::i64, ov::Shape{3}, {-100});
    const auto stride = op::v0::Constant::create(element::i64, ov::Shape{3}, {-1});

    const auto op = make_op(data, begin, end, stride, mask, mask);

    check_static_shape(op.get(),
                       {StaticShape{3, 4, 5}, StaticShape{3}, StaticShape{3}, StaticShape{3}},
                       {StaticShape{3, 4, 5}});
}

TEST_F(StridedSliceStaticShapeInferenceTest, use_begin_end) {
    const auto mask = std::vector<int64_t>(4, 0);

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());
    const auto begin = std::make_shared<op::v0::Parameter>(element::i64, ov::Shape{3});
    const auto end = std::make_shared<op::v0::Parameter>(element::i64, ov::Shape{3});
    const auto stride = std::make_shared<op::v0::Parameter>(element::i64, ov::Shape{3});

    const auto op = make_op(data, begin, end, stride, mask, mask);

    check_static_shape(op.get(), {StaticShape{3, 2, 3}, {1, 0, 0}, {2, 1, 3}, {1, 1, 1}}, {StaticShape{1, 1, 3}});

    check_static_shape(op.get(), {StaticShape{3, 2, 3}, {1, 0, 0}, {2, 2, 3}, {1, 1, 1}}, {StaticShape{1, 2, 3}});

    check_static_shape(op.get(), {StaticShape{3, 2, 3}, {2, 0, 0}, {3, 2, 3}, {1, 1, 2}}, {StaticShape{1, 2, 2}});
}

TEST_F(StridedSliceStaticShapeInferenceTest, ignore_begin_end) {
    const auto begin_mask = std::vector<int64_t>{0, 1, 1};
    const auto end_mask = std::vector<int64_t>(3, 1);

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());
    const auto begin = std::make_shared<op::v0::Parameter>(element::i64, ov::Shape{3});
    const auto end = std::make_shared<op::v0::Parameter>(element::i64, ov::Shape{3});
    const auto stride = std::make_shared<op::v0::Parameter>(element::i64, ov::Shape{3});

    const auto op = make_op(data, begin, end, stride, begin_mask, end_mask);

    check_static_shape(op.get(), {StaticShape{3, 2, 3}, {1, 0, 0}, {0, 0, 0}, {1, 1, 1}}, {StaticShape{2, 2, 3}});
}

TEST_F(StridedSliceStaticShapeInferenceTest, ignore_begin_end_stride_by_two_last_dim) {
    const auto begin_mask = std::vector<int64_t>{1, 0, 1};
    const auto end_mask = std::vector<int64_t>{0, 1, 1};

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());
    const auto begin = std::make_shared<op::v0::Parameter>(element::i64, ov::Shape{3});
    const auto end = std::make_shared<op::v0::Parameter>(element::i64, ov::Shape{3});
    const auto stride = std::make_shared<op::v0::Parameter>(element::i64, ov::Shape{3});

    auto op = make_op(data, begin, end, stride, begin_mask, end_mask);

    check_static_shape(op.get(), {StaticShape{3, 2, 3}, {0, 1, 0}, {2, 0, 0}, {1, 1, 2}}, {StaticShape{2, 1, 2}});
}

TEST_F(StridedSliceStaticShapeInferenceTest, use_reverse_stride_on_last_dimension) {
    const auto mask = std::vector<int64_t>{0, 1, 1};

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());
    const auto begin = std::make_shared<op::v0::Parameter>(element::i64, ov::Shape{3});
    const auto end = std::make_shared<op::v0::Parameter>(element::i64, ov::Shape{3});
    const auto stride = std::make_shared<op::v0::Parameter>(element::i64, ov::Shape{3});

    const auto op = make_op(data, begin, end, stride, mask, mask);

    check_static_shape(op.get(), {StaticShape{3, 2, 3}, {0, 0, 0}, {1, 0, 0}, {1, 1, -1}}, {StaticShape{1, 2, 3}});
}

TEST_F(StridedSliceStaticShapeInferenceTest, default_stride) {
    const auto mask = std::vector<int64_t>{0, 1, 0};

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());
    const auto begin = op::v0::Constant::create(element::i64, ov::Shape{3}, {0, 0, 0});
    const auto end = op::v0::Constant::create(element::i64, ov::Shape{3}, {1, 0, 2});
    const auto op = make_op(data, begin, end, mask, mask);

    input_shapes = ShapeVector{{3, 2, 3}, {3}, {3}};

    shape_inference(op.get(), input_shapes, output_shapes);

    ASSERT_EQ(output_shapes.front(), StaticShape({1, 2, 2}));
}


using StridedSliceCustomShapeInferenceTest = StridedSliceStaticShapeInferenceTest;

TEST_F(StridedSliceCustomShapeInferenceTest, reverse_stride_begin_end_clip_to_dimension) {
    const auto mask = std::vector<int64_t>(4, 0);

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());
    // only support i32
    const auto begin = op::v0::Constant::create(element::i32, ov::Shape{3}, {100});
    const auto end = op::v0::Constant::create(element::i32, ov::Shape{3}, {-100});
    const auto stride = op::v0::Constant::create(element::i32, ov::Shape{3}, {-1});

    const auto op = make_op(data, begin, end, stride, mask, mask);

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 4, 5}, StaticShape{3}, StaticShape{3}, StaticShape{3}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{3, 4, 5}};
    // implementation depends on some output information of the op
    op->set_output_type(0, element::i32, {3, 4, 5});
    unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes);
}

TEST_F(StridedSliceCustomShapeInferenceTest, use_begin_end1) {
    const auto mask = std::vector<int64_t>(4, 0);

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());
    // only supprot i32
    const auto begin = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{3});
    const auto end = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{3});
    const auto stride = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{3});

    const auto op = make_op(data, begin, end, stride, mask, mask);

    int32_t begin_data[] = {1, 0, 0};
    int32_t end_data[] = {2, 1, 3};
    int32_t stride_data[] = {1, 1, 1};
    const std::map<size_t, HostTensorPtr> const_data{
        {1, std::make_shared<HostTensor>(element::i32, ov::Shape{3}, begin_data)},
        {2, std::make_shared<HostTensor>(element::i32, ov::Shape{3}, end_data)},
        {3, std::make_shared<HostTensor>(element::i32, ov::Shape{3}, stride_data)},
    };

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 2, 3}, {3}, {3}, {3}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{1, 1, 3}};
    // implementation depends on some output information of the op
    op->set_output_type(0, element::i32, {1, 1, 3});
    unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes, const_data);
}

TEST_F(StridedSliceCustomShapeInferenceTest, use_begin_end2) {
    const auto mask = std::vector<int64_t>(4, 0);

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());
    // only supprot i32
    const auto begin = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{3});
    const auto end = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{3});
    const auto stride = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{3});

    const auto op = make_op(data, begin, end, stride, mask, mask);

    int32_t begin_data[] = {1, 0, 0};
    int32_t end_data[] = {2, 2, 3};
    int32_t stride_data[] = {1, 1, 1};
    const std::map<size_t, HostTensorPtr> const_data{
        {1, std::make_shared<HostTensor>(element::i32, ov::Shape{3}, begin_data)},
        {2, std::make_shared<HostTensor>(element::i32, ov::Shape{3}, end_data)},
        {3, std::make_shared<HostTensor>(element::i32, ov::Shape{3}, stride_data)},
    };

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 2, 3}, {3}, {3}, {3}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{1, 2, 3}};
    // implementation depends on some output information of the op
    op->set_output_type(0, element::i32, {1, 2, 3});
    unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes, const_data);
}

TEST_F(StridedSliceCustomShapeInferenceTest, use_begin_end3) {
    const auto mask = std::vector<int64_t>(4, 0);

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());
    // only supprot i32
    const auto begin = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{3});
    const auto end = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{3});
    const auto stride = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{3});

    const auto op = make_op(data, begin, end, stride, mask, mask);

    int32_t begin_data[] = {2, 0, 0};
    int32_t end_data[] = {3, 2, 3};
    int32_t stride_data[] = {1, 1, 2};

    const std::map<size_t, HostTensorPtr> const_data{
        {1, std::make_shared<HostTensor>(element::i32, ov::Shape{3}, begin_data)},
        {2, std::make_shared<HostTensor>(element::i32, ov::Shape{3}, end_data)},
        {3, std::make_shared<HostTensor>(element::i32, ov::Shape{3}, stride_data)},
    };

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 2, 3}, {3}, {3}, {3}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{1, 2, 2}};
    // implementation depends on some output information of the op
    op->set_output_type(0, element::i32, {1, 2, 2});
    unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes, const_data);
}

TEST_F(StridedSliceCustomShapeInferenceTest, ignore_begin_end) {
    const auto begin_mask = std::vector<int64_t>{0, 1, 1};
    const auto end_mask = std::vector<int64_t>(3, 1);

    // only support i32
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());
    const auto begin = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{3});
    const auto end = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{3});
    const auto stride = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{3});

    const auto op = make_op(data, begin, end, stride, begin_mask, end_mask);

    int32_t begin_data[] = {1, 0, 0};
    int32_t end_data[] = {0, 0, 0};
    int32_t stride_data[] = {1, 1, 1};

    const std::map<size_t, HostTensorPtr> const_data{
        {1, std::make_shared<HostTensor>(element::i32, ov::Shape{3}, begin_data)},
        {2, std::make_shared<HostTensor>(element::i32, ov::Shape{3}, end_data)},
        {3, std::make_shared<HostTensor>(element::i32, ov::Shape{3}, stride_data)},
    };

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 2, 3}, {3}, {3}, {3}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{2, 2, 3}};
    // implementation depends on some output information of the op
    op->set_output_type(0, element::i32, {2, 2, 3});
    unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes, const_data);
}

TEST_F(StridedSliceCustomShapeInferenceTest, ignore_begin_end_stride_by_two_last_dim) {
    const auto begin_mask = std::vector<int64_t>{1, 0, 1};
    const auto end_mask = std::vector<int64_t>{0, 1, 1};

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());
    // only supprot i32
    const auto begin = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{3});
    const auto end = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{3});
    const auto stride = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{3});

    auto op = make_op(data, begin, end, stride, begin_mask, end_mask);

    int32_t begin_data[] = {0, 1, 0};
    int32_t end_data[] = {2, 0, 0};
    int32_t stride_data[] = {1, 1, 2};

    const std::map<size_t, HostTensorPtr> const_data{
        {1, std::make_shared<HostTensor>(element::i32, ov::Shape{3}, begin_data)},
            {2, std::make_shared<HostTensor>(element::i32, ov::Shape{3}, end_data)},
            {3, std::make_shared<HostTensor>(element::i32, ov::Shape{3}, stride_data)},
    };

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 2, 3}, {3}, {3}, {3}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{2, 1, 2}};
    // implementation depends on some output information of the op
    op->set_output_type(0, element::i32, {2, 1, 2});
    unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes, const_data);
}

TEST_F(StridedSliceCustomShapeInferenceTest, use_reverse_stride_on_last_dimension) {
    const auto mask = std::vector<int64_t>{0, 1, 1};

    // only supprot i32
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());
    const auto begin = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{3});
    const auto end = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{3});
    const auto stride = std::make_shared<op::v0::Parameter>(element::i32, ov::Shape{3});

    const auto op = make_op(data, begin, end, stride, mask, mask);

    int32_t begin_data[] = {0, 0, 0};
    int32_t end_data[] = {1, 0, 0};
    int32_t stride_data[] = {1, 1, -1};

    const std::map<size_t, HostTensorPtr> const_data{
        {1, std::make_shared<HostTensor>(element::i32, ov::Shape{3}, begin_data)},
            {2, std::make_shared<HostTensor>(element::i32, ov::Shape{3}, end_data)},
            {3, std::make_shared<HostTensor>(element::i32, ov::Shape{3}, stride_data)},
    };

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 2, 3}, {3}, {3}, {3}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{1, 2, 3}};
    // implementation depends on some output information of the op
    // op->set_output_type(0, element::i32, {1, 2, 3});
    // TODO ,there is some issue in implementation, this test case can't pass
    // unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes, const_data);
}

TEST_F(StridedSliceCustomShapeInferenceTest, default_stride) {
    const auto mask = std::vector<int64_t>{0, 1, 0};

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());
    // only supprot i32
    const auto begin = op::v0::Constant::create(element::i32, ov::Shape{3}, {0, 0, 0});
    const auto end = op::v0::Constant::create(element::i32, ov::Shape{3}, {1, 0, 2});
    const auto op = make_op(data, begin, end, mask, mask);

    std::vector<StaticShape> static_input_shapes = {{3, 2, 3}, {3}, {3}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{1, 2, 2}};
    // implementation depends on some output information of the op
    // op->set_output_type(0, element::i32, {1, 2, 2});
    // TODO ,there is some issue in implementation, this test case can't pass
    // unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes);
}

