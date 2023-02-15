// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pad_shape_inference.hpp>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace ov::opset10;
using namespace testing;

class PadV1StaticShapeInference : public OpStaticShapeInferenceTest<op::v1::Pad> {
protected:
    void SetUp() override {
        output_shapes.resize(1);
    }
};

TEST_F(PadV1StaticShapeInference, default_ctor) {
    const auto op = make_op();
    op->set_pad_mode(op::PadMode::EDGE);

    int64_t pads_begin[] = {3, 2, 1, 1};
    int32_t pads_end[] = {0, 1, 2, 3};

    const auto const_data =
        std::map<size_t, HostTensorPtr>{{1, std::make_shared<HostTensor>(element::i64, Shape{4}, pads_begin)},
                                        {2, std::make_shared<HostTensor>(element::i32, Shape{4}, pads_end)}};

    input_shapes = ShapeVector{{3, 6, 5, 5}, {4}, {4}};
    shape_inference(op.get(), input_shapes, output_shapes, const_data);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({6, 9, 8, 9}));
}

TEST_F(PadV1StaticShapeInference, pads_begin_end_value_as_constants) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    const auto pads_begin = Constant::create(element::i64, Shape{4}, {3, 2, 1, 0});
    const auto pads_end = Constant::create(element::i64, Shape{4}, {0, 1, 2, 3});
    const auto pad_val = Constant::create(element::f32, Shape{}, {2112});

    const auto op = make_op(data, pads_begin, pads_end, pad_val, op::PadMode::CONSTANT);

    input_shapes = ShapeVector{{3, 6, 5, 5}, {4}, {4}, {}};
    shape_inference(op.get(), input_shapes, output_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({6, 9, 8, 8}));
}

TEST_F(PadV1StaticShapeInference, pads_begin_end_in_constant_map) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto pads_begin = std::make_shared<Parameter>(element::u64, PartialShape::dynamic());
    const auto pads_end = std::make_shared<Parameter>(element::u64, PartialShape::dynamic());

    uint64_t pads_begin_data[] = {0, 2, 2, 0};
    uint32_t pads_end_data[] = {0, 1, 2, 0};

    const auto const_data =
        std::map<size_t, HostTensorPtr>{{1, std::make_shared<HostTensor>(element::u64, Shape{4}, pads_begin_data)},
                                        {2, std::make_shared<HostTensor>(element::u32, Shape{4}, pads_end_data)}};

    const auto op = make_op(data, pads_begin, pads_end, op::PadMode::REFLECT);

    input_shapes = ShapeVector{{3, 6, 5, 1}, {4}, {4}};
    shape_inference(op.get(), input_shapes, output_shapes, const_data);

    EXPECT_EQ(output_shapes.front(), StaticShape({3, 9, 9, 1}));
}

TEST_F(PadV1StaticShapeInference, pads_begin_got_negative_value) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto pads_begin = std::make_shared<Parameter>(element::i8, PartialShape::dynamic());
    const auto pads_end = Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});

    int8_t pads_begin_data[] = {0, -2, -2, 0};

    const auto const_data =
        std::map<size_t, HostTensorPtr>{{1, std::make_shared<HostTensor>(element::i8, Shape{4}, pads_begin_data)}};

    const auto op = make_op(data, pads_begin, pads_end, op::PadMode::REFLECT);
    input_shapes = ShapeVector{{3, SIZE_MAX, 5, 2}, {4}, {4}};

    shape_inference(op.get(), input_shapes, output_shapes, const_data);

    EXPECT_EQ(output_shapes.front(), StaticShape({3, SIZE_MAX, 3, 2}));
}

TEST_F(PadV1StaticShapeInference, pads_end_got_negative_value) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto pads_begin = Constant::create(element::i64, Shape{4}, {1, 1, 2, 1});
    const auto pads_end = std::make_shared<Parameter>(element::i8, PartialShape::dynamic());
    const auto op = make_op(data, pads_begin, pads_end, op::PadMode::REFLECT);

    int8_t pads_end_data[] = {0, -3, -2, 0};

    const auto const_data =
        std::map<size_t, HostTensorPtr>{{2, std::make_shared<HostTensor>(element::i8, Shape{4}, pads_end_data)}};

    input_shapes = ShapeVector{{3, 6, 5, SIZE_MAX}, {4}, {4}};

    shape_inference(op.get(), input_shapes, output_shapes, const_data);

    EXPECT_EQ(output_shapes.front(), StaticShape({4, 4, 5, SIZE_MAX}));
}

TEST_F(PadV1StaticShapeInference, pads_begin_is_empty) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto pads_begin = std::make_shared<Parameter>(element::u64, PartialShape::dynamic());
    const auto pads_end = Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
    const auto op = make_op(data, pads_begin, pads_end, op::PadMode::REFLECT);

    const auto const_data = std::map<size_t, HostTensorPtr>{{1, std::make_shared<HostTensor>(element::u64, Shape{0})}};

    input_shapes = ShapeVector{{3, 6, 5, 2}, {0}, {4}};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes, output_shapes, const_data),
                    NodeValidationFailure,
                    HasSubstr("length of pads_begin mismatches with rank of input"));
}

TEST_F(PadV1StaticShapeInference, pads_end_is_empty) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto pads_begin = Constant::create(element::i64, Shape{4}, {1, 1, 1, 1});
    const auto pads_end = std::make_shared<Parameter>(element::i8, PartialShape::dynamic());
    const auto op = make_op(data, pads_begin, pads_end, op::PadMode::REFLECT);

    const auto const_data = std::map<size_t, HostTensorPtr>{{2, std::make_shared<HostTensor>(element::i8, Shape{0})}};

    input_shapes = ShapeVector{{3, 6, 5, 2}, {4}, {0}};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes, output_shapes, const_data),
                    NodeValidationFailure,
                    HasSubstr("length of pads_end mismatches with rank of input"));
}
