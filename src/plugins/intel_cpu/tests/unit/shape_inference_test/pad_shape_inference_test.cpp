// Copyright (C) 2018-2025 Intel Corporation
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

template <class TOp>
class PadStaticShapeInference : public OpStaticShapeInferenceTest<TOp> {
protected:
    void SetUp() override {
        this->output_shapes.resize(1);
    }
};

TYPED_TEST_SUITE_P(PadStaticShapeInference);

TYPED_TEST_P(PadStaticShapeInference, default_ctor) {
    const auto op = this->make_op();
    op->set_pad_mode(op::PadMode::EDGE);

    int64_t pads_begin[] = {3, 2, 1, 1};
    int32_t pads_end[] = {0, 1, 2, 3};

    const auto const_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i64, ov::Shape{4}, pads_begin}},
                                                                   {2, {element::i32, ov::Shape{4}, pads_end}}};

    this->input_shapes = StaticShapeVector{{3, 6, 5, 5}, {4}, {4}};
    this->output_shapes = shape_inference(op.get(), this->input_shapes, const_data);

    EXPECT_EQ(this->output_shapes.size(), 1);
    EXPECT_EQ(this->output_shapes.front(), StaticShape({6, 9, 8, 9}));
}

TYPED_TEST_P(PadStaticShapeInference, pads_begin_end_value_as_constants) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    const auto pads_begin = Constant::create(element::i64, ov::Shape{4}, {3, 2, 1, 0});
    const auto pads_end = Constant::create(element::i64, ov::Shape{4}, {0, 1, 2, 3});
    const auto pad_val = Constant::create(element::f32, ov::Shape{}, {2112});

    const auto op = this->make_op(data, pads_begin, pads_end, pad_val, op::PadMode::CONSTANT);

    this->input_shapes = StaticShapeVector{{3, 6, 5, 5}, {4}, {4}, {}};
    this->output_shapes = shape_inference(op.get(), this->input_shapes);

    EXPECT_EQ(this->output_shapes.size(), 1);
    EXPECT_EQ(this->output_shapes.front(), StaticShape({6, 9, 8, 8}));
}

TYPED_TEST_P(PadStaticShapeInference, pads_begin_end_in_constant_map) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto pads_begin = std::make_shared<Parameter>(element::u64, PartialShape::dynamic());
    const auto pads_end = std::make_shared<Parameter>(element::u64, PartialShape::dynamic());

    uint64_t pads_begin_data[] = {0, 2, 2, 0};
    uint32_t pads_end_data[] = {0, 1, 2, 0};

    const auto const_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::u64, ov::Shape{4}, pads_begin_data}},
                                                                   {2, {element::u32, ov::Shape{4}, pads_end_data}}};

    const auto op = this->make_op(data, pads_begin, pads_end, op::PadMode::REFLECT);

    this->input_shapes = StaticShapeVector{{3, 6, 5, 1}, {4}, {4}};
    this->output_shapes = shape_inference(op.get(), this->input_shapes, const_data);

    EXPECT_EQ(this->output_shapes.front(), StaticShape({3, 9, 9, 1}));
}

TYPED_TEST_P(PadStaticShapeInference, pads_begin_got_negative_value) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto pads_begin = std::make_shared<Parameter>(element::i8, PartialShape::dynamic());
    const auto pads_end = Constant::create(element::i64, ov::Shape{4}, {0, 0, 0, 0});

    int8_t pads_begin_data[] = {0, -2, -2, 0};

    const auto const_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i8, ov::Shape{4}, pads_begin_data}}};

    const auto op = this->make_op(data, pads_begin, pads_end, op::PadMode::REFLECT);
    this->input_shapes = StaticShapeVector{{3, SIZE_MAX, 5, 2}, {4}, {4}};

    this->output_shapes = shape_inference(op.get(), this->input_shapes, const_data);

    EXPECT_EQ(this->output_shapes.front(), StaticShape({3, SIZE_MAX, 3, 2}));
}

TYPED_TEST_P(PadStaticShapeInference, pads_end_got_negative_value) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto pads_begin = Constant::create(element::i64, ov::Shape{4}, {1, 1, 2, 1});
    const auto pads_end = std::make_shared<Parameter>(element::i8, PartialShape::dynamic());
    const auto op = this->make_op(data, pads_begin, pads_end, op::PadMode::REFLECT);

    int8_t pads_end_data[] = {0, -3, -2, 0};

    const auto const_data = std::unordered_map<size_t, ov::Tensor>{{2, {element::i8, ov::Shape{4}, pads_end_data}}};

    this->input_shapes = StaticShapeVector{{3, 6, 5, SIZE_MAX}, {4}, {4}};

    this->output_shapes = shape_inference(op.get(), this->input_shapes, const_data);

    EXPECT_EQ(this->output_shapes.front(), StaticShape({4, 4, 5, SIZE_MAX}));
}

TYPED_TEST_P(PadStaticShapeInference, pads_begin_is_empty) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto pads_begin = std::make_shared<Parameter>(element::u64, PartialShape::dynamic());
    const auto pads_end = Constant::create(element::i64, ov::Shape{4}, {0, 0, 0, 0});
    const auto op = this->make_op(data, pads_begin, pads_end, op::PadMode::REFLECT);

    const auto const_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::u64, ov::Shape{0}}}};

    this->input_shapes = StaticShapeVector{{3, 6, 5, 2}, {0}, {4}};

    OV_EXPECT_THROW(shape_inference(op.get(), this->input_shapes, const_data),
                    NodeValidationFailure,
                    HasSubstr("length of pads_begin mismatches with rank of input"));
}

TYPED_TEST_P(PadStaticShapeInference, pads_end_is_empty) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto pads_begin = Constant::create(element::i64, ov::Shape{4}, {1, 1, 1, 1});
    const auto pads_end = std::make_shared<Parameter>(element::i8, PartialShape::dynamic());
    const auto op = this->make_op(data, pads_begin, pads_end, op::PadMode::REFLECT);

    const auto const_data = std::unordered_map<size_t, ov::Tensor>{{2, {element::i8, ov::Shape{0}}}};

    this->input_shapes = StaticShapeVector{{3, 6, 5, 2}, {4}, {0}};

    OV_EXPECT_THROW(shape_inference(op.get(), this->input_shapes, const_data),
                    NodeValidationFailure,
                    HasSubstr("length of pads_end mismatches with rank of input"));
}

REGISTER_TYPED_TEST_SUITE_P(PadStaticShapeInference,
                            default_ctor,
                            pads_begin_end_value_as_constants,
                            pads_begin_end_in_constant_map,
                            pads_begin_got_negative_value,
                            pads_end_got_negative_value,
                            pads_begin_is_empty,
                            pads_end_is_empty);

using PadTypes = Types<op::v1::Pad, op::v12::Pad>;
INSTANTIATE_TYPED_TEST_SUITE_P(shape_inference, PadStaticShapeInference, PadTypes);
