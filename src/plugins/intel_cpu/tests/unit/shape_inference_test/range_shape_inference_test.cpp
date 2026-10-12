// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "range_shape_inference.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using std::make_shared;
using testing::ElementsAre;
using testing::HasSubstr;

TEST(StaticShapeInferenceTest, Rangev4_i32) {
    auto start = make_shared<op::v0::Parameter>(element::i32, ov::PartialShape{});
    auto stop = make_shared<op::v0::Parameter>(element::i32, ov::PartialShape{});
    auto step = make_shared<op::v0::Parameter>(element::i32, ov::PartialShape{});
    auto range = make_shared<op::v4::Range>(start, stop, step, element::i32);

    int32_t start_v = 2, stop_v = 0, step_v = -2;
    auto const_data = std::unordered_map<size_t, ov::Tensor>{{0, {element::i32, ov::Shape{}, &start_v}},
                                                             {1, {element::i32, ov::Shape{}, &stop_v}},
                                                             {2, {element::i32, ov::Shape{}, &step_v}}};

    auto output_shapes = shape_inference(range.get(), StaticShapeVector{{}, {}, {}}, const_data);
    EXPECT_THAT(output_shapes, ElementsAre(StaticShape{1}));

    step_v = -1;
    output_shapes = shape_inference(range.get(), StaticShapeVector{{}, {}, {}}, const_data);
    EXPECT_THAT(output_shapes, ElementsAre(StaticShape{2}));

    start_v = -19, stop_v = 19, step_v = 1;
    output_shapes = shape_inference(range.get(), StaticShapeVector{{}, {}, {}}, const_data);
    EXPECT_THAT(output_shapes, ElementsAre(StaticShape{38}));

    step_v = 3;
    output_shapes = shape_inference(range.get(), StaticShapeVector{{}, {}, {}}, const_data);
    EXPECT_THAT(output_shapes, ElementsAre(StaticShape{13}));

    start_v = 20, stop_v = -19, step_v = 1;
    output_shapes = shape_inference(range.get(), StaticShapeVector{{}, {}, {}}, const_data);
    EXPECT_THAT(output_shapes, ElementsAre(StaticShape{0}));
}

TEST(StaticShapeInferenceTest, Rangev4_f32) {
    auto start = make_shared<op::v0::Parameter>(element::f32, ov::PartialShape{});
    auto stop = make_shared<op::v0::Parameter>(element::f32, ov::PartialShape{});
    auto step = make_shared<op::v0::Parameter>(element::f32, ov::PartialShape{});
    auto range = make_shared<op::v4::Range>(start, stop, step, element::f32);

    float start_v = 0.f, stop_v = 1.f, step_v = .25f;
    auto const_data = std::unordered_map<size_t, ov::Tensor>{{0, {element::f32, ov::Shape{}, &start_v}},
                                                             {1, {element::f32, ov::Shape{}, &stop_v}},
                                                             {2, {element::f32, ov::Shape{}, &step_v}}};

    auto output_shapes = shape_inference(range.get(), StaticShapeVector{{}, {}, {}}, const_data);
    EXPECT_THAT(output_shapes, ElementsAre(StaticShape{4}));

    start_v = -1.f;
    output_shapes = shape_inference(range.get(), StaticShapeVector{{}, {}, {}}, const_data);
    EXPECT_THAT(output_shapes, ElementsAre(StaticShape{8}));

    stop_v = .875f;
    output_shapes = shape_inference(range.get(), StaticShapeVector{{}, {}, {}}, const_data);
    EXPECT_THAT(output_shapes, ElementsAre(StaticShape{8}));
}

TEST(StaticShapeInferenceTest, Rangev4_i32_zero_step_from_tensor_accessor) {
    auto start = make_shared<op::v0::Parameter>(element::i32, ov::PartialShape{});
    auto stop = make_shared<op::v0::Parameter>(element::i32, ov::PartialShape{});
    auto step = make_shared<op::v0::Parameter>(element::i32, ov::PartialShape{});
    auto range = make_shared<op::v4::Range>(start, stop, step, element::i32);

    int32_t start_v = 0, stop_v = 10, step_v = 0;
    const auto const_data = std::unordered_map<size_t, ov::Tensor>{{0, {element::i32, ov::Shape{}, &start_v}},
                                                                   {1, {element::i32, ov::Shape{}, &stop_v}},
                                                                   {2, {element::i32, ov::Shape{}, &step_v}}};

    OV_EXPECT_THROW(shape_inference(range.get(), StaticShapeVector{{}, {}, {}}, const_data),
                    NodeValidationFailure,
                    HasSubstr("'step' cannot be zero, nan, or infinite."));
}

TEST(StaticShapeInferenceTest, Rangev4_f32_zero_step_from_tensor_accessor) {
    auto start = make_shared<op::v0::Parameter>(element::f32, ov::PartialShape{});
    auto stop = make_shared<op::v0::Parameter>(element::f32, ov::PartialShape{});
    auto step = make_shared<op::v0::Parameter>(element::f32, ov::PartialShape{});
    auto range = make_shared<op::v4::Range>(start, stop, step, element::f32);

    float start_v = 0.f, stop_v = 1.f, step_v = 0.f;
    const auto const_data = std::unordered_map<size_t, ov::Tensor>{{0, {element::f32, ov::Shape{}, &start_v}},
                                                                   {1, {element::f32, ov::Shape{}, &stop_v}},
                                                                   {2, {element::f32, ov::Shape{}, &step_v}}};

    OV_EXPECT_THROW(shape_inference(range.get(), StaticShapeVector{{}, {}, {}}, const_data),
                    NodeValidationFailure,
                    HasSubstr("'step' cannot be zero, nan, or infinite."));
}

TEST(StaticShapeInferenceTest, Rangev4_f32_inf_nan_step_from_tensor_accessor) {
    auto start = make_shared<op::v0::Parameter>(element::f32, ov::PartialShape{});
    auto stop = make_shared<op::v0::Parameter>(element::f32, ov::PartialShape{});
    auto step = make_shared<op::v0::Parameter>(element::f32, ov::PartialShape{});
    auto range = make_shared<op::v4::Range>(start, stop, step, element::f32);

    float start_v = 0.f, stop_v = 1.f, step_v = std::numeric_limits<float>::infinity();
    const auto const_data = std::unordered_map<size_t, ov::Tensor>{{0, {element::f32, ov::Shape{}, &start_v}},
                                                                   {1, {element::f32, ov::Shape{}, &stop_v}},
                                                                   {2, {element::f32, ov::Shape{}, &step_v}}};

    OV_EXPECT_THROW(shape_inference(range.get(), StaticShapeVector{{}, {}, {}}, const_data),
                    NodeValidationFailure,
                    HasSubstr("'step' cannot be zero, nan, or infinite."));

    step_v = -std::numeric_limits<float>::infinity();
    OV_EXPECT_THROW(shape_inference(range.get(), StaticShapeVector{{}, {}, {}}, const_data),
                    NodeValidationFailure,
                    HasSubstr("'step' cannot be zero, nan, or infinite."));

    step_v = std::nanf("");
    OV_EXPECT_THROW(shape_inference(range.get(), StaticShapeVector{{}, {}, {}}, const_data),
                    NodeValidationFailure,
                    HasSubstr("'step' cannot be zero, nan, or infinite."));
}
