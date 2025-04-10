// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/transpose.hpp"
#include "transformations/cpu_opset/common/pass/permute_slice_n_interpolation.hpp"

using namespace testing;

class PermuteSliceInterpolateTest: public TransformationTestsF {};

TEST_F(PermuteSliceInterpolateTest, 3D) {
    const ov::Shape in_shape{1, 200, 4};
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, in_shape);

        auto slice_start = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, { 0L});
        auto slice_stop  = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1L});
        auto slice_step  = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, { 1L});
        auto slice_axes  = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, { 2L});
        auto slice = std::make_shared<ov::op::v8::Slice>(input,
                                                         slice_start,
                                                         slice_stop,
                                                         slice_step,
                                                         slice_axes);

        auto transpose_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(slice, transpose_order);

        auto interpolate_sizes  = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {100L});
        auto interpolate_scales = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.f});
        auto interpolate_axes   = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2L});
        ov::op::util::InterpolateBase::InterpolateAttrs intp_attr;
        intp_attr.mode = ov::op::util::InterpolateBase::InterpolateMode::LINEAR;
        auto interpolate = std::make_shared<ov::op::v4::Interpolate>(transpose,
                                                                     interpolate_sizes,
                                                                     interpolate_scales,
                                                                     interpolate_axes,
                                                                     intp_attr);

        model = std::make_shared<ov::Model>(ov::NodeVector{interpolate}, ov::ParameterVector{input});
        manager.register_pass<ov::intel_cpu::PermuteSliceAndInterpolation>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, in_shape);

        auto transpose_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(input, transpose_order);

        auto interpolate_sizes  = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {100L});
        auto interpolate_scales = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.f});
        auto interpolate_axes   = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2L});
        ov::op::util::InterpolateBase::InterpolateAttrs intp_attr;
        intp_attr.mode = {ov::op::util::InterpolateBase::InterpolateMode::LINEAR};
        auto interpolate = std::make_shared<ov::op::v4::Interpolate>(transpose,
                                                                     interpolate_sizes,
                                                                     interpolate_scales,
                                                                     interpolate_axes,
                                                                     intp_attr);

        auto slice_start = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, { 0L});
        auto slice_stop  = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1L});
        auto slice_step  = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, { 1L});
        auto slice_axes  = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, { 1L});
        auto slice = std::make_shared<ov::op::v8::Slice>(interpolate,
                                                         slice_start,
                                                         slice_stop,
                                                         slice_step,
                                                         slice_axes);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{slice}, ov::ParameterVector{input});
    }
}

TEST_F(PermuteSliceInterpolateTest, 4D) {
    const ov::Shape in_shape{1, 200, 200, 4};
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, in_shape);

        auto slice_start = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, { 0L});
        auto slice_stop  = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1L});
        auto slice_step  = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, { 1L});
        auto slice_axes  = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, { 3L});
        auto slice = std::make_shared<ov::op::v8::Slice>(input,
                                                         slice_start,
                                                         slice_stop,
                                                         slice_step,
                                                         slice_axes);

        auto transpose_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(slice, transpose_order);

        auto interpolate_sizes  = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {100L, 100L});
        auto interpolate_scales = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2}, {1.f, 1.f});
        auto interpolate_axes   = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {2L, 3L});
        ov::op::util::InterpolateBase::InterpolateAttrs intp_attr;
        intp_attr.mode = ov::op::util::InterpolateBase::InterpolateMode::LINEAR;
        auto interpolate = std::make_shared<ov::op::v4::Interpolate>(transpose,
                                                                     interpolate_sizes,
                                                                     interpolate_scales,
                                                                     interpolate_axes,
                                                                     intp_attr);

        model = std::make_shared<ov::Model>(ov::NodeVector{interpolate}, ov::ParameterVector{input});
        manager.register_pass<ov::intel_cpu::PermuteSliceAndInterpolation>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, in_shape);

        auto transpose_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(input, transpose_order);

        auto interpolate_sizes  = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {100L, 100L});
        auto interpolate_scales = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2}, {1.f, 1.f});
        auto interpolate_axes   = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {2L, 3L});
        ov::op::util::InterpolateBase::InterpolateAttrs intp_attr;
        intp_attr.mode = {ov::op::util::InterpolateBase::InterpolateMode::LINEAR};
        auto interpolate = std::make_shared<ov::op::v4::Interpolate>(transpose,
                                                                     interpolate_sizes,
                                                                     interpolate_scales,
                                                                     interpolate_axes,
                                                                     intp_attr);

        auto slice_start = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, { 0L});
        auto slice_stop  = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1L});
        auto slice_step  = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, { 1L});
        auto slice_axes  = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, { 1L});
        auto slice = std::make_shared<ov::op::v8::Slice>(interpolate,
                                                         slice_start,
                                                         slice_stop,
                                                         slice_step,
                                                         slice_axes);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{slice}, ov::ParameterVector{input});
    }
}

TEST_F(PermuteSliceInterpolateTest, 5D_Add) {
    const ov::Shape in_shape{1, 200, 200, 200, 4};
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, in_shape);

        auto slice_start = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, { 0L});
        auto slice_stop  = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1L});
        auto slice_step  = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, { 1L});
        auto slice_axes  = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, { 4L});
        auto slice = std::make_shared<ov::op::v8::Slice>(input,
                                                         slice_start,
                                                         slice_stop,
                                                         slice_step,
                                                         slice_axes);

        auto transpose_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 4, 1, 2, 3});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(slice, transpose_order);

        auto interpolate_sizes  = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {100L, 100L, 100L});
        auto interpolate_scales = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{3}, {1.f, 1.f, 1.f});
        auto interpolate_axes   = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {2L, 3L, 4L});
        ov::op::util::InterpolateBase::InterpolateAttrs intp_attr;
        intp_attr.mode = ov::op::util::InterpolateBase::InterpolateMode::LINEAR;
        auto interpolate = std::make_shared<ov::op::v4::Interpolate>(transpose,
                                                                     interpolate_sizes,
                                                                     interpolate_scales,
                                                                     interpolate_axes,
                                                                     intp_attr);

        auto add_val = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{1}, {1});
        auto add = std::make_shared<ov::op::v1::Add>(interpolate, add_val);

        model = std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{input});
        manager.register_pass<ov::intel_cpu::PermuteSliceAndInterpolation>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, in_shape);

        auto transpose_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 4, 1, 2, 3});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(input, transpose_order);

        auto interpolate_sizes  = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {100L, 100L, 100L});
        auto interpolate_scales = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{3}, {1.f, 1.f, 1.f});
        auto interpolate_axes   = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {2L, 3L, 4L});
        ov::op::util::InterpolateBase::InterpolateAttrs intp_attr;
        intp_attr.mode = {ov::op::util::InterpolateBase::InterpolateMode::LINEAR};
        auto interpolate = std::make_shared<ov::op::v4::Interpolate>(transpose,
                                                                     interpolate_sizes,
                                                                     interpolate_scales,
                                                                     interpolate_axes,
                                                                     intp_attr);

        auto slice_start = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, { 0L});
        auto slice_stop  = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1L});
        auto slice_step  = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, { 1L});
        auto slice_axes  = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, { 1L});
        auto slice = std::make_shared<ov::op::v8::Slice>(interpolate,
                                                         slice_start,
                                                         slice_stop,
                                                         slice_step,
                                                         slice_axes);

        auto add_val = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{1}, {1});
        auto add = std::make_shared<ov::op::v1::Add>(slice, add_val);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{input});
    }
}
