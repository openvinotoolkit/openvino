// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "low_precision/rt_info/precisions_attribute.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/cpu_opset/arm/pass/align_unsupported_lp_conv_fq_precision.hpp"

using namespace ov::intel_cpu;

namespace {

std::shared_ptr<ov::Node> create_convolution(ov::element::Type input_type,
                                             ov::element::Type weights_type,
                                             std::shared_ptr<ov::op::v0::Parameter>& input) {
    static const size_t spatial_dims = 2;
    ov::Strides strides(spatial_dims, 1);
    ov::CoordinateDiff pads(spatial_dims, 0);

    input = std::make_shared<ov::op::v0::Parameter>(input_type, ov::Shape{1, 3, 16, 16});
    auto weights = ov::op::v0::Constant::create(weights_type, ov::Shape{16, 3, 3, 3}, {1});

    return std::make_shared<ov::op::TypeRelaxed<ov::op::v1::Convolution>>(
        ov::element::TypeVector{ov::element::f32, ov::element::f32},
        ov::element::TypeVector{ov::element::f32},
        ov::op::TemporaryReplaceOutputType(input, ov::element::f32).get(),
        ov::op::TemporaryReplaceOutputType(weights, ov::element::f32).get(),
        strides,
        pads,
        pads,
        strides);
}

std::shared_ptr<ov::op::v0::FakeQuantize> create_fake_quantize(const ov::Output<ov::Node>& input,
                                                               ov::element::Type ranges_precision,
                                                               ov::element::Type output_precision) {
    const bool is_signed = ranges_precision == ov::element::i8;
    const float low = is_signed ? -128.0f : 0.0f;
    const float high = is_signed ? 127.0f : 255.0f;
    auto fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(
        ov::test::utils::make_fake_quantize(input, ranges_precision, 256, {}, {low}, {high}, {low}, {high}));
    fq->set_output_type(0, output_precision, fq->get_output_shape(0));
    return fq;
}

std::shared_ptr<ov::Model> create_model(ov::element::Type input_precision,
                                        ov::element::Type fq_output_precision,
                                        std::vector<ov::element::Type> supported_precisions = {}) {
    std::shared_ptr<ov::op::v0::Parameter> input;
    auto conv = create_convolution(input_precision, ov::element::i8, input);
    auto multiply = std::make_shared<ov::op::v1::Multiply>(
        conv,
        ov::op::v0::Constant::create(ov::element::f32, {1}, {0.5f}));
    auto add = std::make_shared<ov::op::v1::Add>(
        multiply,
        ov::op::v0::Constant::create(ov::element::f32, {1, 16, 1, 1}, {1.5f}));
    auto fq = create_fake_quantize(add, input_precision, fq_output_precision);
    if (!supported_precisions.empty()) {
        fq->output(0).get_rt_info()[ov::PrecisionsAttribute::get_type_info_static()] =
            ov::PrecisionsAttribute(supported_precisions);
    }

    return std::make_shared<ov::Model>(ov::OutputVector{fq}, ov::ParameterVector{input});
}

}  // namespace

TEST_F(TransformationTestsF, AlignUnsupportedLPConvFQPrecision_U8ActivationI8Output_Applied) {
    model = create_model(ov::element::u8, ov::element::i8);
    manager.register_pass<AlignUnsupportedLPConvFQPrecision>();
    model_ref = create_model(ov::element::u8, ov::element::u8);
}

TEST_F(TransformationTestsF, AlignUnsupportedLPConvFQPrecision_I8ActivationU8Output_Applied) {
    model = create_model(ov::element::i8, ov::element::u8);
    manager.register_pass<AlignUnsupportedLPConvFQPrecision>();
    model_ref = create_model(ov::element::i8, ov::element::i8);
}

TEST_F(TransformationTestsF, AlignUnsupportedLPConvFQPrecision_OverridesConflictingPrecisionsAttribute) {
    model = create_model(ov::element::u8, ov::element::i8, {ov::element::i8});
    manager.register_pass<AlignUnsupportedLPConvFQPrecision>();
    model_ref = create_model(ov::element::u8, ov::element::u8, {ov::element::u8});
}

TEST_F(TransformationTestsF, AlignUnsupportedLPConvFQPrecision_U8ActivationF32Output_NotApplied) {
    model = create_model(ov::element::u8, ov::element::f32);
    manager.register_pass<AlignUnsupportedLPConvFQPrecision>();
}

TEST_F(TransformationTestsF, AlignUnsupportedLPConvFQPrecision_AlreadyAligned_NotApplied) {
    model = create_model(ov::element::u8, ov::element::u8);
    manager.register_pass<AlignUnsupportedLPConvFQPrecision>();
}

TEST_F(TransformationTestsF, AlignUnsupportedLPConvFQPrecision_F32Activation_NotApplied) {
    model = create_model(ov::element::f32, ov::element::f32);
    manager.register_pass<AlignUnsupportedLPConvFQPrecision>();
}