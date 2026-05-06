// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "low_precision/fake_quantize_decomposition.hpp"
#include "low_precision/rt_info/precisions_attribute.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/cpu_opset/arm/pass/align_unsupported_lp_conv_fq_precision.hpp"

using namespace ov::intel_cpu;

namespace {

// Create FQ with ranges that naturally resolve to the desired precision:
// u8: output range [0, 255] → unsigned
// i8: output range [-128, 127] → signed
std::shared_ptr<ov::op::v0::FakeQuantize> create_fq_with_natural_precision(
        const ov::Output<ov::Node>& input,
        ov::element::Type natural_precision,
        const std::vector<ov::element::Type>& precisions_attr) {
    float il, ih, ol, oh;
    if (natural_precision == ov::element::u8) {
        il = 0.0f; ih = 255.0f; ol = 0.0f; oh = 255.0f;
    } else {
        il = -128.0f; ih = 127.0f; ol = -128.0f; oh = 127.0f;
    }
    auto fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(
        ov::test::utils::make_fake_quantize(input, ov::element::f32, 256, {}, {il}, {ih}, {ol}, {oh}));
    fq->output(0).get_rt_info()[ov::PrecisionsAttribute::get_type_info_static()] =
        ov::PrecisionsAttribute(precisions_attr);
    return fq;
}

std::shared_ptr<ov::Model> create_model(ov::element::Type conv_input_precision,
                                        ov::element::Type output_fq_natural_precision,
                                        const std::vector<ov::element::Type>& output_fq_precisions_attr) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 16, 16});
    auto input_fq = create_fq_with_natural_precision(input, conv_input_precision, {conv_input_precision});

    auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{16, 3, 3, 3}, {1});
    ov::Strides strides(2, 1);
    ov::CoordinateDiff pads(2, 0);
    auto conv = std::make_shared<ov::op::v1::Convolution>(input_fq, weights, strides, pads, pads, strides);

    // Simulate post-MarkupOptimizations state: conv input(0) has a resolved single precision
    // attribute propagated from the input FQ. The pass reads this via getAttribute<PrecisionsAttribute>.
    conv->input(0).get_rt_info()[ov::PrecisionsAttribute::get_type_info_static()] =
        ov::PrecisionsAttribute({conv_input_precision});

    auto bias = ov::op::v0::Constant::create(ov::element::f32, {1, 16, 1, 1}, {1.5f});
    auto add = std::make_shared<ov::op::v1::Add>(conv, bias);

    auto output_fq = create_fq_with_natural_precision(add, output_fq_natural_precision, output_fq_precisions_attr);

    return std::make_shared<ov::Model>(ov::OutputVector{output_fq}, ov::ParameterVector{input});
}

}  // namespace

// Conv input u8, output FQ naturally i8 → output FQ should be forced to {u8}
TEST_F(TransformationTestsF, ConvAndFQ_U8ActivationI8Output_Applied) {
    const std::vector<ov::element::Type> default_precisions = {ov::element::u8, ov::element::i8};
    model = create_model(ov::element::u8, ov::element::i8, default_precisions);
    manager.register_pass<AlignUnsupportedLPConvFQPrecision>();
    model_ref = create_model(ov::element::u8, ov::element::i8, {ov::element::u8});
}

// Conv input i8, output FQ naturally u8 → output FQ should be forced to {i8}
TEST_F(TransformationTestsF, ConvAndFQ_I8ActivationU8Output_Applied) {
    const std::vector<ov::element::Type> default_precisions = {ov::element::u8, ov::element::i8};
    model = create_model(ov::element::i8, ov::element::u8, default_precisions);
    manager.register_pass<AlignUnsupportedLPConvFQPrecision>();
    model_ref = create_model(ov::element::i8, ov::element::u8, {ov::element::i8});
}

// Both u8 → output FQ not changed by the pass
TEST_F(TransformationTestsF, ConvAndFQ_BothU8_NotApplied) {
    const std::vector<ov::element::Type> default_precisions = {ov::element::u8, ov::element::i8};
    model = create_model(ov::element::u8, ov::element::u8, default_precisions);
    manager.register_pass<AlignUnsupportedLPConvFQPrecision>();
}

// Both i8 → output FQ not changed by the pass
TEST_F(TransformationTestsF, ConvAndFQ_BothI8_NotApplied) {
    const std::vector<ov::element::Type> default_precisions = {ov::element::u8, ov::element::i8};
    model = create_model(ov::element::i8, ov::element::i8, default_precisions);
    manager.register_pass<AlignUnsupportedLPConvFQPrecision>();
}

// Conv input u8, output FQ has only i16 precision → conv_precision not in FQ set → not applied
TEST_F(TransformationTestsF, ConvAndFQ_PrecisionNotInFQSet_NotApplied) {
    const std::vector<ov::element::Type> fq_precisions = {ov::element::i16};
    model = create_model(ov::element::u8, ov::element::i8, fq_precisions);
    manager.register_pass<AlignUnsupportedLPConvFQPrecision>();
}