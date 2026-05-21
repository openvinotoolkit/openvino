// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/cpu_opset/arm/pass/fallback_unsupported_lp_conv_to_fp16.hpp"

using namespace ov::intel_cpu;

namespace {

std::shared_ptr<ov::op::TypeRelaxed<ov::op::v1::Convolution>> create_convolution(const ov::Output<ov::Node>& input,
                                                                                  const ov::Output<ov::Node>& weights,
                                                                                  const std::string& friendly_name) {
    constexpr size_t spatial_dims = 2;
    ov::Strides strides(spatial_dims, 1);
    ov::CoordinateDiff pads(spatial_dims, 0);

    auto convolution = std::make_shared<ov::op::TypeRelaxed<ov::op::v1::Convolution>>(
        ov::element::TypeVector{ov::element::f32, ov::element::f32},
        ov::element::TypeVector{ov::element::f32},
        ov::op::TemporaryReplaceOutputType(input, ov::element::f32).get(),
        ov::op::TemporaryReplaceOutputType(weights, ov::element::f32).get(),
        strides,
        pads,
        pads,
        strides);
    convolution->set_friendly_name(friendly_name);
    return convolution;
}

std::shared_ptr<ov::op::v0::FakeQuantize> create_fake_quantize(const ov::Output<ov::Node>& input,
                                                               ov::element::Type output_type) {
    const bool is_signed = output_type == ov::element::i8;
    const float low = is_signed ? -128.0f : 0.0f;
    const float high = is_signed ? 127.0f : 255.0f;

    auto fake_quantize = ov::as_type_ptr<ov::op::v0::FakeQuantize>(
        ov::test::utils::make_fake_quantize(input, output_type, 256, {}, {low}, {high}, {low}, {high}));
    fake_quantize->set_output_type(0, output_type, fake_quantize->get_output_shape(0));
    fake_quantize->set_friendly_name("FakeQuantize");
    return fake_quantize;
}

template <typename NodeType>
std::shared_ptr<NodeType> find_node(const std::shared_ptr<ov::Model>& model, const std::string& friendly_name) {
    for (const auto& node : model->get_ordered_ops()) {
        if (node->get_friendly_name() == friendly_name) {
            return ov::as_type_ptr<NodeType>(node);
        }
    }

    return nullptr;
}

std::shared_ptr<ov::Model> create_init_graph() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::Shape{1, 3, 16, 16});
    auto input_convert = std::make_shared<ov::op::v0::Convert>(input, ov::element::f32);
    auto zero_point = ov::op::v0::Constant::create(ov::element::f32, {1, 1, 1, 1}, {128.0f});
    auto subtract = std::make_shared<ov::op::v1::Subtract>(input_convert, zero_point);
    subtract->set_friendly_name("Subtract");

    auto weights = ov::op::v0::Constant::create(ov::element::i8, ov::Shape{16, 3, 3, 3}, {1});
    auto convolution = create_convolution(subtract, weights, "Convolution");

    auto scales = ov::op::v0::Constant::create(ov::element::f32, {1}, {0.5f});
    auto multiply = std::make_shared<ov::op::v1::Multiply>(convolution, scales);
    multiply->set_friendly_name("Multiply");

    auto bias = ov::op::v0::Constant::create(ov::element::f32, {1, 16, 1, 1}, {1.5f});
    auto add = std::make_shared<ov::op::v1::Add>(multiply, bias);
    add->set_friendly_name("Add");

    auto clamp = std::make_shared<ov::op::v0::Clamp>(add, 0.0f, 6.0f);
    clamp->set_friendly_name("Clamp");

    return std::make_shared<ov::Model>(ov::OutputVector{create_fake_quantize(clamp, ov::element::u8)},
                                       ov::ParameterVector{input});
}

}  // namespace

TEST(FallbackUnsupportedLPConvToFP16Test, ClampBeforeFQHandledBySuffixMatcher) {
    auto model = create_init_graph();

    ov::pass::Manager manager;
    manager.register_pass<FallbackUnsupportedLPConvToFP16>();
    manager.run_passes(model);

    EXPECT_EQ(find_node<ov::op::v1::Multiply>(model, "Multiply"), nullptr);

    const auto convolution = find_node<ov::op::v1::Convolution>(model, "Convolution");
    ASSERT_NE(convolution, nullptr);

    const auto activation_to_fp16 = ov::as_type_ptr<ov::op::v0::Convert>(convolution->get_input_node_shared_ptr(0));
    ASSERT_NE(activation_to_fp16, nullptr);
    EXPECT_EQ(activation_to_fp16->get_friendly_name(), "Convolution/ActivationToFP16");
    EXPECT_EQ(activation_to_fp16->get_destination_type(), ov::element::f16);

    const auto scaled_weights = ov::as_type_ptr<ov::op::v1::Multiply>(convolution->get_input_node_shared_ptr(1));
    ASSERT_NE(scaled_weights, nullptr);
    EXPECT_EQ(scaled_weights->get_friendly_name(), "Multiply/WeightsScaled");

    const auto weights_to_fp16 = ov::as_type_ptr<ov::op::v0::Convert>(scaled_weights->get_input_node_shared_ptr(0));
    ASSERT_NE(weights_to_fp16, nullptr);
    EXPECT_EQ(weights_to_fp16->get_friendly_name(), "Convolution/WeightsToFP16");
    EXPECT_EQ(weights_to_fp16->get_destination_type(), ov::element::f16);

    const auto scales_reshape = ov::as_type_ptr<ov::op::v1::Reshape>(scaled_weights->get_input_node_shared_ptr(1));
    ASSERT_NE(scales_reshape, nullptr);
    EXPECT_EQ(scales_reshape->get_friendly_name(), "Multiply/ScalesToWeightsShape");

    const auto scales_to_fp16 = ov::as_type_ptr<ov::op::v0::Convert>(scales_reshape->get_input_node_shared_ptr(0));
    ASSERT_NE(scales_to_fp16, nullptr);
    EXPECT_EQ(scales_to_fp16->get_friendly_name(), "Multiply/ScalesToFP16");
    EXPECT_EQ(scales_to_fp16->get_destination_type(), ov::element::f16);

    const auto fake_quantize = find_node<ov::op::v0::FakeQuantize>(model, "FakeQuantize");
    ASSERT_NE(fake_quantize, nullptr);

    const auto clamp = ov::as_type_ptr<ov::op::v0::Clamp>(fake_quantize->get_input_node_shared_ptr(0));
    ASSERT_NE(clamp, nullptr);
    EXPECT_EQ(clamp->get_friendly_name(), "Clamp");

    const auto add = ov::as_type_ptr<ov::op::v1::Add>(clamp->get_input_node_shared_ptr(0));
    ASSERT_NE(add, nullptr);
    EXPECT_EQ(add->get_friendly_name(), "Add");
    EXPECT_EQ(add->get_input_node_shared_ptr(0), convolution);
}