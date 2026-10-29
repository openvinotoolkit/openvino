// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"

#include "openvino/core/model.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/add.hpp"
#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/placeholder.hpp"

#include "plugin/transformations/convert_fc_to_compressed.hpp"
#include "plugin/transformations/repack_gguf_weights.hpp"
#include "ov_ops/fully_connected_compressed.hpp"

#include <array>
#include <cstring>
#include <memory>
#include <vector>

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

namespace {
std::shared_ptr<ov::intel_gpu::op::FullyConnectedCompressed> make_gguf_repack_fc(
    const std::shared_ptr<ov::op::v0::Constant>& weight) {
    auto activation = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                            ov::PartialShape{-1, weight->get_shape()[1]});
    auto absent = std::make_shared<ov::intel_gpu::op::Placeholder>();
    auto scale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {1.0f});
    return std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(activation,
                                                                       weight,
                                                                       absent,
                                                                       scale,
                                                                       absent,
                                                                       ov::element::f32);
}
}

TEST(GGUFRepack, PreservesSharedRawAndIsIdempotent) {
    const std::array<ov::element::Type, 6> formats = {ov::element::gguf_q4_k,
                                                   ov::element::gguf_q5_k,
                                                   ov::element::gguf_q6_k,
                                                   ov::element::gguf_q4_0,
                                                   ov::element::gguf_q4_1,
                                                   ov::element::gguf_q8_0};
    for (const auto& type : formats) {
        SCOPED_TRACE(type.get_type_name());
        const ov::Shape shape{32, 512};
        std::vector<uint8_t> payload(32 * (512 / type.block_elem_count()) * type.block_byte_size());
        for (size_t index = 0; index < payload.size(); ++index) {
            payload[index] = static_cast<uint8_t>((index * 37 + index / 97 + 19) % 256);
        }
        auto raw = std::make_shared<ov::op::v0::Constant>(type, shape, payload.data());
        auto fc = make_gguf_repack_fc(raw);
        auto second_fc = make_gguf_repack_fc(raw);
        auto activation = ov::as_type_ptr<ov::op::v0::Parameter>(fc->get_input_node_shared_ptr(0));
        auto second_activation = ov::as_type_ptr<ov::op::v0::Parameter>(second_fc->get_input_node_shared_ptr(0));
        auto graph = std::make_shared<ov::Model>(ov::OutputVector{fc, raw, second_fc},
                            ov::ParameterVector{activation, second_activation});
        ov::pass::Manager repack;
        repack.register_pass<RepackGGUFWeightsShuffle>();
        repack.run_passes(graph);
        auto packed = ov::as_type_ptr<ov::op::v0::Constant>(fc->get_input_node_shared_ptr(1));
        ASSERT_NE(packed, nullptr);
        EXPECT_NE(packed, raw);
        EXPECT_EQ(packed->get_element_type(), type);
        EXPECT_EQ(packed->get_shape(), shape);
        EXPECT_EQ(packed->get_byte_size(), payload.size());
        EXPECT_EQ(std::memcmp(raw->get_data_ptr(), payload.data(), payload.size()), 0);
        EXPECT_EQ(graph->get_results()[1]->get_input_node_shared_ptr(0), raw);
        auto second_packed = ov::as_type_ptr<ov::op::v0::Constant>(second_fc->get_input_node_shared_ptr(1));
        ASSERT_NE(second_packed, nullptr);
        EXPECT_NE(second_packed, raw);
        EXPECT_EQ(std::memcmp(packed->get_data_ptr(), second_packed->get_data_ptr(), payload.size()), 0);
        std::vector<uint8_t> once(payload.size());
        std::memcpy(once.data(), packed->get_data_ptr(), once.size());
        repack.run_passes(graph);
        auto twice = ov::as_type_ptr<ov::op::v0::Constant>(fc->get_input_node_shared_ptr(1));
        ASSERT_NE(twice, nullptr);
        EXPECT_EQ(twice, packed);
        EXPECT_EQ(std::memcmp(twice->get_data_ptr(), once.data(), once.size()), 0);
        EXPECT_EQ(second_fc->get_input_node_shared_ptr(1), second_packed);
        auto cloned_graph = graph->clone();
        auto cloned_fc = cloned_graph->get_results()[0]->get_input_node_shared_ptr(0);
        auto cloned_weight = cloned_fc->get_input_node_shared_ptr(1);
        repack.run_passes(cloned_graph);
        EXPECT_EQ(cloned_fc->get_input_node_shared_ptr(1), cloned_weight);
        auto cloned_packed = ov::as_type_ptr<ov::op::v0::Constant>(cloned_weight);
        ASSERT_NE(cloned_packed, nullptr);
        EXPECT_EQ(std::memcmp(cloned_packed->get_data_ptr(), once.data(), once.size()), 0);
    }
}

TEST(GGUFRepack, LeavesIneligibleWeightsUnchanged) {
    const std::vector<std::pair<ov::element::Type, ov::Shape>> cases = {
        {ov::element::gguf_q4_k, {17, 256}},
        {ov::element::gguf_q4_0, {16, 32}},
        {ov::element::gguf_q3_k, {16, 256}},
        {ov::element::f32, {16, 256}},
    };
    for (const auto& test_case : cases) {
        auto raw = std::make_shared<ov::op::v0::Constant>(test_case.first, test_case.second);
        auto fc = make_gguf_repack_fc(raw);
        auto activation = ov::as_type_ptr<ov::op::v0::Parameter>(fc->get_input_node_shared_ptr(0));
        auto graph = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{activation});
        ov::pass::Manager repack;
        repack.register_pass<RepackGGUFWeightsShuffle>();
        repack.run_passes(graph);
        EXPECT_EQ(fc->get_input_node_shared_ptr(1), raw);
    }
}

TEST_F(TransformationTestsF, ConvertGGUFFCToCompressed) {
    auto activation = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 256});
    auto weight = std::make_shared<ov::op::v0::Constant>(ov::element::gguf_q4_k, ov::Shape{32, 256});
    auto absent = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0});
    auto fc = std::make_shared<ov::op::internal::FullyConnectedCompressed>(activation,
                                                                        weight,
                                                                        absent,
                                                                        absent,
                                                                        absent,
                                                                        ov::element::f32);
    model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{activation});
    manager.register_pass<ConvertGGUFFullyConnectedCompressed>();

    auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
    auto scale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {1.0f});
    auto no_zero_point = std::make_shared<ov::intel_gpu::op::Placeholder>();
    auto gpu_fc = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(activation,
                                                                             weight,
                                                                             no_bias,
                                                                             scale,
                                                                             no_zero_point,
                                                                             ov::element::f32);
    model_ref = std::make_shared<ov::Model>(ov::OutputVector{gpu_fc}, ov::ParameterVector{activation});
}

TEST_F(TransformationTestsF, ConvertGGUFFCToCompressedPreservesBiasAndPrecision) {
    auto activation = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{2, -1, 256});
    auto weight = std::make_shared<ov::op::v0::Constant>(ov::element::gguf_q4_k, ov::Shape{32, 256});
    auto bias = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{32}, {0.5f});
    auto absent = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0});
    auto fc = std::make_shared<ov::op::internal::FullyConnectedCompressed>(activation,
                                                                        weight,
                                                                        bias,
                                                                        absent,
                                                                        absent,
                                                                        ov::element::f16);
    model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{activation});
    manager.register_pass<ConvertGGUFFullyConnectedCompressed>();

    auto scale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {1.0f});
    auto no_zero_point = std::make_shared<ov::intel_gpu::op::Placeholder>();
    auto gpu_fc = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(activation,
                                                                             weight,
                                                                             bias,
                                                                             scale,
                                                                             no_zero_point,
                                                                             ov::element::f16);
    model_ref = std::make_shared<ov::Model>(ov::OutputVector{gpu_fc}, ov::ParameterVector{activation});
}

TEST(GGUFConversion, RankFourBinaryPreservesBroadcastAndRejectsUnsafeShapes) {
    struct Scenario {
        ov::PartialShape activation;
        ov::PartialShape peer;
        bool move_binary;
        bool shared_fc = false;
        ov::op::AutoBroadcastType broadcast = ov::op::AutoBroadcastType::NUMPY;
    };
    const std::vector<Scenario> scenarios{
        {{1, 1, -1, 256}, {1, 1, -1, 32}, true},
        {{1, 1, 1, 256}, {1, 1, 7, 32}, true},
        {{1, 1, 7, 256}, {1, 1, 1, 32}, true},
        {{2, 1, -1, 256}, {2, 1, -1, 32}, false},
        {{1, 1, -1, 256}, {1, 2, -1, 32}, false},
        {{1, 1, -1, 256}, {32}, false},
        {{1, 1, -1, 256}, {1, 1, -1, 1}, false},
        {{1, 1, -1, 256}, ov::PartialShape::dynamic(), false},
        {{1, 1, -1, 256}, {1, 1, -1, 32}, false, true},
        {{1, 1, 7, 256}, {1, 1, 7, 32}, false, false, ov::op::AutoBroadcastType::NONE},
    };
    for (size_t scenario = 0; scenario < scenarios.size(); ++scenario) {
        SCOPED_TRACE(scenario);
        const auto& spec = scenarios[scenario];
        for (const bool product : {false, true}) {
            for (const bool reverse : {false, true}) {
                SCOPED_TRACE(product);
                SCOPED_TRACE(reverse);
                auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, spec.activation);
                auto peer = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, spec.peer);
                auto weight = std::make_shared<ov::op::v0::Constant>(ov::element::gguf_q4_k, ov::Shape{32, 256});
                auto absent = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0});
                auto fc = std::make_shared<ov::op::internal::FullyConnectedCompressed>(
                    input, weight, absent, absent, absent, ov::element::f32);
                const ov::Output<ov::Node> left = reverse ? peer->output(0) : fc->output(0);
                const ov::Output<ov::Node> right = reverse ? fc->output(0) : peer->output(0);
                std::shared_ptr<ov::Node> binary;
                if (product)
                    binary = std::make_shared<ov::op::v1::Multiply>(left, right, spec.broadcast);
                else
                    binary = std::make_shared<ov::op::v1::Add>(left, right, spec.broadcast);
                binary->set_friendly_name("rank_four_binary");
                ov::OutputVector outputs{binary};
                if (spec.shared_fc)
                    outputs.push_back(fc);
                auto graph = std::make_shared<ov::Model>(outputs, ov::ParameterVector{input, peer});
                const auto expected_shape = binary->get_output_partial_shape(0);
                ov::pass::Manager conversion;
                conversion.register_pass<ConvertGGUFFullyConnectedCompressed>();
                conversion.run_passes(graph);
                auto result = graph->get_results().at(0)->get_input_node_shared_ptr(0);
                EXPECT_EQ(result->get_output_partial_shape(0), expected_shape);
                if (spec.move_binary) {
                    ASSERT_TRUE(ov::is_type<ov::op::v1::Reshape>(result));
                    auto flat_binary = result->get_input_node_shared_ptr(0);
                    EXPECT_EQ(flat_binary->get_type_info(), binary->get_type_info());
                    EXPECT_EQ(flat_binary->get_friendly_name(), "rank_four_binary");
                    auto gpu_fc = flat_binary->get_input_node_shared_ptr(reverse ? 1 : 0);
                    ASSERT_TRUE(ov::is_type<ov::intel_gpu::op::FullyConnectedCompressed>(gpu_fc));
                    EXPECT_EQ(gpu_fc->get_input_partial_shape(0).rank(), ov::Rank(2));
                    EXPECT_EQ(gpu_fc->get_input_node_shared_ptr(1), weight);
                } else {
                    EXPECT_EQ(result, binary);
                    EXPECT_TRUE(ov::is_type<ov::op::v1::Reshape>(binary->get_input_node_shared_ptr(reverse ? 1 : 0)));
                }
            }
        }
    }
}

TEST_F(TransformationTestsF, ConvertGGUFFCLeavesOtherCompressedWeightsUnchanged) {
    auto activation = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 256});
    auto weight = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{32, 256}, {1});
    auto scale = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{32, 1}, {1.0f});
    auto absent = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0});
    auto fc = std::make_shared<ov::op::internal::FullyConnectedCompressed>(activation,
                                                                        weight,
                                                                        absent,
                                                                        scale,
                                                                        absent,
                                                                        ov::element::f32);
    model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{activation});
    model_ref = model->clone();
    manager.register_pass<ConvertGGUFFullyConnectedCompressed>();
}

TEST_F(TransformationTestsF, ConvertFCToCompressed1) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(convert, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 1 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed2) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 1 }, { 1 });
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_const);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 1 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 1 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed3) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 4, 4 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 4, 1 }, { 1 });
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_const);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 4, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { -1, 16 });
        auto reshape = std::make_shared<ov::op::v1::Reshape>(scale, reshape_const, false);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, reshape, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 4 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 4 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed4) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 4, 4 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 1, 4, 1 }, { 1 });
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_const);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 1, 4, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { -1, 16 });
        auto reshape = std::make_shared<ov::op::v1::Reshape>(scale, reshape_const, false);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, reshape, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 16 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 1, 4 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 1, 4 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed5) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 32, 4, 4 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 1, 1 }, { 1 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f32);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 4, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { -1, 16 });
        auto reshape = std::make_shared<ov::op::v1::Reshape>(scale, reshape_const, false);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, reshape, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 32, 16 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 32, 4 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 1 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed6) {
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 4, 4, 32 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 1, 1 }, { 1 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f32);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 4, 1, 32 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 16, -1 });
        auto reshape = std::make_shared<ov::op::v1::Reshape>(scale, reshape_const, false);
        auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose = std::make_shared<ov::op::v1::Transpose>(reshape, transpose_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, transpose, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 16, 32 }, { 1 });
        auto transpose_weights_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose_weights = std::make_shared<ov::op::v1::Transpose>(weights_const, transpose_weights_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 4, 32 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 1 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(
            input1, transpose_weights, no_bias, scale_const, zp_const, ov::element::dynamic, true, false, false);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed7) {
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 4, 4, 32 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 4, 1, 32 }, { 1 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f32);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 4, 1, 32 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 16, -1 });
        auto reshape = std::make_shared<ov::op::v1::Reshape>(scale, reshape_const, false);
        auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose = std::make_shared<ov::op::v1::Transpose>(reshape, transpose_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, transpose, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 16, 32 }, { 1 });
        auto transpose_weights_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose_weights = std::make_shared<ov::op::v1::Transpose>(weights_const, transpose_weights_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 4, 32 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 4, 32 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(
            input1, transpose_weights, no_bias, scale_const, zp_const, ov::element::dynamic, true, false, false);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed8) {
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 4, 4, 32 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 4, 1, 32 }, { 1 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 4, 1, 32 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 16, -1 });
        auto reshape = std::make_shared<ov::op::v1::Reshape>(scale, reshape_const, false);
        auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose = std::make_shared<ov::op::v1::Transpose>(reshape, transpose_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, transpose, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 16, 32 }, { 1 });
        auto transpose_weights_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose_weights = std::make_shared<ov::op::v1::Transpose>(weights_const, transpose_weights_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 4, 32 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 4, 32 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(
            input1, transpose_weights, no_bias, scale_const, zp_const, ov::element::dynamic, true, false, false);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

class TestSubgraph : public ov::op::util::SubGraphOp {
public:
    OPENVINO_OP("TestSubgraph", "Test", ov::op::util::SubGraphOp);

    TestSubgraph() = default;

    TestSubgraph(const OutputVector& args, const std::shared_ptr<ov::Model>& body);

    TestSubgraph(const NodeVector& args, const std::shared_ptr<ov::Model>& body);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

private:
    const ov::Model& body() const {
        return *m_bodies[0];
    }
    ov::Model& body() {
        return *m_bodies[0];
    }
    const std::shared_ptr<ov::Model>& body_ptr() const {
        return m_bodies[0];
    }
    std::shared_ptr<ov::Model>& body_ptr() {
        return m_bodies[0];
    }

};

TestSubgraph::TestSubgraph(const ov::OutputVector& args, const std::shared_ptr<ov::Model>& body)
    : SubGraphOp(args) {
    SubGraphOp::set_function(body);
    constructor_validate_and_infer_types();
    for (size_t i = 0; i < body->get_parameters().size(); ++i)
        m_input_descriptions[0].push_back(std::make_shared<InvariantInputDescription>(i, i));
    for (size_t i = 0; i < body->get_output_size(); ++i)
        m_output_descriptions[0].push_back(std::make_shared<BodyOutputDescription>(i, i));
}

TestSubgraph::TestSubgraph(const ov::NodeVector& args, const std::shared_ptr<ov::Model>& body)
    : TestSubgraph(as_output_vector(args), body) {}

std::shared_ptr<ov::Node> TestSubgraph::clone_with_new_inputs(const ov::OutputVector& inputs) const {
    return std::make_shared<TestSubgraph>(inputs, body().clone());
}

void TestSubgraph::validate_and_infer_types() {
    ov::ParameterVector old_parameters;
    for (auto op : body_ptr()->get_parameters()) {
        old_parameters.push_back(op);
    }

    for (size_t i = 0; i < get_input_size(); ++i) {
        body_ptr()->replace_parameter(
            i,
            std::make_shared<ov::op::v0::Parameter>(get_input_element_type(i), get_input_partial_shape(i)));
    }

    body_ptr()->validate_nodes_and_infer_types();

    for (size_t i = 0; i < body_ptr()->get_parameters().size(); i++) {
        body_ptr()->get_parameters()[i]->set_friendly_name(old_parameters[i]->get_friendly_name());
    }

    set_output_size(body_ptr()->get_output_size());
    for (size_t i = 0; i < get_output_size(); ++i) {
        set_output_type(i, body_ptr()->get_output_element_type(i), body_ptr()->get_output_partial_shape(i));
    }
}

bool TestSubgraph::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("body", body_ptr());
    visitor.on_attribute("input_descriptions", m_input_descriptions[0]);
    visitor.on_attribute("output_descriptions", m_output_descriptions[0]);
    return true;
}

TEST_F(TransformationTestsF, ConvertFCToCompressed9) {
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    {
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 4, 4, 32 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 4, 1, 32 }, { 1 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 4, 1, 32 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 16, -1 });
        auto reshape = std::make_shared<ov::op::v1::Reshape>(scale, reshape_const, false);
        auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose = std::make_shared<ov::op::v1::Transpose>(reshape, transpose_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();

        auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 15});
        auto const_value1 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 1}, {1});
        auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        auto const_value2 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 1}, {1});
        auto add1 = std::make_shared<ov::op::v1::Add>(param1, const_value1);
        auto add2 = std::make_shared<ov::op::v1::Add>(param2, const_value2);
        auto result1 = std::make_shared<ov::op::v0::Result>(add1);
        auto result2 = std::make_shared<ov::op::v0::Result>(add2);
        auto submodel = std::make_shared<ov::Model>(ov::ResultVector{result1, result2}, ov::ParameterVector{param1, param2});
        ParameterVector subgraph_parameters{submodel->inputs().size()};
        OutputVector args{submodel->inputs().size()};
        for (size_t i = 0; i < submodel->inputs().size(); i++) {
            auto const& input = submodel->input(i);
            subgraph_parameters[i] =
                std::make_shared<ov::op::v0::Parameter>(input.get_element_type(), input.get_partial_shape());
            args[i] = subgraph_parameters[i]->output(0);
        }
        auto subgraph_op = std::make_shared<TestSubgraph>(args, submodel);
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(subgraph_op->output(1), transpose, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<ov::op::v0::Result>(subgraph_op->output(0)), fc}, subgraph_parameters);
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 16, 32 }, { 1 });
        auto transpose_weights_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 1, 0 });
        auto transpose_weights = std::make_shared<ov::op::v1::Transpose>(weights_const, transpose_weights_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 4, 32 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 4, 32 }, { 1 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);

        auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 15});
        auto const_value1 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 1}, {1});
        auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        auto const_value2 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 1}, {1});
        auto add1 = std::make_shared<ov::op::v1::Add>(param1, const_value1);
        auto add2 = std::make_shared<ov::op::v1::Add>(param2, const_value2);
        auto result1 = std::make_shared<ov::op::v0::Result>(add1);
        auto result2 = std::make_shared<ov::op::v0::Result>(add2);
        auto submodel = std::make_shared<ov::Model>(ov::ResultVector{result1, result2}, ov::ParameterVector{param1, param2});
        ParameterVector subgraph_parameters{submodel->inputs().size()};
        OutputVector args{submodel->inputs().size()};
        for (size_t i = 0; i < submodel->inputs().size(); i++) {
            auto const& input = submodel->input(i);
            subgraph_parameters[i] =
                std::make_shared<ov::op::v0::Parameter>(input.get_element_type(), input.get_partial_shape());
            args[i] = subgraph_parameters[i]->output(0);
        }
        auto subgraph_op = std::make_shared<TestSubgraph>(args, submodel);
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(
            subgraph_op->output(1), transpose_weights, no_bias, scale_const, zp_const, ov::element::dynamic, true, false, false);

        model_ref =
            std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<ov::op::v0::Result>(subgraph_op->output(0)), fc_compressed}, subgraph_parameters);
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed10) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto zp_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_const);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed11) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 1 }, { 1 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 1 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed12) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        auto weights_param = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{32, 16});
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_param, ov::element::f16);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{32, 1}, {1});
        auto scale = std::make_shared<ov::op::v1::Multiply>(convert, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();

        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1, weights_param});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        auto weights_param = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{32, 16});
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{32, 1}, {1});

        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_param, no_bias, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1, weights_param});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed13) {
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        auto weights_param = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{4, 4, 32});
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {16, 32});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(weights_param, reshape_const, false);
        auto convert = std::make_shared<ov::op::v0::Convert>(reshape, ov::element::f16);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 32}, {1});
        auto scale = std::make_shared<ov::op::v1::Multiply>(convert, scale_const);
        auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {1, 0});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(scale, transpose_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();

        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, transpose, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1, weights_param});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        auto weights_param = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{4, 4, 32});
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {16, 32});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(weights_param, reshape_const, false);
        auto transpose_weights_param = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {1, 0});
        auto transpose_weights = std::make_shared<ov::op::v1::Transpose>(reshape, transpose_weights_param);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 32}, {1});
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(
            input1, transpose_weights, no_bias, scale_const, ov::element::dynamic, true, false, false);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1, weights_param});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed14) {
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        auto weights_param = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{4, 4, 32});
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_param, ov::element::f16);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {16, 32});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(convert, reshape_const, false);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 32}, {1});
        auto scale = std::make_shared<ov::op::v1::Multiply>(reshape, scale_const);
        auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {1, 0});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(scale, transpose_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();

        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, transpose, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1, weights_param});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        auto weights_param = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{4, 4, 32});
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {16, 32});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(weights_param, reshape_const, false);
        auto transpose_weights_param = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {1, 0});
        auto transpose_weights = std::make_shared<ov::op::v1::Transpose>(reshape, transpose_weights_param);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 32}, {1});
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(
            input1, transpose_weights, no_bias, scale_const, ov::element::dynamic, true, false, false);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1, weights_param});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed15) {
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        auto weights_param = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{4, 4, 32});
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_param, ov::element::f16);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{4, 1, 32}, {1});
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{4, 1, 32}, {1});
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {16, 32});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(scale, reshape_const, false);
        auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {1, 0});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(reshape, transpose_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();

        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, transpose, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1, weights_param});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        auto weights_param = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{4, 4, 32});
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {16, 32});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(weights_param, reshape_const, false);
        auto transpose_weights_param = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {1, 0});
        auto transpose_weights = std::make_shared<ov::op::v1::Transpose>(reshape, transpose_weights_param);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{4, 32}, {1});
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{4, 32}, {1});
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);

        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(
            input1, transpose_weights, no_bias, scale_const, zp_const, ov::element::dynamic, true, false, false);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1, weights_param});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed16_i8ZpNotConvertedToU8) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::i8, ov::Shape{ 32, 16 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto zp_const = ov::op::v0::Constant::create(ov::element::i8, ov::Shape{ 32, 1 }, { -128 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        // The i8 ZP constant must be preserved as i8, NOT recast to u8.
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::i8, ov::Shape{ 32, 16 }, { 1 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::i8, ov::Shape{ 32, 1 }, { -128 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed16_i4ScalarZpConvertedToI8) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::i4, ov::Shape{ 32, 16 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto zp_const = ov::op::v0::Constant::create(ov::element::i4, ov::Shape{ 1 }, { -3 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        // oneDNN rejects sub-byte per-tensor zero points, so the scalar i4 ZP must be widened to i8.
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::i4, ov::Shape{ 32, 16 }, { 1 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::i4, ov::Shape{ 1 }, { -3 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::i8);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_convert);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

// 4D non-grouped weight constant: leading dims are unit, only last two are
// real (K, N). is_weight_3d is false, so the reshape branch must collapse
// dims [d0, d1, d2, d3] -> [d0*d1*d2, d3] for weight/scale/zp.
TEST_F(TransformationTestsF, ConvertFCToCompressed17_4dNonGrouped) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 1, 32, 16 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 1, 32, 1 }, { 1 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 1, 1, 32, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 1 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

// 4D grouped weight/scale/zp constants with is_weight_3d=true: the FC sees
// a 3D weight (after 4D->3D reshape), so the reshape_const lambda must
// collapse 4D constants [d0, d1, d2, d3] -> [d0, d1, d2*d3] (grouped path).
TEST_F(TransformationTestsF, ConvertFCToCompressed18_4dGroupedWith3dWeight) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 8 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 2, 4, 2, 4 }, { 1 });
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 2, 4, 2, 1 }, { 1 });
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);
        auto sub = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 2, 4, 2, 1 }, { 1 });
        auto scale = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 3 }, { 2, 4, -1 });
        auto reshape = std::make_shared<ov::op::v1::Reshape>(scale, reshape_const, false);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, reshape, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 8 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{ 2, 4, 8 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 2, 4, 2 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 2, 4, 2 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}
  
TEST_F(TransformationTestsF, ConvertFCToCompressed19_mxfp8e4m3) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 32 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f8e4m3, ov::Shape{ 32, 32 }, { 1 });
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f8e8m0, ov::Shape{ 32, 1 }, { 1 });
        auto scale_convert = std::make_shared<ov::op::v0::Convert>(scale_const, ov::element::f16);
        auto deq_mul = std::make_shared<ov::op::v1::Multiply>(weights_convert, scale_convert);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, deq_mul, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 32 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f8e4m3, ov::Shape{ 32, 32 }, { 1 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f8e8m0, ov::Shape{ 32, 1 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed20_mxfp8_e5m2) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 32 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f8e5m2, ov::Shape{ 32, 32 }, { 1 });
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f8e8m0, ov::Shape{ 32, 1 }, { 1 });
        auto scale_convert = std::make_shared<ov::op::v0::Convert>(scale_const, ov::element::f16);
        auto deq_mul = std::make_shared<ov::op::v1::Multiply>(weights_convert, scale_convert);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, deq_mul, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 32 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f8e5m2, ov::Shape{ 32, 32 }, { 1 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f8e8m0, ov::Shape{ 32, 1 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed21_fp8e4m3) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 32 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f8e4m3, ov::Shape{ 32, 32 }, { 1 });
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto deq_mul = std::make_shared<ov::op::v1::Multiply>(weights_convert, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, deq_mul, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 32 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f8e4m3, ov::Shape{ 32, 32 }, { 1 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed22_fp8_e5m2) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 32 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f8e5m2, ov::Shape{ 32, 32 }, { 1 });
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto deq_mul = std::make_shared<ov::op::v1::Multiply>(weights_convert, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, deq_mul, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 32 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f8e5m2, ov::Shape{ 32, 32 }, { 1 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}
TEST_F(TransformationTestsF, ConvertFCToCompressed23_fp4e2m1) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 32 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f4e2m1, ov::Shape{ 32, 32 }, { 1 });
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto deq_mul = std::make_shared<ov::op::v1::Multiply>(weights_convert, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, deq_mul, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 32 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f4e2m1, ov::Shape{ 32, 32 }, { 1 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertFCToCompressed24_mxfp4e2m1) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f4e2m1, ov::Shape{ 32, 16 }, { 1 });
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f8e8m0, ov::Shape{ 32, 1 }, { 1 });
        auto scale_convert = std::make_shared<ov::op::v0::Convert>(scale_const, ov::element::f16);
        auto scale = std::make_shared<ov::op::v1::Multiply>(weights_convert, scale_convert);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f4e2m1, ov::Shape{ 32, 16 }, { 1 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f8e8m0, ov::Shape{ 32, 1 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input1, weights_const, no_bias, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input1});
    }
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
