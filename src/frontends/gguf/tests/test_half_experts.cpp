// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sstream>
#include <tuple>

#include "node_context.hpp"
#include "op_test_utils.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/pass/serialize.hpp"
#include "ov_ops/gather_matmul.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"

using namespace ov_gguf_test;

namespace {

using HalfExpertCase = std::tuple<ov::element::Type, bool, bool, bool>;

std::shared_ptr<ov::Model> expert_model(const ov::element::Type& precision,
                                        bool rank4,
                                        bool converted,
                                        bool per_expert,
                                        const std::vector<float>& values) {
    const ov::Shape weight_shape = rank4 ? ov::Shape{1, 3, 2, 2} : ov::Shape{3, 2, 2};
    auto constant = ov::op::v0::Constant::create(precision, weight_shape, values);
    constant->set_friendly_name("expert_storage");
    ov::Output<ov::Node> weights = constant;
    if (converted) {
        weights = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f32);
    }
    const ov::PartialShape acts_shape{1, ov::Dimension::dynamic(), per_expert ? 2 : 1, 2};
    const ov::PartialShape ids_shape{1, 1, ov::Dimension::dynamic(), 2};
    auto acts = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, acts_shape);
    auto ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ids_shape);
    auto decoder = SingleOpBuilder()
                       .op("GGML_OP_MUL_MAT_ID")
                       .input("w", weights.get_element_type(), weight_shape)
                       .input("a", ov::element::f32, acts_shape)
                       .input("ids", ov::element::i32, ids_shape)
                       .output("out", ov::element::f32, {1, ov::Dimension::dynamic(), 2, 2})
                       .decoder();
    auto tensors = std::make_shared<TensorMap>();
    (*tensors)["w"] = weights;
    (*tensors)["a"] = acts;
    (*tensors)["ids"] = ids;
    NodeContext context(decoder, tensors);
    auto outputs = get_supported_ops().at("GGML_OP_MUL_MAT_ID")(context);
    return std::make_shared<ov::Model>(outputs, ov::ParameterVector{acts, ids});
}

void check_half_graph(const std::shared_ptr<ov::Model>& model, const ov::element::Type& precision, bool roundtrip) {
    size_t storage_count = 0;
    size_t selected_count = 0;
    size_t expansion_count = 0;
    for (const auto& node : model->get_ordered_ops()) {
        EXPECT_FALSE(ov::is_type<ov::op::internal::GatherMatmul>(node));
        if (const auto storage = ov::as_type_ptr<ov::op::v0::Constant>(node)) {
            if (storage->get_shape() == ov::Shape{3, 2, 2}) {
                ++storage_count;
                EXPECT_EQ(storage->get_element_type(), precision);
                EXPECT_TRUE(ov::is_conversion_disabled(storage, precision, ov::element::f32));
                if (!roundtrip) {
                    EXPECT_TRUE(ov::is_keep_const_precision(storage));
                }
            }
        }
        if (const auto gather = ov::as_type_ptr<ov::op::v8::Gather>(node)) {
            if (gather->get_input_element_type(0) == precision) {
                ++selected_count;
                EXPECT_EQ(gather->get_output_element_type(0), precision);
                EXPECT_TRUE(ov::is_conversion_disabled(gather, precision, ov::element::f32));
            }
        }
        if (const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(node)) {
            if (convert->get_input_element_type(0) == precision &&
                convert->get_destination_type() == ov::element::f32) {
                ++expansion_count;
                EXPECT_TRUE(ov::is_type<ov::op::v8::Gather>(convert->input_value(0).get_node()));
                EXPECT_TRUE(ov::is_decompression(convert));
                if (!roundtrip) {
                    EXPECT_TRUE(ov::constant_folding_is_disabled(convert));
                }
            }
        }
    }
    EXPECT_EQ(storage_count, 1);
    EXPECT_EQ(selected_count, 1);
    EXPECT_EQ(expansion_count, 1);
}

void check_runtime_storage(const ov::CompiledModel& compiled, const ov::element::Type& precision) {
    size_t storage_count = 0;
    for (const auto& node : compiled.get_runtime_model()->get_ordered_ops()) {
        const auto& info = node->get_rt_info();
        if (info.count("layerType") && info.at("layerType").as<std::string>() == "Const" &&
            node->get_output_partial_shape(0) == ov::PartialShape{3, 2, 2}) {
            ++storage_count;
            ASSERT_TRUE(info.count("outputPrecisions"));
            EXPECT_EQ(info.at("outputPrecisions").as<std::string>(), precision.get_type_name());
        }
    }
    EXPECT_EQ(storage_count, 1);
}

void infer_tokens(ov::InferRequest& request,
                  size_t tokens,
                  bool per_expert,
                  const std::vector<float>& expected_two_tokens) {
    const std::vector<float> acts =
        per_expert ? std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8} : std::vector<float>{1, 2, 3, 4};
    const std::vector<int32_t> selected{0, 1, 2, 0};
    std::vector<float> inputs;
    std::vector<float> expected;
    ov::Tensor ids(ov::element::i32, {1, 1, tokens, 2});
    const size_t values_per_token = per_expert ? 4 : 2;
    for (size_t token = 0; token < tokens; ++token) {
        const size_t source = token % 2;
        inputs.insert(inputs.end(),
                      acts.begin() + source * values_per_token,
                      acts.begin() + (source + 1) * values_per_token);
        std::copy_n(selected.begin() + source * 2, 2, ids.data<int32_t>() + token * 2);
        expected.insert(expected.end(),
                        expected_two_tokens.begin() + source * 4,
                        expected_two_tokens.begin() + (source + 1) * 4);
    }
    request.set_input_tensor(0, make_f32_tensor({1, tokens, per_expert ? 2u : 1u, 2}, inputs));
    request.set_input_tensor(1, ids);
    request.infer();
    const auto output = request.get_output_tensor();
    EXPECT_EQ(output.get_element_type(), ov::element::f32);
    EXPECT_EQ(output.get_shape(), (ov::Shape{1, tokens, 2, 2}));
    expect_near(output, expected);
}

class GGUFHalfExperts : public ::testing::TestWithParam<HalfExpertCase> {};

TEST_P(GGUFHalfExperts, PreservesStorageAndDynamicTokenValues) {
    const auto [precision, rank4, converted, per_expert] = GetParam();
    // These are the independent expected values of GGUFOps.MulMatIdNused2 and ActivationsPerExpert.
    const std::vector<float> weights{1, 0, 0, 1, 2, 0, 0, 2, 3, 0, 0, 3};
    const std::vector<float> expected =
        per_expert ? std::vector<float>{1, 2, 6, 8, 15, 18, 7, 8} : std::vector<float>{1, 2, 2, 4, 9, 12, 3, 4};
    auto model = expert_model(precision, rank4, converted, per_expert, weights);
    check_half_graph(model, precision, false);
    ov::Core core;
    const ov::AnyMap properties{{"INFERENCE_PRECISION_HINT", ov::element::f32}, {"DYNAMIC_QUANTIZATION_GROUP_SIZE", 0}};
    for (const bool roundtrip : {false, true}) {
        SCOPED_TRACE(roundtrip ? "serialized" : "original");
        if (roundtrip) {
            std::stringstream xml, bin;
            ov::pass::Serialize(xml, bin).run_on_model(model);
            auto bytes = bin.str();
            ov::Tensor storage(ov::element::u8, {bytes.size()});
            std::memcpy(storage.data(), bytes.data(), bytes.size());
            model = core.read_model(xml.str(), storage);
            check_half_graph(model, precision, true);
        }
        auto compiled = core.compile_model(model, "CPU", properties);
        check_runtime_storage(compiled, precision);
        auto request = compiled.create_infer_request();
        for (const size_t tokens : {1u, 4u, 1u, 2u, 0u, 1u}) {
            SCOPED_TRACE(tokens);
            infer_tokens(request, tokens, per_expert, expected);
        }
    }
    auto f32 = expert_model(ov::element::f32, rank4, false, per_expert, weights);
    auto compiled = core.compile_model(f32, "CPU", properties);
    auto request = compiled.create_infer_request();
    infer_tokens(request, 2, per_expert, expected);
}

INSTANTIATE_TEST_SUITE_P(Storage,
                         GGUFHalfExperts,
                         ::testing::Combine(::testing::Values(ov::element::f16, ov::element::bf16),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Bool()),
                         ([](const ::testing::TestParamInfo<HalfExpertCase>& info) {
                             const auto& [precision, rank4, converted, per_expert] = info.param;
                             return std::string(precision.get_type_name()) + (rank4 ? "_rank4" : "_rank3") +
                                    (converted ? "_converted" : "_direct") + (per_expert ? "_down" : "_shared");
                         }));

TEST(GGUFHalfExpertsRepresented, Bf16ValuesMatchAnalyticAndF32Path) {
    const std::vector<float>
        original{1.003f, -0.701f, 0.334f, 2.007f, -3.011f, 0.127f, 0.913f, 1.009f, 2.019f, -1.117f, 0.703f, -0.227f};
    std::vector<float> represented;
    std::transform(original.begin(), original.end(), std::back_inserter(represented), [](float value) {
        return static_cast<float>(ov::bfloat16(value));
    });
    const auto dot = [&](size_t expert, size_t row, float x, float y) {
        return represented[expert * 4 + row * 2] * x + represented[expert * 4 + row * 2 + 1] * y;
    };
    const std::vector<float> expected{dot(0, 0, 1, 2),
                                      dot(0, 1, 1, 2),
                                      dot(1, 0, 3, 4),
                                      dot(1, 1, 3, 4),
                                      dot(2, 0, 5, 6),
                                      dot(2, 1, 5, 6),
                                      dot(0, 0, 7, 8),
                                      dot(0, 1, 7, 8)};
    ov::Core core;
    for (const auto& precision : {ov::element::bf16, ov::element::f32}) {
        SCOPED_TRACE(precision.get_type_name());
        auto model = expert_model(precision, false, precision == ov::element::bf16, true, represented);
        auto compiled = core.compile_model(
            model,
            "CPU",
            {{"INFERENCE_PRECISION_HINT", ov::element::f32}, {"DYNAMIC_QUANTIZATION_GROUP_SIZE", 0}});
        auto request = compiled.create_infer_request();
        infer_tokens(request, 2, true, expected);
    }
}

}  // namespace
