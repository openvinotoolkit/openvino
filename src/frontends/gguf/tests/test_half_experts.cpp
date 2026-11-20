// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstring>
#include <tuple>

#include "helper_ops/gguf_constant.hpp"
#include "node_context.hpp"
#include "op_test_utils.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/gather_matmul.hpp"
#include "pass/materialize_weights.hpp"
#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"
#include "transformations/rt_info/decompression.hpp"

using namespace ov_gguf_test;

namespace {

enum class WeightStorage { DIRECT, CONVERTED, NATIVE_DEFERRED, CGRAPH_DEFERRED };
using HalfExpertCase = std::tuple<ov::element::Type, bool, WeightStorage, bool>;

std::shared_ptr<ov::Model> expert_model(const ov::element::Type& precision,
                                        bool rank4,
                                        WeightStorage storage,
                                        bool per_expert,
                                        const std::vector<float>& values) {
    const ov::Shape weight_shape = rank4 ? ov::Shape{1, 3, 2, 2} : ov::Shape{3, 2, 2};
    auto constant = ov::op::v0::Constant::create(precision, weight_shape, values);
    constant->set_friendly_name("expert_storage");
    ov::Output<ov::Node> weights = constant;
    if (storage == WeightStorage::CONVERTED) {
        weights = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f32);
    } else if (storage == WeightStorage::NATIVE_DEFERRED || storage == WeightStorage::CGRAPH_DEFERRED) {
        ov::Tensor data(precision, weight_shape);
        std::memcpy(data.data(), constant->get_data_ptr(), data.get_byte_size());
        ov::frontend::gguf::GgufWeight source(data);
        source.native = storage == WeightStorage::NATIVE_DEFERRED;
        weights = std::make_shared<ov::frontend::gguf::GGUFConstant>(source);
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

std::shared_ptr<ov::Model> routing_model(const ov::PartialShape& acts_shape, const ov::PartialShape& ids_shape) {
    const ov::Shape weight_shape{3, 2, 2};
    auto decoder = SingleOpBuilder()
                       .op("GGML_OP_MUL_MAT_ID")
                       .input("w", ov::element::f16, weight_shape)
                       .input("a", ov::element::f32, acts_shape)
                       .input("ids", ov::element::i32, ids_shape)
                       .output("out", ov::element::f32, {1, ids_shape[2], 2, 2})
                       .decoder();
    auto acts = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, acts_shape);
    auto ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ids_shape);
    auto tensors = std::make_shared<TensorMap>();
    (*tensors)["w"] = ov::op::v0::Constant::create(ov::element::f16,
                                                   weight_shape,
                                                   std::vector<float>{1, 0, 0, 1, 2, 0, 0, 2, 3, 0, 0, 3});
    (*tensors)["a"] = acts;
    (*tensors)["ids"] = ids;
    NodeContext context(decoder, tensors);
    return std::make_shared<ov::Model>(get_supported_ops().at("GGML_OP_MUL_MAT_ID")(context),
                                       ov::ParameterVector{acts, ids});
}

void check_expert_graph(const std::shared_ptr<ov::Model>& model, bool deferred) {
    size_t gmm_count = 0;
    size_t deferred_count = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (const auto gmm = ov::as_type_ptr<ov::op::internal::GatherMatmul>(node)) {
            ++gmm_count;
            EXPECT_EQ(gmm->get_input_partial_shape(1), (ov::PartialShape{3, 2, 2}));
            EXPECT_EQ(gmm->get_output_element_type(0), ov::element::f32);
        }
        deferred_count += ov::is_type<ov::frontend::gguf::GGUFConstant>(node);
    }
    EXPECT_EQ(gmm_count, 1);
    EXPECT_EQ(deferred_count, deferred ? 1 : 0);
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

TEST_P(GGUFHalfExperts, UsesGatherMatmulAndPreservesDynamicTokenValues) {
    const auto [precision, rank4, storage, per_expert] = GetParam();
    // These are the independent expected values of GGUFOps.MulMatIdNused2 and ActivationsPerExpert.
    const std::vector<float> weights{1, 0, 0, 1, 2, 0, 0, 2, 3, 0, 0, 3};
    const std::vector<float> expected =
        per_expert ? std::vector<float>{1, 2, 6, 8, 15, 18, 7, 8} : std::vector<float>{1, 2, 2, 4, 9, 12, 3, 4};
    auto model = expert_model(precision, rank4, storage, per_expert, weights);
    const bool deferred = storage == WeightStorage::NATIVE_DEFERRED || storage == WeightStorage::CGRAPH_DEFERRED;
    check_expert_graph(model, deferred);
    if (deferred) {
        EXPECT_TRUE(ov::frontend::gguf::pass::MaterializeGGUFWeights().run_on_model(model));
        model->validate_nodes_and_infer_types();
        check_expert_graph(model, false);
    }
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::MarkCompressedFloatConstants>();
    manager.run_passes(model);
    if (storage != WeightStorage::DIRECT) {
        size_t marked = 0;
        for (const auto& node : model->get_ordered_ops()) {
            if (const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(node);
                convert && convert->get_input_element_type(0) == precision && ov::is_decompression(convert)) {
                ++marked;
            }
        }
        EXPECT_EQ(marked, 1);
    }
    ov::Core core;
    const ov::AnyMap properties{{"INFERENCE_PRECISION_HINT", ov::element::f32}, {"DYNAMIC_QUANTIZATION_GROUP_SIZE", 0}};
    auto compiled = core.compile_model(model, "CPU", properties);
    auto request = compiled.create_infer_request();
    for (const size_t tokens : {1u, 4u, 1u, 2u, 0u, 1u}) {
        SCOPED_TRACE(tokens);
        infer_tokens(request, tokens, per_expert, expected);
    }
}

INSTANTIATE_TEST_SUITE_P(Storage,
                         GGUFHalfExperts,
                         ::testing::Combine(::testing::Values(ov::element::f16, ov::element::bf16),
                                            ::testing::Bool(),
                                            ::testing::Values(WeightStorage::DIRECT,
                                                              WeightStorage::CONVERTED,
                                                              WeightStorage::NATIVE_DEFERRED,
                                                              WeightStorage::CGRAPH_DEFERRED),
                                            ::testing::Bool()),
                         ([](const ::testing::TestParamInfo<HalfExpertCase>& info) {
                             const auto& [precision, rank4, storage, per_expert] = info.param;
                             const auto storage_name = storage == WeightStorage::DIRECT            ? "_direct"
                                                       : storage == WeightStorage::CONVERTED       ? "_converted"
                                                       : storage == WeightStorage::NATIVE_DEFERRED ? "_native"
                                                                                                   : "_cgraph";
                             return std::string(precision.get_type_name()) + (rank4 ? "_rank4" : "_rank3") +
                                    storage_name + (per_expert ? "_down" : "_shared");
                         }));

TEST(GGUFHalfExpertsRouting, StaticActivationLayoutsAndTokenCounts) {
    struct Case {
        ov::Shape activations_shape;
        size_t tokens;
        bool per_expert;
    };
    const std::vector<int32_t> ids_values{0, 1, 2, 0};
    const std::vector<Case> cases{{{2, 2}, 2, false},
                                  {{1, 2, 2}, 2, false},
                                  {{1, 1, 2, 2}, 2, false},
                                  {{2, 2, 2}, 2, true},
                                  {{1, 2, 2, 2}, 2, true},
                                  {{2, 1, 2, 2}, 2, true},
                                  {{1, 2, 1, 2, 2}, 2, true},
                                  {{2, 2}, 1, true}};
    for (const auto& [activations_shape, tokens, per_expert] : cases) {
        SCOPED_TRACE(activations_shape);
        SCOPED_TRACE(tokens);
        const ov::Shape ids_shape{1, 1, tokens, 2};
        const ov::Shape output_shape{1, tokens, 2, 2};
        auto model = routing_model(activations_shape, ids_shape);
        auto request = ov::Core{}
                           .compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32))
                           .create_infer_request();
        std::vector<float> values =
            per_expert ? std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8} : std::vector<float>{1, 2, 3, 4};
        std::vector<float> expected =
            per_expert ? std::vector<float>{1, 2, 6, 8, 15, 18, 7, 8} : std::vector<float>{1, 2, 2, 4, 9, 12, 3, 4};
        values.resize(ov::shape_size(activations_shape));
        expected.resize(tokens * 4);
        request.set_input_tensor(0, make_f32_tensor(activations_shape, values));
        ov::Tensor selected(ov::element::i32, ids_shape);
        std::copy_n(ids_values.begin(), tokens * 2, selected.data<int32_t>());
        request.set_input_tensor(1, selected);
        request.infer();
        EXPECT_EQ(request.get_output_tensor().get_shape(), output_shape);
        expect_near(request.get_output_tensor(), expected);
    }
}

TEST(GGUFHalfExpertsRouting, StaticActivationsAndDynamicIds) {
    const ov::PartialShape ids_shape{1, 1, ov::Dimension::dynamic(), 2};
    const std::vector<int32_t> ids_values{0, 1, 2, 0};
    for (const auto& acts_shape : {ov::Shape{2, 2}, ov::Shape{1, 2, 2}, ov::Shape{1, 1, 2, 2}}) {
        SCOPED_TRACE(acts_shape);
        auto model = routing_model(acts_shape, ids_shape);
        size_t gmm_count = 0;
        for (const auto& node : model->get_ordered_ops()) {
            if (const auto gmm = ov::as_type_ptr<ov::op::internal::GatherMatmul>(node)) {
                ++gmm_count;
                EXPECT_EQ(gmm->get_input_partial_shape(0)[0], ov::Dimension(2));
            }
        }
        EXPECT_EQ(gmm_count, 1);
        auto request = ov::Core{}
                           .compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32))
                           .create_infer_request();
        request.set_input_tensor(0, make_f32_tensor(acts_shape, {1, 2, 3, 4}));
        for (const size_t tokens : {2u, 1u}) {
            ov::Tensor selected(ov::element::i32, {1, 1, tokens, 2});
            std::copy_n(ids_values.begin(), tokens * 2, selected.data<int32_t>());
            request.set_input_tensor(1, selected);
            request.infer();
            EXPECT_EQ(request.get_output_tensor().get_shape(), (ov::Shape{1, tokens, 2, 2}));
            expect_near(request.get_output_tensor(),
                        tokens == 2 ? std::vector<float>{1, 2, 2, 4, 9, 12, 3, 4} : std::vector<float>{1, 2, 6, 8});
        }
    }
}

TEST(GGUFHalfExpertsRouting, DynamicActivationsAndStaticIds) {
    const ov::PartialShape acts_shape{1, ov::Dimension::dynamic(), 2, 2};
    const ov::Shape ids_shape{1, 1, 2, 2};
    auto model = routing_model(acts_shape, ids_shape);
    auto request =
        ov::Core{}.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32)).create_infer_request();
    ov::Tensor selected(ov::element::i32, ids_shape);
    const std::vector<int32_t> ids_values{0, 1, 2, 0};
    std::copy(ids_values.begin(), ids_values.end(), selected.data<int32_t>());
    request.set_input_tensor(1, selected);
    request.set_input_tensor(0, make_f32_tensor({1, 1, 2, 2}, {1, 2, 3, 4}));
    request.infer();
    EXPECT_EQ(request.get_output_tensor().get_shape(), (ov::Shape{1, 2, 2, 2}));
    expect_near(request.get_output_tensor(), {1, 2, 2, 4, 9, 12, 3, 4});
    request.set_input_tensor(0, make_f32_tensor({1, 2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}));
    request.infer();
    EXPECT_EQ(request.get_output_tensor().get_shape(), (ov::Shape{1, 2, 2, 2}));
    expect_near(request.get_output_tensor(), {1, 2, 6, 8, 15, 18, 7, 8});
    request.set_input_tensor(0, make_f32_tensor({1, 3, 2, 2}, std::vector<float>(12, 1)));
    EXPECT_THROW(request.infer(), ov::Exception);
}

TEST(GGUFHalfExpertsRouting, DynamicLeadingBatchAxis) {
    const ov::PartialShape acts_shape{ov::Dimension::dynamic(), 1, 2, 2};
    const ov::Shape ids_shape{1, 1, 2, 2};
    auto model = routing_model(acts_shape, ids_shape);
    auto request =
        ov::Core{}.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32)).create_infer_request();
    ov::Tensor selected(ov::element::i32, ids_shape);
    const std::vector<int32_t> ids_values{0, 1, 2, 0};
    std::copy(ids_values.begin(), ids_values.end(), selected.data<int32_t>());
    request.set_input_tensor(1, selected);
    request.set_input_tensor(0, make_f32_tensor({1, 1, 2, 2}, {1, 2, 3, 4}));
    request.infer();
    EXPECT_EQ(request.get_output_tensor().get_shape(), (ov::Shape{1, 2, 2, 2}));
    expect_near(request.get_output_tensor(), {1, 2, 2, 4, 9, 12, 3, 4});
    request.set_input_tensor(0, make_f32_tensor({2, 1, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}));
    request.infer();
    EXPECT_EQ(request.get_output_tensor().get_shape(), (ov::Shape{1, 2, 2, 2}));
    expect_near(request.get_output_tensor(), {1, 2, 6, 8, 15, 18, 7, 8});
}

TEST(GGUFHalfExpertsRouting, MixedLayoutWithEmptyTokens) {
    const ov::PartialShape acts_shape{1, ov::Dimension::dynamic(), 2, 2};
    const ov::Shape ids_shape{1, 1, 0, 2};
    auto model = routing_model(acts_shape, ids_shape);
    auto request =
        ov::Core{}.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32)).create_infer_request();
    request.set_input_tensor(0, make_f32_tensor({1, 0, 2, 2}, {}));
    request.set_input_tensor(1, ov::Tensor(ov::element::i32, ids_shape));
    request.infer();
    EXPECT_EQ(request.get_output_tensor().get_shape(), (ov::Shape{1, 0, 2, 2}));
}

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
        auto model = expert_model(precision,
                                  false,
                                  precision == ov::element::bf16 ? WeightStorage::CONVERTED : WeightStorage::DIRECT,
                                  true,
                                  represented);
        auto compiled = core.compile_model(
            model,
            "CPU",
            {{"INFERENCE_PRECISION_HINT", ov::element::f32}, {"DYNAMIC_QUANTIZATION_GROUP_SIZE", 0}});
        auto request = compiled.create_infer_request();
        infer_tokens(request, 2, true, expected);
    }
}

}  // namespace
