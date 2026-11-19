// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>

#include "op_test_utils.hpp"
#include "openvino/op/gated_delta_net.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "utils.hpp"

namespace {

using namespace ov::op;

enum class QkSlices { Adjacent, Offset, Gap, Strided, SeparateSources, ExplicitShape };
class GGUFQkNormalization : public testing::TestWithParam<QkSlices> {};

// eps^2 underflows to a subnormal, so the translator keeps ggml's explicit normalization.
TEST_P(GGUFQkNormalization, PreservesIndependentRowsAndClamp) {
    constexpr float eps = 1e-20f;
    constexpr int64_t batch = 2, heads = 2, width = 4, features = 4 * heads * width;
    const auto layout = GetParam();
    const auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{batch, 1, -1, features});
    x->set_friendly_name("x");
    ov::ParameterVector parameters{x};
    ov::Output<ov::Node> k_source = x;
    if (layout == QkSlices::SeparateSources) {
        const auto y = std::make_shared<v0::Parameter>(ov::element::f32, x->get_partial_shape());
        y->set_friendly_name("y");
        parameters.push_back(y);
        k_source = y;
    }
    const int64_t step = layout == QkSlices::Strided ? 2 : 1;
    const int64_t first = layout == QkSlices::Offset ? width : 0;
    const int64_t middle = first + step * heads * width;
    const int64_t k_first = middle + (layout == QkSlices::Gap ? width : 0);
    const auto slice = [&](const ov::Output<ov::Node>& input, int64_t begin) {
        return std::make_shared<v8::Slice>(input,
                                           v0::Constant::create(ov::element::i64, {1}, {begin}),
                                           v0::Constant::create(ov::element::i64, {1}, {begin + step * heads * width}),
                                           v0::Constant::create(ov::element::i64, {1}, {step}),
                                           v0::Constant::create(ov::element::i64, {1}, {3}));
    };
    const bool special_zero = layout != QkSlices::ExplicitShape;
    const auto shape =
        v0::Constant::create<int64_t>(ov::element::i64, {4}, {special_zero ? 0 : batch, -1, heads, width});
    const auto q = std::make_shared<v1::Reshape>(slice(x, first), shape, special_zero);
    const auto k = std::make_shared<v1::Reshape>(slice(k_source, k_first), shape, special_zero);
    const ov::PartialShape qkv_shape{batch, -1, heads, width}, gate_shape{batch, -1, heads, 1};
    const auto decoder = std::make_shared<ov_gguf_test::SingleOpDecoder>(
        "GGML_OP_GATED_DELTA_NET",
        std::vector<ov_gguf_test::TensorDesc>{{"q", ov::element::f32, qkv_shape},
                                              {"k", ov::element::f32, qkv_shape},
                                              {"v", ov::element::f32, qkv_shape},
                                              {"g", ov::element::f32, gate_shape},
                                              {"beta", ov::element::f32, gate_shape},
                                              {"state", ov::element::f32, {batch, heads, width, width}}},
        std::vector<ov_gguf_test::TensorDesc>{},
        ov_gguf_test::TensorDesc{"out", ov::element::f32, qkv_shape},
        std::map<std::string, ov::Any>{{"split_outputs", true}, {"fuse_qk_l2norm", true}, {"qk_l2_norm_eps", eps}});
    auto values = std::make_shared<ov::frontend::gguf::TensorMap>();
    for (const auto& entry : decoder->get_model_inputs())
        values->emplace(entry.first, entry.second);
    values->at("q") = q;
    values->at("k") = k;
    const ov::frontend::gguf::NodeContext context(decoder, values);
    const auto outputs = ov::frontend::gguf::op::translate_gated_delta_net(context);
    const auto gdn = ov::as_type_ptr<internal::GatedDeltaNet>(outputs[0].get_node_shared_ptr());
    ASSERT_NE(gdn, nullptr);
    EXPECT_FALSE(gdn->get_fuse_qk_l2norm());
    EXPECT_GT(gdn->get_q_l2_norm_eps(), 0.f);
    EXPECT_GT(gdn->get_k_l2_norm_eps(), 0.f);
    const bool joined = layout == QkSlices::Adjacent || layout == QkSlices::Offset;
    EXPECT_EQ(ov::is_type<v1::Split>(gdn->get_input_node_shared_ptr(0)), joined);
    if (joined) {
        EXPECT_EQ(gdn->get_input_node_shared_ptr(0), gdn->get_input_node_shared_ptr(1));
        EXPECT_EQ(gdn->input_value(0).get_index(), 0u);
        EXPECT_EQ(gdn->input_value(1).get_index(), 1u);
    }
    const auto model =
        std::make_shared<ov::Model>(ov::OutputVector{gdn->input_value(0), gdn->input_value(1), q}, parameters);
    const auto reference = std::make_shared<ov::Model>(
        ov::OutputVector{ov::frontend::gguf::make_l2_norm(q, eps), ov::frontend::gguf::make_l2_norm(k, eps), q},
        parameters);
    ov::Core core;
    auto compiled = core.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32));
    auto compiled_ref = core.compile_model(reference, "CPU", ov::hint::inference_precision(ov::element::f32));
    for (const size_t tokens : {1u, 3u}) {
        const ov::Shape input_shape{batch, 1, tokens, features};
        std::vector<float> data(ov::shape_size(input_shape));
        for (size_t i = 0; i < data.size(); ++i)
            data[i] = i % 12 < 4 ? 0.f : (i % 12 < 8 ? 1e-8f : 0.25f * static_cast<float>(i % 7) - 0.5f);
        auto request = compiled.create_infer_request();
        auto ref_request = compiled_ref.create_infer_request();
        for (size_t i = 0; i < parameters.size(); ++i) {
            const auto tensor = ov_gguf_test::make_f32_tensor(input_shape, data);
            request.set_input_tensor(i, tensor);
            ref_request.set_input_tensor(i, tensor);
        }
        request.infer();
        ref_request.infer();
        for (size_t i = 0; i < 3; ++i) {
            const auto expected = ref_request.get_output_tensor(i);
            EXPECT_EQ(request.get_output_tensor(i).get_shape(), expected.get_shape());
            ov_gguf_test::expect_near(
                request.get_output_tensor(i),
                {expected.data<const float>(), expected.data<const float>() + expected.get_size()},
                1e-6f);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(GGUFOps,
                         GGUFQkNormalization,
                         testing::Values(QkSlices::Adjacent,
                                         QkSlices::Offset,
                                         QkSlices::Gap,
                                         QkSlices::Strided,
                                         QkSlices::SeparateSources,
                                         QkSlices::ExplicitShape));

std::shared_ptr<ov::Model> make_fused_qk_norm_gdn(int64_t heads, int64_t width) {
    const ov::PartialShape qkv_shape{1, -1, heads, width}, gate_shape{1, -1, heads, 1};
    const auto decoder = std::make_shared<ov_gguf_test::SingleOpDecoder>(
        "GGML_OP_GATED_DELTA_NET",
        std::vector<ov_gguf_test::TensorDesc>{{"q", ov::element::f32, qkv_shape},
                                              {"k", ov::element::f32, qkv_shape},
                                              {"v", ov::element::f32, qkv_shape},
                                              {"g", ov::element::f32, gate_shape},
                                              {"beta", ov::element::f32, gate_shape},
                                              {"state", ov::element::f32, {1, heads, width, width}}},
        std::vector<ov_gguf_test::TensorDesc>{},
        ov_gguf_test::TensorDesc{"out", ov::element::f32, qkv_shape},
        std::map<std::string, ov::Any>{{"split_outputs", true}, {"fuse_qk_l2norm", true}, {"qk_l2_norm_eps", 1e-6f}});
    auto values = std::make_shared<ov::frontend::gguf::TensorMap>();
    ov::ParameterVector parameters;
    for (const auto& name : {"q", "k", "v", "g", "beta", "state"}) {
        const auto& input = decoder->get_model_inputs().at(name);
        values->emplace(name, input);
        parameters.push_back(ov::as_type_ptr<v0::Parameter>(input));
    }
    const ov::frontend::gguf::NodeContext context(decoder, values);
    return std::make_shared<ov::Model>(ov::frontend::gguf::op::translate_gated_delta_net(context), parameters);
}

// The squared eps makes the L2-norm reciprocal of a zero row exceed the f16 range; low-precision
// kernels must keep it out of f16 so zero Q/K rows stay finite.
class GGUFGatedDeltaNetZeroQk : public testing::TestWithParam<int64_t> {};

TEST_P(GGUFGatedDeltaNetZeroQk, FusedQkNormKeepsZeroRowsFiniteInF16) {
    constexpr int64_t heads = 2, tokens = 2;
    const int64_t width = GetParam();
    ov::Core core;
    auto request =
        core.compile_model(make_fused_qk_norm_gdn(heads, width), "CPU", ov::hint::inference_precision(ov::element::f16))
            .create_infer_request();
    const ov::Shape qkv{1, tokens, heads, static_cast<size_t>(width)}, gate{1, tokens, heads, 1};
    std::vector<float> v(ov::shape_size(qkv));
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = 0.01f * static_cast<float>(i % 17) - 0.08f;
    request.set_input_tensor(0, ov_gguf_test::make_f32_tensor(qkv, std::vector<float>(v.size(), 0.f)));
    request.set_input_tensor(1, ov_gguf_test::make_f32_tensor(qkv, std::vector<float>(v.size(), 0.f)));
    request.set_input_tensor(2, ov_gguf_test::make_f32_tensor(qkv, v));
    request.set_input_tensor(3, ov_gguf_test::make_f32_tensor(gate, std::vector<float>(heads * tokens, -0.1f)));
    request.set_input_tensor(4, ov_gguf_test::make_f32_tensor(gate, std::vector<float>(heads * tokens, 0.5f)));
    const ov::Shape state{1, heads, static_cast<size_t>(width), static_cast<size_t>(width)};
    request.set_input_tensor(5, ov_gguf_test::make_f32_tensor(state, std::vector<float>(ov::shape_size(state), 0.1f)));
    request.infer();
    for (size_t i = 0; i < 2; ++i) {
        const auto out = request.get_output_tensor(i);
        const auto* data = out.data<const float>();
        for (size_t j = 0; j < out.get_size(); ++j)
            ASSERT_TRUE(std::isfinite(data[j])) << "output " << i << " element " << j;
    }
}

INSTANTIATE_TEST_SUITE_P(GGUFOps, GGUFGatedDeltaNetZeroQk, testing::Values(int64_t{128}, int64_t{256}));

TEST(GGUFOps, GatedDeltaNetFusesQkNormWithSquaredEps) {
    const ov::PartialShape qkv_shape{1, -1, 2, 4}, gate_shape{1, -1, 2, 1};
    const auto decoder = std::make_shared<ov_gguf_test::SingleOpDecoder>(
        "GGML_OP_GATED_DELTA_NET",
        std::vector<ov_gguf_test::TensorDesc>{{"q", ov::element::f32, qkv_shape},
                                              {"k", ov::element::f32, qkv_shape},
                                              {"v", ov::element::f32, qkv_shape},
                                              {"g", ov::element::f32, gate_shape},
                                              {"beta", ov::element::f32, gate_shape},
                                              {"state", ov::element::f32, {1, 2, 4, 4}}},
        std::vector<ov_gguf_test::TensorDesc>{},
        ov_gguf_test::TensorDesc{"out", ov::element::f32, qkv_shape},
        std::map<std::string, ov::Any>{{"split_outputs", true}, {"fuse_qk_l2norm", true}, {"qk_l2_norm_eps", 1e-6f}});
    auto values = std::make_shared<ov::frontend::gguf::TensorMap>();
    for (const auto& entry : decoder->get_model_inputs())
        values->emplace(entry.first, entry.second);
    const ov::frontend::gguf::NodeContext context(decoder, values);
    const auto outputs = ov::frontend::gguf::op::translate_gated_delta_net(context);
    const auto gdn = ov::as_type_ptr<internal::GatedDeltaNet>(outputs[0].get_node_shared_ptr());
    ASSERT_NE(gdn, nullptr);
    EXPECT_TRUE(gdn->get_fuse_qk_l2norm());
    EXPECT_FLOAT_EQ(gdn->get_q_l2_norm_eps(), 1e-12f);
    EXPECT_FLOAT_EQ(gdn->get_k_l2_norm_eps(), 1e-12f);
    EXPECT_EQ(gdn->input_value(0), values->at("q"));
    EXPECT_EQ(gdn->input_value(1), values->at("k"));
}

}  // namespace
