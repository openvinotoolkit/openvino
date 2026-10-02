// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Runtime registration, shared decoder blocks, and non-decoder architecture construction.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "gguf_writer.hpp"
#include "gtest/gtest.h"
#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/extension/decoder_transformation.hpp"
#include "openvino/frontend/gguf/adapt_to_genai.hpp"
#include "openvino/frontend/gguf/builder/graph_context.hpp"
#include "openvino/frontend/gguf/extension/architecture.hpp"
#include "openvino/frontend/gguf/frontend.hpp"
#include "openvino/frontend/gguf/make_stateful.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/openvino.hpp"
#include "openvino/util/file_util.hpp"

using namespace ov_gguf_test;
using ov::frontend::gguf::ArchitectureDefinition;
using ov::frontend::gguf::ArchitectureExtension;
using ov::frontend::gguf::BuildContext;
using ov::frontend::gguf::DecoderOptions;
using ov::frontend::gguf::GgufGraph;
using ov::frontend::gguf::GgufGraphContext;
using ov::frontend::gguf::GgufMetadata;
using ov::frontend::gguf::make_decoder_architecture;
using ov::frontend::gguf::ModelBuilder;
using ov::frontend::gguf::RegistrationMode;
using ov::frontend::gguf::RopeMode;

namespace {

// Remove the generated files even when a test fails.
class ScratchDir {
public:
    ScratchDir() : m_path(ov::test::utils::generateTestFilePrefix() + "_gguf_arch_ext") {
        ov::util::create_directory_recursive(std::filesystem::path(m_path));
    }
    ~ScratchDir() {
        std::error_code ec;
        std::filesystem::remove_all(std::filesystem::path(m_path), ec);
    }
    const std::string& path() const {
        return m_path;
    }

private:
    std::string m_path;
};

// Small nonzero fixture shared by registration, structure, and stateful numerical tests.
std::string write_decoder_gguf(const std::string& dir, const std::string& arch = "qwen3", uint32_t layers = 2) {
    const auto path = ov::util::path_join({dir, arch + ".gguf"}).string();
    GgufWriter writer;
    writer.kv_str("general.architecture", arch);
    writer.kv_u32(arch + ".block_count", layers);
    writer.kv_u32(arch + ".embedding_length", 8);
    writer.kv_u32(arch + ".attention.head_count", 2);
    writer.kv_u32(arch + ".attention.head_count_kv", 1);
    writer.kv_u32(arch + ".rope.dimension_count", 4);
    writer.kv_u32(arch + ".context_length", 32);
    writer.kv_f32(arch + ".attention.layer_norm_rms_epsilon", 1e-5f);
    writer.kv_f32(arch + ".rope.scaling.factor", 4.f);
    writer.kv_f32(arch + ".attn_logit_softcapping", 2.f);
    writer.kv_u32(arch + ".attention.sliding_window", 2);
    const auto weight = [&](const std::string& name, const std::vector<uint64_t>& shape, bool norm = false) {
        size_t count = 1;
        for (auto d : shape)
            count *= d;
        std::vector<float> values(count);
        for (size_t i = 0; i < count; ++i)
            values[i] = norm ? 1.f : 0.1f * std::sin(float(i + 1));
        writer.tensor(name, shape, values);
    };
    weight("token_embd.weight", {8, 16});
    weight("output_norm.weight", {8}, true);
    for (uint32_t layer = 0; layer < layers; ++layer) {
        const auto prefix = "blk." + std::to_string(layer) + ".";
        weight(prefix + "attn_norm.weight", {8}, true);
        weight(prefix + "attn_q.weight", {8, 8});
        weight(prefix + "attn_k.weight", {8, 4});
        weight(prefix + "attn_v.weight", {8, 4});
        weight(prefix + "attn_output.weight", {8, 8});
        weight(prefix + "attn_q_norm.weight", {4}, true);
        weight(prefix + "attn_k_norm.weight", {4}, true);
        weight(prefix + "ffn_norm.weight", {8}, true);
        weight(prefix + "ffn_gate.weight", {8, 12});
        weight(prefix + "ffn_up.weight", {8, 12});
        weight(prefix + "ffn_down.weight", {12, 8});
    }
    return writer.write(path) ? path : std::string{};
}

constexpr const char* kUnknownArch = "my-qwen3";

std::shared_ptr<ov::Model> convert_with(const std::string& path,
                                        const std::vector<ov::Extension::Ptr>& extensions = {}) {
    ov::frontend::gguf::FrontEnd frontend;
    for (const auto& extension : extensions)
        frontend.add_extension(extension);
    return frontend.convert(frontend.load(path));
}

std::map<std::string, size_t> op_histogram(const std::shared_ptr<ov::Model>& model) {
    std::map<std::string, size_t> hist;
    for (const auto& node : model->get_ops()) {
        ++hist[node->get_type_info().name];
    }
    return hist;
}

}  // namespace

TEST(GGUFArchitectureExtension, UnknownArchitectureIsRejectedWithoutAnExtension) {
    ScratchDir scratch;
    const auto path = write_decoder_gguf(scratch.path(), kUnknownArch);
    ASSERT_FALSE(path.empty());

    OV_EXPECT_THROW(convert_with(path),
                    ov::Exception,
                    testing::AllOf(testing::HasSubstr("does not support architecture"),
                                   testing::HasSubstr("ArchitectureExtension")));
}

TEST(GGUFArchitectureExtension, DecoderDefinitionMatchesBuiltInGraph) {
    ScratchDir scratch;

    const auto ref_path = write_decoder_gguf(scratch.path());
    ASSERT_FALSE(ref_path.empty());
    const auto reference = convert_with(ref_path, {});
    ASSERT_TRUE(reference);
    const auto expected_hist = op_histogram(reference);

    // Enable the identical topology under an unrecognized architecture name.
    ScratchDir scratch2;
    const auto path = write_decoder_gguf(scratch2.path(), kUnknownArch);
    ASSERT_FALSE(path.empty());
    const auto ext = std::make_shared<ArchitectureExtension>(kUnknownArch, RopeMode::Neox);

    const auto model = convert_with(path, {ext});
    ASSERT_TRUE(model);

    EXPECT_EQ(model->get_ops().size(), reference->get_ops().size());
    EXPECT_EQ(model->inputs().size(), reference->inputs().size());
    EXPECT_EQ(op_histogram(model), expected_hist)
        << "an architecture enabled by extension must build the same graph as the built-in path";
}

TEST(GGUFArchitectureExtension, DecoderRopeModeReachesTheBuilder) {
    ScratchDir scratch;
    const auto path = write_decoder_gguf(scratch.path(), kUnknownArch);
    ASSERT_FALSE(path.empty());

    const auto neox = convert_with(path, {std::make_shared<ArchitectureExtension>(kUnknownArch, RopeMode::Neox)});
    ASSERT_TRUE(neox);

    const auto normal = convert_with(path, {std::make_shared<ArchitectureExtension>(kUnknownArch, RopeMode::Normal)});
    ASSERT_TRUE(normal);

    // NORMAL and NEOX lower to different rotation subgraphs, so the two graphs must differ.
    EXPECT_NE(op_histogram(neox), op_histogram(normal))
        << "the registered RoPE mode did not affect the graph, so it is not reaching the builder";
}

TEST(GGUFArchitectureExtension, DecoderOptionsSelectTheActivation) {
    ScratchDir scratch;
    const auto path = write_decoder_gguf(scratch.path(), kUnknownArch);
    ASSERT_FALSE(path.empty());

    const auto plain = convert_with(path, {std::make_shared<ArchitectureExtension>(kUnknownArch, RopeMode::Neox)});
    ASSERT_TRUE(plain);

    // SwiGLU and GeGLU use the same tensor layout; the option must select the activation.
    auto hooked_ext = std::make_shared<ArchitectureExtension>(
        make_decoder_architecture(kUnknownArch, RopeMode::Neox, [](const GgufMetadata&) {
            DecoderOptions options;
            options.geglu = true;
            return options;
        }));
    const auto hooked = convert_with(path, {hooked_ext});
    ASSERT_TRUE(hooked);

    EXPECT_NE(op_histogram(plain), op_histogram(hooked)) << "decoder options did not change the activation";
}

namespace {

// Custom topology reuses the very same decoder blocks as the built-in family.
class Qwen3Builder : public ModelBuilder {
public:
    explicit Qwen3Builder(const BuildContext& ctx) : m_ctx(ctx) {}
    std::shared_ptr<GgufGraph> build() override {
        GgufGraphContext ctx(m_ctx);
        const auto dimensions = ctx.configure_decoder(RopeMode::Neox);
        auto tensors = ctx.tensors();
        auto cur = ctx.build_inp_embd(tensors.require("token_embd.weight"));
        ctx.build_inp_pos();
        ctx.build_attn_inp_kv();
        for (int layer = 0; layer < dimensions.layers; ++layer) {
            auto norm = ctx.build_norm(cur, tensors.layer(layer, "attn_norm.weight"), dimensions.norm_epsilon);
            cur = ctx.node("GGML_OP_ADD", {ctx.decoder_attention(layer, norm), cur});
            norm = ctx.build_norm(cur, tensors.layer(layer, "ffn_norm.weight"), dimensions.norm_epsilon);
            cur = ctx.node("GGML_OP_ADD", {ctx.decoder_ffn(layer, norm), cur});
        }
        cur = ctx.build_norm(cur, tensors.require("output_norm.weight"), dimensions.norm_epsilon);
        auto output = tensors("output.weight");
        if (!output)
            output = tensors.require("token_embd.weight");
        ctx.set_output(ctx.node("GGML_OP_MUL_MAT", {output, cur}));
        return ctx.finish();
    }

private:
    BuildContext m_ctx;
};

constexpr uint32_t kVisEmbd = 32;
constexpr uint32_t kVisLayers = 2;
constexpr uint32_t kVisHeads = 4;
constexpr uint32_t kVisPatches = 16;
constexpr uint32_t kVisFF = 64;
constexpr uint32_t kVisProjDim = 48;

// Vision and audio files share architecture "clip"; metadata distinguishes them.
std::string write_vision_gguf(const std::string& dir) {
    GgufWriter w;
    w.kv_str("general.architecture", "clip");
    w.kv_bool("clip.has_vision_encoder", true);
    w.kv_u32("clip.vision.embedding_length", kVisEmbd);
    w.kv_u32("clip.vision.block_count", kVisLayers);
    w.kv_u32("clip.vision.attention.head_count", kVisHeads);
    w.kv_u32("clip.vision.feed_forward_length", kVisFF);
    w.kv_u32("clip.vision.projection_dim", kVisProjDim);
    w.kv_f32("clip.vision.attention.layer_norm_epsilon", 1e-5f);

    // Dims are in GGUF on-disk order (fastest-varying first).
    w.tensor("v.position_embd.weight", {kVisEmbd, kVisPatches});
    for (uint32_t il = 0; il < kVisLayers; ++il) {
        const std::string p = "v.blk." + std::to_string(il) + ".";
        w.tensor(p + "ln1.weight", {kVisEmbd});
        w.tensor(p + "attn_q.weight", {kVisEmbd, kVisEmbd});
        w.tensor(p + "attn_k.weight", {kVisEmbd, kVisEmbd});
        w.tensor(p + "attn_v.weight", {kVisEmbd, kVisEmbd});
        w.tensor(p + "attn_out.weight", {kVisEmbd, kVisEmbd});
        w.tensor(p + "ln2.weight", {kVisEmbd});
        w.tensor(p + "ffn_up.weight", {kVisEmbd, kVisFF});
        w.tensor(p + "ffn_down.weight", {kVisFF, kVisEmbd});
    }
    w.tensor("v.post_ln.weight", {kVisEmbd});
    w.tensor("mm.0.weight", {kVisEmbd, kVisProjDim});

    const std::string path = ov::util::path_join({dir, "mmproj-vision.gguf"}).string();
    return w.write(path) ? path : std::string{};
}

// Patch embeddings -> non-causal transformer blocks -> projector. No decoder configuration or KV cache.
class VisionEncoderBuilder : public ModelBuilder {
public:
    explicit VisionEncoderBuilder(const BuildContext& ctx) : m_ctx(ctx) {}

    std::shared_ptr<GgufGraph> build() override {
        GgufGraphContext ctx(m_ctx);
        const auto& meta = ctx.metadata();
        auto tensors = ctx.tensors();

        const int64_t n_embd = meta.get_int("clip.vision.embedding_length").value_or(0);
        const int n_layer = static_cast<int>(meta.get_int("clip.vision.block_count").value_or(0));
        const int64_t n_head = meta.get_int("clip.vision.attention.head_count").value_or(0);
        const float eps = static_cast<float>(meta.get_float("clip.vision.attention.layer_norm_epsilon").value_or(1e-5));
        const int64_t n_patches = kVisPatches;
        const int64_t head_size = n_embd / n_head;

        // One embedding per image patch.
        auto cur = ctx.add_input("inp_patches", ov::element::f32, ov::PartialShape({1, 1, n_patches, n_embd}));
        cur = ctx.node("GGML_OP_ADD", {cur, tensors.require("v.position_embd.weight")});

        for (int il = 0; il < n_layer; ++il) {
            const std::string p = "v.blk." + std::to_string(il) + ".";
            auto residual = cur;

            cur = ctx.build_norm(cur, tensors.require(p + "ln1.weight"), eps);

            // Non-causal attention: softmax(Q @ K^T / sqrt(head_size)) @ V.
            auto q = ctx.node("GGML_OP_MUL_MAT", {tensors.require(p + "attn_q.weight"), cur});
            auto k = ctx.node("GGML_OP_MUL_MAT", {tensors.require(p + "attn_k.weight"), cur});
            auto v = ctx.node("GGML_OP_MUL_MAT", {tensors.require(p + "attn_v.weight"), cur});

            for (auto* value : {&q, &k, &v}) {
                *value = ctx.node("GGML_OP_RESHAPE",
                                  {*value},
                                  6,
                                  {{"reshape_target", std::vector<int64_t>{1, n_patches, n_head, head_size}}});
            }
            // Q/K contract over head_size; V contracts over patches.
            q = ctx.node("GGML_OP_PERMUTE", {q}, 1, {{"perm", std::vector<int64_t>{0, 2, 1, 3}}});
            k = ctx.node("GGML_OP_PERMUTE", {k}, 1, {{"perm", std::vector<int64_t>{0, 2, 1, 3}}});
            v = ctx.node("GGML_OP_PERMUTE", {v}, 1, {{"perm", std::vector<int64_t>{0, 2, 3, 1}}});

            auto kq = ctx.node("GGML_OP_MUL_MAT", {k, q});
            kq = ctx.node("GGML_OP_SCALE",
                          {kq},
                          0,
                          {{"scale", 1.0f / std::sqrt(static_cast<float>(head_size))}, {"bias", 0.0f}});
            kq = ctx.node("GGML_OP_SOFT_MAX", {kq});
            auto kqv = ctx.node("GGML_OP_MUL_MAT", {v, kq});

            cur = ctx.node("GGML_OP_PERMUTE", {kqv}, 1, {{"perm", std::vector<int64_t>{0, 2, 1, 3}}});
            cur = ctx.node("GGML_OP_RESHAPE",
                           {ctx.node("GGML_OP_CONT", {cur}, 1)},
                           6,
                           {{"reshape_target", std::vector<int64_t>{1, 1, n_patches, n_embd}}});
            cur = ctx.node("GGML_OP_MUL_MAT", {tensors.require(p + "attn_out.weight"), cur});
            cur = ctx.node("GGML_OP_ADD", {cur, residual});

            residual = cur;
            cur = ctx.build_norm(cur, tensors.require(p + "ln2.weight"), eps);
            cur = ctx.node("GGML_OP_MUL_MAT", {tensors.require(p + "ffn_up.weight"), cur});
            cur = ctx.node("GGML_UNARY_OP_GELU", {cur});
            cur = ctx.node("GGML_OP_MUL_MAT", {tensors.require(p + "ffn_down.weight"), cur});
            cur = ctx.node("GGML_OP_ADD", {cur, residual});
        }

        cur = ctx.build_norm(cur, tensors.require("v.post_ln.weight"), eps);
        cur = ctx.node("GGML_OP_MUL_MAT", {tensors.require("mm.0.weight"), cur});
        ctx.set_output(cur);
        return ctx.finish();
    }

private:
    BuildContext m_ctx;
};

}  // namespace

TEST(GGUFArchitectureExtension, NonDecoderFileIsRejectedWithoutAnExtension) {
    ScratchDir scratch;
    const auto path = write_vision_gguf(scratch.path());
    ASSERT_FALSE(path.empty());

    OV_EXPECT_THROW(convert_with(path),
                    ov::Exception,
                    testing::AllOf(testing::HasSubstr("decoder family"), testing::HasSubstr("ArchitectureExtension")));
}

TEST(GGUFArchitectureExtension, NonDecoderFamilyConvertsEndToEnd) {
    ScratchDir scratch;
    const auto path = write_vision_gguf(scratch.path());
    ASSERT_FALSE(path.empty());

    // Match both architecture "clip" and its vision flag.
    auto ext = std::make_shared<ArchitectureExtension>(
        ArchitectureDefinition{"clip.vision",
                               "clip",
                               [](const BuildContext& c) {
                                   return std::make_shared<VisionEncoderBuilder>(c);
                               },
                               [](const GgufMetadata& m) {
                                   return m.get_bool("clip.has_vision_encoder").value_or(false);
                               }});

    const auto model = convert_with(path, {ext});
    ASSERT_TRUE(model);

    EXPECT_EQ(model->inputs().size(), 1u) << "a vision encoder takes patches and nothing else -- "
                                          << "no tokens, no positions, no KV caches";
    EXPECT_EQ(model->inputs()[0].get_any_name(), "inp_patches");
    EXPECT_EQ(model->outputs().size(), 1u) << "no KV caches means no cache outputs";

    auto hist = op_histogram(model);
    EXPECT_EQ(hist["Softmax"], kVisLayers) << "one non-causal attention softmax per encoder block";
    EXPECT_GT(hist["MatMul"], 0u);
    EXPECT_EQ(hist["Parameter"], 1u);
}

TEST(GGUFArchitectureExtension, AmbiguousClaimIsReportedNotGuessed) {
    ScratchDir scratch;
    const auto path = write_vision_gguf(scratch.path());
    ASSERT_FALSE(path.empty());

    const auto claim_everything = [](const GgufMetadata&) {
        return true;
    };
    const auto factory = [](const BuildContext& c) {
        return std::make_shared<VisionEncoderBuilder>(c);
    };
    auto first = std::make_shared<ArchitectureExtension>(
        ArchitectureDefinition{"first-claimant", "clip", factory, claim_everything});
    auto second = std::make_shared<ArchitectureExtension>(
        ArchitectureDefinition{"second-claimant", "clip", factory, claim_everything});

    OV_EXPECT_THROW(convert_with(path, {first, second}),
                    ov::Exception,
                    testing::AllOf(testing::HasSubstr("first-claimant"), testing::HasSubstr("second-claimant")));
}

TEST(GGUFArchitectureExtension, SharedDecoderBlocksMatchNumericallyAcrossPrefillAndDecode) {
    ScratchDir scratch;
    const auto path = write_decoder_gguf(scratch.path(), "qwen3");
    ASSERT_FALSE(path.empty());
    const auto load = [&](bool custom) {
        ov::frontend::gguf::FrontEnd frontend;
        if (custom) {
            frontend.add_extension(std::make_shared<ArchitectureExtension>(
                ArchitectureDefinition{"qwen3",
                                       "qwen3",
                                       [](const BuildContext& ctx) {
                                           return std::make_shared<Qwen3Builder>(ctx);
                                       }},
                RegistrationMode::Replace));
        }
        frontend.add_extension(std::make_shared<ov::frontend::DecoderTransformationExtension>(
            ov::frontend::gguf::pass::GGUFMakeStateful()));
        frontend.add_extension(
            std::make_shared<ov::frontend::DecoderTransformationExtension>(ov::frontend::gguf::pass::AdaptToGenAI()));
        return frontend.convert(frontend.load(path));
    };
    auto builtin = load(false);
    auto custom = load(true);
    ASSERT_EQ(builtin->get_variables().size(), 4);
    ASSERT_EQ(custom->get_variables().size(), 4);
    ov::Core core;
    const auto request = [&](const std::shared_ptr<ov::Model>& model) {
        return core
            .compile_model(model,
                           "CPU",
                           ov::hint::inference_precision(ov::element::f32),
                           ov::num_streams(1),
                           ov::inference_num_threads(4),
                           ov::hint::dynamic_quantization_group_size(0),
                           ov::hint::kv_cache_precision(ov::element::f16))
            .create_infer_request();
    };
    auto reference = request(builtin);
    auto extension = request(custom);
    size_t past = 0;
    for (size_t tokens : {3u, 1u, 2u}) {
        ov::Tensor ids(ov::element::i64, {1, tokens});
        ov::Tensor pos(ov::element::i64, {1, tokens});
        for (size_t i = 0; i < tokens; ++i) {
            ids.data<int64_t>()[i] = static_cast<int64_t>((past + i + 1) % 16);
            pos.data<int64_t>()[i] = static_cast<int64_t>(past + i);
        }
        ov::Tensor mask(ov::element::i64, {1, past + tokens});
        std::fill_n(mask.data<int64_t>(), mask.get_size(), 1);
        ov::Tensor beam(ov::element::i32, {1});
        *beam.data<int32_t>() = 0;
        for (auto* infer : {&reference, &extension}) {
            infer->set_tensor("input_ids", ids);
            infer->set_tensor("position_ids", pos);
            infer->set_tensor("attention_mask", mask);
            infer->set_tensor("beam_idx", beam);
            infer->infer();
            const auto states = infer->query_state();
            ASSERT_EQ(states.size(), 4);
            for (const auto& state : states)
                EXPECT_EQ(state.get_state().get_size(), (past + tokens) * 4);
        }
        auto a = reference.get_output_tensor();
        auto b = extension.get_output_tensor();
        ASSERT_GE(a.get_size(), 16);
        ASSERT_GE(b.get_size(), 16);
        const auto* last_a = a.data<float>() + a.get_size() - 16;
        const auto* last_b = b.data<float>() + b.get_size() - 16;
        for (size_t i = 0; i < 16; ++i) {
            ASSERT_TRUE(std::isfinite(last_a[i]));
            EXPECT_NEAR(last_a[i], last_b[i], 1e-5f);
        }
        past += tokens;
    }
}

namespace {
class CustomOpBuilder : public ModelBuilder {
public:
    explicit CustomOpBuilder(const BuildContext& context) : m_context(context) {}

    std::shared_ptr<GgufGraph> build() override {
        GgufGraphContext graph(m_context);
        auto input = graph.add_input("features", ov::element::f32, {1, 1, -1, 4});
        auto expanded = graph.node("TEST_EXPAND_FEATURES", {input});
        // The following operation can query the converter's inferred width immediately.
        graph.set_output(graph.node("GGML_OP_RESHAPE",
                                    {expanded},
                                    6,
                                    {{"reshape_target", std::vector<int64_t>{1, 1, -1, expanded.ne(0)}}}));
        return graph.finish();
    }

private:
    BuildContext m_context;
};
}  // namespace

TEST(GGUFArchitectureExtension, ConverterRegisteredAfterLoadInfersBuilderValues) {
    ScratchDir scratch;
    const auto path = write_decoder_gguf(scratch.path(), kUnknownArch);
    ASSERT_FALSE(path.empty());
    ov::frontend::gguf::FrontEnd frontend;
    frontend.add_extension(std::make_shared<ArchitectureExtension>(
        ArchitectureDefinition{kUnknownArch, kUnknownArch, [](const BuildContext& context) {
                                   return std::make_shared<CustomOpBuilder>(context);
                               }}));
    auto input_model = frontend.load(path);
    size_t calls = 0;
    const auto register_converter = [&](size_t copies) {
        frontend.add_extension(std::make_shared<ov::frontend::ConversionExtension>(
            "TEST_EXPAND_FEATURES",
            [&, copies](const ov::frontend::NodeContext& context) {
                ++calls;
                return ov::OutputVector{
                    std::make_shared<ov::op::v0::Concat>(ov::OutputVector(copies, context.get_input(0)), 3)};
            }));
    };
    register_converter(2);
    auto first = frontend.convert(input_model);
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(first->output().get_partial_shape(), (ov::PartialShape{1, 1, -1, 8}));
    register_converter(3);
    auto second = frontend.convert(input_model);
    EXPECT_EQ(calls, 2);
    EXPECT_EQ(second->output().get_partial_shape(), (ov::PartialShape{1, 1, -1, 12}));
    EXPECT_EQ(first->output().get_partial_shape(), (ov::PartialShape{1, 1, -1, 8}));
    for (size_t tokens : {1u, 3u}) {
        ov::Tensor data(ov::element::f32, {1, 1, tokens, 4});
        std::iota(data.data<float>(), data.data<float>() + data.get_size(), 0.f);
        ov::TensorVector result{ov::Tensor(ov::element::f32, {1, 1, tokens, 12})};
        ASSERT_TRUE(second->evaluate(result, {data}));
        EXPECT_EQ(result[0].get_shape(), (ov::Shape{1, 1, tokens, 12}));
        for (size_t t = 0; t < tokens; ++t) {
            for (size_t i = 0; i < 12; ++i) {
                EXPECT_EQ(result[0].data<float>()[12 * t + i], data.data<float>()[4 * t + i % 4]);
            }
        }
    }
}
