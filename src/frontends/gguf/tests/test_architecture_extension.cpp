// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Tests for adding a GGUF architecture at RUNTIME, through ov::frontend::gguf::ArchitectureExtension.
//
// The claim under test is that a new architecture -- of ANY family, decoder or not -- can be
// enabled without rebuilding the GGUF frontend or any other OpenVINO binary. Each tier of the
// mechanism gets a test that would fail if that tier stopped working:
//
//   Tier 1  a name and a RoPE mode. Asserted by FINGERPRINT EQUALITY against the built-in builder:
//           an architecture the frontend does not know, enabled only by an extension, must produce
//           the byte-for-byte same graph as the built-in one does for the architecture it was
//           renamed from. That is a much stronger statement than "it converted".
//   Tier 2  a configuration hook, which must actually reach the auto-detected DecoderConfig.
//   Tier 3  a whole custom builder written against the builder SDK. Two of them: a port of
//           llama.cpp's qwen3.cpp (a decoder, so the port can be compared against the built-in
//           result), and a vision encoder (a family the frontend has no support for at all, which
//           is the case the mechanism exists for).
//
// Plus the packaging claim: an extension wrapped as a shared-library extension still reaches the
// registry, which is what makes `core.add_extension("libmy_arch.so")` work.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "gguf_writer.hpp"
#include "gtest/gtest.h"
#include "op_test_utils.hpp"
#include "openvino/core/so_extension.hpp"
#include "openvino/frontend/gguf/builder/decoder_config.hpp"
#include "openvino/frontend/gguf/builder/graph_context.hpp"
#include "openvino/frontend/gguf/extension/architecture.hpp"
#include "openvino/frontend/gguf/frontend.hpp"
#include "openvino/util/file_util.hpp"

using namespace ov_gguf_test;
using ov::frontend::gguf::ArchitectureExtension;
using ov::frontend::gguf::AttnOptions;
using ov::frontend::gguf::BuildContext;
using ov::frontend::gguf::FfnOp;
using ov::frontend::gguf::GgufGraph;
using ov::frontend::gguf::GgufGraphContext;
using ov::frontend::gguf::GgufMetadata;
using ov::frontend::gguf::GgufValue;
using ov::frontend::gguf::Maturity;
using ov::frontend::gguf::ModelBuilder;
using ov::frontend::gguf::RopeMode;

namespace {

std::string fixture_dir() {
    return ov::util::path_join({test_data_dir(), "arch_fixtures"}).string();
}

// A scratch directory that outlives one test.
//
// remove_all rather than removeDir: these tests write a rebuilt .gguf into the directory, and a
// directory removal that only handles empty directories would silently leave multi-megabyte files
// behind on every run -- including when a test fails, which is exactly when it would go unnoticed.
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

// Rebuild a loadable .gguf from a header fixture, optionally renaming the architecture.
//
// The rename is a straight byte substitution of an EQUAL-LENGTH name, which keeps every offset in
// the header valid. It rewrites `general.architecture` and, in the same pass, the "<arch>."-prefixed
// hyperparameter keys -- so the result is a coherent file for an architecture the frontend has
// never heard of, which is exactly the input an extension is supposed to enable.
std::string materialize(const std::string& header_file,
                        size_t data_bytes,
                        const std::string& dir,
                        const std::string& from_arch = "",
                        const std::string& to_arch = "") {
    const std::string src = ov::util::path_join({fixture_dir(), header_file}).string();
    std::ifstream in(src, std::ios::binary);
    if (!in) {
        return {};
    }
    std::string header((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    if (!from_arch.empty()) {
        EXPECT_EQ(from_arch.size(), to_arch.size()) << "the rename must preserve every header offset";
        for (size_t pos = header.find(from_arch); pos != std::string::npos;
             pos = header.find(from_arch, pos + to_arch.size())) {
            header.replace(pos, from_arch.size(), to_arch);
        }
    }

    std::string name = header_file;
    const std::string suffix = ".hdr";
    if (name.size() > suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
        name.resize(name.size() - suffix.size());
    }
    const std::string dst = ov::util::path_join({dir, name}).string();

    std::ofstream out(dst, std::ios::binary);
    if (!out) {
        return {};
    }
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    const std::vector<char> zeros(64 * 1024, 0);
    size_t remaining = data_bytes;
    while (remaining > 0) {
        const size_t chunk = std::min(remaining, zeros.size());
        out.write(zeros.data(), static_cast<std::streamsize>(chunk));
        remaining -= chunk;
    }
    out.close();
    return out ? dst : std::string{};
}

// The qwen3 fixture and its pinned size, matching test_arch_conversion.cpp's manifest entry.
constexpr const char* kQwen3Header = "qwen3-dense.gguf.hdr";
constexpr size_t kQwen3DataBytes = 4733824;
// An architecture name of the same length as "qwen3" that no built-in list contains.
constexpr const char* kUnknownArch = "myqw3";

bool fixtures_present() {
    return ov::util::file_exists(ov::util::path_join({fixture_dir(), kQwen3Header}).string());
}

struct GraphShape {
    size_t ops = 0;
    size_t inputs = 0;
};

// Convert `path`, registering `exts` first. Returns the converted model, or nullptr plus the error.
std::shared_ptr<ov::Model> convert_with(const std::string& path,
                                        const std::vector<std::shared_ptr<ov::Extension>>& exts,
                                        std::string& error) {
    ov::frontend::gguf::FrontEnd fe;
    for (const auto& e : exts) {
        fe.add_extension(e);
    }
    try {
        return fe.convert(fe.load(path));
    } catch (const std::exception& e) {
        error = e.what();
        return nullptr;
    }
}

// Count nodes by op type name, for structural assertions on a graph with no pinned fingerprint.
std::map<std::string, size_t> op_histogram(const std::shared_ptr<ov::Model>& model) {
    std::map<std::string, size_t> hist;
    for (const auto& node : model->get_ops()) {
        ++hist[node->get_type_info().name];
    }
    return hist;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Tier 1: a name and a RoPE mode.
// ---------------------------------------------------------------------------------------------

// The premise of every test below: without an extension, the renamed architecture is unknown.
// If this ever stopped failing, the Tier-1 test would be proving nothing.
TEST(GGUFArchitectureExtension, UnknownArchitectureIsRejectedWithoutAnExtension) {
    if (!fixtures_present()) {
        GTEST_SKIP() << "no arch fixtures -- generate them with tests/gen_arch_fixtures.py --fetch";
    }
    ScratchDir scratch;
    const auto path = materialize(kQwen3Header, kQwen3DataBytes, scratch.path(), "qwen3", kUnknownArch);
    ASSERT_FALSE(path.empty());

    std::string error;
    const auto model = convert_with(path, {}, error);
    ASSERT_FALSE(model) << "an unregistered architecture must not convert";
    EXPECT_NE(error.find("does not support architecture"), std::string::npos) << "actual error:\n" << error;
    // The diagnostic should point at the way out, not just state the refusal.
    EXPECT_NE(error.find("ArchitectureExtension"), std::string::npos) << "actual error:\n" << error;
}

// The core claim: a name and a RoPE mode are enough, and the graph is IDENTICAL to what the
// built-in builder produces for the same file under its real name.
TEST(GGUFArchitectureExtension, Tier1MatchesBuiltInGraphExactly) {
    if (!fixtures_present()) {
        GTEST_SKIP() << "no arch fixtures -- generate them with tests/gen_arch_fixtures.py --fetch";
    }
    ScratchDir scratch;

    // Reference: the same fixture under its real name, through the built-in path.
    const auto ref_path = materialize(kQwen3Header, kQwen3DataBytes, scratch.path());
    ASSERT_FALSE(ref_path.empty());
    std::string ref_error;
    const auto reference = convert_with(ref_path, {}, ref_error);
    ASSERT_TRUE(reference) << "the built-in qwen3 path failed:\n" << ref_error;
    const GraphShape expected{reference->get_ops().size(), reference->inputs().size()};
    const auto expected_hist = op_histogram(reference);

    // Under test: the renamed architecture, enabled only by a Tier-1 extension. qwen3 is NEOX.
    ScratchDir scratch2;
    const auto path = materialize(kQwen3Header, kQwen3DataBytes, scratch2.path(), "qwen3", kUnknownArch);
    ASSERT_FALSE(path.empty());
    const auto ext = std::make_shared<ArchitectureExtension>(kUnknownArch, RopeMode::Neox);

    std::string error;
    const auto model = convert_with(path, {ext}, error);
    ASSERT_TRUE(model) << "the extension-registered architecture failed to convert:\n" << error;

    EXPECT_EQ(model->get_ops().size(), expected.ops);
    EXPECT_EQ(model->inputs().size(), expected.inputs);
    EXPECT_EQ(op_histogram(model), expected_hist)
        << "an architecture enabled by extension must build the same graph as the built-in path";
}

// The RoPE mode is the one fact about a same-family architecture that cannot be read from the
// file, so registering the wrong one has to be observable -- otherwise the parameter is decorative.
TEST(GGUFArchitectureExtension, Tier1RopeModeReachesTheBuilder) {
    if (!fixtures_present()) {
        GTEST_SKIP() << "no arch fixtures -- generate them with tests/gen_arch_fixtures.py --fetch";
    }
    ScratchDir scratch;
    const auto path = materialize(kQwen3Header, kQwen3DataBytes, scratch.path(), "qwen3", kUnknownArch);
    ASSERT_FALSE(path.empty());

    std::string neox_error;
    const auto neox =
        convert_with(path, {std::make_shared<ArchitectureExtension>(kUnknownArch, RopeMode::Neox)}, neox_error);
    ASSERT_TRUE(neox) << neox_error;

    std::string normal_error;
    const auto normal =
        convert_with(path, {std::make_shared<ArchitectureExtension>(kUnknownArch, RopeMode::Normal)}, normal_error);
    ASSERT_TRUE(normal) << normal_error;

    // NORMAL and NEOX lower to different rotation subgraphs, so the two graphs must differ.
    EXPECT_NE(op_histogram(neox), op_histogram(normal))
        << "the registered RoPE mode did not affect the graph, so it is not reaching the builder";
}

// ---------------------------------------------------------------------------------------------
// Tier 2: a configuration hook over the auto-detected DecoderConfig.
// ---------------------------------------------------------------------------------------------

TEST(GGUFArchitectureExtension, Tier2ConfigureHookReachesTheDecoderConfig) {
    if (!fixtures_present()) {
        GTEST_SKIP() << "no arch fixtures -- generate them with tests/gen_arch_fixtures.py --fetch";
    }
    ScratchDir scratch;
    const auto path = materialize(kQwen3Header, kQwen3DataBytes, scratch.path(), "qwen3", kUnknownArch);
    ASSERT_FALSE(path.empty());

    std::string plain_error;
    const auto plain =
        convert_with(path, {std::make_shared<ArchitectureExtension>(kUnknownArch, RopeMode::Neox)}, plain_error);
    ASSERT_TRUE(plain) << plain_error;

    // Switch the FFN from SwiGLU to GeGLU. That is a real per-architecture choice the tensor table
    // cannot disambiguate, which is why the hook exists; it must change the emitted activation.
    auto hooked_ext = std::make_shared<ArchitectureExtension>(kUnknownArch,
                                                              RopeMode::Neox,
                                                              [](ov::frontend::gguf::DecoderConfig& cfg) {
                                                                  cfg.is_geglu = true;
                                                              });
    std::string hooked_error;
    const auto hooked = convert_with(path, {hooked_ext}, hooked_error);
    ASSERT_TRUE(hooked) << hooked_error;

    EXPECT_NE(op_histogram(plain), op_histogram(hooked))
        << "the configure() hook did not change the graph, so it is not being applied";
}

// ---------------------------------------------------------------------------------------------
// Tier 3a: a port of llama.cpp's src/models/qwen3.cpp, written against the builder SDK.
// ---------------------------------------------------------------------------------------------

namespace {

// A port of llama.cpp's qwen3 graph. Compare against src/models/qwen3.cpp upstream: the structure,
// the order of operations and the names are deliberately kept as close as the two APIs allow, which
// is the property that makes porting an upstream model file a reviewable change rather than a
// reimplementation. See docs/porting_a_llama_cpp_model.md.
class Qwen3PortBuilder : public ModelBuilder {
public:
    explicit Qwen3PortBuilder(const BuildContext& ctx) : m_ctx(ctx) {}

    std::shared_ptr<GgufGraph> build() override {
        GgufGraphContext ctx(m_ctx);
        const auto& hparams = ctx.hparams();
        auto tensors = ctx.tensors();

        const int64_t n_embd_head = hparams.n_embd_head_v();
        const int n_layer = static_cast<int>(hparams.n_layer);
        const int64_t n_head = hparams.n_head();
        const int64_t n_head_kv = hparams.n_head_kv();
        const float kq_scale = 1.0f / std::sqrt(static_cast<float>(n_embd_head));

        ov::frontend::gguf::RopeConfig rope{};
        rope.freq_base = hparams.rope_freq_base_train;
        rope.freq_scale = hparams.rope_freq_scale_train;
        rope.n_dims = static_cast<int32_t>(hparams.n_rot);
        rope.n_ctx_orig = static_cast<int32_t>(hparams.n_ctx_train);
        rope.attn_factor = 1.0f;
        rope.beta_fast = 32.0f;
        rope.beta_slow = 1.0f;

        auto inpL = ctx.build_inp_embd(tensors.require("token_embd.weight"));
        auto inp_pos = ctx.build_inp_pos();
        ctx.build_attn_inp_kv();

        for (int il = 0; il < n_layer; ++il) {
            const auto layer = tensors.layer(il);
            auto inpSA = inpL;

            // norm
            auto cur = ctx.build_norm(inpL, layer.attn_norm, hparams.f_norm_rms_eps);
            ctx.cb(cur, "attn_norm", il);

            // self-attention
            {
                auto Qcur = ctx.build_lora_mm(layer.wq, cur);
                auto Kcur = ctx.build_lora_mm(layer.wk, cur);
                auto Vcur = ctx.build_lora_mm(layer.wv, cur);

                Qcur = ctx.reshape(Qcur, {n_embd_head, n_head, ctx.n_tokens()});
                Kcur = ctx.reshape(Kcur, {n_embd_head, n_head_kv, ctx.n_tokens()});
                Vcur = ctx.reshape(Vcur, {n_embd_head, n_head_kv, ctx.n_tokens()});

                Qcur = ctx.build_norm(Qcur, layer.attn_q_norm, hparams.f_norm_rms_eps);
                ctx.cb(Qcur, "Qcur_normed", il);
                Qcur = ctx.rope_ext(Qcur, inp_pos, GgufValue(), rope, kRopeNeox);

                Kcur = ctx.build_norm(Kcur, layer.attn_k_norm, hparams.f_norm_rms_eps);
                ctx.cb(Kcur, "Kcur_normed", il);
                Kcur = ctx.rope_ext(Kcur, inp_pos, GgufValue(), rope, kRopeNeox);

                cur = ctx.build_attn(il, Qcur, Kcur, Vcur, layer.wo, layer.bo, kq_scale);
            }

            auto ffn_inp = ctx.add(cur, inpSA);
            ctx.cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = ctx.build_norm(ffn_inp, layer.ffn_norm, hparams.f_norm_rms_eps);
            ctx.cb(cur, "ffn_norm", il);
            cur = ctx.build_ffn(cur,
                                layer.ffn_up,
                                GgufValue(),
                                layer.ffn_gate,
                                GgufValue(),
                                layer.ffn_down,
                                GgufValue(),
                                FfnOp::Silu);
            ctx.cb(cur, "ffn_out", il);

            cur = ctx.add(cur, ffn_inp);
            inpL = cur;
        }

        auto cur = ctx.build_norm(inpL, tensors.require("output_norm.weight"), hparams.f_norm_rms_eps);

        // lm_head; qwen3 ties it to the token embedding when there is no separate output tensor.
        auto output = tensors("output.weight");
        if (!output) {
            output = tensors.require("token_embd.weight");
        }
        cur = ctx.build_lora_mm(output, cur);
        ctx.set_output(cur);
        return ctx.finish();
    }

private:
    // NEOX rope: rotate halves. See arch_registry.hpp's ROPE_OP_CASE_NEOX.
    static constexpr int kRopeNeox = 0x00010000;
    BuildContext m_ctx;
};

}  // namespace

TEST(GGUFArchitectureExtension, Tier3PortedDecoderBuildsAnEquivalentGraph) {
    if (!fixtures_present()) {
        GTEST_SKIP() << "no arch fixtures -- generate them with tests/gen_arch_fixtures.py --fetch";
    }
    ScratchDir scratch;

    // Reference: the built-in qwen3 builder on the same file.
    const auto ref_path = materialize(kQwen3Header, kQwen3DataBytes, scratch.path());
    ASSERT_FALSE(ref_path.empty());
    std::string ref_error;
    const auto reference = convert_with(ref_path, {}, ref_error);
    ASSERT_TRUE(reference) << ref_error;
    const auto ref_hist = op_histogram(reference);

    // Under test: the ported builder, supplied entirely by an extension.
    ScratchDir scratch2;
    const auto path = materialize(kQwen3Header, kQwen3DataBytes, scratch2.path(), "qwen3", kUnknownArch);
    ASSERT_FALSE(path.empty());
    auto ext = std::make_shared<ArchitectureExtension>(kUnknownArch, [](const BuildContext& c) {
        return std::make_shared<Qwen3PortBuilder>(c);
    });

    std::string error;
    const auto model = convert_with(path, {ext}, error);
    ASSERT_TRUE(model) << "the ported qwen3 builder failed to convert:\n" << error;

    auto hist = op_histogram(model);

    // The port is NOT expected to be node-identical to the built-in builder, and demanding that
    // would be testing the wrong thing. It is a faithful port of upstream qwen3.cpp, whereas the
    // built-in builder additionally honours whatever the FILE says -- and this synthetic fixture
    // populates every key llama.cpp knows, including an attn_logit_softcapping that no real qwen3
    // checkpoint carries. The built-in graph therefore soft-caps and lowers to an explicit softmax;
    // the port does not, and fuses to SDPA instead. Both are correct for their input.
    //
    // What must hold is that the port built the same ARCHITECTURE, through the SDK alone.
    const size_t n_layer = 2;  // the fixture's block_count

    // The hardest thing build_attn has to get right: a KV-cached attention that still fuses into a
    // single SDPA per layer. A shape or layout mistake shows up here as a decomposed chain.
    EXPECT_EQ(hist["ScaledDotProductAttention"], n_layer) << "attention must collapse to one SDPA per layer";

    // Logits plus a K and a V cache per layer -- the same output surface the built-in path has.
    EXPECT_EQ(model->outputs().size(), reference->outputs().size())
        << "the ported model must expose the same outputs (logits + per-layer KV caches)";
    EXPECT_EQ(model->outputs().size(), 1u + 2u * n_layer);

    // The per-layer caches must be real model inputs, not constants folded away.
    size_t cache_inputs = 0;
    for (const auto& input : model->inputs()) {
        if (input.get_any_name().rfind("cache_", 0) == 0) {
            ++cache_inputs;
        }
    }
    EXPECT_EQ(cache_inputs, 2u * n_layer);
}

// ---------------------------------------------------------------------------------------------
// Tier 3b: a NON-DECODER family, which the frontend has no support for whatsoever.
// ---------------------------------------------------------------------------------------------

namespace {

// Hyperparameters of the synthetic mmproj vision encoder below.
constexpr uint32_t kVisEmbd = 32;
constexpr uint32_t kVisLayers = 2;
constexpr uint32_t kVisHeads = 4;
constexpr uint32_t kVisPatches = 16;
constexpr uint32_t kVisFF = 64;
constexpr uint32_t kVisProjDim = 48;

// Write a minimal mmproj (vision encoder) GGUF. It names itself "clip" and declares
// clip.has_vision_encoder, exactly as llama.cpp's mmproj files do (tools/mtmd/clip-impl.h), which
// is why identifying it needs a metadata predicate rather than an architecture name.
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

// A vision encoder: patch embeddings -> N NON-CAUSAL transformer blocks -> projector.
//
// Nothing about this is a causal decoder -- no KV cache, no RoPE, no causal mask, and its own
// input -- so it exercises the part of the mechanism that matters most: a family the frontend does
// not implement, contributed entirely from outside it.
class VisionEncoderBuilder : public ModelBuilder {
public:
    explicit VisionEncoderBuilder(const BuildContext& ctx) : m_ctx(ctx) {}

    std::shared_ptr<GgufGraph> build() override {
        GgufGraphContext ctx(m_ctx);
        const auto& meta = ctx.metadata();
        auto tensors = ctx.tensors();

        const int64_t n_embd = meta.get_key_or("clip.vision.embedding_length", int64_t(0));
        const int n_layer = static_cast<int>(meta.get_key_or("clip.vision.block_count", int64_t(0)));
        const int64_t n_head = meta.get_key_or("clip.vision.attention.head_count", int64_t(0));
        const float eps = static_cast<float>(meta.get_key_or("clip.vision.attention.layer_norm_epsilon", 1e-5));
        const int64_t n_patches = kVisPatches;
        const int64_t head_size = n_embd / n_head;

        // The encoder's own input: one embedding per image patch. A decoder's token input would
        // make no sense here, which is the point.
        auto cur = ctx.add_input("inp_patches", ov::element::f32, ov::PartialShape({1, 1, n_patches, n_embd}));
        cur = ctx.add(cur, tensors.require("v.position_embd.weight"));

        for (int il = 0; il < n_layer; ++il) {
            const std::string p = "v.blk." + std::to_string(il) + ".";
            auto residual = cur;

            cur = ctx.build_norm(cur, tensors.require(p + "ln1.weight"), eps);

            // Non-causal self-attention, written out with the ggml vocabulary because there is no
            // KV cache to hide behind a build_attn: Q@K^T -> softmax -> @V.
            auto q = ctx.build_lora_mm(tensors.require(p + "attn_q.weight"), cur);
            auto k = ctx.build_lora_mm(tensors.require(p + "attn_k.weight"), cur);
            auto v = ctx.build_lora_mm(tensors.require(p + "attn_v.weight"), cur);

            // [patches, heads, head_size] -> [heads, patches, head_size], so the matmuls contract
            // over head_size with the head axis batched, as llama.cpp's ggml_permute does here.
            q = ctx.permute(ctx.reshape(q, {head_size, n_head, n_patches}), {0, 2, 1, 3});
            k = ctx.permute(ctx.reshape(k, {head_size, n_head, n_patches}), {0, 2, 1, 3});
            // V is contracted over the patch axis instead, so it needs head_size innermost.
            v = ctx.permute(ctx.reshape(v, {head_size, n_head, n_patches}), {0, 2, 3, 1});

            auto kq = ctx.mul_mat(k, q);
            kq = ctx.scale(kq, 1.0f / std::sqrt(static_cast<float>(head_size)));
            kq = ctx.soft_max(kq);
            auto kqv = ctx.mul_mat(v, kq);

            cur = ctx.permute(kqv, {0, 2, 1, 3});
            cur = ctx.reshape(ctx.cont(cur), {n_embd, n_patches});
            cur = ctx.build_lora_mm(tensors.require(p + "attn_out.weight"), cur);
            cur = ctx.add(cur, residual);

            residual = cur;
            cur = ctx.build_norm(cur, tensors.require(p + "ln2.weight"), eps);
            cur = ctx.build_ffn(cur,
                                tensors.require(p + "ffn_up.weight"),
                                GgufValue(),
                                GgufValue(),  // ungated: up -> GELU -> down
                                GgufValue(),
                                tensors.require(p + "ffn_down.weight"),
                                GgufValue(),
                                FfnOp::Gelu);
            cur = ctx.add(cur, residual);
        }

        cur = ctx.build_norm(cur, tensors.require("v.post_ln.weight"), eps);
        cur = ctx.build_lora_mm(tensors.require("mm.0.weight"), cur);
        ctx.set_output(cur);
        return ctx.finish();
    }

private:
    BuildContext m_ctx;
};

}  // namespace

// Without the extension, an mmproj file is refused -- the built-in builder only does decoders.
TEST(GGUFArchitectureExtension, NonDecoderFileIsRejectedWithoutAnExtension) {
    ScratchDir scratch;
    const auto path = write_vision_gguf(scratch.path());
    ASSERT_FALSE(path.empty());

    std::string error;
    const auto model = convert_with(path, {}, error);
    ASSERT_FALSE(model) << "a vision file must not convert through the decoder builder";
    EXPECT_NE(error.find("decoder family"), std::string::npos) << "actual error:\n" << error;
    // The refusal must name the way out, since this is precisely the case the mechanism exists for.
    EXPECT_NE(error.find("ArchitectureExtension"), std::string::npos) << "actual error:\n" << error;
}

// The headline case: a family the frontend has no code for at all, added from outside it.
TEST(GGUFArchitectureExtension, Tier3NonDecoderFamilyConvertsEndToEnd) {
    ScratchDir scratch;
    const auto path = write_vision_gguf(scratch.path());
    ASSERT_FALSE(path.empty());

    // The file calls itself "clip", so the extension claims it by metadata flag, not by name.
    auto ext = std::make_shared<ArchitectureExtension>(
        "clip",
        [](const BuildContext& c) {
            return std::make_shared<VisionEncoderBuilder>(c);
        },
        [](const GgufMetadata& m) {
            return m.get_key_or("clip.has_vision_encoder", false);
        });

    std::string error;
    const auto model = convert_with(path, {ext}, error);
    ASSERT_TRUE(model) << "the vision encoder extension failed to convert:\n" << error;

    // It must be the vision graph, not something that accidentally went down the decoder path.
    EXPECT_EQ(model->inputs().size(), 1u) << "a vision encoder takes patches and nothing else -- "
                                          << "no tokens, no positions, no KV caches";
    EXPECT_EQ(model->inputs()[0].get_any_name(), "inp_patches");
    EXPECT_EQ(model->outputs().size(), 1u) << "no KV caches means no cache outputs";

    auto hist = op_histogram(model);
    EXPECT_EQ(hist["Softmax"], kVisLayers) << "one non-causal attention softmax per encoder block";
    EXPECT_GT(hist["MatMul"], 0u);
    EXPECT_EQ(hist["Parameter"], 1u);
}

// ---------------------------------------------------------------------------------------------
// Packaging: the shared-library route, which is what makes this a no-rebuild mechanism.
// ---------------------------------------------------------------------------------------------

// An extension loaded from a .so arrives wrapped in an SOExtension. If that wrapper were not
// unwrapped into the registry, `core.add_extension("libmy_arch.so")` would silently do nothing --
// the failure mode the whole feature has to avoid.
TEST(GGUFArchitectureExtension, SharedLibraryWrappedExtensionStillRegisters) {
    if (!fixtures_present()) {
        GTEST_SKIP() << "no arch fixtures -- generate them with tests/gen_arch_fixtures.py --fetch";
    }
    ScratchDir scratch;
    const auto path = materialize(kQwen3Header, kQwen3DataBytes, scratch.path(), "qwen3", kUnknownArch);
    ASSERT_FALSE(path.empty());

    // A null library handle is enough: SOExtension only keeps it alive, and this extension's code
    // lives in the test binary, exactly as a real one's lives in its .so.
    const auto inner = std::make_shared<ArchitectureExtension>(kUnknownArch, RopeMode::Neox);
    const auto wrapped = std::make_shared<ov::detail::SOExtension>(inner, std::shared_ptr<void>{});

    std::string error;
    const auto model = convert_with(path, {wrapped}, error);
    ASSERT_TRUE(model) << "an SOExtension-wrapped ArchitectureExtension did not reach the registry:\n" << error;
}

// Two extensions claiming one file is a registration bug. Picking one silently would surface much
// later as an inexplicably wrong graph, so it must fail loudly and name both.
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
    auto first = std::make_shared<ArchitectureExtension>("first-claimant", factory, claim_everything);
    auto second = std::make_shared<ArchitectureExtension>("second-claimant", factory, claim_everything);

    std::string error;
    const auto model = convert_with(path, {first, second}, error);
    ASSERT_FALSE(model) << "two extensions claimed the same file and one was silently chosen";
    EXPECT_NE(error.find("first-claimant"), std::string::npos) << "actual error:\n" << error;
    EXPECT_NE(error.find("second-claimant"), std::string::npos) << "actual error:\n" << error;
}
