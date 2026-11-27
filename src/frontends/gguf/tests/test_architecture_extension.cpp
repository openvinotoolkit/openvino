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
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "builder/api/metadata_store.hpp"
#include "builder/arch_registry.hpp"
#include "builder/gguf_builder_decoder.hpp"
#include "builder/gguf_graph.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "gguf_writer.hpp"
#include "gtest/gtest.h"
#include "model_contract.hpp"
#include "op_test_utils.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/extension/decoder_transformation.hpp"
#include "openvino/frontend/gguf/adapt_to_genai.hpp"
#include "openvino/frontend/gguf/builder/graph_context.hpp"
#include "openvino/frontend/gguf/extension/architecture.hpp"
#include "openvino/frontend/gguf/extension/genai.hpp"
#include "openvino/frontend/gguf/frontend.hpp"
#include "openvino/frontend/gguf/make_stateful.hpp"
#include "openvino/frontend/gguf/set_rows_op.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/op/util/variable_extension.hpp"
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

// qwen35 file with one NextN block; empty per-layer arrays use scalar metadata.
struct MtpFixture {
    std::vector<uint32_t> kv_heads;
    std::vector<uint32_t> recurrent;
    std::vector<uint32_t> sliding;
    uint32_t nextn_kv_heads = 1;
    uint32_t nextn_swa_head_size = 0;
    std::vector<std::string> hadamard_names;
    bool tied_hadamard = false;
    bool explicit_untied_hadamard = false;
    bool inverse_embedding_hadamard = false;
    bool separate_embedding = false;
    uint32_t nextn_layers = 1;
};

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
std::string write_decoder_gguf(const std::string& dir,
                               const std::string& arch = "qwen3",
                               uint32_t layers = 2,
                               bool layer_norm = false,
                               const std::optional<MtpFixture>& mtp = std::nullopt) {
    const auto fixture = mtp.value_or(MtpFixture{});
    const auto& hadamard_names = fixture.hadamard_names;
    const auto path = ov::util::path_join({dir, arch + ".gguf"}).string();
    GgufWriter writer;
    writer.kv_str("general.architecture", arch);
    writer.kv_u32(arch + ".block_count", layers);
    writer.kv_u32(arch + ".embedding_length", 8);
    writer.kv_u32(arch + ".attention.head_count", 2);
    if (fixture.kv_heads.empty())
        writer.kv_u32(arch + ".attention.head_count_kv", 1);
    else
        writer.kv_u32_array(arch + ".attention.head_count_kv", fixture.kv_heads);
    if (!fixture.recurrent.empty())
        writer.kv_u32_array(arch + ".attention.recurrent_layers", fixture.recurrent);
    if (!fixture.sliding.empty())
        writer.kv_u32_array(arch + ".attention.sliding_window_pattern", fixture.sliding);
    if (fixture.nextn_swa_head_size) {
        writer.kv_u32(arch + ".attention.sliding_window", 4);
        writer.kv_u32(arch + ".attention.key_length_swa", fixture.nextn_swa_head_size);
        writer.kv_u32(arch + ".rope.dimension_count_swa", fixture.nextn_swa_head_size);
    }
    writer.kv_u32(arch + ".rope.dimension_count", 4);
    writer.kv_u32(arch + ".context_length", 32);
    writer.kv_f32(arch + (layer_norm ? ".attention.layer_norm_epsilon" : ".attention.layer_norm_rms_epsilon"), 1e-5f);
    writer.kv_f32(arch + ".rope.scaling.factor", 4.f);
    if (mtp) {
        writer.kv_u32(arch + ".nextn_predict_layers", fixture.nextn_layers);
        writer.kv_u32(arch + ".full_attention_interval", 1);
        writer.kv_u32(arch + ".ssm.conv_kernel", 4);
        writer.kv_u32(arch + ".ssm.state_size", 4);
        writer.kv_u32(arch + ".ssm.group_count", 1);
        writer.kv_u32(arch + ".ssm.time_step_rank", 2);
        writer.kv_u32(arch + ".ssm.inner_size", 8);
        writer.kv_u32_array(arch + ".rope.dimension_sections", {1, 1, 0, 0});
        if (!hadamard_names.empty()) {
            const std::string p = "prism.hadamard.";
            writer.kv_u32(p + "version", fixture.tied_hadamard ? 2 : 1);
            if (fixture.tied_hadamard || fixture.explicit_untied_hadamard)
                writer.kv_bool(p + "tied_output", fixture.tied_hadamard);
            if (fixture.tied_hadamard || fixture.inverse_embedding_hadamard)
                writer.kv_str_array(p + "inverse_weight_names", {"token_embd.weight"});
            writer.kv_u32(p + "block_size", 8);
            writer.kv_str(p + "transform", "normalized-sylvester-walsh-hadamard");
            writer.kv_str(p + "axis", "input-last-dimension");
            writer.kv_str(p + "sign_mode", "identity");
            writer.kv_str_array(p + "weight_names", hadamard_names);
        }
    } else {
        writer.kv_f32(arch + ".attn_logit_softcapping", 2.f);
        writer.kv_u32(arch + ".attention.sliding_window", 2);
    }
    const auto weight = [&](const std::string& name, const std::vector<uint64_t>& shape, bool norm = false) {
        writer.filled_tensor(name, shape, norm);
    };
    weight("token_embd.weight", {8, 16});
    weight("output_norm.weight", {8}, true);
    if (!hadamard_names.empty() && !fixture.tied_hadamard)
        weight("output.weight", {8, 16});
    if (layer_norm)
        weight("output_norm.bias", {8});
    for (uint32_t layer = 0; layer < layers; ++layer) {
        const auto prefix = "blk." + std::to_string(layer) + ".";
        weight(prefix + "attn_norm.weight", {8}, true);
        if (layer_norm)
            weight(prefix + "attn_norm.bias", {8});
        const bool nextn = mtp && layer == layers - 1;
        const auto head_size = nextn && fixture.nextn_swa_head_size ? fixture.nextn_swa_head_size : 4u;
        weight(prefix + "attn_q.weight", {8, mtp ? 4 * head_size : 8u});
        const auto kv_width = nextn ? head_size * fixture.nextn_kv_heads : 4u;
        weight(prefix + "attn_k.weight", {8, kv_width});
        weight(prefix + "attn_v.weight", {8, kv_width});
        weight(prefix + "attn_output.weight", {mtp ? 2 * head_size : 8u, 8});
        weight(prefix + "attn_q_norm.weight", {head_size}, true);
        weight(prefix + "attn_k_norm.weight", {head_size}, true);
        weight(prefix + "ffn_norm.weight", {8}, true);
        if (layer_norm)
            weight(prefix + "ffn_norm.bias", {8});
        weight(prefix + "ffn_gate.weight", {8, 12});
        weight(prefix + "ffn_up.weight", {8, 12});
        weight(prefix + "ffn_down.weight", {12, 8});
        if (nextn) {
            weight(prefix + "nextn.eh_proj.weight", {16, 8});
            weight(prefix + "nextn.enorm.weight", {8}, true);
            weight(prefix + "nextn.hnorm.weight", {8}, true);
            weight(prefix + "nextn.shared_head_norm.weight", {8}, true);
            if (fixture.separate_embedding)
                weight(prefix + "nextn.embed_tokens.weight", {8, 16});
            if (!hadamard_names.empty())
                weight(prefix + "nextn.shared_head_head.weight", {8, 16});
        }
    }
    return writer.write(path) ? path : std::string{};
}

std::string write_qwen3vl_gguf(const std::string& dir, const std::vector<std::string>& tags) {
    const auto path = ov::util::path_join({dir, "qwen3vl.gguf"}).string();
    GgufWriter writer;
    writer.kv_str("general.architecture", "qwen3vl");
    writer.kv_str_array("general.tags", tags);
    writer.kv_u32("qwen3vl.n_deepstack_layers", 0);
    writer.filled_tensor("token_embd.weight", {8, 16});
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

TEST(GGUFArchitectureExtension, DeferredArchitecturesAreRejectedWithoutAnExtension) {
    for (const auto* architecture : {"qwen3next"}) {
        SCOPED_TRACE(architecture);
        ScratchDir scratch;
        const auto path = write_decoder_gguf(scratch.path(), architecture);
        ASSERT_FALSE(path.empty());
        ov::frontend::gguf::FrontEnd frontend;
        OV_EXPECT_THROW(frontend.load(path),
                        ov::Exception,
                        testing::AllOf(testing::HasSubstr("native GGUF builder does not support architecture"),
                                       testing::HasSubstr(architecture)));
        frontend.add_extension(std::make_shared<ArchitectureExtension>(architecture, RopeMode::Neox));
        EXPECT_NO_THROW(frontend.load(path));
    }
}

TEST(GGUFArchitectureExtension, BuiltInDecodersAcceptSupportedTensorTypes) {
    using namespace ov::frontend::gguf;
    ov::Core core;
    for (const auto* architecture : {"afmoe", "qwen3tts", "qwen3vlmoe"}) {
        SCOPED_TRACE(architecture);
        const auto arrays = cnpy::npz_load(test_data_dir() + "/arch_accuracy/" + architecture + ".npz");
        const TemporaryGguf fixture(npz_array(arrays, "model"));
        for (const auto type : {GGUF_TYPE_BF16, GGUF_TYPE_Q8_0}) {
            SCOPED_TRACE(type);
            auto loaded = get_gguf_data(fixture.path);
            auto& weights = std::get<1>(loaded);
            quantize_weight(weights, "token_embd.weight", type);
            const detail::MetadataStore meta_store{std::get<0>(loaded)};
            const GgufMetadata meta(meta_store);
            const auto definition = ArchRegistry{}.find(meta);
            ASSERT_TRUE(definition);
            detail::WeightStore store{weights};
            const auto builder = definition->factory(BuildContext{meta, architecture, &store});
            std::shared_ptr<GgufDecoder> decoder = std::make_shared<GgufBuilderDecoder>(builder->build());
            FrontEnd frontend;
            frontend.add_extension(std::make_shared<GenAIExtension>(GenAIExtension::InputMode::IDS_TO_LOGITS));
            const auto model = frontend.convert(frontend.load(decoder));
            EXPECT_NO_THROW(core.compile_model(model,
                                               "CPU",
                                               ov::hint::inference_precision(ov::element::f32),
                                               ov::hint::dynamic_quantization_group_size(0)));
        }
    }
}

TEST(GGUFArchitectureExtension, Qwen3ASRTaskTagRemainsRegistered) {
    ScratchDir scratch;
    const auto path = write_qwen3vl_gguf(scratch.path(), {"automatic-speech-recognition"});
    ASSERT_FALSE(path.empty());
    ov::frontend::gguf::FrontEnd frontend;
    EXPECT_NO_THROW(frontend.load(path));
}

TEST(GGUFArchitectureExtension, Qwen3VLDenseWithoutDeepStackRemainsRegistered) {
    for (const auto& tags : {std::vector<std::string>{}, std::vector<std::string>{"image-text-to-text"}}) {
        ScratchDir scratch;
        const auto path = write_qwen3vl_gguf(scratch.path(), tags);
        ASSERT_FALSE(path.empty());
        ov::frontend::gguf::FrontEnd frontend;
        EXPECT_NO_THROW(frontend.load(path));
    }
}

TEST(GGUFArchitectureExtension, UnsupportedDiagnosticIncludesRegisteredArchitectures) {
    ScratchDir scratch;
    const auto path = write_decoder_gguf(scratch.path(), "unknown-decoder");
    ASSERT_FALSE(path.empty());
    ov::frontend::gguf::FrontEnd frontend;
    frontend.add_extension(std::make_shared<ArchitectureExtension>("custom-decoder", RopeMode::Neox));
    OV_EXPECT_THROW(frontend.convert(frontend.load(path)),
                    ov::Exception,
                    testing::AllOf(testing::HasSubstr("Supported:"), testing::HasSubstr("custom-decoder")));
    OV_EXPECT_THROW(convert_with(path), ov::Exception, testing::Not(testing::HasSubstr("custom-decoder")));
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

TEST(GGUFGenAIExtension, RegistrationIsLocalAndConversionCanBeRepeated) {
    ScratchDir scratch;
    const auto path = write_decoder_gguf(scratch.path());
    ASSERT_FALSE(path.empty());
    ov::frontend::gguf::FrontEnd frontend;
    auto extension = std::make_shared<ov::frontend::gguf::GenAIExtension>();
    frontend.add_extension(extension);
    const auto input = frontend.load(path);
    for (size_t i = 0; i < 2; ++i) {
        const auto model = frontend.convert(input);
        EXPECT_EQ(model->get_variables().size(), 4);
        EXPECT_NO_THROW(model->input("input_ids"));
        EXPECT_NO_THROW(model->input("attention_mask"));
        EXPECT_NO_THROW(model->input("position_ids"));
        EXPECT_NO_THROW(model->input("beam_idx"));
        EXPECT_EQ(model->output("logits").get_partial_shape().rank(), 3);
        EXPECT_FALSE(extension->get_embedding_model());
    }
    const auto plain = convert_with(path);
    EXPECT_TRUE(plain->get_variables().empty());
    EXPECT_NO_THROW(plain->input("inp_tokens"));
    OV_EXPECT_THROW(frontend.add_extension(std::make_shared<ov::frontend::gguf::GenAIExtension>()),
                    ov::Exception,
                    testing::HasSubstr("already registered"));
}

TEST(GGUFGenAIExtension, MtpDraftAcceptsTargetOnlyLayerMetadata) {
    using namespace ov::frontend::gguf;
    // Target-only arrays must match the explicit full-layer reference; the second case infers
    // two NextN KV heads from a projection using the smaller sliding-window head size.
    const std::vector<std::pair<MtpFixture, MtpFixture>> cases{
        {MtpFixture{{1}, {0}, {0}}, MtpFixture{{1, 1}, {0, 1}, {0, 0}}},
        {MtpFixture{{1}, {0}, {0, 1}, 2, 2}, MtpFixture{{1, 2}, {0, 1}, {0, 1}, 2, 2}},
    };
    for (const auto& [target_only, all_layers] : cases) {
        SCOPED_TRACE(target_only.nextn_swa_head_size);
        ScratchDir target_only_dir;
        ScratchDir all_layers_dir;
        const auto load = [&](const std::string& dir, const MtpFixture& fixture) {
            const auto path = write_decoder_gguf(dir, "qwen35", 2, false, fixture);
            FrontEnd frontend;
            auto models = adapt_to_genai(frontend.convert(frontend.load(path, LoadOptions{true})));
            return models.at("mtp_draft");
        };
        const auto target_only_model = load(target_only_dir.path(), target_only);
        const auto all_layers_model = load(all_layers_dir.path(), all_layers);
        ov::Core core;
        const auto compile = [&](const std::shared_ptr<ov::Model>& model) {
            return core
                .compile_model(model,
                               "CPU",
                               ov::hint::inference_precision(ov::element::f32),
                               ov::hint::dynamic_quantization_group_size(0),
                               ov::hint::kv_cache_precision(ov::element::f32))
                .create_infer_request();
        };
        auto actual = compile(target_only_model);
        auto reference = compile(all_layers_model);
        size_t past = 0;
        for (const size_t tokens : {3u, 1u}) {
            ov::Tensor ids(ov::element::i64, {1, tokens}), positions(ov::element::i64, {1, tokens});
            ov::Tensor mask(ov::element::i64, {1, past + tokens}), beam(ov::element::i32, {1});
            ov::Tensor hidden(ov::element::f32, {1, tokens, 8});
            std::fill_n(mask.data<int64_t>(), mask.get_size(), 1);
            *beam.data<int32_t>() = 0;
            for (size_t i = 0; i < tokens; ++i) {
                ids.data<int64_t>()[i] = static_cast<int64_t>(past + i + 1);
                positions.data<int64_t>()[i] = static_cast<int64_t>(past + i);
            }
            for (size_t i = 0; i < hidden.get_size(); ++i)
                hidden.data<float>()[i] = 0.1f * std::sin(static_cast<float>(i + past));
            for (auto* request : {&actual, &reference}) {
                request->set_tensor("input_ids", ids);
                request->set_tensor("position_ids", positions);
                request->set_tensor("attention_mask", mask);
                request->set_tensor("beam_idx", beam);
                request->set_tensor("hidden_states", hidden);
                request->infer();
            }
            for (const auto* name : {"logits", "last_hidden_state"}) {
                const auto actual_output = actual.get_tensor(name);
                const auto reference_output = reference.get_tensor(name);
                ASSERT_EQ(actual_output.get_shape(), reference_output.get_shape());
                for (size_t i = 0; i < actual_output.get_size(); ++i)
                    EXPECT_NEAR(actual_output.data<const float>()[i], reference_output.data<const float>()[i], 1e-5f);
            }
            past += tokens;
        }
    }
}

TEST(GGUFGenAIExtension, MtpLoadOptionFiltersUnusedHadamardWeights) {
    using namespace ov::frontend::gguf;
    // The draft uses its NextN head, so output.weight (or the tied token_embd head) is unused;
    // target-only names leave nothing rotated.
    struct HadamardCase {
        std::vector<std::string> names;
        bool tied = false;
        bool explicit_untied = false;
        bool inverse_embedding = false;
    };
    const std::vector<HadamardCase> cases{
        {{"output.weight", "blk.1.nextn.shared_head_head.weight"}},
        {{"output.weight", "blk.0.attn_q.weight"}},
        {{"blk.0.attn_q.weight", "blk.1.nextn.shared_head_head.weight"}, true},
        {{"blk.0.attn_q.weight"}, true},
        {{"blk.0.attn_q.weight"}, false, true},
        {{"output.weight"}, false, true, true},
    };
    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.names.back() + (test_case.tied ? " tied" : ""));
        ScratchDir scratch;
        MtpFixture fixture;
        fixture.hadamard_names = test_case.names;
        fixture.tied_hadamard = test_case.tied;
        fixture.explicit_untied_hadamard = test_case.explicit_untied;
        fixture.inverse_embedding_hadamard = test_case.inverse_embedding;
        const auto path = write_decoder_gguf(scratch.path(), "qwen35", 2, false, fixture);
        ov::frontend::gguf::FrontEnd frontend;
        const auto input = frontend.load(path, LoadOptions{true});
        auto models = adapt_to_genai(frontend.convert(input), {pass::AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS_TEXT});
        const auto model = models.at("language");
        EXPECT_NO_THROW(model->output("last_hidden_state"));
        const auto draft = models.at("mtp_draft");
        ASSERT_TRUE(draft);
        EXPECT_EQ(draft->input("hidden_states").get_partial_shape(), (ov::PartialShape{-1, -1, 8}));
        EXPECT_EQ(draft->output("logits").get_partial_shape(), (ov::PartialShape{-1, -1, 16}));
    }
}

TEST(GGUFGenAIExtension, MtpIsOptionalAndSurvivesModelClone) {
    using namespace ov::frontend::gguf;
    ScratchDir scratch;
    const auto path = write_decoder_gguf(scratch.path(), "qwen35", 2, false, MtpFixture{});
    FrontEnd frontend;
    ASSERT_TRUE(frontend.supported(path, LoadOptions{true}));
    auto ordinary = adapt_to_genai(frontend.convert(frontend.load(path)));
    EXPECT_EQ(ordinary.count("mtp_draft"), 0);
    EXPECT_THROW(ordinary.at("language")->output("last_hidden_state"), ov::Exception);

    auto combined = frontend.convert(frontend.load(path, LoadOptions{true}));
    EXPECT_TRUE(combined->has_rt_info("gguf_mtp_components"));
    EXPECT_NO_THROW(combined->input("mtp.inp_tokens"));
    std::set<size_t> offsets;
    for (const auto& operation : combined->get_ops()) {
        const auto& info = operation->get_rt_info();
        const auto it = info.find(ov::WeightlessCacheAttribute::get_type_info_static());
        if (it != info.end()) {
            EXPECT_TRUE(offsets.insert(it->second.as<ov::WeightlessCacheAttribute>().bin_offset).second);
        }
    }
    EXPECT_GT(offsets.size(), 1);
    auto models = adapt_to_genai(combined->clone());
    ASSERT_EQ(models.count("language"), 1);
    ASSERT_EQ(models.count("mtp_draft"), 1);
    EXPECT_NO_THROW(models.at("language")->output("last_hidden_state"));
    EXPECT_NO_THROW(models.at("mtp_draft")->input("hidden_states"));

    auto legacy = combined->clone();
    auto& info = legacy->get_rt_info().at("gguf_mtp_components").as<ov::AnyMap>();
    info.erase("target_sinks");
    info.erase("target_variables");
    info.erase("draft_variable_prefix");
    info.erase("draft_io_prefix");
    info.erase("draft_name");
    EXPECT_EQ(adapt_to_genai(legacy).count("mtp_draft"), 1);
}

TEST(GGUFGenAIExtension, MtpWithoutNextNExposesOnlyTarget) {
    using namespace ov::frontend::gguf;
    ScratchDir scratch;
    MtpFixture fixture;
    fixture.nextn_layers = 0;
    const auto path = write_decoder_gguf(scratch.path(), "qwen35", 2, false, fixture);
    FrontEnd frontend;
    const auto models = adapt_to_genai(frontend.convert(frontend.load(path, LoadOptions{true})));
    EXPECT_EQ(models.count("mtp_draft"), 0);
    EXPECT_NO_THROW(models.at("language")->output("last_hidden_state"));
}

TEST(GGUFGenAIExtension, MtpLoadOptionsSupportPathListsAndTrailingMmap) {
    using namespace ov::frontend::gguf;
    ScratchDir scratch;
    const auto path = write_decoder_gguf(scratch.path(), "qwen35", 2, false, MtpFixture{});
    FrontEnd frontend;
    const std::vector<std::string> paths{path};
    ASSERT_TRUE(frontend.supported(paths, LoadOptions{true}, false));
    const auto input = frontend.load(paths, LoadOptions{true}, false);
    for (size_t i = 0; i < 2; ++i) {
        const auto models = adapt_to_genai(frontend.convert(input));
        ASSERT_EQ(models.count("language"), 1);
        ASSERT_EQ(models.count("mtp_draft"), 1);
    }
    EXPECT_FALSE(frontend.supported(path, LoadOptions{true}, LoadOptions{true}));
}

TEST(GGUFGenAIExtension, MtpLoadOptionsRejectLegacyGenAIExtension) {
    using namespace ov::frontend::gguf;
    ScratchDir scratch;
    const auto path = write_decoder_gguf(scratch.path(), "qwen35", 2, false, MtpFixture{});
    FrontEnd legacy;
    legacy.add_extension(std::make_shared<GenAIExtension>());
    OV_EXPECT_THROW(legacy.convert(legacy.load(path, LoadOptions{true})),
                    ov::Exception,
                    testing::HasSubstr("MTP conversion requires adapt_to_genai"));
}

TEST(GGUFGenAIExtension, MtpTransformationRunsBeforeLoweringOnEachComponent) {
    using namespace ov::frontend::gguf;
    for (const bool has_draft : {false, true}) {
        SCOPED_TRACE(has_draft);
        ScratchDir scratch;
        MtpFixture fixture;
        fixture.nextn_layers = has_draft ? 1 : 0;
        const auto path = write_decoder_gguf(scratch.path(), "qwen35", 2, false, fixture);
        std::vector<bool> saw_set_rows;
        FrontEnd frontend;
        frontend.add_extension(std::make_shared<ov::frontend::DecoderTransformationExtension>(
            [&](const std::shared_ptr<ov::Model>& model) {
                const auto nodes = model->get_ops();
                saw_set_rows.push_back(std::any_of(nodes.begin(), nodes.end(), [](const auto& node) {
                    return ov::is_type<SetRows>(node);
                }));
                const auto index = static_cast<int64_t>(saw_set_rows.size());
                model->set_friendly_name("extension_component_" + std::to_string(index));
                model->get_rt_info()["extension_component"] = index;
                for (const auto* name : {"extra", "mtp.extra"}) {
                    auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
                    parameter->set_friendly_name(name);
                    parameter->output(0).set_names({name});
                    model->add_parameters({parameter});
                    auto result = std::make_shared<ov::op::v0::Result>(parameter);
                    result->set_friendly_name(std::string{name} + ".result");
                    model->add_results({result});
                    if (std::string{name} == "extra")
                        model->add_results({std::make_shared<ov::op::v0::Result>(parameter)});
                }
                auto contract = read_model_contract(model);
                contract.genai->preserved_inputs.insert(contract.genai->preserved_inputs.end(), {"extra", "mtp.extra"});
                write_model_contract(model, contract);
                return true;
            }));
        const auto combined = frontend.convert(frontend.load(path, LoadOptions{true}));
        ASSERT_EQ(saw_set_rows.size(), has_draft ? 2 : 1);
        EXPECT_TRUE(std::all_of(saw_set_rows.begin(), saw_set_rows.end(), [](bool found) {
            return found;
        }));
        EXPECT_EQ(combined->get_friendly_name(), "extension_component_1");
        if (has_draft) {
            EXPECT_NO_THROW(combined->input("mtp.extra"));
            EXPECT_NO_THROW(combined->input("mtp.mtp.extra"));
            EXPECT_NO_THROW(combined->input("mtp.mtp.mtp.extra"));
        }
        const auto models = adapt_to_genai(combined->clone());
        ASSERT_EQ(models.count("language"), 1);
        EXPECT_EQ(models.at("language")->get_friendly_name(), "extension_component_1");
        EXPECT_EQ(models.at("language")->get_rt_info().at("extension_component").as<int64_t>(), 1);
        EXPECT_NO_THROW(models.at("language")->input("extra"));
        EXPECT_NO_THROW(models.at("language")->input("mtp.extra"));
        EXPECT_EQ(models.count("mtp_draft"), has_draft ? 1 : 0);
        if (has_draft) {
            EXPECT_EQ(models.at("mtp_draft")->get_friendly_name(), "extension_component_2");
            EXPECT_EQ(models.at("mtp_draft")->get_rt_info().at("extension_component").as<int64_t>(), 2);
            EXPECT_NO_THROW(models.at("mtp_draft")->input("extra"));
            EXPECT_NO_THROW(models.at("mtp_draft")->input("mtp.extra"));
            const auto draft_results = models.at("mtp_draft")->get_results();
            EXPECT_EQ(std::count_if(draft_results.begin(),
                                    draft_results.end(),
                                    [](const auto& result) {
                                        return result->input_value(0).get_names().count("extra") != 0;
                                    }),
                      2);
            const auto cloned_draft = models.at("mtp_draft")->clone();
            EXPECT_NO_THROW(cloned_draft->input("extra"));
            EXPECT_NO_THROW(cloned_draft->input("mtp.extra"));
        } else {
            FrontEnd stateful;
            stateful.add_extension(
                std::make_shared<ov::frontend::DecoderTransformationExtension>(pass::GGUFMakeStateful()));
            const auto converted = stateful.convert(stateful.load(path, LoadOptions{true}));
            EXPECT_EQ(converted->get_sinks().size(), 4);
            EXPECT_EQ(adapt_to_genai(converted).count("language"), 1);
        }
    }
}

TEST(GGUFGenAIExtension, MtpTransformationAddedOutputNamesSurviveSplitAndClone) {
    using namespace ov::frontend::gguf;
    ScratchDir scratch;
    const auto path = write_decoder_gguf(scratch.path(), "qwen35", 2, false, MtpFixture{});
    size_t visits = 0;
    FrontEnd frontend;
    frontend.add_extension(
        std::make_shared<ov::frontend::DecoderTransformationExtension>([&](const std::shared_ptr<ov::Model>& model) {
            const bool draft = ++visits == 2;
            auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
            input->set_friendly_name("extension_input");
            input->output(0).set_names({"extension_input"});
            model->add_parameters({input});
            auto value =
                std::make_shared<ov::op::v1::Add>(input, ov::op::v0::Constant::create(ov::element::f32, {1}, {1.f}));
            if (draft)
                value->output(0).set_names({"extension_output", "draft_alias", "mtp.draft_alias"});
            else
                value->output(0).set_names({"extension_output"});
            model->add_output(value->output(0));
            auto passthrough = std::make_shared<ov::op::v0::Result>(input);
            passthrough->output(0).set_names({"extension_passthrough"});
            model->add_results({passthrough});
            auto contract = read_model_contract(model);
            contract.genai->preserved_inputs.push_back("extension_input");
            write_model_contract(model, contract);
            return true;
        }));
    const auto combined = frontend.convert(frontend.load(path, LoadOptions{true}));
    ASSERT_EQ(visits, 2);
    const auto prefix =
        combined->get_rt_info().at("gguf_mtp_components").as<ov::AnyMap>().at("draft_io_prefix").as<std::string>();
    EXPECT_EQ(prefix, "mtp.");
    EXPECT_NO_THROW(combined->output("extension_output"));
    EXPECT_NO_THROW(combined->output(prefix + "extension_output"));
    EXPECT_NO_THROW(combined->output(prefix + "extension_passthrough"));
    EXPECT_NO_THROW(combined->output(prefix + "draft_alias"));
    EXPECT_NO_THROW(combined->output(prefix + "mtp.draft_alias"));
    EXPECT_TRUE(combined->output(prefix + "extension_output").get_names().count(prefix + "extension_output"));
    EXPECT_FALSE(combined->output(prefix + "extension_output").get_names().count("extension_output"));

    const auto models = adapt_to_genai(combined);
    ov::Core core;
    for (const auto& name : {std::string{"language"}, std::string{"mtp_draft"}}) {
        SCOPED_TRACE(name);
        const auto cloned = models.at(name)->clone();
        EXPECT_NO_THROW(cloned->input("extension_input"));
        EXPECT_NO_THROW(cloned->output("extension_output"));
        EXPECT_NO_THROW(cloned->output("extension_passthrough"));
        EXPECT_TRUE(cloned->output("extension_output").get_names().count("extension_output"));
        if (name == "mtp_draft") {
            EXPECT_NO_THROW(cloned->output("draft_alias"));
            EXPECT_NO_THROW(cloned->output("mtp.draft_alias"));
            const auto compiled = core.compile_model(models.at(name), "CPU");
            EXPECT_NO_THROW(compiled.input("extension_input"));
            EXPECT_NO_THROW(compiled.output("extension_output"));
            EXPECT_NO_THROW(compiled.output("extension_passthrough"));
            EXPECT_NO_THROW(compiled.output("draft_alias"));
            EXPECT_NO_THROW(compiled.output("mtp.draft_alias"));
        }
    }
}

TEST(GGUFGenAIExtension, MtpStatefulTransformationsPreserveIndependentStates) {
    using namespace ov::frontend::gguf;
    ScratchDir scratch;
    const auto path = write_decoder_gguf(scratch.path(), "qwen35", 2, false, MtpFixture{});
    const auto first = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{ov::PartialShape{1}, ov::element::f32, "custom_state"});
    const auto second = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{ov::PartialShape{1}, ov::element::f32, "mtp.custom_state"});
    size_t visits = 0;
    FrontEnd frontend;
    frontend.add_extension(
        std::make_shared<ov::frontend::DecoderTransformationExtension>([&](const std::shared_ptr<ov::Model>& model) {
            const auto scale = ++visits == 1 ? 1.f : 10.f;
            for (const auto& [variable, increment] :
                 std::vector<std::pair<ov::op::util::Variable::Ptr, float>>{{first, scale}, {second, 2.f * scale}}) {
                const auto init = ov::op::v0::Constant::create(ov::element::f32, {1}, {0.f});
                const auto read = std::make_shared<ov::op::v6::ReadValue>(init, variable);
                const auto delta = ov::op::v0::Constant::create(ov::element::f32, {1}, {increment});
                model->add_sinks(
                    {std::make_shared<ov::op::v6::Assign>(std::make_shared<ov::op::v1::Add>(read, delta), variable)});
            }
            const auto init = ov::op::v0::Constant::create(ov::element::f32, {1}, {0.f});
            const auto read = std::make_shared<ov::op::v3::ReadValue>(init, "legacy_state");
            const auto delta = ov::op::v0::Constant::create(ov::element::f32, {1}, {3.f * scale});
            model->add_sinks(
                {std::make_shared<ov::op::v3::Assign>(std::make_shared<ov::op::v1::Add>(read, delta), "legacy_state")});
            return true;
        }));
    frontend.add_extension(std::make_shared<ov::frontend::DecoderTransformationExtension>(pass::GGUFMakeStateful()));
    const auto combined = frontend.convert(frontend.load(path, LoadOptions{true}));
    ASSERT_EQ(visits, 2);
    EXPECT_EQ(first->get_info().variable_id, "custom_state");
    EXPECT_EQ(second->get_info().variable_id, "mtp.custom_state");
    ASSERT_EQ(combined->get_sinks().size(), 10);
    ASSERT_EQ(combined->get_variables().size(), 10);
    std::set<std::string> ids;
    for (const auto& variable : combined->get_variables())
        EXPECT_TRUE(ids.insert(variable->get_info().variable_id).second);
    const auto& info = combined->get_rt_info().at("gguf_mtp_components").as<ov::AnyMap>();
    const auto& prefix = info.at("draft_variable_prefix").as<std::string>();
    EXPECT_NE(prefix, "mtp.");
    EXPECT_EQ(ids.count(prefix + "custom_state"), 1);
    EXPECT_EQ(ids.count(prefix + "mtp.custom_state"), 1);
    EXPECT_EQ(ids.count(prefix + "legacy_state"), 1);

    ov::Core core;
    const auto compile = [&](const std::shared_ptr<ov::Model>& model) {
        return core
            .compile_model(model,
                           "CPU",
                           ov::hint::inference_precision(ov::element::f32),
                           ov::hint::dynamic_quantization_group_size(0),
                           ov::hint::kv_cache_precision(ov::element::f32))
            .create_infer_request();
    };
    auto combined_request = compile(combined->clone());
    EXPECT_EQ(combined_request.query_state().size(), 10);
    const auto models = adapt_to_genai(combined->clone());
    FrontEnd reference_frontend;
    const auto reference = adapt_to_genai(reference_frontend.convert(reference_frontend.load(path, LoadOptions{true})));
    struct Requests {
        std::string name;
        ov::InferRequest actual;
        ov::InferRequest reference;
    };
    std::vector<Requests> requests;
    for (const auto& name : {std::string{"language"}, std::string{"mtp_draft"}}) {
        const auto model = models.at(name);
        ASSERT_EQ(model->get_sinks().size(), 5);
        ASSERT_EQ(model->get_variables().size(), 5);
        for (const auto& variable_id : {"custom_state", "mtp.custom_state", "legacy_state"})
            EXPECT_TRUE(model->get_variable_by_id(variable_id));
        for (const auto& node : model->get_ordered_ops()) {
            if (const auto variable_op = std::dynamic_pointer_cast<ov::op::util::VariableExtension>(node)) {
                EXPECT_EQ(variable_op->get_variable_id(), variable_op->get_variable()->get_info().variable_id);
            }
        }
        requests.push_back({name, compile(model->clone()), compile(reference.at(name))});
    }
    const auto custom_states = [](ov::InferRequest& request) {
        std::map<std::string, float> values;
        for (const auto& state : request.query_state()) {
            if (state.get_name() == "custom_state" || state.get_name() == "mtp.custom_state" ||
                state.get_name() == "legacy_state")
                values[state.get_name()] = state.get_state().data<const float>()[0];
        }
        return values;
    };

    size_t past = 0;
    for (const size_t tokens : {3u, 1u}) {
        ov::Tensor ids_tensor(ov::element::i64, {1, tokens}), positions(ov::element::i64, {1, tokens});
        ov::Tensor mask(ov::element::i64, {1, past + tokens}), beam(ov::element::i32, {1});
        ov::Tensor hidden(ov::element::f32, {1, tokens, 8});
        std::fill_n(mask.data<int64_t>(), mask.get_size(), 1);
        *beam.data<int32_t>() = 0;
        for (size_t i = 0; i < tokens; ++i) {
            ids_tensor.data<int64_t>()[i] = static_cast<int64_t>(past + i + 1);
            positions.data<int64_t>()[i] = static_cast<int64_t>(past + i);
        }
        for (size_t i = 0; i < hidden.get_size(); ++i)
            hidden.data<float>()[i] = 0.1f * std::sin(static_cast<float>(i + past));
        for (auto& pair : requests) {
            SCOPED_TRACE(pair.name);
            for (auto* request : {&pair.actual, &pair.reference}) {
                request->set_tensor("input_ids", ids_tensor);
                request->set_tensor("position_ids", positions);
                request->set_tensor("attention_mask", mask);
                request->set_tensor("beam_idx", beam);
                if (pair.name == "mtp_draft")
                    request->set_tensor("hidden_states", hidden);
                request->infer();
            }
            for (const auto* name : {"logits", "last_hidden_state"}) {
                const auto actual = pair.actual.get_tensor(name);
                const auto expected = pair.reference.get_tensor(name);
                ASSERT_EQ(actual.get_shape(), expected.get_shape());
                for (size_t i = 0; i < actual.get_size(); ++i)
                    EXPECT_NEAR(actual.data<const float>()[i], expected.data<const float>()[i], 1e-5f);
            }
            const auto values = custom_states(pair.actual);
            ASSERT_EQ(values.size(), 3);
            const float scale = pair.name == "language" ? 1.f : 10.f;
            const float steps = past == 0 ? 1.f : 2.f;
            EXPECT_FLOAT_EQ(values.at("custom_state"), scale * steps);
            EXPECT_FLOAT_EQ(values.at("mtp.custom_state"), 2.f * scale * steps);
            EXPECT_FLOAT_EQ(values.at("legacy_state"), 3.f * scale * steps);
        }
        past += tokens;
    }
    requests.front().actual.reset_state();
    const auto draft_states = custom_states(requests.back().actual);
    ASSERT_EQ(draft_states.size(), 3);
    EXPECT_FLOAT_EQ(draft_states.at("custom_state"), 20.f);
    EXPECT_FLOAT_EQ(draft_states.at("mtp.custom_state"), 40.f);
    EXPECT_FLOAT_EQ(draft_states.at("legacy_state"), 60.f);
    ov::Tensor ids_tensor(ov::element::i64, {1, 1}), positions(ov::element::i64, {1, 1});
    ov::Tensor mask(ov::element::i64, {1, 1}), beam(ov::element::i32, {1});
    *ids_tensor.data<int64_t>() = 1;
    *positions.data<int64_t>() = 0;
    *mask.data<int64_t>() = 1;
    *beam.data<int32_t>() = 0;
    auto& target = requests.front().actual;
    target.set_tensor("input_ids", ids_tensor);
    target.set_tensor("position_ids", positions);
    target.set_tensor("attention_mask", mask);
    target.set_tensor("beam_idx", beam);
    target.infer();
    const auto reset_target_states = custom_states(target);
    ASSERT_EQ(reset_target_states.size(), 3);
    EXPECT_FLOAT_EQ(reset_target_states.at("custom_state"), 1.f);
    EXPECT_FLOAT_EQ(reset_target_states.at("mtp.custom_state"), 2.f);
    EXPECT_FLOAT_EQ(reset_target_states.at("legacy_state"), 3.f);
}

TEST(GGUFGenAIExtension, MtpComponentsRejectUnsupportedInputs) {
    using namespace ov::frontend::gguf;
    MtpFixture separate_embedding;
    separate_embedding.separate_embedding = true;
    struct Case {
        std::string arch;
        uint32_t layers;
        std::optional<MtpFixture> mtp;
        std::string component;
        std::string error;
    };
    const std::string per_layer = "per-layer settings must describe every decoder layer";
    const std::vector<Case> cases{
        {"qwen3", 2, std::nullopt, "mtp", "MTP is not supported for architecture"},
        {"qwen35", 0, MtpFixture{}, "mtp", "qwen35.block_count"},
        {"qwen35", 1, MtpFixture{}, "mtp", "invalid number of NextN blocks"},
        {"qwen35", 2, separate_embedding, "mtp", "shared token embeddings"},
        {"qwen35", 2, MtpFixture{{0, 1, 1}}, "mtp", per_layer},
        {"qwen35", 2, MtpFixture{{}, {1, 0, 0}}, "mtp", per_layer},
        {"qwen35", 2, MtpFixture{{}, {}, {0, 0, 0}}, "mtp", per_layer},
    };
    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.error);
        ScratchDir scratch;
        const auto path = write_decoder_gguf(scratch.path(), test_case.arch, test_case.layers, false, test_case.mtp);
        FrontEnd frontend;
        OV_EXPECT_THROW(frontend.convert(frontend.load(path, LoadOptions{test_case.component == "mtp"})),
                        ov::Exception,
                        testing::HasSubstr(test_case.error));
    }
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
        if (!m_ctx.component.empty())
            ctx.acknowledge_component();
        return ctx.finish();
    }

private:
    BuildContext m_ctx;
};

TEST(GGUFArchitectureExtension, ExternalBuilderCanAcknowledgeSelectedComponent) {
    ScratchDir scratch;
    const auto path = write_decoder_gguf(scratch.path(), "qwen3");
    ASSERT_FALSE(path.empty());

    ov::frontend::gguf::FrontEnd frontend;
    frontend.add_extension(std::make_shared<ArchitectureExtension>(
        ArchitectureDefinition{"qwen3",
                               "qwen3",
                               [](const BuildContext& context) {
                                   return std::make_shared<Qwen3Builder>(context);
                               }},
        RegistrationMode::Replace));
    const auto model =
        ov::frontend::gguf::adapt_to_genai(frontend.convert(frontend.load(path, ov::frontend::gguf::LoadOptions{true})))
            .at("language");
    ASSERT_TRUE(model);
    EXPECT_EQ(model->output().get_partial_shape(), (ov::PartialShape{-1, -1, 16}));
}

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
        frontend.add_extension(std::make_shared<ov::frontend::gguf::GenAIExtension>());
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

TEST(GGUFArchitectureExtension, TypedUnknownLayerNormFactoryControlsScalesAndLayerRope) {
    using namespace ov::frontend::gguf;
    ScratchDir scratch;
    const auto path = write_decoder_gguf(scratch.path(), "custom-normalization", 2, true);
    ASSERT_FALSE(path.empty());
    const auto definition = make_decoder_architecture("custom-normalization", [](const BuildContext& context) {
        DecoderParameters defaults;
        defaults.normalization.kind = NormalizationKind::Layer;
        auto parameters = read_decoder_parameters(context.metadata, defaults);
        parameters.attention.rope_mode = RopeMode::Neox;
        parameters.scales.embedding = 2.5f;
        parameters.scales.residual = 0.375f;
        parameters.scales.logits = 0.75f;
        parameters.layers.resize(2);
        parameters.layers[0].sliding_window = false;
        parameters.layers[1].sliding_window = true;
        auto local = parameters.attention.rope;
        local.freq_base = 32000.f;
        local.freq_scale = 0.5f;
        parameters.layers[1].rope = local;
        return parameters;
    });
    auto loaded = get_gguf_data(path);
    auto& [metadata, weights, mappings] = loaded;
    detail::MetadataStore store{metadata};
    detail::WeightStore tensors{weights};
    const BuildContext context{GgufMetadata(store), "custom-normalization", &tensors};
    const auto graph = definition.factory(context)->build();
    EXPECT_TRUE(graph->use_per_op_rope);
    const auto op = [&](const std::string& name) -> const GgufOp& {
        const auto found = std::find_if(graph->nodes.begin(), graph->nodes.end(), [&](const GgufOp& node) {
            return node.name == name;
        });
        OPENVINO_ASSERT(found != graph->nodes.end(), "missing expected operation ", name);
        return *found;
    };
    EXPECT_EQ(op("blk.0.attn_norm.norm").op_type, "GGML_OP_NORM");
    EXPECT_FLOAT_EQ(op("embd_scaled").attributes.at("scale").as<float>(), 2.5f);
    EXPECT_FLOAT_EQ(op("blk.0.attn_out_scaled").attributes.at("scale").as<float>(), 0.375f);
    EXPECT_FLOAT_EQ(op("result_output_scaled").attributes.at("scale").as<float>(), 0.75f);
    const auto extension = std::make_shared<ArchitectureExtension>(definition);
    const auto converted = convert_with(path, {extension});
    ASSERT_TRUE(converted);
    EXPECT_EQ(converted->output(0).get_partial_shape().rank().get_length(), 4);
}

TEST(GGUFArchitectureExtension, TypedGatedDeltaNetFactoryScalesBothResidualBranches) {
    using namespace ov::frontend::gguf;
    ScratchDir scratch;
    const auto path = write_decoder_gguf(scratch.path(), "custom-recurrent");
    ASSERT_FALSE(path.empty());
    auto loaded = get_gguf_data(path);
    auto& [metadata, weights, mappings] = loaded;
    for (int layer = 0; layer < 2; ++layer) {
        const auto prefix = "blk." + std::to_string(layer) + ".";
        const std::map<std::string, ov::Shape> shapes{{"attn_qkv.weight", {12, 8}},
                                                      {"attn_gate.weight", {4, 8}},
                                                      {"ssm_alpha.weight", {1, 8}},
                                                      {"ssm_beta.weight", {1, 8}},
                                                      {"ssm_conv1d.weight", {12, 3}},
                                                      {"ssm_a", {1}},
                                                      {"ssm_dt.bias", {1}},
                                                      {"ssm_norm.weight", {4}},
                                                      {"ssm_out.weight", {8, 4}}};
        for (const auto& [name, shape] : shapes) {
            ov::Tensor weight(ov::element::f32, shape);
            std::fill_n(weight.data<float>(), weight.get_size(), 0.1f);
            weights[prefix + name] = weight;
        }
    }
    detail::MetadataStore store{metadata};
    detail::WeightStore tensors{weights};
    const BuildContext context{GgufMetadata(store), "custom-recurrent", &tensors};
    for (float scale : {1.0f, 0.375f}) {
        const auto definition = make_decoder_architecture("custom-recurrent", [scale](const BuildContext& context) {
            auto parameters = read_decoder_parameters(context.metadata);
            parameters.hybrid.kind = RecurrentKind::GatedDeltaNet;
            parameters.hybrid.conv_kernel = 3;
            parameters.hybrid.state_size = 4;
            parameters.hybrid.group_count = 1;
            parameters.hybrid.time_step_rank = 1;
            parameters.hybrid.inner_size = 4;
            parameters.hybrid.recurrent_layers = {1, 1};
            parameters.scales.residual = scale;
            return parameters;
        });
        const auto graph = definition.factory(context)->build();
        const auto op = [&](const std::string& name) -> const GgufOp& {
            const auto found = std::find_if(graph->nodes.begin(), graph->nodes.end(), [&](const GgufOp& node) {
                return node.name == name;
            });
            OPENVINO_ASSERT(found != graph->nodes.end(), "missing expected operation ", name);
            return *found;
        };
        for (int layer = 0; layer < 2; ++layer) {
            const auto prefix = "blk." + std::to_string(layer) + ".";
            const auto expected_attention = scale == 1.0f ? prefix + "linear_attn_out" : prefix + "attn_out_scaled";
            const auto expected_ffn = scale == 1.0f ? prefix + "ffn_out" : prefix + "ffn_out_scaled";
            if (scale != 1.0f) {
                for (const auto& name : {"attn_out_scaled", "ffn_out_scaled"}) {
                    EXPECT_EQ(op(prefix + name).op_type, "GGML_OP_SCALE");
                    EXPECT_FLOAT_EQ(op(prefix + name).attributes.at("scale").as<float>(), scale);
                }
            }
            if (layer == 0) {
                EXPECT_EQ(op(prefix + "ffn_inp").input_names.front(), expected_attention);
            } else {
                EXPECT_EQ(op(prefix + "attn_out_g").input_names.front(), expected_attention);
                EXPECT_EQ(op(prefix + "ffn_inp").input_names.front(), prefix + "attn_out_g");
            }
            EXPECT_EQ(op(prefix + "l_out").input_names, (std::vector<std::string>{expected_ffn, prefix + "ffn_inp"}));
        }
    }
}
