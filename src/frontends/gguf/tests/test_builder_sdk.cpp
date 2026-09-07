// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <numeric>

#include "builder/arch_registry.hpp"
#include "builder/decoder_config.hpp"
#include "builder/gguf_builder_decoder.hpp"
#include "builder/gguf_graph.hpp"
#include "builder/sdk/metadata_store.hpp"
#include "gtest/gtest.h"
#include "openvino/frontend/gguf/builder/graph_context.hpp"
#include "openvino/frontend/gguf/frontend.hpp"
#include "openvino/frontend/gguf/make_stateful.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/manager.hpp"
#include "projector.hpp"

using namespace ov::frontend::gguf;

namespace {
struct Environment {
    std::unordered_map<std::string, GGUFMetaData> metadata;
    std::unordered_map<std::string, ov::Tensor> weights;
    std::unordered_map<std::string, GgufTensorType> qtypes;
    detail::MetadataStore store{metadata};
    detail::WeightStore tensors{weights, qtypes};
    BuildContext context{GgufMetadata(store), "test", &tensors};

    Environment() {
        metadata["general.architecture"] = std::string("test");
    }
    void integer(const std::string& key, uint32_t value) {
        ov::Tensor t(ov::element::u32, {});
        *t.data<uint32_t>() = value;
        metadata[key] = t;
    }
    void real(const std::string& key, float value) {
        ov::Tensor t(ov::element::f32, {});
        *t.data<float>() = value;
        metadata[key] = t;
    }
    void architecture(const std::string& name) {
        std::unordered_map<std::string, GGUFMetaData> renamed;
        for (const auto& entry : metadata)
            renamed[entry.first.rfind("test.", 0) == 0 ? name + entry.first.substr(4) : entry.first] = entry.second;
        metadata = std::move(renamed);
        metadata["general.architecture"] = name;
    }

    void decoder() {
        integer("test.block_count", 2);
        integer("test.embedding_length", 32);
        integer("test.attention.head_count", 4);
        real("test.attention.layer_norm_rms_epsilon", 1e-5f);
    }
};

std::shared_ptr<ov::Model> convert(const std::shared_ptr<GgufGraph>& graph) {
    FrontEnd frontend;
    std::shared_ptr<GgufDecoder> decoder = std::make_shared<GgufBuilderDecoder>(graph);
    return frontend.convert(frontend.load(decoder));
}

ArchitectureDefinition handler(const std::string& id, ArchitectureDefinition::MatchFn match = {}) {
    return {id,
            "test",
            [](const BuildContext&) -> std::shared_ptr<ModelBuilder> {
                return {};
            },
            std::move(match)};
}
}  // namespace

TEST(GGUFBuilderSDK, MetadataReadsReuseNumericConversionAcrossWidths) {
    Environment env;
    const auto& metadata = env.context.metadata;
    const auto store = [&](const std::string& key, const ov::op::v0::Constant& value) {
        ov::TensorVector output;
        ASSERT_TRUE(value.evaluate(output, {}));
        env.metadata[key] = output.front();
    };
    for (auto type : {ov::element::boolean,
                      ov::element::i8,
                      ov::element::u8,
                      ov::element::i16,
                      ov::element::u16,
                      ov::element::i32,
                      ov::element::u32,
                      ov::element::i64,
                      ov::element::u64,
                      ov::element::f32,
                      ov::element::f64}) {
        SCOPED_TRACE(type.get_type_name());
        store("scalar", ov::op::v0::Constant(type, {}, 1));
        store("array", ov::op::v0::Constant(type, {3}, 1));
        EXPECT_EQ(metadata.get_float("scalar"), 1.0);
        EXPECT_EQ(metadata.get_float_array("array"), (std::vector<double>{1, 1, 1}));
        EXPECT_FALSE(metadata.get_float("array"));
        if (type.is_real()) {
            EXPECT_FALSE(metadata.get_int("scalar"));
            EXPECT_TRUE(metadata.get_int_array("array").empty());
        } else {
            EXPECT_EQ(metadata.get_int("scalar"), 1);
            EXPECT_EQ(metadata.get_bool("scalar"), true);
            EXPECT_EQ(metadata.get_int_array("array"), (std::vector<int64_t>{1, 1, 1}));
        }
    }
    store("signed", ov::op::v0::Constant(ov::element::i64, {}, -7));
    EXPECT_EQ(metadata.get_int("signed"), -7);
    store("unsigned", ov::op::v0::Constant(ov::element::u64, {}, uint64_t{1} << 63));
    EXPECT_EQ(metadata.get_float("unsigned"), double(uint64_t{1} << 63));
}

TEST(GGUFBuilderSDK, MetadataMissingAndIncompatibleValuesRemainOptional) {
    Environment env;
    const auto& metadata = env.context.metadata;
    env.metadata["string"] = std::string("value");
    env.metadata["strings"] = std::vector<std::string>{"a", "b"};
    env.metadata["integers"] = std::vector<int32_t>{-2, 3};
    env.metadata["integer"] = 7;
    env.metadata["float"] = 0.5f;
    for (const auto* key : {"missing", "string", "strings"}) {
        EXPECT_FALSE(metadata.get_int(key));
        EXPECT_FALSE(metadata.get_float(key));
        EXPECT_FALSE(metadata.get_bool(key));
        EXPECT_TRUE(metadata.get_int_array(key).empty());
        EXPECT_TRUE(metadata.get_float_array(key).empty());
    }
    EXPECT_FALSE(metadata.has("missing"));
    EXPECT_TRUE(metadata.has("string"));
    EXPECT_EQ(metadata.get_str("string"), "value");
    EXPECT_EQ(metadata.get_str("missing").value_or("fallback"), "fallback");
    EXPECT_EQ(metadata.get_str_array("strings"), (std::vector<std::string>{"a", "b"}));
    EXPECT_EQ(metadata.get_int_array("integers"), (std::vector<int64_t>{-2, 3}));
    EXPECT_EQ(metadata.get_int("integer"), 7);
    EXPECT_EQ(metadata.get_float("float"), 0.5);
}

TEST(GGUFBuilderSDK, LogicalWeightDimensionsPreserveVectorsAndExpertAxes) {
    Environment env;
    env.weights["norm.weight"] = ov::Tensor(ov::element::f32, {32});
    env.weights["experts.weight"] = ov::Tensor(ov::element::f32, {2, 3, 4});
    GgufGraphContext graph(env.context);
    auto vector = graph.tensors().require("norm.weight");
    EXPECT_EQ(vector.ne(0), 32);
    EXPECT_EQ(vector.ne(1), 1);
    auto experts = graph.tensors().require("experts.weight");
    EXPECT_EQ(experts.ne(0), 4);
    EXPECT_EQ(experts.ne(1), 3);
    EXPECT_EQ(experts.ne(2), 2);
}

TEST(GGUFBuilderSDK, GeneralReshapeDoesNotCopyTheAttentionBatchAxis) {
    Environment env;
    GgufGraphContext graph(env.context);
    auto input = graph.add_input("x", ov::element::f32, {2, 3, 4, 5});
    graph.set_output(graph.reshape(input, {10, 3, 4}));
    auto model = convert(graph.finish());
    EXPECT_EQ(model->output().get_shape(), (ov::Shape{1, 4, 3, 10}));
    ov::Tensor data(ov::element::f32, {2, 3, 4, 5});
    std::iota(data.data<float>(), data.data<float>() + data.get_size(), 0.0f);
    ov::TensorVector output{ov::Tensor(ov::element::f32, {1, 4, 3, 10})};
    ASSERT_TRUE(model->evaluate(output, {data}));
    for (size_t i = 0; i < data.get_size(); ++i)
        EXPECT_EQ(output[0].data<float>()[i], float(i));
}

TEST(GGUFBuilderSDK, ExplicitInferredReshapeRemainsDynamicAcrossTokenCounts) {
    Environment env;
    GgufGraphContext graph(env.context);
    auto input = graph.add_input("x", ov::element::f32, {1, 1, -1, 8});
    graph.set_output(graph.reshape(input, {4, 2, -1}));
    auto model = convert(graph.finish());
    for (size_t tokens : {1u, 3u, 7u}) {
        ov::Tensor data(ov::element::f32, {1, 1, tokens, 8});
        std::iota(data.data<float>(), data.data<float>() + data.get_size(), 1.0f);
        ov::TensorVector output{ov::Tensor(ov::element::f32, {1, tokens, 2, 4})};
        ASSERT_TRUE(model->evaluate(output, {data}));
        EXPECT_EQ(output[0].get_shape(), (ov::Shape{1, tokens, 2, 4}));
        for (size_t i = 0; i < data.get_size(); ++i)
            EXPECT_EQ(output[0].data<float>()[i], float(i + 1));
    }
}

TEST(GGUFBuilderSDK, DecoderOptionsAreResolvedBeforeRopePlans) {
    Environment env;
    env.decoder();
    env.real("test.rope.scaling.factor", 4.0f);
    DecoderOptions options;
    options.sliding_window = 16;
    options.swa_rope_frequency_base = 2000.f;
    options.swa_rope_dimensions = 4;
    GgufGraphContext graph(env.context);
    auto dimensions = graph.configure_decoder(RopeMode::Neox, options);
    EXPECT_EQ(dimensions.embedding, 32);
    const auto layer = graph.decoder_layer_parameters(0);
    EXPECT_EQ(layer.kv_heads, 4);
    EXPECT_EQ(layer.rope.n_dims, 4);
    EXPECT_FLOAT_EQ(layer.rope.freq_base, 2000.f);
    auto input = graph.add_input("x", ov::element::f32, {1, 1, 1, 32});
    graph.set_output(input);
    auto built = graph.finish();
    EXPECT_FLOAT_EQ(built->rope_config.freq_scale, 0.25f);
    EXPECT_TRUE(built->use_per_op_rope);
    EXPECT_EQ(built->swa_window_size, 16);
    DecoderOptions invalid;
    invalid.swa_rope_dimensions = 33;
    GgufGraphContext other(env.context);
    EXPECT_THROW(other.configure_decoder(RopeMode::Neox, invalid), ov::Exception);
}

TEST(GGUFBuilderSDK, RecurrentStateDeclarationReachesMakeStateful) {
    Environment env;
    GgufGraphContext graph(env.context);
    auto state = graph.add_input("state", ov::element::f32, {1, 1, 1, 4});
    auto input = graph.add_input("x", ov::element::f32, {1, 1, 1, 4});
    auto update = graph.add(state, input);
    graph.add_recurrent_state(state, update);
    graph.set_output(graph.scale(update, 2));
    auto model = convert(graph.finish());
    ASSERT_TRUE(model->get_rt_info().count(pass::gguf_recurrent_states_key()));
    ov::pass::Manager passes;
    passes.register_pass<pass::GGUFMakeStateful>();
    passes.run_passes(model);
    EXPECT_EQ(model->inputs().size(), 1);
    EXPECT_EQ(model->outputs().size(), 1);
    EXPECT_EQ(model->get_sinks().size(), 1);
}

TEST(GGUFBuilderSDK, SlidingWindowDeclarationSurvivesConversion) {
    Environment env;
    GgufGraphContext graph(env.context);
    graph.set_sliding_window(32);
    graph.set_output(graph.add_input("x", ov::element::f32, {1, 1, 1, 4}));
    auto model = convert(graph.finish());
    EXPECT_EQ(model->get_rt_info().at(pass::gguf_swa_window_key()).as<int64_t>(), 32);
    EXPECT_THROW(graph.set_sliding_window(64), ov::Exception);
    EXPECT_THROW(graph.finish(), ov::Exception);
}

TEST(GGUFArchitectureRegistry, DisjointHandlersForTheSameArchitectureCoexist) {
    Environment env;
    env.integer("modality", 1);
    ArchRegistry registry({handler("vision",
                                   [](const GgufMetadata& m) {
                                       return m.get_int("modality") == 1;
                                   }),
                           handler("audio", [](const GgufMetadata& m) {
                               return m.get_int("modality") == 2;
                           })});
    ASSERT_TRUE(registry.find(env.context.metadata));
    EXPECT_EQ(registry.find(env.context.metadata)->id, "vision");
    env.integer("modality", 2);
    EXPECT_EQ(registry.find(env.context.metadata)->id, "audio");
    env.metadata["general.architecture"] = std::string("unrelated");
    EXPECT_FALSE(registry.find(env.context.metadata));
}

TEST(GGUFArchitectureRegistry, ReplacementIsExplicitAndInstancesAreIsolated) {
    Environment env;
    ArchRegistry first({handler("test")});
    ArchRegistry second({handler("test")});
    EXPECT_THROW(first.add(handler("test")), ov::Exception);
    auto replacement = handler("test");
    replacement.maturity = Maturity::Verified;
    first.add(replacement, RegistrationMode::Replace);
    EXPECT_EQ(first.find(env.context.metadata)->maturity, Maturity::Verified);
    EXPECT_EQ(second.find(env.context.metadata)->maturity, Maturity::Experimental);
    EXPECT_THROW(first.add(handler("missing"), RegistrationMode::Replace), ov::Exception);
    auto invalid = handler("invalid");
    invalid.factory = {};
    EXPECT_THROW(first.add(invalid), ov::Exception);
}

// Promotion changes only where the definition is registered, including its custom builder.
TEST(GGUFArchitectureRegistry, PromotedDefinitionUsesTheSameBuilderAndDispatch) {
    Environment env;
    env.metadata["general.architecture"] = std::string("example-projector");
    env.context.arch = "example-projector";
    env.weights["projection.weight"] = ov::Tensor(ov::element::f32, {3, 2});
    const auto definition = example::projector_architecture();
    ArchRegistry native({definition});
    ArchRegistry external(std::vector<ArchitectureDefinition>{});
    external.add_extension(std::make_shared<ArchitectureExtension>(definition));
    auto first = native.find(env.context.metadata)->factory(env.context)->build();
    auto second = external.find(env.context.metadata)->factory(env.context)->build();
    ASSERT_EQ(first->nodes.size(), second->nodes.size());
    EXPECT_EQ(first->model_output_names, second->model_output_names);
    for (size_t i = 0; i < first->nodes.size(); ++i) {
        EXPECT_EQ(first->nodes[i].op_type, second->nodes[i].op_type);
        EXPECT_EQ(first->nodes[i].output_shape, second->nodes[i].output_shape);
    }
}

TEST(GGUFBuilderSDK, SingleHeadSplitPreservesDynamicTokens) {
    Environment env;
    GgufGraphContext graph(env.context);
    auto input = graph.add_input("x", ov::element::f32, {1, 1, -1, 8});
    auto heads = graph.split_heads(input, 1, 8);
    graph.set_output(heads);
    graph.set_output(graph.merge_heads(heads));
    auto model = convert(graph.finish());
    ov::Tensor data(ov::element::f32, {1, 1, 3, 8});
    std::iota(data.data<float>(), data.data<float>() + data.get_size(), 1.0f);
    ov::TensorVector results{ov::Tensor(ov::element::f32, {1, 3, 1, 8}), ov::Tensor(ov::element::f32, {1, 1, 3, 8})};
    ASSERT_TRUE(model->evaluate(results, {data}));
    EXPECT_EQ(results[0].get_shape(), (ov::Shape{1, 3, 1, 8}));
    EXPECT_EQ(results[1].get_shape(), data.get_shape());
}

TEST(GGUFBuilderSDK, Gemma2DefaultsIncludeSlidingWindowAnd27BAttentionScale) {
    Environment env;
    env.decoder();
    env.integer("test.block_count", 46);
    env.integer("test.embedding_length", 4608);
    env.integer("test.attention.head_count", 32);
    env.integer("test.attention.key_length", 128);
    env.architecture("gemma2");
    DecoderConfig config(decoder_config_from_meta(env.metadata), env.weights);
    EXPECT_EQ(config.swa_window_size, 4096);
    EXPECT_TRUE(config.layer_is_swa(0));
    EXPECT_FALSE(config.layer_is_swa(1));
    EXPECT_FLOAT_EQ(config.layer_kq_scale(0), 1.f / 12.f);
}

TEST(GGUFBuilderSDK, ErnieInterleavesDenseAndExpertLayers) {
    Environment env;
    env.decoder();
    env.integer("test.block_count", 4);
    env.integer("test.interleave_moe_layer_step", 2);
    env.weights["blk.1.ffn_gate_exps.weight"] = ov::Tensor(ov::element::f32, {4, 48, 32});
    env.architecture("ernie4_5-moe");
    for (uint32_t dense_lead : {0u, 1u}) {
        env.integer("ernie4_5-moe.leading_dense_block_count", dense_lead);
        DecoderConfig config(decoder_config_from_meta(env.metadata), env.weights);
        EXPECT_FALSE(config.layer_is_moe(0));
        EXPECT_TRUE(config.layer_is_moe(1));
        EXPECT_FALSE(config.layer_is_moe(2));
        EXPECT_TRUE(config.layer_is_moe(3));
        EXPECT_TRUE(config.expert_weights_norm);
    }
}

TEST(GGUFBuilderSDK, Exaone64LayerDefaultsIncludeLocalRopeAndSlidingWindow) {
    Environment env;
    env.decoder();
    env.integer("test.block_count", 64);
    env.architecture("exaone4");
    DecoderConfig config(decoder_config_from_meta(env.metadata), env.weights);
    EXPECT_TRUE(config.post_norm_only);
    EXPECT_TRUE(config.rope_on_swa_only);
    EXPECT_EQ(config.swa_window_size, 4096);
    EXPECT_TRUE(config.layer_is_swa(0));
    EXPECT_TRUE(config.layer_is_swa(2));
    EXPECT_FALSE(config.layer_is_swa(3));
    EXPECT_TRUE(config.layer_is_swa(4));
}

TEST(GGUFBuilderSDK, ArchitectureOptionsOverrideAmbiguousTensorSemantics) {
    Environment env;
    env.decoder();
    DecoderOptions options;
    options.qk_norm_after_rope = true;
    options.post_norm_only = true;
    options.normalize_expert_weights = true;
    options.rope_skip_period = 4;
    DecoderConfig config(decoder_config_from_meta(env.metadata), env.weights, RopeMode::Neox, options);
    EXPECT_TRUE(config.qk_norm_after_rope);
    EXPECT_TRUE(config.post_norm_only);
    EXPECT_TRUE(config.expert_weights_norm);
    EXPECT_EQ(config.rope_skip_period, 4);
    options.rope_skip_period = -1;
    EXPECT_THROW(DecoderConfig(decoder_config_from_meta(env.metadata), env.weights, RopeMode::Neox, options),
                 ov::Exception);
}
