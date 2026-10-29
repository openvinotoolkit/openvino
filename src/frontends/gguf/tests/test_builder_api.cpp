// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <limits>
#include <numeric>

#include "builder/api/metadata_store.hpp"
#include "builder/arch_registry.hpp"
#include "builder/decoder_config.hpp"
#include "builder/gguf_builder_decoder.hpp"
#include "builder/gguf_graph.hpp"
#include "gtest/gtest.h"
#include "openvino/core/type/float16.hpp"
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
    DecoderParameters parameters(RopeMode rope = RopeMode::Normal, const DecoderOptions& options = {}) const {
        return resolve_decoder_parameters(context.metadata, rope, options);
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

TEST(GGUFBuilderAPI, MetadataReadsReuseNumericConversionAcrossWidths) {
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

TEST(GGUFBuilderAPI, MetadataMissingAndIncompatibleValuesRemainOptional) {
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
    for (float value : {0.5f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        env.metadata["float"] = value;
        EXPECT_FALSE(metadata.get_int("float"));
        EXPECT_FALSE(metadata.get_bool("float"));
    }
}

TEST(GGUFBuilderAPI, RejectsValuesFromAnotherGraphWithMatchingNames) {
    Environment env;
    GgufGraphContext graph(env.context), other(env.context);
    auto local = graph.add_input("state", ov::element::f32, {1, 1, 1, 4});
    auto foreign = other.add_input("state", ov::element::f32, {1, 1, 1, 4});
    EXPECT_THROW(graph.node("GGML_OP_ADD", {local, foreign}), ov::Exception);
    EXPECT_THROW(graph.build_norm(foreign, {}, 1e-5f), ov::Exception);
    EXPECT_THROW(graph.build_norm(local, foreign, 1e-5f), ov::Exception);
    EXPECT_THROW(graph.set_output(foreign), ov::Exception);
    EXPECT_THROW(graph.add_recurrent_state(local, foreign), ov::Exception);
    EXPECT_THROW(graph.add_recurrent_state(foreign, local), ov::Exception);
    graph.set_output(local);
    EXPECT_NO_THROW(convert(graph.finish()));
}

TEST(GGUFBuilderAPI, GeneratedNamesDoNotOverwriteInputs) {
    Environment env;
    GgufGraphContext graph(env.context);
    auto input = graph.add_input("GGML_OP_SCALE_0", ov::element::f32, {1, 1, 1, 4});
    auto output = graph.node("GGML_OP_SCALE", {input}, 0, {{"scale", 2.f}});
    EXPECT_NE(output.name(), input.name());
    EXPECT_THROW(graph.add_input(output.name(), ov::element::f32, input.shape()), ov::Exception);
    graph.set_output(input);
    graph.set_output(output);
    auto model = convert(graph.finish());
    auto data = ov::op::v0::Constant::create(ov::element::f32, {1, 1, 1, 4}, {1, 2, 3, 4});
    ov::TensorVector inputs;
    ov::TensorVector outputs{ov::Tensor(ov::element::f32, {1, 1, 1, 4}), ov::Tensor(ov::element::f32, {1, 1, 1, 4})};
    ASSERT_TRUE(data->evaluate(inputs, {}));
    ASSERT_TRUE(model->evaluate(outputs, inputs));
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(outputs[0].data<float>()[i], float(i + 1));
        EXPECT_EQ(outputs[1].data<float>()[i], float(2 * (i + 1)));
    }
}

TEST(GGUFBuilderAPI, NormalizationPreservesInputNamesAndAcceptsGraphValues) {
    Environment env;
    GgufGraphContext graph(env.context);
    auto input = graph.add_input("norm_0.rms", ov::element::f32, {1, 1, 1, 4});
    auto output = graph.build_norm(input, input, 0.f);
    graph.set_output(input);
    graph.set_output(output);
    auto model = convert(graph.finish());
    auto data = ov::op::v0::Constant::create(ov::element::f32, {1, 1, 1, 4}, {1, 2, 3, 4});
    ov::TensorVector inputs;
    ov::TensorVector outputs{ov::Tensor(ov::element::f32, {1, 1, 1, 4}), ov::Tensor(ov::element::f32, {1, 1, 1, 4})};
    ASSERT_TRUE(data->evaluate(inputs, {}));
    ASSERT_TRUE(model->evaluate(outputs, inputs));
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(outputs[0].data<float>()[i], float(i + 1));
        EXPECT_NEAR(outputs[1].data<float>()[i], float((i + 1) * (i + 1)) / std::sqrt(7.5f), 1e-6f);
    }
}

TEST(GGUFBuilderAPI, LogicalWeightDimensionsPreserveVectorsAndExpertAxes) {
    GgufValue empty;
    EXPECT_TRUE(empty.shape().rank().is_dynamic());
    EXPECT_EQ(empty.type(), ov::element::dynamic);
    EXPECT_EQ(empty.ne(0), -1);
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
    EXPECT_THROW(graph.node("GGML_OP_ADD", {empty, vector}), ov::Exception);
}

TEST(GGUFBuilderAPI, GeneralReshapeDoesNotCopyTheAttentionBatchAxis) {
    Environment env;
    GgufGraphContext graph(env.context);
    auto input = graph.add_input("x", ov::element::f32, {2, 3, 4, 5});
    graph.set_output(
        graph.node("GGML_OP_RESHAPE", {input}, 6, {{"reshape_target", std::vector<int64_t>{1, 4, 3, 10}}}));
    auto model = convert(graph.finish());
    EXPECT_EQ(model->output().get_shape(), (ov::Shape{1, 4, 3, 10}));
    ov::Tensor data(ov::element::f32, {2, 3, 4, 5});
    std::iota(data.data<float>(), data.data<float>() + data.get_size(), 0.0f);
    ov::TensorVector output{ov::Tensor(ov::element::f32, {1, 4, 3, 10})};
    ASSERT_TRUE(model->evaluate(output, {data}));
    for (size_t i = 0; i < data.get_size(); ++i)
        EXPECT_EQ(output[0].data<float>()[i], float(i));
}

TEST(GGUFBuilderAPI, ExplicitInferredReshapeRemainsDynamicAcrossTokenCounts) {
    Environment env;
    GgufGraphContext graph(env.context);
    auto input = graph.add_input("x", ov::element::f32, {1, 1, -1, 8});
    graph.set_output(
        graph.node("GGML_OP_RESHAPE", {input}, 6, {{"reshape_target", std::vector<int64_t>{1, -1, 2, 4}}}));
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

TEST(GGUFBuilderAPI, DecoderOptionsAreResolvedBeforeRopePlans) {
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

TEST(GGUFBuilderAPI, RopeScalingDefaultsAndPositiveFactors) {
    Environment env;
    env.decoder();
    EXPECT_FLOAT_EQ(env.parameters().attention.rope.freq_scale, 1.0f);
    for (const float factor : {0.0f, 1.0f, 4.0f}) {
        env.real("test.rope.scaling.factor", factor);
        EXPECT_FLOAT_EQ(env.parameters().attention.rope.freq_scale, factor == 0.0f ? 1.0f : 1.0f / factor);
    }
}

TEST(GGUFBuilderAPI, RejectsInvalidRopeScalingFactors) {
    for (const float factor : {-1.0f,
                               std::numeric_limits<float>::infinity(),
                               std::numeric_limits<float>::quiet_NaN(),
                               std::numeric_limits<float>::denorm_min()}) {
        SCOPED_TRACE(factor);
        Environment env;
        env.decoder();
        env.real("test.rope.scaling.factor", factor);
        EXPECT_THROW(env.parameters(), ov::Exception);
    }
}

TEST(GGUFBuilderAPI, RecurrentStateDeclarationReachesMakeStateful) {
    Environment env;
    GgufGraphContext graph(env.context);
    auto state = graph.add_input("state", ov::element::f32, {1, 1, 1, 4});
    auto input = graph.add_input("x", ov::element::f32, {1, 1, 1, 4});
    auto update = graph.node("GGML_OP_ADD", {state, input});
    graph.add_recurrent_state(state, update);
    const auto output = graph.node("GGML_OP_SCALE", {update}, 0, {{"scale", float{2}}, {"bias", 0.0f}});
    graph.set_primary_output(output);
    const auto built = graph.finish();
    ASSERT_EQ(built->model_output_names.front(), output.name());
    auto model = convert(built);
    ASSERT_TRUE(model->get_rt_info().count(pass::gguf_recurrent_states_key()));
    ov::pass::Manager passes;
    passes.register_pass<pass::GGUFMakeStateful>();
    passes.run_passes(model);
    EXPECT_EQ(model->inputs().size(), 1);
    EXPECT_EQ(model->outputs().size(), 1);
    EXPECT_EQ(model->get_sinks().size(), 1);
}

// A named output can expose an input or another output; every name stays addressable.
TEST(GGUFBuilderAPI, NamedOutputsKeepTheNamesOfTheValuesTheyExpose) {
    Environment env;
    GgufGraphContext graph(env.context);
    auto input = graph.add_input("x", ov::element::f32, {1, 1, 1, 4});
    graph.set_output(input, "first");
    graph.set_output(input, "second");
    auto model = convert(graph.finish());
    EXPECT_NO_THROW(model->input("x"));
    EXPECT_NO_THROW(model->output("first"));
    EXPECT_NO_THROW(model->output("second"));
}

TEST(GGUFBuilderAPI, SlidingWindowDeclarationSurvivesConversion) {
    Environment env;
    GgufGraphContext graph(env.context);
    graph.set_sliding_window(32);
    graph.set_output(graph.add_input("x", ov::element::f32, {1, 1, 1, 4}));
    auto model = convert(graph.finish());
    EXPECT_EQ(model->get_rt_info().at(pass::gguf_swa_window_key()).as<int64_t>(), 32);
    EXPECT_THROW(graph.set_sliding_window(64), ov::Exception);
    EXPECT_THROW(graph.finish(), ov::Exception);
}

TEST(GGUFArchitectureRegistry, SupportedListIncludesCustomFamiliesAndExtensions) {
    ArchRegistry registry;
    const auto builtin_names = registry.supported_archs();
    EXPECT_EQ(builtin_names.count("clip"), 1);
    EXPECT_EQ(builtin_names.count("mamba2"), 1);
    EXPECT_EQ(builtin_names.count("nemotron_h"), 1);
    EXPECT_EQ(builtin_names.count("qwen3"), 1);
    Environment env;
    env.metadata["general.architecture"] = std::string("clip");
    EXPECT_FALSE(registry.find(env.context.metadata));
    env.metadata["clip.vision.projector_type"] = std::string("mlp");
    ASSERT_TRUE(registry.find(env.context.metadata));
    EXPECT_EQ(registry.find(env.context.metadata)->id, "clip.mmproj");

    registry.add_extension(std::make_shared<ArchitectureExtension>("custom-decoder", RopeMode::Neox));
    EXPECT_EQ(registry.supported_archs().size(), builtin_names.size() + 1);
    EXPECT_EQ(registry.supported_archs().count("custom-decoder"), 1);
    EXPECT_NE(registry.describe_supported().find("custom-decoder"), std::string::npos);
    EXPECT_EQ(ArchRegistry{}.supported_archs(), builtin_names);
}

TEST(GGUFArchitectureRegistry, SupportedNamesTrackReplacementAndDeduplicateHandlers) {
    ArchRegistry registry({handler("vision"), handler("audio")});
    EXPECT_EQ(registry.supported_archs(), (std::set<std::string>{"test"}));
    EXPECT_EQ(registry.describe_supported(), "test");
    auto replacement = handler("vision");
    replacement.architecture = "other";
    registry.add_extension(std::make_shared<ArchitectureExtension>(replacement, RegistrationMode::Replace));
    EXPECT_EQ(registry.supported_archs(), (std::set<std::string>{"other", "test"}));
    replacement.id = "audio";
    registry.add_extension(std::make_shared<ArchitectureExtension>(replacement, RegistrationMode::Replace));
    EXPECT_EQ(registry.supported_archs(), (std::set<std::string>{"other"}));
    EXPECT_EQ(registry.describe_supported(), "other");
}

TEST(GGUFArchitectureRegistry, Lfm2AudioVocoderListsOnlyCanonicalName) {
    ArchRegistry registry;
    EXPECT_EQ(registry.supported_archs().count("lfm2.audio_vocoder"), 1);
    Environment env;
    const std::string publisher_name = "this model cannot be used as LLM, use it via --model-vocoder in TTS examples";
    EXPECT_EQ(registry.supported_archs().count(publisher_name), 0);
    EXPECT_EQ(registry.describe_supported().find(publisher_name), std::string::npos);
    env.metadata["general.architecture"] = std::string("lfm2.audio_vocoder");
    EXPECT_FALSE(registry.find(env.context.metadata));
    env.integer("depthformer_n_layer", 2);
    EXPECT_FALSE(registry.find(env.context.metadata));
    env.integer("depthformer_n_embd", 16);
    const auto definition = registry.find(env.context.metadata);
    ASSERT_TRUE(definition);
    EXPECT_EQ(definition->architecture, "lfm2.audio_vocoder");
    EXPECT_TRUE(definition->factory);

    auto replacement = *registry.find(env.context.metadata);
    EXPECT_EQ(replacement.id, "lfm2.audio_vocoder");
    registry.add_extension(std::make_shared<ArchitectureExtension>(replacement, RegistrationMode::Replace));
    EXPECT_NO_THROW(registry.find(env.context.metadata));
    ASSERT_TRUE(registry.find(env.context.metadata));
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
    replacement.match = [](const GgufMetadata&) {
        return false;
    };
    first.add(replacement, RegistrationMode::Replace);
    EXPECT_FALSE(first.find(env.context.metadata));
    EXPECT_TRUE(second.find(env.context.metadata));
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
        EXPECT_EQ(first->values->at(first->nodes[i].output_name).get_partial_shape(),
                  second->values->at(second->nodes[i].output_name).get_partial_shape());
    }
}

TEST(GGUFBuilderAPI, SingleHeadSplitPreservesDynamicTokens) {
    Environment env;
    GgufGraphContext graph(env.context);
    auto input = graph.add_input("x", ov::element::f32, {1, 1, -1, 8});
    auto heads = graph.node("GGML_OP_RESHAPE",
                            {input},
                            6,
                            {{"reshape_target", std::vector<int64_t>{0, -1, 1, 8}}, {"special_zero", true}});
    graph.set_output(heads);
    graph.set_output(graph.node("GGML_OP_RESHAPE", {heads}, 2, {{"merge_heads", true}}));
    auto model = convert(graph.finish());
    ov::Tensor data(ov::element::f32, {1, 1, 3, 8});
    std::iota(data.data<float>(), data.data<float>() + data.get_size(), 1.0f);
    ov::TensorVector results{ov::Tensor(ov::element::f32, {1, 3, 1, 8}), ov::Tensor(ov::element::f32, {1, 1, 3, 8})};
    ASSERT_TRUE(model->evaluate(results, {data}));
    EXPECT_EQ(results[0].get_shape(), (ov::Shape{1, 3, 1, 8}));
    EXPECT_EQ(results[1].get_shape(), data.get_shape());
}

TEST(GGUFBuilderAPI, Gemma2DefaultsIncludeSlidingWindowAnd27BAttentionScale) {
    Environment env;
    env.decoder();
    env.integer("test.block_count", 46);
    env.integer("test.embedding_length", 4608);
    env.integer("test.attention.head_count", 32);
    env.integer("test.attention.key_length", 128);
    env.architecture("gemma2");
    DecoderConfig config(env.parameters(), env.weights);
    EXPECT_EQ(config.swa_window_size, 4096);
    EXPECT_TRUE(config.layer_is_swa(0));
    EXPECT_FALSE(config.layer_is_swa(1));
    EXPECT_FLOAT_EQ(config.layer_kq_scale(0), 1.f / 12.f);
}

TEST(GGUFBuilderAPI, Gemma2RejectsZeroAttentionHeadsWithOrWithoutKeyLength) {
    for (bool explicit_key_length : {false, true}) {
        SCOPED_TRACE(explicit_key_length);
        Environment env;
        env.decoder();
        env.integer("test.block_count", 46);
        env.integer("test.attention.head_count", 0);
        if (explicit_key_length) {
            env.integer("test.attention.key_length", 128);
        }
        env.architecture("gemma2");
        EXPECT_THROW(env.parameters(), ov::Exception);
    }
}

TEST(GGUFBuilderAPI, SlidingPolicyIsResolvedBeforeInferringNormalizationWidth) {
    for (const bool sinks : {false, true}) {
        for (const bool full_width : {false, true}) {
            SCOPED_TRACE(sinks);
            SCOPED_TRACE(full_width);
            Environment env;
            env.decoder();
            auto parameters = read_decoder_parameters(env.context.metadata);
            parameters.dimensions.head_size = 128;
            parameters.attention.sliding_head_size = 64;
            parameters.attention.sliding_period = 2;
            parameters.attention.window_size = sinks ? 0 : 16;
            if (sinks) {
                env.weights["blk.0.attn_sinks.weight"] = ov::Tensor(ov::element::f32, {4});
            }
            env.weights["blk.0.attn_q_norm.weight"] = ov::Tensor(ov::element::f32, {full_width ? 256u : 64u});
            DecoderConfig config(parameters, env.weights);
            EXPECT_TRUE(config.layer_is_swa(0));
            EXPECT_EQ(config.layer_head_size(0), 64);
            EXPECT_EQ(config.layer_head_size(1), 128);
            EXPECT_TRUE(config.has_qk_norm);
            EXPECT_EQ(config.qk_norm_full, full_width);
            parameters.normalization.qk_full_width = !full_width;
            EXPECT_EQ(DecoderConfig(parameters, env.weights).qk_norm_full, !full_width);
        }
    }
}

TEST(GGUFBuilderAPI, OneLayerBooleanSlidingPatternRemainsALayerFlag) {
    for (const bool sliding : {false, true}) {
        SCOPED_TRACE(sliding);
        Environment env;
        env.decoder();
        env.integer("test.block_count", 1);
        env.integer("test.attention.key_length", 128);
        env.integer("test.attention.key_length_swa", 64);
        env.integer("test.rope.dimension_count", 64);
        env.real("test.rope.freq_base", 32000.f);
        env.real("test.rope.freq_base_swa", 10000.f);
        ov::Tensor pattern(ov::element::boolean, {1});
        *pattern.data<char>() = sliding;
        env.metadata["test.attention.sliding_window_pattern"] = pattern;
        const auto parameters = read_decoder_parameters(env.context.metadata);
        EXPECT_EQ(parameters.attention.sliding_layers, (std::vector<int32_t>{sliding ? 1 : 0}));
        EXPECT_EQ(parameters.attention.sliding_period, 0);
        DecoderConfig config(parameters, env.weights);
        EXPECT_EQ(config.layer_is_swa(0), sliding);
        EXPECT_EQ(config.layer_head_size(0), sliding ? 64 : 128);
        EXPECT_FLOAT_EQ(config.layer_rope_config(0).freq_base, sliding ? 10000.f : 32000.f);
        env.integer("test.attention.sliding_window_pattern", 2);
        const auto periodic = read_decoder_parameters(env.context.metadata);
        EXPECT_TRUE(periodic.attention.sliding_layers.empty());
        EXPECT_EQ(periodic.attention.sliding_period, 2);
    }
}

TEST(GGUFBuilderAPI, SlidingWindowDefaultsMatchExaoneMoeAndPlamo3) {
    for (const auto* family : {"exaone-moe", "plamo3"}) {
        SCOPED_TRACE(family);
        Environment env;
        env.decoder();
        env.integer("test.block_count", 8);
        env.integer("test.attention.sliding_window", 128);
        env.real("test.rope.freq_base", 1000000.f);
        env.architecture(family);
        DecoderConfig config(env.parameters(), env.weights);
        const bool exaone = std::string(family) == "exaone-moe";
        EXPECT_EQ(config.swa_layer_pattern, exaone ? 4 : 8);
        EXPECT_TRUE(config.layer_is_swa(0));
        EXPECT_FALSE(config.layer_is_swa(exaone ? 3 : 7));
        EXPECT_EQ(config.rope_on_swa_only, exaone);
        EXPECT_FLOAT_EQ(config.rope_freq_base_swa, exaone ? 1000000.f : 10000.f);
    }
}

TEST(GGUFBuilderAPI, Glm4MoeDefaultsToSigmoidRouting) {
    Environment env;
    env.decoder();
    env.architecture("glm4moe");
    EXPECT_TRUE(DecoderConfig(env.parameters(), env.weights).moe_sigmoid_gating);
    env.integer("glm4moe.expert_gating_func", 0);
    EXPECT_TRUE(DecoderConfig(env.parameters(), env.weights).moe_sigmoid_gating);
    env.integer("glm4moe.expert_gating_func", 1);
    EXPECT_FALSE(DecoderConfig(env.parameters(), env.weights).moe_sigmoid_gating);
}

TEST(GGUFBuilderAPI, ErnieInterleavesDenseAndExpertLayers) {
    Environment env;
    env.decoder();
    env.integer("test.block_count", 4);
    env.integer("test.interleave_moe_layer_step", 2);
    env.integer("test.expert_count", 4);
    env.integer("test.expert_used_count", 2);
    for (int layer = 0; layer < 4; ++layer) {
        const auto prefix = "blk." + std::to_string(layer) + ".";
        if (layer % 2 == 0) {
            env.weights[prefix + "ffn_gate.weight"] = ov::Tensor(ov::element::f32, {48, 32});
            env.weights[prefix + "ffn_up.weight"] = ov::Tensor(ov::element::f32, {48, 32});
            env.weights[prefix + "ffn_down.weight"] = ov::Tensor(ov::element::f32, {32, 48});
        } else {
            env.weights[prefix + "ffn_gate_inp.weight"] = ov::Tensor(ov::element::f32, {4, 32});
            env.weights[prefix + "ffn_gate_exps.weight"] = ov::Tensor(ov::element::f32, {4, 48, 32});
            env.weights[prefix + "ffn_up_exps.weight"] = ov::Tensor(ov::element::f32, {4, 48, 32});
            env.weights[prefix + "ffn_down_exps.weight"] = ov::Tensor(ov::element::f32, {4, 32, 48});
        }
    }
    env.architecture("ernie4_5-moe");
    for (uint32_t dense_lead : {0u, 1u}) {
        SCOPED_TRACE(dense_lead);
        env.integer("ernie4_5-moe.leading_dense_block_count", dense_lead);
        DecoderConfig config(env.parameters(), env.weights);
        EXPECT_FALSE(config.layer_is_moe(0));
        EXPECT_TRUE(config.layer_is_moe(1));
        EXPECT_FALSE(config.layer_is_moe(2));
        EXPECT_TRUE(config.layer_is_moe(3));
        EXPECT_TRUE(config.expert_weights_norm);
        GgufGraphContext graph(env.context);
        graph.configure_decoder(RopeMode::Neox);
        auto value = graph.add_input("x", ov::element::f32, {1, 1, -1, 32});
        for (int layer = 0; layer < 4; ++layer) {
            value = graph.decoder_ffn(layer, value);
        }
        graph.set_output(value);
        auto model = convert(graph.finish());
        EXPECT_EQ(model->output().get_partial_shape(), (ov::PartialShape{1, 1, -1, 32}));
    }
}

TEST(GGUFBuilderAPI, Exaone64LayerDefaultsIncludeLocalRopeAndSlidingWindow) {
    Environment env;
    env.decoder();
    env.integer("test.block_count", 64);
    env.architecture("exaone4");
    DecoderConfig config(env.parameters(), env.weights);
    EXPECT_TRUE(config.post_norm_only);
    EXPECT_TRUE(config.rope_on_swa_only);
    EXPECT_EQ(config.swa_window_size, 4096);
    EXPECT_TRUE(config.layer_is_swa(0));
    EXPECT_TRUE(config.layer_is_swa(2));
    EXPECT_FALSE(config.layer_is_swa(3));
    EXPECT_TRUE(config.layer_is_swa(4));
}

TEST(GGUFBuilderAPI, ArchitectureOptionsOverrideAmbiguousTensorSemantics) {
    Environment env;
    env.decoder();
    DecoderOptions options;
    options.qk_norm_after_rope = true;
    options.post_norm_only = true;
    options.normalize_expert_weights = true;
    options.rope_skip_period = 4;
    DecoderConfig config(env.parameters(RopeMode::Neox, options), env.weights);
    EXPECT_TRUE(config.qk_norm_after_rope);
    EXPECT_TRUE(config.post_norm_only);
    EXPECT_TRUE(config.expert_weights_norm);
    EXPECT_EQ(config.rope_skip_period, 4);
    options.rope_skip_period = -1;
    EXPECT_THROW(DecoderConfig(env.parameters(RopeMode::Neox, options), env.weights), ov::Exception);
}

TEST(GGUFBuilderAPI, DevstralReadsYarnAndAttentionTemperatureMetadata) {
    Environment env;
    env.decoder();
    env.metadata["test.rope.scaling.type"] = std::string("yarn");
    env.real("test.rope.scaling.factor", 48.0f);
    env.real("test.rope.scaling.yarn_beta_fast", 4.0f);
    env.real("test.rope.scaling.yarn_beta_slow", 1.0f);
    env.real("test.rope.scaling.yarn_log_multiplier", 1.0f);
    env.integer("test.rope.scaling.original_context_length", 8192);
    env.real("test.attention.temperature_scale", 0.1f);
    env.architecture("mistral3");
    DecoderConfig config(env.parameters(), env.weights);
    EXPECT_FLOAT_EQ(config.rope_config.beta_fast, 4.0f);
    EXPECT_FLOAT_EQ(config.rope_config.beta_slow, 1.0f);
    EXPECT_FLOAT_EQ(config.rope_config.attn_factor, 1.0f / (1.0f + 0.1f * std::log(48.0f)));
    EXPECT_FLOAT_EQ(config.attention_temperature_scale, 0.1f);
    env.integer("mistral3.rope.scaling.original_context_length", 0);
    EXPECT_THROW(DecoderConfig(env.parameters(), env.weights), ov::Exception);
}

TEST(GGUFBuilderAPI, BroadcastAndConcatShapesComeFromOpenVINO) {
    Environment env;
    GgufGraphContext graph(env.context);
    auto a = graph.add_input("a", ov::element::f32, {2, 1, -1, 4});
    auto b = graph.add_input("b", ov::element::f32, {1, 3, 1, 4});
    auto sum = graph.node("GGML_OP_ADD", {a, b});
    EXPECT_EQ(sum.shape(), (ov::PartialShape{2, 3, -1, 4}));
    auto joined = graph.node("GGML_OP_CONCAT", {sum, sum}, 0, {{"concat_axis", 0}});
    EXPECT_EQ(joined.shape(), (ov::PartialShape{2, 3, -1, 8}));
    auto output = graph.node("GGML_OP_TRANSPOSE", {joined});
    EXPECT_EQ(output.shape(), (ov::PartialShape{2, 3, 8, -1}));
    graph.set_output(output);
    auto model = convert(graph.finish());
    for (size_t tokens : {1u, 3u}) {
        ov::Tensor input_a(ov::element::f32, {2, 1, tokens, 4});
        ov::Tensor input_b(ov::element::f32, {1, 3, 1, 4});
        std::fill_n(input_a.data<float>(), input_a.get_size(), 2.f);
        std::fill_n(input_b.data<float>(), input_b.get_size(), 3.f);
        ov::TensorVector results{ov::Tensor(ov::element::f32, {2, 3, 8, tokens})};
        ASSERT_TRUE(model->evaluate(results, {input_a, input_b}));
        EXPECT_EQ(results[0].get_shape(), (ov::Shape{2, 3, 8, tokens}));
        for (size_t i = 0; i < results[0].get_size(); ++i) {
            EXPECT_EQ(results[0].data<float>()[i], 5.f);
        }
    }
}

TEST(GGUFBuilderAPI, GenericNodesNeedOnlySemanticAttributes) {
    Environment env;
    GgufGraphContext graph(env.context);
    auto input = graph.add_input("x", ov::element::f32, {1, 1, -1, 5});
    auto selected = graph.node("GGML_OP_TOP_K", {input}, 0, {{"k", int64_t{2}}});
    EXPECT_EQ(selected.shape(), (ov::PartialShape{1, 1, -1, 2}));
    EXPECT_EQ(selected.type(), ov::element::i32);
    auto repeated = graph.node("GGML_OP_REPEAT", {selected}, 0, {{"repeats", std::vector<int64_t>{1, 2, 1, 3}}});
    EXPECT_EQ(repeated.shape(), (ov::PartialShape{1, 2, -1, 6}));
    graph.set_output(repeated);
    auto model = convert(graph.finish());
    EXPECT_EQ(model->output().get_partial_shape(), repeated.shape());
}

TEST(GGUFBuilderAPI, MergeHeadsAcceptsRuntimeTokenAndHeadDimensions) {
    Environment env;
    GgufGraphContext graph(env.context);
    auto input = graph.add_input("x", ov::element::f32, {2, -1, 3, -1});
    auto merged = graph.node("GGML_OP_RESHAPE", {input}, 2, {{"merge_heads", true}});
    EXPECT_EQ(merged.shape(), (ov::PartialShape{2, 1, -1, -1}));
    graph.set_output(merged);
    auto model = convert(graph.finish());
    for (const ov::Shape& shape : {ov::Shape{2, 1, 3, 4}, ov::Shape{2, 5, 3, 2}}) {
        ov::Tensor input_data(ov::element::f32, shape);
        std::iota(input_data.data<float>(), input_data.data<float>() + input_data.get_size(), 0.f);
        ov::TensorVector result{ov::Tensor(ov::element::f32, {2, 1, shape[1], 3 * shape[3]})};
        ASSERT_TRUE(model->evaluate(result, {input_data}));
        EXPECT_EQ(result[0].get_shape(), (ov::Shape{2, 1, shape[1], 3 * shape[3]}));
        for (size_t i = 0; i < input_data.get_size(); ++i) {
            EXPECT_EQ(result[0].data<float>()[i], input_data.data<float>()[i]);
        }
    }
}

TEST(GGUFBuilderAPI, ViewReusesWholeConcatInputsAndPreservesNames) {
    Environment env;
    GgufGraphContext graph(env.context);
    auto prefix = graph.add_input("prefix", ov::element::f32, {1, 2, 1, -1});
    auto suffix = graph.add_input("suffix", ov::element::f32, {1, 2, 1, 3});
    auto packed = graph.node("GGML_OP_CONCAT", {prefix, suffix}, 0, {{"concat_axis", 0}});
    for (const auto& slice : std::vector<std::vector<int64_t>>{{3, 0, -3}, {3, -3, 3}, {3, 0, -2}})
        graph.set_output(graph.node("GGML_OP_VIEW", {packed}, 3, {{"view_slice", slice}}));
    auto model = convert(graph.finish());
    EXPECT_EQ(model->get_results()[0]->input_value(0), model->input("prefix"));
    EXPECT_EQ(model->get_results()[1]->input_value(0), model->input("suffix"));
    // A view spanning part of the suffix still needs the concatenated tensor.
    const auto partial = model->get_results()[2]->get_input_node_shared_ptr(0);
    EXPECT_STREQ(partial->get_type_name(), "Slice");
    EXPECT_STREQ(partial->get_input_node_shared_ptr(0)->get_type_name(), "Concat");
    EXPECT_EQ(model->input("prefix").get_node()->get_friendly_name(), "prefix");
    EXPECT_EQ(model->input("suffix").get_node()->get_friendly_name(), "suffix");
}

TEST(GGUFBuilderAPI, GenericNodesValidateAttributesThroughConverters) {
    for (const auto& invalid : std::vector<std::pair<std::string, std::map<std::string, ov::Any>>>{
             {"GGML_OP_RESHAPE", {{"reshape_target", std::vector<int64_t>{1, -1, -1, 4}}}},
             {"GGML_OP_PERMUTE", {{"perm", std::vector<int64_t>{0, 1, 1, 3}}}},
             {"GGML_OP_CONCAT", {{"concat_axis", 4}}}}) {
        SCOPED_TRACE(invalid.first);
        Environment env;
        GgufGraphContext graph(env.context);
        auto input = graph.add_input("x", ov::element::f32, {1, 1, 2, 4});
        const auto inputs =
            invalid.first == "GGML_OP_CONCAT" ? std::vector<GgufValue>{input, input} : std::vector<GgufValue>{input};
        const int op_case = invalid.first == "GGML_OP_PERMUTE" ? 1 : 0;
        EXPECT_THROW(graph.node(invalid.first, inputs, op_case, invalid.second), ov::Exception);
    }
}

TEST(GGUFBuilderAPI, GenericRopeNodesPreserveModelPositionContract) {
    for (bool multimodal : {false, true}) {
        for (bool per_op : {false, true}) {
            SCOPED_TRACE(testing::Message() << "multimodal=" << multimodal << ", per_op=" << per_op);
            Environment env;
            GgufGraphContext graph(env.context);
            RopeConfig config;
            config.n_dims = 4;
            config.n_ctx_orig = 128;
            config.freq_base = 10000.f;
            config.freq_scale = config.attn_factor = 1.f;
            config.beta_fast = 32.f;
            config.beta_slow = 1.f;
            config.per_op = per_op;
            config.is_imrope = multimodal;
            graph.configure_rope(config);
            auto input = graph.add_input("x", ov::element::f32, {1, -1, 2, 4});
            auto positions = graph.build_inp_pos();
            auto rotated =
                graph.node("GGML_OP_ROPE", {input, positions}, (multimodal ? 2 : 1) << 16, {{"rope_config", config}});
            EXPECT_EQ(rotated.shape(), input.shape());
            graph.set_output(rotated);
            auto built = graph.finish();
            EXPECT_EQ(built->values->count("rope_cos"), per_op ? 0u : 1u);
            auto model = convert(built);
            EXPECT_EQ(model->get_rt_info().count(pass::gguf_imrope_key()), multimodal ? 1u : 0u);
            EXPECT_EQ(model->inputs().size(), 2u);
            EXPECT_EQ(model->output().get_partial_shape(), input.shape());
        }
    }
}

TEST(GGUFBuilderAPI, MatmulProducesF32FromF16Operands) {
    Environment env;
    GgufGraphContext graph(env.context);
    auto weights = graph.add_input("w", ov::element::f16, {1, 1, 2, 2});
    auto input = graph.add_input("x", ov::element::f16, {1, 1, -1, 2});
    auto product = graph.node("GGML_OP_MUL_MAT", {weights, input});
    EXPECT_EQ(product.type(), ov::element::f32);
    graph.set_output(product);
    auto model = convert(graph.finish());
    ov::Tensor w(ov::element::f16, {1, 1, 2, 2});
    const std::vector<ov::float16> values{256.f, 256.f, 1.f, -1.f};
    std::copy(values.begin(), values.end(), w.data<ov::float16>());
    for (size_t tokens : {1u, 3u}) {
        ov::Tensor x(ov::element::f16, {1, 1, tokens, 2});
        std::fill_n(x.data<ov::float16>(), x.get_size(), ov::float16(256.f));
        ov::TensorVector result{ov::Tensor(ov::element::f32, {1, 1, tokens, 2})};
        ASSERT_TRUE(model->evaluate(result, {w, x}));
        for (size_t t = 0; t < tokens; ++t) {
            // Converting an F16 MatMul result afterward would already have overflowed.
            EXPECT_EQ(result[0].data<float>()[2 * t], 131072.f);
            EXPECT_EQ(result[0].data<float>()[2 * t + 1], 0.f);
        }
    }
}

TEST(GGUFBuilderAPI, GetRowsDerivesFloatingAndIntegerResultTypes) {
    for (auto type : {ov::element::f16, ov::element::i32}) {
        SCOPED_TRACE(type);
        Environment env;
        GgufGraphContext graph(env.context);
        auto data = graph.add_input("data", type, {1, 1, 3, 2});
        auto indices = graph.add_input("indices", ov::element::i32, {1, 1, 1, 2});
        auto rows = graph.node("GGML_OP_GET_ROWS", {data, indices});
        const auto expected_type = type == ov::element::i32 ? ov::element::i32 : ov::element::f32;
        EXPECT_EQ(rows.type(), expected_type);
        graph.set_output(rows);
        auto model = convert(graph.finish());
        ov::Tensor input(type, {1, 1, 3, 2});
        for (size_t i = 0; i < input.get_size(); ++i) {
            if (type == ov::element::i32) {
                input.data<int32_t>()[i] = static_cast<int32_t>(i);
            } else {
                input.data<ov::float16>()[i] = ov::float16(static_cast<float>(i));
            }
        }
        ov::Tensor ids(ov::element::i32, {1, 1, 1, 2});
        ids.data<int32_t>()[0] = 2;
        ids.data<int32_t>()[1] = 0;
        ov::TensorVector result{ov::Tensor(expected_type, {1, 1, 2, 2})};
        ASSERT_TRUE(model->evaluate(result, {input, ids}));
        const std::vector<int32_t> expected{4, 5, 0, 1};
        for (size_t i = 0; i < expected.size(); ++i) {
            if (type == ov::element::i32) {
                EXPECT_EQ(result[0].data<int32_t>()[i], expected[i]);
            } else {
                EXPECT_EQ(result[0].data<float>()[i], expected[i]);
            }
        }
    }
}

TEST(GGUFBuilderAPI, ElementwiseNodesPreserveF16) {
    Environment env;
    GgufGraphContext graph(env.context);
    auto input = graph.add_input("x", ov::element::f16, {1, 1, 1, 4});
    auto sum = graph.node("GGML_OP_ADD", {input, input});
    auto ratio = graph.node("GGML_OP_DIV", {sum, input});
    auto joined = graph.node("GGML_OP_CONCAT", {ratio, ratio}, 0, {{"concat_axis", 0}});
    auto output = graph.node("GGML_OP_FILL", {joined}, 0, {{"fill_value", 3.f}});
    for (const auto& value : {sum, ratio, joined, output}) {
        EXPECT_EQ(value.type(), ov::element::f16);
    }
    graph.set_output(output);
    auto model = convert(graph.finish());
    EXPECT_EQ(model->output().get_element_type(), ov::element::f16);
}

TEST(GGUFBuilderAPI, CopyAndStateWritesUseDestinationType) {
    Environment env;
    GgufGraphContext graph(env.context);
    auto destination = graph.add_input("dst", ov::element::f16, {1, 1, 1, 4});
    auto indices = graph.add_input("ids", ov::element::i32, {1, 1, 1, 1});
    auto source = graph.add_input("src", ov::element::f32, {1, 1, 1, 4});
    auto copy = graph.node("GGML_OP_CPY", {source, destination});
    auto set = graph.node("GGML_OP_SET", {destination, source}, 0, {{"set_offset_elems", int64_t{0}}});
    auto rows = graph.node("GGML_OP_SET_ROWS", {source, indices, destination});
    for (const auto& value : {copy, set, rows}) {
        EXPECT_EQ(value.type(), ov::element::f16);
        graph.set_output(value);
    }
    auto model = convert(graph.finish());
    for (const auto& output : model->outputs()) {
        EXPECT_EQ(output.get_element_type(), ov::element::f16);
    }
}

TEST(GGUFBuilderAPI, CastAndIm2colRequireExplicitDestinationType) {
    for (bool cast : {true, false}) {
        Environment env;
        GgufGraphContext graph(env.context);
        auto input = graph.add_input("x", ov::element::f32, {1, 1, 1, 3});
        const auto op = cast ? "GGML_OP_CPY" : "GGML_OP_IM2COL";
        auto inputs = std::vector<GgufValue>{input};
        std::map<std::string, ov::Any> attrs;
        if (!cast) {
            auto kernel = graph.add_input("kernel", ov::element::f32, {1, 1, 1, 2});
            inputs = {kernel, input};
            attrs["im2col_params"] = std::vector<int32_t>{1, 0, 0, 0, 1, 0, 0};
        }
        EXPECT_THROW(graph.node(op, inputs, 0, attrs), ov::Exception);
        attrs["dst_type"] = ov::element::f16;
        auto output = graph.node(op, inputs, 0, attrs);
        EXPECT_EQ(output.type(), ov::element::f16);
    }
}

TEST(GGUFBuilderAPI, MiniCpmLogitScaleUsesDivisorWithAndWithoutMetadata) {
    Environment env;
    env.decoder();
    env.integer("test.embedding_length", 2304);
    env.architecture("minicpm");
    EXPECT_FLOAT_EQ(env.parameters().scales.logits, 1.0f / 9.0f);
    env.real("minicpm.logit_scale", 9.0f);
    EXPECT_FLOAT_EQ(env.parameters().scales.logits, 1.0f / 9.0f);
    env.real("minicpm.logit_scale", 4.0f);
    EXPECT_FLOAT_EQ(env.parameters().scales.logits, 0.25f);
    DecoderOptions options;
    options.logit_scale = 0.75f;
    EXPECT_FLOAT_EQ(env.parameters(RopeMode::Normal, options).scales.logits, 0.75f);
}

TEST(GGUFBuilderAPI, ExpertNormalizationMetadataOverridesBuilderDefaults) {
    for (const bool normalize_by_default : {false, true}) {
        SCOPED_TRACE(normalize_by_default);
        Environment env;
        env.decoder();
        DecoderParameters defaults;
        defaults.experts.normalize_weights = normalize_by_default;
        EXPECT_EQ(read_decoder_parameters(env.context.metadata, defaults).experts.normalize_weights,
                  normalize_by_default);
        for (const bool normalize : {false, true}) {
            SCOPED_TRACE(normalize);
            ov::Tensor normalization(ov::element::boolean, {});
            *normalization.data<char>() = normalize;
            env.metadata["test.expert_weights_norm"] = normalization;
            EXPECT_EQ(read_decoder_parameters(env.context.metadata, defaults).experts.normalize_weights, normalize);
        }
    }
}

TEST(GGUFBuilderAPI, BuiltinExpertNormalizationPreservesLlamaPolicyAndOptions) {
    for (const auto* architecture :
         {"hunyuan-moe", "minimax-m2", "qwen3moe", "qwen35moe", "ernie4_5-moe", "mellum", "gemma4"}) {
        SCOPED_TRACE(architecture);
        Environment env;
        env.decoder();
        env.architecture(architecture);
        EXPECT_TRUE(env.parameters().experts.normalize_weights);
        ov::Tensor normalization(ov::element::boolean, {});
        *normalization.data<char>() = false;
        env.metadata[std::string(architecture) + ".expert_weights_norm"] = normalization;
        EXPECT_TRUE(env.parameters().experts.normalize_weights);
        DecoderOptions options;
        options.normalize_expert_weights = false;
        EXPECT_FALSE(env.parameters(RopeMode::Normal, options).experts.normalize_weights);
    }
}

TEST(GGUFBuilderAPI, ZeroMetadataScalesDisableMultiplication) {
    Environment env;
    env.decoder();
    DecoderParameters defaults;
    defaults.scales.embedding = 12.0f;
    defaults.scales.residual = 1.4f;
    auto parameters = read_decoder_parameters(env.context.metadata, defaults);
    EXPECT_FLOAT_EQ(parameters.scales.embedding, 12.0f);
    EXPECT_FLOAT_EQ(parameters.scales.residual, 1.4f);
    defaults.scales.embedding = 0.0f;
    defaults.scales.residual = 0.0f;
    parameters = read_decoder_parameters(env.context.metadata, defaults);
    EXPECT_FLOAT_EQ(parameters.scales.embedding, 0.0f);
    EXPECT_FLOAT_EQ(parameters.scales.residual, 0.0f);
    for (float scale : {0.0f, 0.5f, 2.0f}) {
        env.real("test.embedding_scale", scale);
        env.real("test.residual_scale", scale);
        parameters = read_decoder_parameters(env.context.metadata, defaults);
        EXPECT_FLOAT_EQ(parameters.scales.embedding, scale == 0.0f ? 1.0f : scale);
        EXPECT_FLOAT_EQ(parameters.scales.residual, scale == 0.0f ? 1.0f : scale);
    }
}

TEST(GGUFBuilderAPI, TypedCustomLayerNormScalesAndLayerSettingsAreResolvedTogether) {
    Environment env;
    env.decoder();
    env.metadata.erase("test.attention.layer_norm_rms_epsilon");
    env.real("test.attention.layer_norm_epsilon", 3e-4f);
    env.real("test.rope.scaling.factor", 4.0f);
    env.architecture("custom-layernorm");
    DecoderParameters defaults;
    defaults.normalization.kind = NormalizationKind::Layer;
    auto parameters = read_decoder_parameters(env.context.metadata, defaults);
    parameters.scales.embedding = 2.5f;
    parameters.scales.residual = 0.375f;
    parameters.scales.logits = 0.75f;
    parameters.scales.attention = 0.25f;
    parameters.attention.window_size = 17;
    parameters.layers.resize(2);
    parameters.layers[0].sliding_window = false;
    parameters.layers[1].sliding_window = true;
    parameters.layers[1].query_heads = 8;
    parameters.layers[1].kv_heads = 2;
    parameters.layers[1].head_size = 4;
    parameters.layers[1].attention_scale = 0.625f;
    auto local_rope = parameters.attention.rope;
    local_rope.n_dims = 4;
    local_rope.freq_base = 32000.f;
    local_rope.freq_scale = 0.5f;
    parameters.layers[1].rope = local_rope;
    parameters.model_contract.batchable_recurrent = true;
    parameters.model_contract.image_attention = ModelContract::ImageAttentionMode::SLIDING_ONLY;
    DecoderConfig resolved(parameters, env.weights);
    EXPECT_TRUE(resolved.layer_norm);
    EXPECT_FLOAT_EQ(resolved.rms_eps, 3e-4f);
    EXPECT_FLOAT_EQ(resolved.embedding_scale, 2.5f);
    EXPECT_FLOAT_EQ(resolved.residual_scale, 0.375f);
    EXPECT_FLOAT_EQ(resolved.logit_scale, 0.75f);
    EXPECT_FLOAT_EQ(resolved.layer_kq_scale(0), 0.25f);
    EXPECT_FLOAT_EQ(resolved.layer_kq_scale(1), 0.625f);
    GgufGraphContext graph(env.context);
    const auto dimensions = graph.configure_decoder(parameters);
    EXPECT_EQ(dimensions.layers, 2);
    EXPECT_FLOAT_EQ(dimensions.norm_epsilon, resolved.rms_eps);
    const auto global = graph.decoder_layer_parameters(0);
    const auto local = graph.decoder_layer_parameters(1);
    EXPECT_FALSE(global.sliding_window);
    EXPECT_TRUE(local.sliding_window);
    EXPECT_EQ(local.query_heads, 8);
    EXPECT_EQ(local.kv_heads, 2);
    EXPECT_EQ(local.head_size, 4);
    EXPECT_FLOAT_EQ(local.attention_scale, resolved.layer_kq_scale(1));
    EXPECT_FLOAT_EQ(global.rope.freq_scale, 0.25f);
    EXPECT_FLOAT_EQ(local.rope.freq_base, 32000.f);
    EXPECT_FLOAT_EQ(local.rope.freq_scale, 0.5f);
    auto input = graph.add_input("x", ov::element::f32, {1, 1, 1, 32});
    graph.set_output(input);
    const auto built = graph.finish();
    EXPECT_TRUE(built->use_per_op_rope);
    EXPECT_EQ(built->swa_window_size, 17);
    EXPECT_TRUE(built->model_contract.batchable_recurrent);
    EXPECT_EQ(built->model_contract.image_attention, ModelContract::ImageAttentionMode::SLIDING_ONLY);
}

TEST(GGUFBuilderAPI, TypedSettingsRejectPartialLayersAndIncompatiblePresentMetadata) {
    Environment env;
    env.decoder();
    auto parameters = read_decoder_parameters(env.context.metadata);
    parameters.layers.resize(1);
    EXPECT_THROW(DecoderConfig(parameters, env.weights), ov::Exception);
    parameters.layers.resize(2);
    parameters.layers[1].head_size = 0;
    EXPECT_THROW(DecoderConfig(parameters, env.weights), ov::Exception);
    env.metadata["test.embedding_scale"] = std::string("invalid");
    EXPECT_THROW(read_decoder_parameters(env.context.metadata), ov::Exception);
}

TEST(GGUFBuilderAPI, ConvenienceOptionsSelectUnknownLayerNormBeforeMetadataReading) {
    Environment env;
    env.decoder();
    env.metadata.erase("test.attention.layer_norm_rms_epsilon");
    env.real("test.attention.layer_norm_epsilon", 2e-4f);
    DecoderOptions options;
    options.normalization = NormalizationKind::Layer;
    options.embedding_scale = 3.0f;
    GgufGraphContext graph(env.context);
    EXPECT_FLOAT_EQ(graph.configure_decoder(RopeMode::Normal, options).norm_epsilon, 2e-4f);
}

TEST(GGUFBuilderAPI, LayerRopeOriginalContextUsesItsOwnTable) {
    Environment env;
    env.decoder();
    auto parameters = read_decoder_parameters(env.context.metadata);
    EXPECT_FALSE(DecoderConfig(parameters, env.weights).use_per_op_rope);
    parameters.layers.resize(2);
    auto rope = parameters.attention.rope;
    rope.n_ctx_orig += 128;
    parameters.layers[1].rope = rope;
    DecoderConfig config(parameters, env.weights);
    EXPECT_TRUE(config.use_per_op_rope);
    EXPECT_EQ(config.layer_rope_config(1).n_ctx_orig, rope.n_ctx_orig);
}

TEST(GGUFBuilderAPI, UnsignedMetadataOverflowCannotWrapToSignedValues) {
    Environment env;
    env.decoder();
    ov::Tensor scalar(ov::element::u64, {});
    scalar.data<uint64_t>()[0] = uint64_t{1} << 63;
    env.metadata["test.embedding_length"] = scalar;
    EXPECT_FALSE(env.context.metadata.get_int("test.embedding_length"));
    EXPECT_THROW(read_decoder_parameters(env.context.metadata), ov::Exception);
    ov::Tensor values(ov::element::u64, {2});
    values.data<uint64_t>()[0] = 1;
    values.data<uint64_t>()[1] = (uint64_t{1} << 63) + 1;
    env.metadata["test.attention.head_count_kv"] = values;
    EXPECT_TRUE(env.context.metadata.get_int_array("test.attention.head_count_kv").empty());
}

TEST(GGUFBuilderAPI, UnsetPoolingSentinelIsPreservedWithoutRelaxingDimensionChecks) {
    Environment env;
    env.decoder();
    ov::Tensor sentinel(ov::element::u32, {});
    sentinel.data<uint32_t>()[0] = std::numeric_limits<uint32_t>::max();
    env.metadata["test.pooling_type"] = sentinel;
    EXPECT_EQ(read_decoder_parameters(env.context.metadata).pooling_type, -1);
    env.metadata["test.embedding_length"] = sentinel;
    EXPECT_THROW(read_decoder_parameters(env.context.metadata), ov::Exception);
}

TEST(GGUFBuilderAPI, LayerRopeCannotChangeTheSharedPositionLayout) {
    Environment env;
    env.decoder();
    auto parameters = read_decoder_parameters(env.context.metadata);
    parameters.layers.resize(2);
    auto rope = parameters.attention.rope;
    rope.is_imrope = true;
    parameters.layers[1].rope = rope;
    EXPECT_THROW(DecoderConfig(parameters, env.weights), ov::Exception);
}

TEST(GGUFBuilderAPI, ScalarRecurrentMetadataAppliesToEveryLayer) {
    Environment env;
    env.decoder();
    for (const int value : {0, 1}) {
        env.integer("test.attention.recurrent_layers", value);
        auto parameters = read_decoder_parameters(env.context.metadata);
        EXPECT_EQ(parameters.hybrid.recurrent_layers, (std::vector<int32_t>{value, value}));
        EXPECT_NO_THROW(DecoderConfig(parameters, env.weights));
    }
}

TEST(GGUFBuilderAPI, QueryHeadArraysDescribeHybridAndVariableWidthLayers) {
    Environment env;
    env.decoder();
    ov::Tensor heads(ov::element::u32, {2});
    heads.data<uint32_t>()[0] = 2;
    heads.data<uint32_t>()[1] = 1;
    env.metadata["test.attention.head_count"] = heads;
    auto parameters = read_decoder_parameters(env.context.metadata);
    EXPECT_EQ(parameters.dimensions.query_heads, 2);
    EXPECT_EQ(DecoderConfig(parameters, env.weights).layer_n_head(1), 1);
    heads.data<uint32_t>()[1] = 0;
    EXPECT_THROW(read_decoder_parameters(env.context.metadata), ov::Exception);
    env.integer("test.attention.recurrent_layers", 1);
    parameters = read_decoder_parameters(env.context.metadata);
    EXPECT_NO_THROW(DecoderConfig(parameters, env.weights));
}

TEST(GGUFBuilderAPI, ZeroQueryAndKvHeadsDescribeLayersWithoutAttention) {
    Environment env;
    env.decoder();
    ov::Tensor query_heads(ov::element::u32, {2});
    query_heads.data<uint32_t>()[0] = 2;
    query_heads.data<uint32_t>()[1] = 0;
    ov::Tensor kv_heads(ov::element::u32, {2});
    kv_heads.data<uint32_t>()[0] = 1;
    kv_heads.data<uint32_t>()[1] = 0;
    env.metadata["test.attention.head_count"] = query_heads;
    env.metadata["test.attention.head_count_kv"] = kv_heads;
    EXPECT_NO_THROW(DecoderConfig(read_decoder_parameters(env.context.metadata), env.weights));
    kv_heads.data<uint32_t>()[1] = 1;
    EXPECT_THROW(read_decoder_parameters(env.context.metadata), ov::Exception);
}
