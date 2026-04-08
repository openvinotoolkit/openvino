// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <regex>
#include <string>
#include <vector>

#include "openvino/op/assign.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"

namespace {
struct CacheStateBlock {
    std::shared_ptr<ov::op::v6::ReadValue> read_value;
    std::shared_ptr<ov::op::v6::Assign> assign;
    ov::Output<ov::Node> output;
};

// Creates a cache block connected to gather with beam_idx, using an explicit variable_id.
CacheStateBlock make_cache_state(const ov::Output<ov::Node>& beam_idx,
                                 const std::string& var_name,
                                 ov::element::Type precision,
                                 const ov::PartialShape& var_shape,
                                 const ov::Shape& init_shape) {
    auto variable =
        std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{var_shape, precision, var_name});

    auto num_elements = ov::shape_size(init_shape);
    auto init = ov::op::v0::Constant::create(precision, init_shape, std::vector<float>(num_elements, 0.0f));
    auto read_value = std::make_shared<ov::op::v6::ReadValue>(init, variable);

    auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto gather = std::make_shared<ov::op::v8::Gather>(read_value, beam_idx, gather_axis);

    auto assign = std::make_shared<ov::op::v6::Assign>(gather, variable);

    return {read_value, assign, gather->output(0)};
}

// Creates a cache block NOT connected to beam_idx, using an explicit variable_id.
CacheStateBlock make_cache_state(const std::string& var_name,
                                 ov::element::Type precision,
                                 const ov::PartialShape& var_shape,
                                 const ov::Shape& init_shape) {
    auto variable =
        std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{var_shape, precision, var_name});

    auto init = ov::op::v0::Constant::create(precision, init_shape, std::vector<float>(ov::shape_size(init_shape), 0.0f));
    auto read_value = std::make_shared<ov::op::v6::ReadValue>(init, variable);

    auto assign = std::make_shared<ov::op::v6::Assign>(read_value, variable);

    return {read_value, assign, read_value->output(0)};
}

// Standard LLM KV cache: variable_id = "past_key_values.N.keypresent.N.key"
CacheStateBlock make_past_key_values_kv_state(const ov::Output<ov::Node>& beam_idx,
                                              size_t layer_idx,
                                              const std::string& key_or_value,
                                              ov::element::Type precision,
                                              const ov::PartialShape& var_shape,
                                              const ov::Shape& init_shape) {
    const std::string idx = std::to_string(layer_idx);
    const std::string var_name = "past_key_values." + idx + "." + key_or_value +
                                 "present." + idx + "." + key_or_value;
    return make_cache_state(beam_idx, var_name, precision, var_shape, init_shape);
}

// Whisper decoder KV cache: variable_id = "past_key_values.N.decoder.keypresent.N.decoder.key"
CacheStateBlock make_whisper_decoder_kv_state(const ov::Output<ov::Node>& beam_idx,
                                              size_t layer_idx,
                                              const std::string& key_or_value,
                                              ov::element::Type precision,
                                              const ov::PartialShape& var_shape,
                                              const ov::Shape& init_shape) {
    const std::string idx = std::to_string(layer_idx);
    const std::string var_name = "past_key_values." + idx + ".decoder." + key_or_value +
                                 "present." + idx + ".decoder." + key_or_value;
    return make_cache_state(beam_idx, var_name, precision, var_shape, init_shape);
}

// Whisper encoder KV cache: numeric variable_id, NOT connected to beam_idx.
CacheStateBlock make_whisper_encoder_kv_state(const std::string& numeric_id,
                                              ov::element::Type precision,
                                              const ov::PartialShape& var_shape,
                                              const ov::Shape& init_shape) {
    return make_cache_state(numeric_id, precision, var_shape, init_shape);
}

CacheStateBlock make_cache_params_kv_state(const ov::Output<ov::Node>& beam_idx,
                                           size_t layer_idx,
                                           const std::string& key_or_value,
                                           ov::element::Type precision,
                                           const ov::PartialShape& var_shape,
                                           const ov::Shape& init_shape) {
    const std::string var_name = "cache_params.past." + key_or_value + "." + std::to_string(layer_idx) +
                                 "cache_params.present." + key_or_value + "." + std::to_string(layer_idx);
    return make_cache_state(beam_idx, var_name, precision, var_shape, init_shape);
}

// Creates a linear cache block (conv) NOT connected to beam_idx.
CacheStateBlock make_cache_params_gated_short_conv_state(size_t layer_idx,
                                                         ov::element::Type precision,
                                                         const ov::Shape& shape) {
    const std::string var_name = "cache_params.past.conv." + std::to_string(layer_idx) +
                                 "cache_params.present.conv." + std::to_string(layer_idx);
    return make_cache_state(var_name, precision, shape, shape);
}

// Creates a linear cache block (conv, ssm) connected to beam_idx via Gather.
CacheStateBlock make_cache_params_lin_state(const ov::Output<ov::Node>& beam_idx,
                                           size_t layer_idx,
                                           const std::string& cache_type,
                                           ov::element::Type precision,
                                           const ov::PartialShape& var_shape,
                                           const ov::Shape& init_shape) {
    const std::string var_name = "cache_params.past." + cache_type + "." + std::to_string(layer_idx) +
                                 "cache_params.present." + cache_type + "." + std::to_string(layer_idx);
    return make_cache_state(beam_idx, var_name, precision, var_shape, init_shape);
}

// Builds a model matching TinyLlama-1.1B caches (scaled down to 2 layers):
//   - Standard KV cache naming: past_key_values.N.key/value  (f32, {?,4,?,64})
//   - All connected via beam_idx.
std::shared_ptr<ov::Model> build_model_with_tinyllama_like_cache(size_t num_layers = 2) {
    auto stub_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 64});
    stub_input->output(0).set_names({"stub_input"});

    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    beam_idx->output(0).set_names({"beam_idx"});

    ov::SinkVector sinks;
    for (size_t layer = 0; layer < num_layers; ++layer) {
        auto key_block = make_past_key_values_kv_state(
            beam_idx, layer, "key", ov::element::f32, ov::PartialShape{-1, 4, -1, 64}, {1, 4, 0, 64});
        auto value_block = make_past_key_values_kv_state(
            beam_idx, layer, "value", ov::element::f32, ov::PartialShape{-1, 4, -1, 64}, {1, 4, 0, 64});
        sinks.push_back(key_block.assign);
        sinks.push_back(value_block.assign);
    }

    auto result = std::make_shared<ov::op::v0::Result>(stub_input);
    return std::make_shared<ov::Model>(ov::ResultVector{result},
                                       sinks,
                                       ov::ParameterVector{stub_input, beam_idx},
                                       "model_with_tinyllama_like_cache");
}

// Builds a model matching Whisper decoder caches (scaled down to 2 layers):
//   - 2 decoder KV layers: past_key_values.N.decoder.key/value (f32, {?,8,?,64}), connected to beam_idx
//   - 2 encoder KV layers: numeric variable_ids (e.g. "242", "244"), NOT connected to beam_idx
// The decoder naming doesn't match the StatefulToStateless regex, so "input_restored." prefix is expected.
// The encoder numeric ids don't match and they also are NOT connected to beam_idx, so they remain unchanged..
std::shared_ptr<ov::Model> build_model_with_whisper_decoder_like_cache(size_t num_layers = 2) {
    auto stub_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 64});
    stub_input->output(0).set_names({"stub_input"});

    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    beam_idx->output(0).set_names({"beam_idx"});

    ov::SinkVector sinks;
    size_t encoder_id_counter = 242;
    for (size_t layer = 0; layer < num_layers; ++layer) {
        // Decoder KV cache (connected to beam_idx, uses "decoder" in variable_id)
        auto dec_key = make_whisper_decoder_kv_state(
            beam_idx, layer, "key", ov::element::f32, ov::PartialShape{-1, 8, -1, 64}, {1, 8, 0, 64});
        auto dec_value = make_whisper_decoder_kv_state(
            beam_idx, layer, "value", ov::element::f32, ov::PartialShape{-1, 8, -1, 64}, {1, 8, 0, 64});
        sinks.push_back(dec_key.assign);
        sinks.push_back(dec_value.assign);

        // Encoder KV cache (NOT connected to beam_idx, numeric variable_ids)
        auto enc_key = make_whisper_encoder_kv_state(
            std::to_string(encoder_id_counter++), ov::element::f32, ov::PartialShape{-1, 8, -1, 64}, {1, 8, 0, 64});
        auto enc_value = make_whisper_encoder_kv_state(
            std::to_string(encoder_id_counter++), ov::element::f32, ov::PartialShape{-1, 8, -1, 64}, {1, 8, 0, 64});
        sinks.push_back(enc_key.assign);
        sinks.push_back(enc_value.assign);
    }

    auto result = std::make_shared<ov::op::v0::Result>(stub_input);
    return std::make_shared<ov::Model>(ov::ResultVector{result},
                                       sinks,
                                       ov::ParameterVector{stub_input, beam_idx},
                                       "model_with_whisper_decoder_like_cache");
}

// Builds a model matching LFM2-1.2B caches (scaled down to 2 layers):
//   - 2 KV-cache layers: cache_params.past.key.<idx> / value.<idx>  (f32, {?,8,?,64})
//   - 2 Gated Short Convolution (conv) layers: cache_params.past.conv.<idx> (f32, {?,2048,3})
// KV caches are connected via beam_idx; conv caches are not.
std::shared_ptr<ov::Model> build_model_with_lfm2_like_cache(size_t num_layers = 2) {
    auto stub_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 64});
    stub_input->output(0).set_names({"stub_input"});

    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    beam_idx->output(0).set_names({"beam_idx"});

    ov::SinkVector sinks;
    for (size_t layer = 0; layer < num_layers; ++layer) {
        // KV cache (connected to beam_idx)
        auto key_block =
            make_cache_params_kv_state(beam_idx, layer, "key", ov::element::f32,
                ov::PartialShape{-1, 8, -1, 64}, {1, 8, 0, 64});
        auto value_block =
            make_cache_params_kv_state(beam_idx, layer, "value", ov::element::f32,
                ov::PartialShape{-1, 8, -1, 64}, {1, 8, 0, 64});
        sinks.push_back(key_block.assign);
        sinks.push_back(value_block.assign);

        // Gated Short Convolution cache (NOT connected to beam_idx)
        auto conv_block = make_cache_params_gated_short_conv_state(layer, ov::element::f32, {1, 2048, 3});
        sinks.push_back(conv_block.assign);
    }

    auto result = std::make_shared<ov::op::v0::Result>(stub_input);
    return std::make_shared<ov::Model>(ov::ResultVector{result},
                                       sinks,
                                       ov::ParameterVector{stub_input, beam_idx},
                                       "model_with_lfm2_like_cache");
}

// Builds a model matching Qwen3.5-2B caches (scaled down to 2 layers):
//   - 2 KV-cache layers: cache_params.past.key.<idx> / value.<idx>  (f32, {?,2,?,256})
//   - 2 GatedDeltaNet (ssm) layers: cache_params.past.ssm.<idx> (f32, {?,16,128,128})
//   - 2 Convolution1D (conv) layers: cache_params.past.conv.<idx> (f32, {?,6144,4})
// All cache types are connected via beam_idx Gather in Qwen3.5-2B.
std::shared_ptr<ov::Model> build_model_with_qwen35_like_cache(size_t num_layers = 2) {
    auto stub_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 64});
    stub_input->output(0).set_names({"stub_input"});

    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    beam_idx->output(0).set_names({"beam_idx"});

    ov::SinkVector sinks;
    for (size_t layer = 0; layer < num_layers; ++layer) {
        // KV cache
        auto key_block =
            make_cache_params_kv_state(beam_idx, layer, "key", ov::element::f32, ov::PartialShape{-1, 2, -1, 256}, {1, 2, 0, 256});
        auto value_block =
            make_cache_params_kv_state(beam_idx, layer, "value", ov::element::f32, ov::PartialShape{-1, 2, -1, 256}, {1, 2, 0, 256});
        sinks.push_back(key_block.assign);
        sinks.push_back(value_block.assign);

        // GatedDeltaNet SSM cache
        auto ssm_block =
            make_cache_params_lin_state(beam_idx, layer, "ssm", ov::element::f32, ov::PartialShape{-1, 16, 128, 128}, {1, 16, 128, 128});
        sinks.push_back(ssm_block.assign);

        // Conv1D cache
        auto conv_block =
            make_cache_params_lin_state(beam_idx, layer, "conv", ov::element::f32, ov::PartialShape{-1, 6144, 4}, {1, 6144, 4});
        sinks.push_back(conv_block.assign);
    }

    auto result = std::make_shared<ov::op::v0::Result>(stub_input);
    return std::make_shared<ov::Model>(ov::ResultVector{result},
                                       sinks,
                                       ov::ParameterVector{stub_input, beam_idx},
                                       "model_with_qwen35_like_cache");
}

bool has_input_with_name(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    for (const auto& input : model->inputs()) {
        for (const auto& input_name : input.get_names()) {
            if (input_name.find(name) != std::string::npos) {
                return true;
            }
        }
    }
    return false;
}

bool has_output_with_name(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    for (const auto& output : model->outputs()) {
        for (const auto& output_name : output.get_names()) {
            if (output_name.find(name) != std::string::npos) {
                return true;
            }
        }
    }
    return false;
}

std::vector<std::string> collect_input_names(const std::shared_ptr<ov::Model>& model) {
    std::vector<std::string> names;
    for (const auto& input : model->inputs()) {
        names.push_back(input.get_any_name());
    }
    return names;
}

std::vector<std::string> collect_output_names(const std::shared_ptr<ov::Model>& model) {
    std::vector<std::string> names;
    for (const auto& output : model->outputs()) {
        names.push_back(output.get_any_name());
    }
    return names;
}

TEST(StatefulToStatelessTest, TinyLlama_ConvertsToStateless) {
    auto model = build_model_with_tinyllama_like_cache();

    ASSERT_EQ(model->get_sinks().size(), 4u);
    ASSERT_EQ(model->get_variables().size(), 4u);

    ASSERT_NO_THROW(ov::pass::StatefulToStateless().run_on_model(model));

    EXPECT_EQ(model->get_sinks().size(), 0u);
    EXPECT_EQ(model->get_variables().size(), 0u);

    for (size_t layer = 0; layer < 2; ++layer) {
        const auto idx = std::to_string(layer);
        EXPECT_TRUE(has_input_with_name(model, "past_key_values." + idx + ".key"));
        EXPECT_TRUE(has_input_with_name(model, "past_key_values." + idx + ".value"));
        EXPECT_TRUE(has_output_with_name(model, "present." + idx + ".key"));
        EXPECT_TRUE(has_output_with_name(model, "present." + idx + ".value"));
    }

    // Verify ordering: key(0), value(0), key(1), value(1)
    auto input_names = collect_input_names(model);
    std::vector<std::string> kv_inputs;
    for (const auto& name : input_names) {
        if (name.find("past_key_values") != std::string::npos) {
            kv_inputs.push_back(name);
        }
    }
    ASSERT_EQ(kv_inputs.size(), 4u);
    EXPECT_NE(kv_inputs[0].find("0.key"), std::string::npos);
    EXPECT_NE(kv_inputs[1].find("0.value"), std::string::npos);
    EXPECT_NE(kv_inputs[2].find("1.key"), std::string::npos);
    EXPECT_NE(kv_inputs[3].find("1.value"), std::string::npos);

    // Verify output ordering: present.0.key, present.0.value, present.1.key, present.1.value
    auto output_names = collect_output_names(model);
    std::vector<std::string> kv_outputs;
    for (const auto& name : output_names) {
        if (name.find("present") != std::string::npos) {
            kv_outputs.push_back(name);
        }
    }
    ASSERT_EQ(kv_outputs.size(), 4u);
    EXPECT_NE(kv_outputs[0].find("0.key"), std::string::npos);
    EXPECT_NE(kv_outputs[1].find("0.value"), std::string::npos);
    EXPECT_NE(kv_outputs[2].find("1.key"), std::string::npos);
    EXPECT_NE(kv_outputs[3].find("1.value"), std::string::npos);
}

// Whisper decoder naming ("past_key_values.N.decoder.key") doesn't match the
// StatefulToStateless regex, so converted parameters get "input_restored." prefix.
// Encoder states (numeric variable_ids) are not connected to beam_idx and don't match
// any naming convention, so they remain unchanged.
TEST(StatefulToStatelessTest, WhisperDecoder_DecoderStatesGetRestoredPrefix) {
    auto model = build_model_with_whisper_decoder_like_cache();

    // 2 layers * (dec_key + dec_value + enc_key + enc_value) = 8 sinks
    ASSERT_EQ(model->get_sinks().size(), 8u);
    ASSERT_EQ(model->get_variables().size(), 8u);

    ASSERT_NO_THROW(ov::pass::StatefulToStateless().run_on_model(model));

    // Decoder states (connected to beam_idx) get converted with "input_restored." prefix
    for (size_t layer = 0; layer < 2; ++layer) {
        const auto idx = std::to_string(layer);
        const auto decoder_key_var =
            "past_key_values." + idx + ".decoder.keypresent." + idx + ".decoder.key";
        const auto decoder_value_var =
            "past_key_values." + idx + ".decoder.valuepresent." + idx + ".decoder.value";

        EXPECT_TRUE(has_input_with_name(model, "input_restored." + decoder_key_var))
            << "Decoder key layer " << layer << " should get input_restored prefix";
        EXPECT_TRUE(has_input_with_name(model, "input_restored." + decoder_value_var))
            << "Decoder value layer " << layer << " should get input_restored prefix";
        EXPECT_TRUE(has_output_with_name(model, "output_restored." + decoder_key_var))
            << "Decoder key layer " << layer << " should get output_restored prefix";
        EXPECT_TRUE(has_output_with_name(model, "output_restored." + decoder_value_var))
            << "Decoder value layer " << layer << " should get output_restored prefix";
    }

    // Encoder states (numeric ids, not connected to beam_idx) remain as sinks/variables
    EXPECT_EQ(model->get_sinks().size(), 4u) << "Encoder states should remain as sinks";
    EXPECT_EQ(model->get_variables().size(), 4u) << "Encoder variables should remain";
}


// Verifies full conversion: sinks/variables removed, beam_idx removed,
// KV inputs -> past_key_values naming, conv inputs -> cache_params naming.
TEST(StatefulToStatelessTest, LFM2_ConvertsToStateless) {
    auto model = build_model_with_lfm2_like_cache();

    // 2 layers x (key + value + conv) = 6 sinks
    ASSERT_EQ(model->get_sinks().size(), 6u);
    ASSERT_EQ(model->get_variables().size(), 6u);

    ASSERT_NO_THROW(ov::pass::StatefulToStateless().run_on_model(model));

    EXPECT_EQ(model->get_sinks().size(), 0u);
    EXPECT_EQ(model->get_variables().size(), 0u);

    // KV cache key/value -> mapped to standard "past_key_values" / "present" naming
    for (size_t layer = 0; layer < 2; ++layer) {
        const auto idx = std::to_string(layer);
        EXPECT_TRUE(has_input_with_name(model, "past_key_values." + idx + ".key"));
        EXPECT_TRUE(has_input_with_name(model, "past_key_values." + idx + ".value"));
        EXPECT_TRUE(has_output_with_name(model, "present." + idx + ".key"));
        EXPECT_TRUE(has_output_with_name(model, "present." + idx + ".value"));
    }

    // Conv cache -> keeps cache_params naming
    for (size_t layer = 0; layer < 2; ++layer) {
        const auto idx = std::to_string(layer);
        EXPECT_TRUE(has_input_with_name(model, "cache_params.past.conv." + idx));
        EXPECT_TRUE(has_output_with_name(model, "cache_params.present.conv." + idx));
    }
}

// Verifies ordering: KV caches first (key before value, layer 0 before 1), then conv in natural order.
TEST(StatefulToStatelessTest, LFM2_KVCacheBeforeConv) {
    auto model = build_model_with_lfm2_like_cache();
    ov::pass::StatefulToStateless().run_on_model(model);

    auto input_names = collect_input_names(model);

    std::vector<std::string> cache_inputs;
    for (const auto& name : input_names) {
        if (name.find("past_key_values") != std::string::npos ||
            name.find("cache_params") != std::string::npos) {
            cache_inputs.push_back(name);
        }
    }

    // 2 layers: key(0), value(0), key(1), value(1), conv(0), conv(1)
    ASSERT_EQ(cache_inputs.size(), 6u);
    EXPECT_NE(cache_inputs[0].find("0.key"), std::string::npos);
    EXPECT_NE(cache_inputs[1].find("0.value"), std::string::npos);
    EXPECT_NE(cache_inputs[2].find("1.key"), std::string::npos);
    EXPECT_NE(cache_inputs[3].find("1.value"), std::string::npos);

    // Conv caches after KV
    EXPECT_NE(cache_inputs[4].find("conv"), std::string::npos);
    EXPECT_NE(cache_inputs[5].find("conv"), std::string::npos);

    // Verify same ordering for outputs: present.0.key, present.0.value, ..., cache_params.present.conv.0, ...
    auto output_names = collect_output_names(model);
    std::vector<std::string> cache_outputs;
    for (const auto& name : output_names) {
        if (name.find("present") != std::string::npos) {
            cache_outputs.push_back(name);
        }
    }
    ASSERT_EQ(cache_outputs.size(), 6u);
    EXPECT_NE(cache_outputs[0].find("0.key"), std::string::npos);
    EXPECT_NE(cache_outputs[1].find("0.value"), std::string::npos);
    EXPECT_NE(cache_outputs[2].find("1.key"), std::string::npos);
    EXPECT_NE(cache_outputs[3].find("1.value"), std::string::npos);
    EXPECT_NE(cache_outputs[4].find("conv"), std::string::npos);
    EXPECT_NE(cache_outputs[5].find("conv"), std::string::npos);
}

// Verifies that shapes from the real model are preserved through conversion.
TEST(StatefulToStatelessTest, LFM2_ShapesAndElementsTypeArePreserved) {
    auto model = build_model_with_lfm2_like_cache(1);  // 1 layer for simplicity
    ov::pass::StatefulToStateless().run_on_model(model);

    for (const auto& input : model->inputs()) {
        const auto& name = input.get_any_name();
        const auto& shape = input.get_partial_shape();

        if (name.find("past_key_values") != std::string::npos) {
            // KV cache: {?,8,?,64}, f32
            ASSERT_EQ(shape.rank().get_length(), 4) << "KV input " << name << " should be rank 4";
            EXPECT_EQ(shape[1].get_length(), 8) << "Num heads mismatch for " << name;
            EXPECT_EQ(shape[3].get_length(), 64) << "Head dim mismatch for " << name;
            EXPECT_EQ(input.get_element_type(), ov::element::f32) << "Element type mismatch for " << name;
        } else if (name.find("cache_params.past.conv") != std::string::npos) {
            // Conv cache: {1, 2048, 3}, f32
            ASSERT_EQ(shape.rank().get_length(), 3) << "Conv input " << name << " should be rank 3";
            EXPECT_EQ(shape[1].get_length(), 2048) << "Conv 1st dim mismatch for " << name;
            EXPECT_EQ(shape[2].get_length(), 3) << "Conv kernel size mismatch for " << name;
            EXPECT_EQ(input.get_element_type(), ov::element::f32) << "Element type mismatch for " << name;
        }
    }
}

// Verifies full conversion with all three cache types.
TEST(StatefulToStatelessTest, Qwen35_ConvertsToStateless) {
    auto model = build_model_with_qwen35_like_cache();

    // 2 layers x (key + value + ssm + conv) = 8 sinks
    ASSERT_EQ(model->get_sinks().size(), 8u);
    ASSERT_EQ(model->get_variables().size(), 8u);

    ASSERT_NO_THROW(ov::pass::StatefulToStateless().run_on_model(model));

    EXPECT_EQ(model->get_sinks().size(), 0u);
    EXPECT_EQ(model->get_variables().size(), 0u);

    for (size_t layer = 0; layer < 2; ++layer) {
        const auto idx = std::to_string(layer);
        // KV -> standard naming
        EXPECT_TRUE(has_input_with_name(model, "past_key_values." + idx + ".key"));
        EXPECT_TRUE(has_input_with_name(model, "past_key_values." + idx + ".value"));
        EXPECT_TRUE(has_output_with_name(model, "present." + idx + ".key"));
        EXPECT_TRUE(has_output_with_name(model, "present." + idx + ".value"));

        // SSM -> cache_params naming
        EXPECT_TRUE(has_input_with_name(model, "cache_params.past.ssm." + idx));
        EXPECT_TRUE(has_output_with_name(model, "cache_params.present.ssm." + idx));

        // Conv -> cache_params naming
        EXPECT_TRUE(has_input_with_name(model, "cache_params.past.conv." + idx));
        EXPECT_TRUE(has_output_with_name(model, "cache_params.present.conv." + idx));
    }
}

// Verifies ordering: KV first (key before value), then ssm/conv in natural order.
TEST(StatefulToStatelessTest, Qwen35_KVCacheBeforeLinearCaches) {
    auto model = build_model_with_qwen35_like_cache();
    ov::pass::StatefulToStateless().run_on_model(model);

    auto input_names = collect_input_names(model);

    int kv_last_pos = -1;
    int lin_first_pos = static_cast<int>(input_names.size());

    for (int i = 0; i < static_cast<int>(input_names.size()); ++i) {
        if (input_names[i].find("past_key_values") != std::string::npos) {
            kv_last_pos = i;
        }
        if (input_names[i].find("cache_params") != std::string::npos) {
            lin_first_pos = std::min(lin_first_pos, i);
        }
    }

    EXPECT_LT(kv_last_pos, lin_first_pos)
        << "KV cache parameters should appear before linear cache parameters";

    // Same check for outputs
    auto output_names = collect_output_names(model);
    int kv_out_last_pos = -1;
    int lin_out_first_pos = static_cast<int>(output_names.size());
    for (int i = 0; i < static_cast<int>(output_names.size()); ++i) {
        if (output_names[i].find("present") != std::string::npos &&
            output_names[i].find("cache_params") == std::string::npos) {
            kv_out_last_pos = i;
        }
        if (output_names[i].find("cache_params.present") != std::string::npos) {
            lin_out_first_pos = std::min(lin_out_first_pos, i);
        }
    }
    EXPECT_LT(kv_out_last_pos, lin_out_first_pos)
        << "KV cache outputs should appear before linear cache outputs";
}

// Verifies KV key-before-value ordering within each layer.
TEST(StatefulToStatelessTest, Qwen35_KeyBeforeValueOrdering) {
    auto model = build_model_with_qwen35_like_cache();
    ov::pass::StatefulToStateless().run_on_model(model);

    auto input_names = collect_input_names(model);

    std::vector<std::string> kv_inputs;
    for (const auto& name : input_names) {
        if (name.find("past_key_values") != std::string::npos) {
            kv_inputs.push_back(name);
        }
    }

    // Expected: key(0), value(0), key(1), value(1)
    ASSERT_EQ(kv_inputs.size(), 4u);
    EXPECT_NE(kv_inputs[0].find("0.key"), std::string::npos);
    EXPECT_NE(kv_inputs[1].find("0.value"), std::string::npos);
    EXPECT_NE(kv_inputs[2].find("1.key"), std::string::npos);
    EXPECT_NE(kv_inputs[3].find("1.value"), std::string::npos);

    // Same ordering for outputs
    auto output_names = collect_output_names(model);
    std::vector<std::string> kv_outputs;
    for (const auto& name : output_names) {
        if (name.find("present") != std::string::npos &&
            name.find("cache_params") == std::string::npos) {
            kv_outputs.push_back(name);
        }
    }
    ASSERT_EQ(kv_outputs.size(), 4u);
    EXPECT_NE(kv_outputs[0].find("0.key"), std::string::npos);
    EXPECT_NE(kv_outputs[1].find("0.value"), std::string::npos);
    EXPECT_NE(kv_outputs[2].find("1.key"), std::string::npos);
    EXPECT_NE(kv_outputs[3].find("1.value"), std::string::npos);
}

// Verifies shapes and element types from the real Qwen3.5 model are preserved.
TEST(StatefulToStatelessTest, Qwen35_ShapesAndElementTypesPreserved) {
    auto model = build_model_with_qwen35_like_cache(1);
    ov::pass::StatefulToStateless().run_on_model(model);

    for (const auto& input : model->inputs()) {
        const auto& name = input.get_any_name();
        const auto& shape = input.get_partial_shape();

        if (name.find("past_key_values") != std::string::npos) {
            // KV cache: {?,2,?,256}, f32
            ASSERT_EQ(shape.rank().get_length(), 4) << "KV input " << name << " should be rank 4";
            EXPECT_EQ(shape[1].get_length(), 2) << "Num heads mismatch for " << name;
            EXPECT_EQ(shape[3].get_length(), 256) << "Head dim mismatch for " << name;
            EXPECT_EQ(input.get_element_type(), ov::element::f32) << "Element type mismatch for " << name;
        } else if (name.find("cache_params.past.ssm") != std::string::npos) {
            // SSM cache: {1, 16, 128, 128}, f32
            ASSERT_EQ(shape.rank().get_length(), 4) << "SSM input " << name << " should be rank 4";
            EXPECT_EQ(shape[1].get_length(), 16);
            EXPECT_EQ(shape[2].get_length(), 128);
            EXPECT_EQ(shape[3].get_length(), 128);
            EXPECT_EQ(input.get_element_type(), ov::element::f32) << "Element type mismatch for " << name;
        } else if (name.find("cache_params.past.conv") != std::string::npos) {
            // Conv cache: {1, 6144, 4}, f32
            ASSERT_EQ(shape.rank().get_length(), 3) << "Conv input " << name << " should be rank 3";
            EXPECT_EQ(shape[1].get_length(), 6144);
            EXPECT_EQ(shape[2].get_length(), 4);
            EXPECT_EQ(input.get_element_type(), ov::element::f32) << "Element type mismatch for " << name;
        }
    }
}
}  // namespace