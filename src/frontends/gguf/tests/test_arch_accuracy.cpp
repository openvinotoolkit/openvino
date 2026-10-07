// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>

#include "cnpy.h"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "gtest/gtest.h"
#include "op_test_utils.hpp"
#include "openvino/frontend/extension/decoder_transformation.hpp"
#include "openvino/frontend/gguf/adapt_mmproj_to_genai.hpp"
#include "openvino/frontend/gguf/adapt_to_genai.hpp"
#include "openvino/frontend/gguf/extension/genai.hpp"
#include "openvino/frontend/gguf/frontend.hpp"
#include "openvino/frontend/gguf/make_stateful.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/paged_gated_delta_net.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"

namespace {
class GGUFArchitectureAccuracy : public ::testing::TestWithParam<const char*> {};

// Check batching and beam reordering against independent stateful requests. The
// same synthetic checkpoints are qualified against llama.cpp below.
void check_batched_decode(const std::shared_ptr<ov::Model>& model) {
    ov::Core core;
    auto compiled = core.compile_model(model,
                                       "CPU",
                                       ov::hint::inference_precision(ov::element::f32),
                                       ov::num_streams(1),
                                       ov::inference_num_threads(4),
                                       ov::hint::dynamic_quantization_group_size(0),
                                       ov::hint::kv_cache_precision(ov::element::f16));
    auto first = compiled.create_infer_request();
    auto second = compiled.create_infer_request();
    auto batch = compiled.create_infer_request();
    const auto infer = [](ov::InferRequest& request,
                          size_t batch_size,
                          const std::vector<int64_t>& ids,
                          const std::vector<int64_t>& mask,
                          const std::vector<int64_t>& positions,
                          const std::vector<int32_t>& beams) {
        const auto set_tensor =
            [&](const char* name, const ov::element::Type& type, const ov::Shape& shape, const auto& values) {
                ov::Tensor tensor(type, shape);
                std::memcpy(tensor.data(), values.data(), tensor.get_byte_size());
                request.set_tensor(name, tensor);
            };
        set_tensor("input_ids", ov::element::i64, {batch_size, ids.size() / batch_size}, ids);
        set_tensor("attention_mask", ov::element::i64, {batch_size, mask.size() / batch_size}, mask);
        set_tensor("position_ids", ov::element::i64, {batch_size, positions.size() / batch_size}, positions);
        set_tensor("beam_idx", ov::element::i32, {batch_size}, beams);
        request.infer();
        auto output = request.get_tensor("logits");
        return std::vector<float>(output.data<const float>(), output.data<const float>() + output.get_size());
    };
    const auto compare =
        [](const std::vector<float>& actual, const std::vector<float>& first, const std::vector<float>& second) {
            auto expected = first;
            expected.insert(expected.end(), second.begin(), second.end());
            ASSERT_EQ(actual.size(), expected.size());
            ov_gguf_test::expect_nmse_below(ov_gguf_test::nmse(actual.data(), expected.data(), actual.size()), 1e-5);
        };
    auto a = infer(first, 1, {1, 2, 3}, {1, 1, 1}, {0, 1, 2}, {0});
    auto b = infer(second, 1, {2, 3}, {1, 1}, {0, 1}, {0});
    compare(infer(batch, 2, {1, 2, 3, 0, 2, 3}, {1, 1, 1, 0, 1, 1}, {0, 1, 2, 0, 0, 1}, {0, 0}), a, b);
    a = infer(first, 1, {4}, {1, 1, 1, 1}, {3}, {0});
    b = infer(second, 1, {5}, {1, 1, 1}, {2}, {0});
    compare(infer(batch, 2, {5, 4}, {0, 1, 1, 1, 1, 1, 1, 1}, {2, 3}, {1, 0}), b, a);
}

TEST(GGUFMultimodalBackboneAdaptation, QwenAndGemmaSupportBatchesAndPagedAttention) {
    for (const auto* family : {"qwen35", "qwen35moe", "qwen35moe-fused", "gemma4-mqa", "gemma4-moe", "gemma4-ple"}) {
        SCOPED_TRACE(family);
        auto arrays = cnpy::npz_load(ov_gguf_test::test_data_dir() + "/arch_accuracy/" + family + ".npz");
        const ov_gguf_test::TemporaryGguf temporary(ov_gguf_test::npz_array(arrays, "model"));
        ov::frontend::gguf::FrontEnd frontend;
        frontend.add_extension(std::make_shared<ov::frontend::gguf::GenAIExtension>());
        auto model = frontend.convert(frontend.load(temporary.path));
        check_batched_decode(model);
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::SDPAToPagedAttention>();
        ASSERT_NO_THROW(manager.run_passes(model));
        // Multiple tokens expose accidental [tokens,tokens,...] M-RoPE broadcasts.
        model->reshape({{"input_ids", {5}}, {"position_ids", {5}}});
        const bool gemma = std::string(family).find("gemma4") == 0;
        size_t attention = 0, recurrent = 0;
        for (const auto& node : model->get_ops()) {
            if (auto pa = ov::as_type_ptr<ov::op::PagedAttentionExtension>(node)) {
                ++attention;
                const auto& query_shape = pa->get_input_partial_shape(0);
                EXPECT_EQ(query_shape.rank(), 2);
                EXPECT_EQ(query_shape[0], 5);
                EXPECT_TRUE(query_shape[1] == 64 || (gemma && query_shape[1] == 32));
            }
            if (auto gdn = ov::as_type_ptr<ov::op::internal::PagedGatedDeltaNet>(node)) {
                ++recurrent;
                EXPECT_EQ(gdn->get_input_partial_shape(0)[0], 5);
            }
        }
        EXPECT_EQ(attention, gemma ? (std::string(family) == "gemma4-ple" ? 4 : 2) : 1);
        EXPECT_EQ(recurrent, gemma ? 0 : 3);
        EXPECT_TRUE(model->get_sinks().empty());
    }
}

// Gemma3/Gemma4 scale token lookups only, as in llama.cpp, so the scale belongs to the embedding
// model and injected media embeddings reach the decoder unscaled.
TEST(GGUFMultimodalBackboneAdaptation, GemmaEmbeddingModelOwnsTokenScaling) {
    for (const auto* family : {"gemma3", "gemma4-mqa", "gemma4-ple"}) {
        SCOPED_TRACE(family);
        auto arrays = cnpy::npz_load(ov_gguf_test::test_data_dir() + "/arch_accuracy/" + family + ".npz");
        const ov_gguf_test::TemporaryGguf temporary(ov_gguf_test::npz_array(arrays, "model"));
        const auto convert = [&] {
            ov::frontend::gguf::FrontEnd frontend;
            frontend.add_extension(std::make_shared<ov::frontend::DecoderTransformationExtension>(
                ov::frontend::gguf::pass::GGUFMakeStateful()));
            return frontend.convert(frontend.load(temporary.path));
        };
        auto token_model = convert();
        ov::frontend::gguf::pass::AdaptToGenAI().run_on_model(token_model);
        ov::frontend::gguf::FrontEnd frontend;
        auto extension = std::make_shared<ov::frontend::gguf::GenAIExtension>(
            ov::frontend::gguf::GenAIExtension::InputMode::EMBEDS_TO_LOGITS);
        frontend.add_extension(extension);
        const auto input = frontend.load(temporary.path);
        auto embedded_model = frontend.convert(input);
        ASSERT_TRUE(extension->get_embedding_model());
        EXPECT_EQ(bool(extension->get_per_layer_embedding_model()), std::string(family) == "gemma4-ple");
        embedded_model = frontend.convert(input);
        const auto embedding_model = extension->get_embedding_model();
        const auto width = embedding_model->output("inputs_embeds").get_partial_shape()[2].get_length();
        const auto embedding_ops = embedding_model->get_ops();
        const bool scaled =
            std::any_of(embedding_ops.begin(), embedding_ops.end(), [&](const std::shared_ptr<ov::Node>& node) {
                if (!ov::is_type<ov::op::v1::Multiply>(node))
                    return false;
                auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node->get_input_node_shared_ptr(1));
                return constant && ov::shape_size(constant->get_shape()) == 1 &&
                       std::abs(constant->cast_vector<float>()[0] - std::sqrt(float(width))) < 1e-4f;
            });
        EXPECT_TRUE(scaled) << "embedding model does not apply sqrt(n_embd)";

        ov::Core core;
        const auto compile = [&](const std::shared_ptr<ov::Model>& model) {
            return core.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32))
                .create_infer_request();
        };
        auto tokens = compile(token_model), values = compile(embedded_model), lookup = compile(embedding_model);
        ov::Tensor ids(ov::element::i64, {1, 3}), mask(ov::element::i64, {1, 3}), positions(ov::element::i64, {1, 3});
        for (int64_t i = 0; i < 3; ++i) {
            ids.data<int64_t>()[i] = i + 1;
            positions.data<int64_t>()[i] = i;
        }
        std::fill_n(mask.data<int64_t>(), 3, 1);
        ov::Tensor beam(ov::element::i32, {1});
        beam.data<int32_t>()[0] = 0;
        lookup.set_tensor("input_ids", ids);
        lookup.infer();
        for (auto* request : {&tokens, &values}) {
            request->set_tensor("attention_mask", mask);
            request->set_tensor("position_ids", positions);
            request->set_tensor("beam_idx", beam);
        }
        tokens.set_tensor("input_ids", ids);
        values.set_tensor("inputs_embeds", lookup.get_tensor("inputs_embeds"));
        for (const auto& input : embedded_model->inputs()) {
            if (input.get_names().count("token_type_ids")) {
                ov::Tensor types(ov::element::i64, {1, 3});
                std::fill_n(types.data<int64_t>(), 3, 0);
                values.set_tensor("token_type_ids", types);
            }
            if (input.get_names().count("per_layer_inputs")) {
                auto per_layer = compile(extension->get_per_layer_embedding_model());
                per_layer.set_tensor("input_ids", ids);
                per_layer.infer();
                values.set_tensor("per_layer_inputs", per_layer.get_tensor("per_layer_inputs"));
            }
        }
        tokens.infer();
        values.infer();
        const auto expected = tokens.get_output_tensor(), actual = values.get_output_tensor();
        ASSERT_EQ(actual.get_shape(), expected.get_shape());
        ov_gguf_test::expect_nmse_below(
            ov_gguf_test::nmse(actual.data<const float>(), expected.data<const float>(), actual.get_size()),
            1e-10);
    }
}

// Adapters registered as transformation extensions run during conversion and must already see
// the architecture and projector metadata they select their changes by.
TEST(GGUFMultimodalBackboneAdaptation, AdaptersRegisteredAsExtensionsSeeModelMetadata) {
    using ov::frontend::DecoderTransformationExtension;
    namespace pass = ov::frontend::gguf::pass;
    const auto convert = [](const std::string& fixture, const std::vector<std::shared_ptr<ov::Extension>>& passes) {
        auto arrays = cnpy::npz_load(ov_gguf_test::test_data_dir() + "/" + fixture);
        const ov_gguf_test::TemporaryGguf temporary(ov_gguf_test::npz_array(arrays, "model"));
        ov::frontend::gguf::FrontEnd frontend;
        for (const auto& extension : passes)
            frontend.add_extension(extension);
        return frontend.convert(frontend.load(temporary.path));
    };
    const auto names = [](const auto& ports) {
        std::set<std::string> result;
        for (const auto& port : ports)
            result.insert(port.get_any_name());
        return result;
    };
    const auto language = convert("arch_accuracy/gemma3.npz",
                                  {std::make_shared<DecoderTransformationExtension>(pass::GGUFMakeStateful()),
                                   std::make_shared<DecoderTransformationExtension>(
                                       pass::AdaptToGenAI(pass::AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS))});
    EXPECT_EQ(names(language->inputs()).count("token_type_ids"), 1);
    const auto vision = convert("mmproj_accuracy/qwen3vl_merger.npz",
                                {std::make_shared<DecoderTransformationExtension>(
                                    pass::AdaptMmprojToGenAI(pass::AdaptMmprojToGenAI::Modality::VISION))});
    EXPECT_EQ(names(vision->outputs()), (std::set<std::string>{"image_features", "deepstack_features.0"}));
}

// The Gated-DeltaNet key-head count divides the value-head count; a missing or zero one must
// fail conversion with a message instead of dividing by zero.
TEST(GGUFMultimodalBackboneAdaptation, GatedDeltaNetRejectsMissingOrZeroGroupCount) {
    auto arrays = cnpy::npz_load(ov_gguf_test::test_data_dir() + "/arch_accuracy/qwen35.npz");
    const auto& model = ov_gguf_test::npz_array(arrays, "model");
    const std::string key = "qwen35.ssm.group_count";
    for (const bool missing : {false, true}) {
        SCOPED_TRACE(missing ? "missing" : "zero");
        std::vector<uint8_t> bytes(model.data<uint8_t>(), model.data<uint8_t>() + model.num_vals);
        const auto at = std::search(bytes.begin(), bytes.end(), key.begin(), key.end());
        ASSERT_NE(at, bytes.end());
        if (missing) {
            at[key.size() - 1] = 'X';  // the loader then defaults the count to zero
        } else {
            const auto value = at + key.size() + sizeof(uint32_t);  // after the u32 type tag
            std::fill(value, value + sizeof(uint32_t), 0);
        }
        cnpy::NpyArray patched({bytes.size()}, sizeof(uint8_t), false);
        std::copy(bytes.begin(), bytes.end(), patched.data<uint8_t>());
        const ov_gguf_test::TemporaryGguf temporary(patched);
        ov::frontend::gguf::FrontEnd frontend;
        OV_EXPECT_THROW(frontend.convert(frontend.load(temporary.path)),
                        ov::Exception,
                        testing::HasSubstr("ssm.group_count"));
    }
}

// The grouped-query KV broadcast may reach MakeStateful as Multiply(ones, Unsqueeze(kv)); the
// cache read behind it must still be found, so the fused cache gets its batch-shaped initializer.
TEST(GGUFMultimodalBackboneAdaptation, MakeStatefulFindsCachesBehindReversedGroupedQueryMultiply) {
    auto arrays = cnpy::npz_load(ov_gguf_test::test_data_dir() + "/arch_accuracy/qwen3.npz");
    const ov_gguf_test::TemporaryGguf temporary(ov_gguf_test::npz_array(arrays, "model"));
    for (const bool reversed : {false, true}) {
        SCOPED_TRACE(reversed ? "ones * kv" : "kv * ones");
        size_t rewritten = 0;
        const auto to_multiply = [&](const std::shared_ptr<ov::Model>& model) {
            for (const auto& node : model->get_ordered_ops()) {
                auto broadcast = ov::as_type_ptr<ov::op::v3::Broadcast>(node);
                auto shape = broadcast ? ov::as_type_ptr<ov::op::v0::Constant>(broadcast->get_input_node_shared_ptr(1))
                                       : nullptr;
                if (!shape || !ov::is_type<ov::op::v0::Unsqueeze>(broadcast->get_input_node_ptr(0)))
                    continue;
                auto ones = ov::op::v0::Constant::create(broadcast->get_output_element_type(0),
                                                         ov::Shape(shape->cast_vector<size_t>()),
                                                         {1.f});
                const auto kv = broadcast->input_value(0);
                ov::replace_node(broadcast,
                                 reversed ? std::make_shared<ov::op::v1::Multiply>(ones, kv)
                                          : std::make_shared<ov::op::v1::Multiply>(kv, ones));
                broadcast->input(0).replace_source_output(ones);  // the replaced node no longer reads kv
                ++rewritten;
            }
            return rewritten > 0;
        };
        ov::frontend::gguf::FrontEnd frontend;
        frontend.add_extension(std::make_shared<ov::frontend::DecoderTransformationExtension>(to_multiply));
        frontend.add_extension(std::make_shared<ov::frontend::DecoderTransformationExtension>(
            ov::frontend::gguf::pass::GGUFMakeStateful()));
        std::shared_ptr<ov::Model> model;
        ASSERT_NO_THROW(model = frontend.convert(frontend.load(temporary.path)));
        ASSERT_GT(rewritten, 0u);
        size_t batch_initialized = 0;
        for (const auto& node : model->get_ordered_ops()) {
            if (ov::is_type<ov::op::v6::ReadValue>(node) && node->get_input_size() == 1 &&
                ov::is_type<ov::op::v3::Broadcast>(node->get_input_node_ptr(0)))
                ++batch_initialized;
        }
        EXPECT_EQ(batch_initialized, rewritten);
    }
}

TEST(GGUFEmbeddingAccuracy, TokenEmbeddingsAndPoolingMatchLlamaCPU) {
    for (const auto* family :
         {"llama-embed", "llama-embed-noncausal", "llama-embed-mean", "llama-embed-cls", "llama-embed-last"}) {
        SCOPED_TRACE(family);
        auto arrays = cnpy::npz_load(ov_gguf_test::test_data_dir() + "/arch_accuracy/" + family + ".npz");
        const ov_gguf_test::TemporaryGguf temporary(ov_gguf_test::npz_array(arrays, "model"));
        ov::frontend::gguf::FrontEnd frontend;
        auto model = frontend.convert(frontend.load(temporary.path));
        ASSERT_EQ(model->get_results().size(), 1);
        EXPECT_NO_THROW(model->output("embeddings"));
        for (const auto mode : {ov::frontend::gguf::pass::AdaptToGenAI::InputMode::IDS_TO_LOGITS,
                                ov::frontend::gguf::pass::AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS}) {
            ov::frontend::gguf::pass::AdaptToGenAI adapter(mode);
            EXPECT_FALSE(adapter.run_on_model(model));
        }
        ov::Core core;
        auto compiled = core.compile_model(model,
                                           "CPU",
                                           ov::hint::inference_precision(ov::element::f32),
                                           ov::hint::dynamic_quantization_group_size(0));
        auto request = compiled.create_infer_request();
        EXPECT_TRUE(request.query_state().empty());
        const auto infer = [&](size_t count, bool compare_reference) {
            for (const auto& input : compiled.inputs()) {
                const auto name = input.get_any_name();
                if (name == "token_len_per_seq") {
                    ov::Tensor lengths(ov::element::i64, {1});
                    lengths.data<int64_t>()[0] = count;
                    request.set_tensor(name, lengths);
                } else if (name == "self_kq_mask") {
                    ov::Tensor mask(ov::element::f32, {1, 1, count, count});
                    for (size_t q = 0; q < count; ++q)
                        for (size_t k = 0; k < count; ++k)
                            mask.data<float>()[q * count + k] = k <= q ? 0.f : -INFINITY;
                    request.set_tensor(name, mask);
                } else {
                    ASSERT_TRUE(name == "inp_tokens" || name == "inp_pos");
                    ov::Tensor values(ov::element::i32, {1, 1, 1, count});
                    for (size_t i = 0; i < count; ++i)
                        values.data<int32_t>()[i] = static_cast<int32_t>(i) + (name == "inp_tokens" ? 1 : 0);
                    request.set_tensor(name, values);
                }
            }
            request.infer();
            auto output = request.get_tensor("embeddings");
            const auto& expected = ov_gguf_test::npz_array(arrays, "embeddings");
            const bool pooled = std::string(family) != "llama-embed" && std::string(family) != "llama-embed-noncausal";
            ASSERT_EQ(output.get_shape(), (ov::Shape{pooled ? 1 : count, expected.shape.back()}));
            if (compare_reference || std::string(family) == "llama-embed") {
                ov_gguf_test::expect_nmse_below(
                    ov_gguf_test::nmse(output.data<const float>(), expected.data<float>(), output.get_size()),
                    1e-5);
            }
        };
        infer(3, true);
        infer(1, false);
        infer(2, false);
        infer(3, true);
    }
}

TEST(GGUFEmbeddingAccuracy, RealCheckpointTokenEmbeddingsMatchLlamaCPU) {
    const auto* directory = std::getenv("OV_GGUF_EMBEDDING_DATA");
    if (!directory)
        GTEST_SKIP() << "Set OV_GGUF_EMBEDDING_DATA for real embedding checkpoint validation";
    const auto base = std::filesystem::path(directory) / "llama-embed";
    std::ifstream file(base.string() + ".bin", std::ios::binary);
    ASSERT_TRUE(file);
    int32_t width = 0, count = 0;
    file.read(reinterpret_cast<char*>(&width), sizeof(width));
    file.read(reinterpret_cast<char*>(&count), sizeof(count));
    ASSERT_GT(width, 0);
    ASSERT_GT(count, 0);
    ASSERT_LE(count, 64);
    std::vector<int32_t> tokens(count);
    std::vector<float> expected(static_cast<size_t>(count) * width);
    file.read(reinterpret_cast<char*>(tokens.data()), tokens.size() * sizeof(int32_t));
    file.read(reinterpret_cast<char*>(expected.data()), expected.size() * sizeof(float));
    ASSERT_TRUE(file);
    ov::frontend::gguf::FrontEnd frontend;
    auto model = frontend.convert(frontend.load(base.string() + ".gguf"));
    ov::Core core;
    auto compiled = core.compile_model(model,
                                       "CPU",
                                       ov::hint::inference_precision(ov::element::f32),
                                       ov::hint::dynamic_quantization_group_size(0),
                                       ov::inference_num_threads(4));
    auto request = compiled.create_infer_request();
    for (const auto& input : compiled.inputs()) {
        const auto name = input.get_any_name();
        if (name == "token_len_per_seq") {
            ov::Tensor lengths(ov::element::i64, {1});
            lengths.data<int64_t>()[0] = count;
            request.set_tensor(name, lengths);
        } else if (name == "self_kq_mask") {
            ov::Tensor mask(ov::element::f32, {1, 1, size_t(count), size_t(count)});
            for (int q = 0; q < count; ++q)
                for (int k = 0; k < count; ++k)
                    mask.data<float>()[q * count + k] = k <= q ? 0.f : -INFINITY;
            request.set_tensor(name, mask);
        } else {
            ASSERT_TRUE(name == "inp_tokens" || name == "inp_pos");
            ov::Tensor values(ov::element::i32, {1, 1, 1, size_t(count)});
            for (int i = 0; i < count; ++i)
                values.data<int32_t>()[i] = name == "inp_tokens" ? tokens[i] : i;
            request.set_tensor(name, values);
        }
    }
    request.infer();
    auto output = request.get_tensor("embeddings");
    ASSERT_EQ(output.get_shape(), (ov::Shape{size_t(count), size_t(width)}));
    const auto metric = ov_gguf_test::nmse(output.data<const float>(), expected.data(), expected.size());
    RecordProperty("nmse", std::to_string(metric.value()));
    ov_gguf_test::expect_nmse_below(metric, 1e-5);
}

TEST_P(GGUFArchitectureAccuracy, PrefillAndCachedDecodeMatchLlamaCPU) {
    const char* override_dir = std::getenv("OV_GGUF_ACCURACY_DATA");
    const bool mamba = std::string(GetParam()).find("mamba2") == 0 || std::string(GetParam()) == "nemotron_h";
    const auto directory = override_dir ? std::filesystem::path(override_dir)
                                        : std::filesystem::path(ov_gguf_test::test_data_dir()) / "arch_accuracy";
    ASSERT_TRUE(std::filesystem::exists(directory)) << "Missing architecture reference data: " << directory;
    const auto base = directory / GetParam();
    const bool real_checkpoint = override_dir && !std::filesystem::exists(base.string() + ".npz");
    std::vector<std::vector<int64_t>> schedule{{1, 2, 3}, {4}, {5, 6}};
    std::vector<float> reference;
    int32_t vocab = 0;
    std::string model_path = base.string() + ".gguf";
    std::unique_ptr<ov_gguf_test::TemporaryGguf> temporary;
    if (real_checkpoint) {
        std::ifstream file(base.string() + ".bin", std::ios::binary);
        ASSERT_TRUE(file) << base;
        file.read(reinterpret_cast<char*>(&vocab), sizeof(vocab));
        ASSERT_GT(vocab, 6);
        ASSERT_LT(vocab, 1000000);
        std::ifstream token_file(base.string() + ".bin.tokens");
        if (token_file) {
            schedule.clear();
            size_t count = 0;
            while (token_file >> count) {
                ASSERT_GT(count, 0);
                ASSERT_LE(count, 64);
                std::vector<int64_t> ids(count);
                for (auto& token : ids)
                    ASSERT_TRUE(token_file >> token);
                schedule.push_back(std::move(ids));
            }
        }
        ASSERT_FALSE(schedule.empty());
        reference.resize(schedule.size() * vocab);
        file.read(reinterpret_cast<char*>(reference.data()), reference.size() * sizeof(float));
        ASSERT_TRUE(file);
    } else {
        const auto arrays = cnpy::npz_load(base.string() + ".npz");
        const auto array = [&](const std::string& name) -> const cnpy::NpyArray& {
            return ov_gguf_test::npz_array(arrays, name);
        };
        const auto& logits = array("logits");
        ASSERT_EQ(logits.shape.size(), 2);
        ASSERT_EQ(logits.shape[0], 3);
        vocab = static_cast<int32_t>(logits.shape[1]);
        const auto* values = logits.data<float>();
        reference.assign(values, values + 3 * vocab);
        temporary = std::make_unique<ov_gguf_test::TemporaryGguf>(array("model"));
        model_path = temporary->path;
    }
    ov::frontend::gguf::FrontEnd fe;
    fe.add_extension(std::make_shared<ov::frontend::gguf::GenAIExtension>());
    auto model = fe.convert(fe.load(model_path));
    if (mamba) {
        EXPECT_EQ(model->input("input_ids").get_partial_shape(), (ov::PartialShape{1, -1}));
        EXPECT_NO_THROW(model->input("beam_idx"));
    }
    ov::Core core;
    auto compiled = core.compile_model(model,
                                       "CPU",
                                       ov::hint::inference_precision(ov::element::f32),
                                       ov::num_streams(1),
                                       ov::inference_num_threads(4),
                                       ov::hint::dynamic_quantization_group_size(0),
                                       ov::hint::kv_cache_precision(ov::element::f16));
    auto request = compiled.create_infer_request();
    const auto infer = [&](const std::vector<int64_t>& tokens, size_t past) {
        const auto count = tokens.size();
        ov::Tensor ids(ov::element::i64, {1, count});
        ov::Tensor positions(ov::element::i64, ids.get_shape());
        ov::Tensor mask(ov::element::i64, {1, past + count});
        ov::Tensor beam(ov::element::i32, {1});
        *beam.data<int32_t>() = 0;
        for (size_t i = 0; i < count; ++i) {
            ids.data<int64_t>()[i] = tokens[i];
            positions.data<int64_t>()[i] = static_cast<int64_t>(past + i);
        }
        std::fill_n(mask.data<int64_t>(), mask.get_size(), 1);
        request.set_tensor("input_ids", ids);
        request.set_tensor("position_ids", positions);
        request.set_tensor("attention_mask", mask);
        if (std::any_of(compiled.inputs().begin(), compiled.inputs().end(), [](const ov::Output<const ov::Node>& p) {
                return p.get_names().count("beam_idx");
            }))
            request.set_tensor("beam_idx", beam);
        request.infer();
        return request.get_output_tensor();
    };
    size_t past = 0;
    size_t step = 0;
    size_t matching_tokens = 0;
    for (const auto& tokens : schedule) {
        SCOPED_TRACE("past=" + std::to_string(past) + ", tokens=" + std::to_string(tokens.size()));
        const auto result = infer(tokens, past);
        ASSERT_EQ(result.get_element_type(), ov::element::f32);
        ASSERT_GE(result.get_size(), static_cast<size_t>(vocab));
        const auto* actual = result.data<const float>() + result.get_size() - vocab;
        const auto* expected = reference.data() + step++ * vocab;
        const auto metric = ov_gguf_test::nmse(actual, expected, static_cast<size_t>(vocab));
        ASSERT_TRUE(metric.all_finite());
        ASSERT_GT(metric.reference_norm(), 1e-12) << "Reference must contain nonzero logits";
        const auto predicted = std::max_element(actual, actual + vocab) - actual;
        const auto wanted = std::max_element(expected, expected + vocab) - expected;
        matching_tokens += predicted == wanted;
        RecordProperty("top1_match_step_" + std::to_string(step), predicted == wanted ? 1 : 0);
        RecordProperty("nmse_step_" + std::to_string(step), std::to_string(metric.value()));
        if (real_checkpoint) {
            // Real checkpoints can use different quantization arithmetic. Check the first prediction
            // and continuation agreement; keep their full-logit metrics in the XML report.
            if (step == 1) {
                EXPECT_EQ(predicted, wanted);
            }
        } else {
            EXPECT_LT(metric.value(), 1e-5) << "Normalized MSE against llama.cpp CPU";
        }
        past += tokens.size();
    }
    if (mamba) {
        const auto states = request.query_state();
        ASSERT_FALSE(states.empty());
        for (auto state : states)
            state.reset();
        const auto logits = infer(schedule.front(), 0);
        const auto* actual = logits.data<const float>() + logits.get_size() - vocab;
        const auto metric = ov_gguf_test::nmse(actual, reference.data(), static_cast<size_t>(vocab));
        ASSERT_TRUE(metric.all_finite());
        ASSERT_GT(metric.reference_norm(), 1e-12);
        if (real_checkpoint)
            EXPECT_EQ(std::max_element(actual, actual + vocab) - actual,
                      std::max_element(reference.begin(), reference.begin() + vocab) - reference.begin());
        else
            EXPECT_LT(metric.value(), 1e-5) << "Fresh prefill after resetting recurrent states";
    }
    if (real_checkpoint) {
        EXPECT_GE(matching_tokens * 10, schedule.size() * 9) << "Fewer than 90% of greedy choices match";
    }
}

INSTANTIATE_TEST_SUITE_P(Architectures,
                         GGUFArchitectureAccuracy,
                         ::testing::Values("qwen35",
                                           "qwen35-mixed",
                                           "qwen35moe",
                                           "qwen35moe-fused",
                                           "gemma4-mqa",
                                           "gemma4-moe",
                                           "gemma4-ple",
                                           "nemotron_h",
                                           "mamba2",
                                           "mamba2-tied",
                                           "llama",
                                           "qwen2",
                                           "qwen3",
                                           "phi3",
                                           "minicpm",
                                           "olmoe",
                                           "hunyuan-dense",
                                           "hunyuan-moe",
                                           "exaone-moe",
                                           "exaone-moe-nextn",
                                           "glm4moe",
                                           "jais2",
                                           "minimax-m2",
                                           "plamo3",
                                           "qwen3moe",
                                           "gemma",
                                           "gemma2",
                                           "gemma3",
                                           "exaone4",
                                           "ernie4_5-moe",
                                           "bailingmoe2",
                                           "maincoder",
                                           "mistral3",
                                           "smollm3",
                                           "mellum",
                                           "muse-glimmer",
                                           "deepseek2-ocr",
                                           "devstral-small",
                                           "devstral-small2",
                                           "devstral2"),
                         [](const ::testing::TestParamInfo<const char*>& info) {
                             std::string name = info.param;
                             std::replace(name.begin(), name.end(), '-', '_');
                             return name;
                         });
}  // namespace
