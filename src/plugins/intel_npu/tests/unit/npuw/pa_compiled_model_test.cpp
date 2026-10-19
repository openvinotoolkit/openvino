// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pa_compiled_model.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <random>
#include <set>
#include <string>

#include "llm_test_helpers.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "openvino/runtime/core.hpp"

namespace {

using ov::npuw::PACompiledModel;

std::string str(const ov::AnyMap& config, const char* key) {
    return config.at(key).as<std::string>();
}

ov::npuw::util::NPUDesc npu_with_compiler_dq() {
    ov::npuw::util::NPUDesc desc;
    desc.arch = "4000";
    desc.max_tiles = 6;
    desc.compiler_dq = true;
    desc.compiler_matmul_gate = true;
    return desc;
}

TEST(PAVariantConfig, PartitionsThePAOpsOffNPU) {
    const auto config = PACompiledModel::make_variant_config({}, "CPU", "bank0", std::nullopt);

    EXPECT_EQ(str(config, "NPU_USE_NPUW"), "YES");
    EXPECT_EQ(str(config, "NPUW_DEVICES"), "NPU,CPU");
    EXPECT_EQ(str(config, "NPUW_ONLINE_PIPELINE"), "REP");
    EXPECT_EQ(str(config, "NPUW_ONLINE_ISOLATE"), "Op:PagedAttentionExtension/pa");
    EXPECT_EQ(str(config, "NPUW_ONLINE_AVOID"), "Op:PagedAttentionExtension/NPU");
    EXPECT_EQ(str(config, "NPUW_ONLINE_KEEP_BLOCKS_TAGGED"), "pa");
    EXPECT_EQ(str(config, "NPUW_FOLD"), "YES");
    EXPECT_EQ(str(config, "NPUW_SUBMODEL_DEVICE"), "last:CPU");
    EXPECT_EQ(str(config, "NPUW_WEIGHTS_BANK"), "bank0");
    EXPECT_EQ(str(config, "NPU_COMPILATION_MODE_PARAMS"),
              "compute-layers-with-higher-precision=Sqrt,Power,ReduceMean,Add_RMSNorm");
}

TEST(PAVariantConfig, StartsFromTheLLMPathNPUDefaults) {
    for (const auto& npudesc : {std::optional<ov::npuw::util::NPUDesc>{}, std::make_optional(npu_with_compiler_dq())}) {
        const auto defaults = ov::npuw::util::get_default_common_config(npudesc);
        const auto config = PACompiledModel::make_variant_config({}, "CPU", "bank0", npudesc);
        for (const auto& [key, value] : defaults) {
            if (key == "NPUW_DEVICES" || key == "NPUW_SLICE_OUT" || key == "NPUW_FUNCALL_FOR_ALL") {
                continue;
            }
            ASSERT_EQ(config.count(key), 1u) << key;
            EXPECT_EQ(config.at(key).as<std::string>(), value.as<std::string>()) << key;
        }
        // The PA model picks its logits rows itself, in a tail with a dynamic
        // row count that cannot be a function.
        EXPECT_EQ(config.count("NPUW_SLICE_OUT"), 0u);
        EXPECT_EQ(config.count("NPUW_FUNCALL_FOR_ALL"), 0u);
    }

    const auto config = PACompiledModel::make_variant_config({}, "CPU", "bank0", npu_with_compiler_dq());
    EXPECT_EQ(str(config, "NPUW_DQ"), "YES");
    EXPECT_EQ(str(config, "NPU_COMPILER_DYNAMIC_QUANTIZATION"), "YES");
    EXPECT_EQ(str(config, "NPUW_FUNCALL_ASYNC"), "YES");
    EXPECT_EQ(str(config, "NPU_TILES"), "6");
}

TEST(PAVariantConfig, ForwardsTheCallerPropertiesButNotTheWholeModelChoices) {
    const ov::AnyMap properties = {{"NPUW_PA", "YES"},
                                   {"NPUW_PLAN", "plan.xml"},
                                   {"NPUW_SUBMODEL_DEVICE", "0:CPU"},
                                   {"NPUW_DUMP_SUBS", "YES"},
                                   {"KV_CACHE_PRECISION", "f16"}};
    const auto config = PACompiledModel::make_variant_config(properties, "CPU", "bank1", std::nullopt);

    EXPECT_EQ(config.count("NPUW_PA"), 0u);
    EXPECT_EQ(config.count("NPUW_PLAN"), 0u);
    EXPECT_EQ(str(config, "NPUW_SUBMODEL_DEVICE"), "last:CPU");
    EXPECT_EQ(str(config, "NPUW_DUMP_SUBS"), "YES");
    EXPECT_EQ(str(config, "KV_CACHE_PRECISION"), "f16");
}

TEST(PAVariantConfig, CallerPropertiesOverrideTheDefaults) {
    const ov::AnyMap properties = {{"NPUW_DEVICES", "CPU"},
                                   {"NPUW_ONLINE_AVOID", "Op:Select/NPU"},
                                   {"NPU_USE_NPUW", "NO"},
                                   {"NPUW_WEIGHTS_BANK", "caller"}};
    const auto config = PACompiledModel::make_variant_config(properties, "CPU", "bank2", std::nullopt);

    EXPECT_EQ(str(config, "NPUW_DEVICES"), "CPU");
    EXPECT_EQ(str(config, "NPUW_ONLINE_AVOID"), "Op:Select/NPU");
    EXPECT_EQ(str(config, "NPUW_ONLINE_ISOLATE"), "Op:PagedAttentionExtension/pa");
    // A variant is always an NPUW model with the bank the front-end chose.
    EXPECT_EQ(str(config, "NPU_USE_NPUW"), "YES");
    EXPECT_EQ(str(config, "NPUW_WEIGHTS_BANK"), "bank2");
}

TEST(PAVariantConfig, CallerCompilationModeParamsExtendTheDefaults) {
    const ov::AnyMap properties = {
        {"NPU_COMPILATION_MODE_PARAMS", "compute-layers-with-higher-precision=MVN optimization-level=3"}};
    const auto config = PACompiledModel::make_variant_config(properties, "CPU", "bank3", std::nullopt);

    const auto params = str(config, "NPU_COMPILATION_MODE_PARAMS");
    EXPECT_NE(params.find("Add_RMSNorm"), std::string::npos) << params;
    EXPECT_NE(params.find("MVN"), std::string::npos) << params;
    EXPECT_NE(params.find("optimization-level=3"), std::string::npos) << params;
}

// The synthetic LLM as the GenAI CB pipeline hands it to a device: PA ops on
// the caches, and the LM head behind a Gather of the sampled token rows.
std::shared_ptr<ov::Model> make_pipeline_pa_model() {
    auto model = ov::test::npuw::build_llm_test_model();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::SDPAToPagedAttention>();
    manager.run_passes(model);

    std::shared_ptr<ov::Node> lm_head = model->output("logits").get_node()->input_value(0).get_node_shared_ptr();
    while (!ov::is_type<ov::op::v0::MatMul>(lm_head)) {
        lm_head = lm_head->input_value(0).get_node_shared_ptr();
    }
    auto sampled = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1});
    sampled->set_friendly_name("sampled_tokens_indices");
    sampled->output(0).get_tensor().set_names({"sampled_tokens_indices"});
    auto rows = std::make_shared<ov::op::v8::Gather>(lm_head->input_value(0),
                                                     sampled,
                                                     ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
    lm_head->input(0).replace_source_output(rows);
    model->add_parameters({sampled});
    model->validate_nodes_and_infer_types();
    return model;
}

// Runs a prefill of `prompt` tokens and one decode step for a single sequence,
// and returns the logits of each, as the CB pipeline would drive the model.
std::vector<std::vector<float>> prefill_and_decode(ov::CompiledModel& compiled, std::size_t prompt) {
    constexpr std::size_t kBlockSize = 32;
    constexpr std::size_t kBlocks = 8;
    auto request = compiled.create_infer_request();
    for (const auto& port : compiled.inputs()) {
        const auto& name = port.get_any_name();
        if (name.rfind("key_cache.", 0) == 0 || name.rfind("value_cache.", 0) == 0) {
            auto shape = port.get_partial_shape();
            shape[0] = static_cast<int64_t>(kBlocks);
            ov::Tensor cache(port.get_element_type(), shape.to_shape());
            std::fill_n(static_cast<uint8_t*>(cache.data()), cache.get_byte_size(), uint8_t{0});
            request.set_tensor(name, cache);
        }
    }
    const auto i32 = [](std::vector<int32_t> v) {
        ov::Tensor t(ov::element::i32, {v.size()});
        std::copy(v.begin(), v.end(), t.data<int32_t>());
        return t;
    };
    const auto i64 = [](std::vector<int64_t> v) {
        ov::Tensor t(ov::element::i64, {v.size()});
        std::copy(v.begin(), v.end(), t.data<int64_t>());
        return t;
    };
    std::vector<std::vector<float>> logits;
    std::size_t past = 0;
    for (const std::size_t tokens : {prompt, std::size_t{1}}) {
        std::vector<int64_t> ids(tokens), positions(tokens);
        for (std::size_t i = 0; i < tokens; ++i) {
            ids[i] = static_cast<int64_t>(1 + (past + i) * 7 % 250);
            positions[i] = static_cast<int64_t>(past + i);
        }
        const auto blocks = static_cast<int32_t>((past + tokens + kBlockSize - 1) / kBlockSize);
        std::vector<int32_t> table(blocks);
        for (int32_t b = 0; b < blocks; ++b) {
            table[b] = b;
        }
        request.set_tensor("input_ids", i64(ids));
        request.set_tensor("position_ids", i64(positions));
        request.set_tensor("past_lens", i32({static_cast<int32_t>(past)}));
        request.set_tensor("subsequence_begins", i32({0, static_cast<int32_t>(tokens)}));
        request.set_tensor("block_indices", i32(table));
        request.set_tensor("block_indices_begins", i32({0, blocks}));
        ov::Tensor max_context_len(ov::element::i32, {});
        *max_context_len.data<int32_t>() = static_cast<int32_t>(past + tokens);
        request.set_tensor("max_context_len", max_context_len);
        request.set_tensor("sampled_tokens_indices", i64({static_cast<int64_t>(tokens - 1)}));
        request.infer();
        const auto out = request.get_tensor("logits");
        logits.emplace_back(out.data<float>(), out.data<float>() + out.get_size());
        past += tokens;
    }
    return logits;
}

// The prefill runs as a 128-token chunk on its variant plus a dynamic tail,
// the decode on the 1-token variant, all through NPUW on CPU.
TEST(PACompiledModelEndToEnd, VariantsCompileAndMatchTheDynamicModel) {
    const auto model = make_pipeline_pa_model();
    // The model meets the chunkable contract, so the variants serve it.
    const std::set<std::string> controls = {"input_ids",
                                            "position_ids",
                                            "past_lens",
                                            "subsequence_begins",
                                            "block_indices",
                                            "block_indices_begins",
                                            "max_context_len",
                                            "sampled_tokens_indices"};
    for (const auto& input : model->inputs()) {
        const auto& name = input.get_any_name();
        const bool cache = name.rfind("key_cache.", 0) == 0 || name.rfind("value_cache.", 0) == 0;
        ASSERT_TRUE(cache || controls.count(name) > 0) << "unexpected input " << name;
    }
    ASSERT_EQ(model->input("input_ids").get_partial_shape().rank().get_length(), 1);
    ASSERT_EQ(model->input("position_ids").get_partial_shape().rank().get_length(), 1);

    const ov::AnyMap common = {{ov::hint::inference_precision.name(), ov::element::f32},
                               {ov::hint::kv_cache_precision.name(), ov::element::f16}};
    ov::AnyMap npuw = common;
    npuw.insert({{"NPU_USE_NPUW", "YES"}, {"NPUW_PA", "YES"}, {"NPUW_DEVICES", "CPU"}, {"NPUW_F16IC", "NO"}});

    ov::Core core;
    auto reference = core.compile_model(model->clone(), "CPU", common);
    ov::CompiledModel front_end;
    ASSERT_NO_THROW(front_end = core.compile_model(model->clone(), "NPU", npuw));

    const auto expected = prefill_and_decode(reference, 130);
    const auto actual = prefill_and_decode(front_end, 130);
    ASSERT_EQ(expected.size(), actual.size());
    for (std::size_t step = 0; step < expected.size(); ++step) {
        ASSERT_EQ(expected[step].size(), actual[step].size());
        float scale = 1.f;
        for (const auto v : expected[step]) {
            scale = std::max(scale, std::fabs(v));
        }
        for (std::size_t i = 0; i < expected[step].size(); ++i) {
            EXPECT_NEAR(expected[step][i], actual[step][i], 1e-3f * scale) << "step " << step << " at " << i;
        }
    }
}

// A token-major trunk as the PA model has it: a weight MatMul over
// [tokens, 1, hidden], elementwise ops on it, and the RoPE position outer
// product, a MatMul with K = 1 between two activations.
std::shared_ptr<ov::Model> make_token_major_model(std::size_t tokens) {
    const auto t = static_cast<int64_t>(tokens);
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{t, 1, 16});
    auto y = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{t, 1, 12});
    auto freq = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{t, 4, 1});
    auto pos = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{t, 1, 1});
    std::vector<float> w(12 * 16), b(12);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& v : w) {
        v = dist(rng);
    }
    for (auto& v : b) {
        v = dist(rng);
    }
    auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{12, 16}, w);
    auto linear = std::make_shared<ov::op::v0::MatMul>(x, weights, false, true);
    auto bias = std::make_shared<ov::op::v1::Add>(linear, ov::op::v0::Constant::create(ov::element::f32, {12}, b));
    auto gate = std::make_shared<ov::op::v1::Multiply>(bias, y);
    auto rope = std::make_shared<ov::op::v0::MatMul>(freq, pos);
    return std::make_shared<ov::Model>(ov::OutputVector{gate, rope}, ov::ParameterVector{x, y, freq, pos});
}

ov::TensorVector evaluate(const std::shared_ptr<ov::Model>& model) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-2.f, 2.f);
    ov::TensorVector inputs;
    for (const auto& param : model->get_parameters()) {
        ov::Tensor tensor(ov::element::f32, param->get_shape());
        for (std::size_t i = 0; i < tensor.get_size(); ++i) {
            tensor.data<float>()[i] = dist(rng);
        }
        inputs.push_back(tensor);
    }
    ov::TensorVector outputs;
    for (const auto& result : model->outputs()) {
        outputs.emplace_back(ov::element::f32, result.get_shape());
    }
    EXPECT_TRUE(model->evaluate(outputs, inputs));
    return outputs;
}

TEST(PARelayoutTokenMajor, MovesTheTokenCountOutOfTheBatchPosition) {
    constexpr std::size_t tokens = 8;
    const auto reference = make_token_major_model(tokens);
    auto model = reference->clone();
    ov::npuw::pa::relayout_token_major(model, tokens);

    std::size_t matmuls = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (!ov::is_type<ov::op::v0::MatMul>(node) && !ov::is_type<ov::op::v1::Multiply>(node) &&
            !ov::is_type<ov::op::v1::Add>(node)) {
            continue;
        }
        matmuls += ov::is_type<ov::op::v0::MatMul>(node) ? 1 : 0;
        for (const auto& input : node->inputs()) {
            const auto& shape = input.get_shape();
            EXPECT_FALSE(shape.size() >= 3 && shape[0] == tokens) << node << " input " << shape;
        }
    }
    // The outer product became a Multiply; the weight MatMul stays.
    EXPECT_EQ(matmuls, 1u);

    const auto expected = evaluate(reference);
    const auto actual = evaluate(model);
    ASSERT_EQ(expected.size(), actual.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        ASSERT_EQ(expected[i].get_shape(), actual[i].get_shape());
        for (std::size_t j = 0; j < expected[i].get_size(); ++j) {
            EXPECT_FLOAT_EQ(expected[i].data<float>()[j], actual[i].data<float>()[j]) << "output " << i << " at " << j;
        }
    }
}

TEST(PARelayoutTokenMajor, LeavesTheOneTokenVariantAlone) {
    auto model = make_token_major_model(1);
    const auto ops = model->get_ops().size();
    ov::npuw::pa::relayout_token_major(model, 1);
    EXPECT_EQ(model->get_ops().size(), ops);
}

}  // namespace
