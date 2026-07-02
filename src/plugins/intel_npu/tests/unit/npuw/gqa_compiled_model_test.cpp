// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "gqa_compiled_model.hpp"
#include "llm_test_helpers.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/runtime/properties.hpp"

namespace {

using ov::test::npuw::MockSubCompiledModel;
using ov::test::npuw::NullPlugin;
using ov::test::npuw::build_llm_test_model;

template <class Op>
std::size_t count_ops(const std::shared_ptr<ov::Model>& model) {
    const auto ops = model->get_ops();
    return std::count_if(ops.begin(), ops.end(), [](const auto& op) {
        return ov::is_type<Op>(op);
    });
}

std::shared_ptr<ov::Model> build_group_query_attention_model() {
    auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4, 1, 16});
    auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 1, 16});
    auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 1, 16});
    auto past_key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 8, 16});
    auto past_value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 8, 16});
    auto seqlens_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    auto total_sequence_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});

    auto gqa = std::make_shared<ov::op::internal::GroupQueryAttention>(
        ov::OutputVector{query, key, value, past_key, past_value, seqlens_k, total_sequence_length},
        4,
        2,
        0.0f,
        false,
        false);

    ov::ResultVector results = {std::make_shared<ov::op::v0::Result>(gqa->output(0)),
                                std::make_shared<ov::op::v0::Result>(gqa->output(1)),
                                std::make_shared<ov::op::v0::Result>(gqa->output(2))};
    ov::ParameterVector params = {query, key, value, past_key, past_value, seqlens_k, total_sequence_length};
    return std::make_shared<ov::Model>(results, params, "gqa_model");
}

std::shared_ptr<ov::Model> build_unqdq_model(const ov::element::Type& input_type = ov::element::f32) {
    auto input = std::make_shared<ov::op::v0::Parameter>(input_type, ov::Shape{1, 4});
    auto input_low = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {-1.0f});
    auto input_high = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f});
    auto output_low = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});
    auto output_high = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {255.0f});
    auto fake_quantize =
        std::make_shared<ov::op::v0::FakeQuantize>(input, input_low, input_high, output_low, output_high, 256);
    auto quantized_convert = std::make_shared<ov::op::v0::Convert>(fake_quantize, ov::element::u16);
    auto dequantized_convert = std::make_shared<ov::op::v0::Convert>(quantized_convert, ov::element::f32);
    auto zero_point = std::make_shared<ov::op::v0::Convert>(
        ov::op::v0::Constant::create(ov::element::u16, ov::Shape{}, {128}),
        ov::element::f32);
    auto subtract = std::make_shared<ov::op::v1::Subtract>(dequantized_convert, zero_point);
    auto scale = std::make_shared<ov::op::v0::Convert>(
        ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {0.1f}),
        ov::element::f32);
    auto multiply = std::make_shared<ov::op::v1::Multiply>(subtract, scale);
    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(multiply)},
                                       ov::ParameterVector{input},
                                       "gqa_unqdq_model");
}

std::shared_ptr<ov::Model> build_hidden_states_model(std::size_t tokens) {
    auto input_hidden_states =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, tokens, 16});
    input_hidden_states->set_friendly_name("input_hidden_states");
    return std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(input_hidden_states)},
        ov::ParameterVector{input_hidden_states},
        "gqa_hidden_states_model");
}

std::shared_ptr<ov::Model> build_conv_to_matmul_model() {
    auto activation = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 8, 3});
    auto weights = std::make_shared<ov::op::v0::Parameter>(ov::element::i8, ov::Shape{4, 3, 1, 1});
    auto scale = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{4, 1, 1, 1});

    auto transpose_in = std::make_shared<ov::op::v1::Transpose>(
        activation,
        ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{0, 3, 1, 2}));
    auto scaled_weights = std::make_shared<ov::op::v1::Multiply>(
        std::make_shared<ov::op::v0::Convert>(weights, ov::element::f32),
        std::make_shared<ov::op::v0::Convert>(scale, ov::element::f32));
    auto convolution = std::make_shared<ov::op::v1::Convolution>(transpose_in,
                                                                 scaled_weights,
                                                                 ov::Strides{1, 1},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::Strides{1, 1});
    auto transpose_out = std::make_shared<ov::op::v1::Transpose>(
        convolution,
        ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{0, 2, 3, 1}));

    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(transpose_out)},
                                       ov::ParameterVector{activation, weights, scale},
                                       "gqa_conv_to_matmul_model");
}

std::shared_ptr<ov::Model> build_dumped_gqa_conv_model() {
    auto activation = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 512, 5120});
    auto weights = std::make_shared<ov::op::v0::Parameter>(ov::element::i4, ov::Shape{5120, 5120, 1, 1});
    auto scale = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{5120});

    auto converted_activation = std::make_shared<ov::op::v0::Convert>(activation, ov::element::f32);
    auto unsqueezed_activation = std::make_shared<ov::op::v0::Unsqueeze>(
        converted_activation,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}));
    auto transposed_activation = std::make_shared<ov::op::v1::Transpose>(
        unsqueezed_activation,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2}));
    auto scaled_weights = std::make_shared<ov::op::v1::Multiply>(
        std::make_shared<ov::op::v0::Convert>(weights, ov::element::f32),
        std::make_shared<ov::op::v1::Reshape>(
            scale,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {5120, 1, 1, 1}),
            true));
    auto convolution = std::make_shared<ov::op::v1::Convolution>(transposed_activation,
                                                                scaled_weights,
                                                                ov::Strides{1, 1},
                                                                ov::CoordinateDiff{0, 0},
                                                                ov::CoordinateDiff{0, 0},
                                                                ov::Strides{1, 1});
    auto transpose_out = std::make_shared<ov::op::v1::Transpose>(
        convolution,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 3, 1}));

    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(transpose_out)},
                                       ov::ParameterVector{activation, weights, scale},
                                       "dumped_gqa_conv_model");
}

std::shared_ptr<ov::Model> build_conv_to_matmul_and_unqdq_model() {
    const auto conv_model = build_conv_to_matmul_model();
    const auto unqdq_model = build_unqdq_model(ov::element::f16);

    ov::ResultVector results;
    for (const auto& result : conv_model->get_results()) {
        results.push_back(std::make_shared<ov::op::v0::Result>(result->input_value(0)));
    }
    for (const auto& result : unqdq_model->get_results()) {
        results.push_back(std::make_shared<ov::op::v0::Result>(result->input_value(0)));
    }

    ov::ParameterVector parameters = conv_model->get_parameters();
    const auto unqdq_parameters = unqdq_model->get_parameters();
    parameters.insert(parameters.end(), unqdq_parameters.begin(), unqdq_parameters.end());

    return std::make_shared<ov::Model>(results, parameters, "gqa_conv_to_matmul_and_unqdq_model");
}

struct CompileCall {
    ov::AnyMap props;
    std::shared_ptr<ov::Model> model;
};

class RecordingFactory {
public:
    ov::npuw::GQACompiledModel::CompiledModelFactory make_factory() {
        return [this](const std::shared_ptr<ov::Model>& model,
                      const std::shared_ptr<const ov::IPlugin>& plugin,
                      const ov::AnyMap& props) -> std::shared_ptr<ov::npuw::ICompiledModel> {
            m_calls.push_back({props, model});
            return std::make_shared<MockSubCompiledModel>(model, plugin, props);
        };
    }

    const CompileCall& only_call() const {
        OPENVINO_ASSERT(m_calls.size() == 1u, "Expected a single compile call");
        return m_calls.front();
    }

private:
    std::vector<CompileCall> m_calls;
};

class PropertyForwardingMockCompiledModel final : public ov::npuw::ICompiledModel {
public:
    PropertyForwardingMockCompiledModel(const std::shared_ptr<ov::Model>& model,
                                       const std::shared_ptr<const ov::IPlugin>& plugin)
        : ov::npuw::ICompiledModel(model, plugin) {}

    void export_model(std::ostream&) const override {}
    std::shared_ptr<const ov::Model> get_runtime_model() const override {
        return {};
    }
    void set_property(const ov::AnyMap& properties) override {
        last_set_properties = properties;
    }
    ov::Any get_property(const std::string& name) const override {
        if (name == "NPUW_FOLD") {
            return true;
        }
        return {};
    }

private:
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override {
        return {};
    }

public:
    ov::AnyMap last_set_properties;
};


class GQACompiledModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        m_plugin = std::make_shared<NullPlugin>();
    }

    static ov::AnyMap base_props() {
        return {{"NPUW_GQA", "YES"}};
    }

    static void merge_props(ov::AnyMap& dst, const ov::AnyMap& src) {
        for (const auto& [key, value] : src) {
            dst[key] = value;
        }
    }

    std::unique_ptr<ov::npuw::GQACompiledModel> create_compiled_model(const std::shared_ptr<ov::Model>& model,
                                                                      const ov::AnyMap& extra_props,
                                                                      RecordingFactory& recorder) const {
        auto props = base_props();
        merge_props(props, extra_props);
        return std::make_unique<ov::npuw::GQACompiledModel>(model, m_plugin, props, recorder.make_factory());
    }

    std::shared_ptr<ov::IPlugin> m_plugin;
};

TEST_F(GQACompiledModelTest, AddsExpectedNpuwDefaultsBeforeInnerCompilation) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::GQACompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_test_model(), {}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_EQ(call.props.at("NPUW_ONLINE_PIPELINE").as<std::string>(), "REP");
    EXPECT_EQ(call.props.at("NPUW_DEVICES").as<std::string>(), "NPU");
    EXPECT_EQ(call.props.at("NPUW_ONLINE_ISOLATE").as<std::string>(), "ATTN");
    EXPECT_EQ(call.props.at("NPUW_ATTN").as<std::string>(), "STATIC");
    EXPECT_EQ(call.props.at("NPUW_ONLINE_KEEP_BLOCK_SIZE").as<std::string>(), "2");
    EXPECT_EQ(call.props.at("NPUW_FOLD").as<std::string>(), "YES");
    EXPECT_EQ(call.props.at(ov::cache_mode.name()).as<ov::CacheMode>(), ov::CacheMode::OPTIMIZE_SPEED);
    EXPECT_EQ(call.props.at("NPUW_UNQDQ").as<std::string>(), "YES");
    EXPECT_EQ(call.props.count("NPUW_FUNCALL_ASYNC"), 0u);
    EXPECT_EQ(call.props.count("NPUW_UNFOLD_IREQS"), 0u);
}

TEST_F(GQACompiledModelTest, AppliesFoldOnlyAttnForGenerateStyleModels) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::GQACompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_hidden_states_model(1), {}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_EQ(call.props.at("NPUW_ONLINE_PIPELINE").as<std::string>(), "REP");
    EXPECT_EQ(call.props.at("NPUW_DEVICES").as<std::string>(), "NPU");
    EXPECT_EQ(call.props.at("NPUW_FOLD").as<std::string>(), "YES");
    EXPECT_EQ(call.props.at(ov::cache_mode.name()).as<ov::CacheMode>(), ov::CacheMode::OPTIMIZE_SPEED);
    EXPECT_EQ(call.props.at("NPUW_UNQDQ").as<std::string>(), "YES");
    EXPECT_EQ(call.props.at("NPUW_FUNCALL_ASYNC").as<std::string>(), "YES");
    EXPECT_EQ(call.props.at("NPUW_UNFOLD_IREQS").as<std::string>(), "YES");
    EXPECT_EQ(call.props.at("NPUW_FOLD_ONLY").as<std::string>(), "attn");
    EXPECT_EQ(call.props.count("NPUW_ONLINE_ISOLATE"), 0u);
    EXPECT_EQ(call.props.count("NPUW_ATTN"), 0u);
    EXPECT_EQ(call.props.count("NPUW_ONLINE_KEEP_BLOCK_SIZE"), 0u);
}

TEST_F(GQACompiledModelTest, KeepsAttnIsolationDefaultsForPrefillStyleModels) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::GQACompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_hidden_states_model(8), {}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_EQ(call.props.at("NPUW_DEVICES").as<std::string>(), "NPU");
    EXPECT_EQ(call.props.at("NPUW_ONLINE_ISOLATE").as<std::string>(), "ATTN");
    EXPECT_EQ(call.props.at("NPUW_FOLD").as<std::string>(), "YES");
    EXPECT_EQ(call.props.count("NPUW_FUNCALL_ASYNC"), 0u);
    EXPECT_EQ(call.props.count("NPUW_UNFOLD_IREQS"), 0u);
}

TEST_F(GQACompiledModelTest, KeepsUserProvidedLowLevelOverrides) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::GQACompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_test_model(),
                                                     {{"NPUW_ONLINE_PIPELINE", "REG"},
                                                      {"NPUW_DEVICES", "CPU"},
                                                      {"NPUW_ONLINE_ISOLATE", "COMPUTE"},
                                                      {"NPUW_ATTN", "STATIC"},
                                                      {"NPUW_FOLD", "NO"},
                                                      {"NPUW_FUNCALL_ASYNC", "NO"},
                                                      {"NPUW_UNFOLD_IREQS", "NO"}},
                                                     recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_EQ(call.props.at("NPUW_ONLINE_PIPELINE").as<std::string>(), "REG");
    EXPECT_EQ(call.props.at("NPUW_DEVICES").as<std::string>(), "CPU");
    EXPECT_EQ(call.props.at("NPUW_ONLINE_ISOLATE").as<std::string>(), "COMPUTE");
    EXPECT_EQ(call.props.at("NPUW_ATTN").as<std::string>(), "STATIC");
    EXPECT_EQ(call.props.at("NPUW_FOLD").as<std::string>(), "NO");
    EXPECT_EQ(call.props.at("NPUW_FUNCALL_ASYNC").as<std::string>(), "NO");
    EXPECT_EQ(call.props.at("NPUW_UNFOLD_IREQS").as<std::string>(), "NO");
}

TEST_F(GQACompiledModelTest, PassesGqaModelThroughWithoutDecomposition) {
    auto model = build_group_query_attention_model();
    ASSERT_GT(count_ops<ov::op::internal::GroupQueryAttention>(model), 0u);

    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::GQACompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(model, {}, recorder));
    ASSERT_NE(compiled, nullptr);

    // GQA model is passed through unchanged — the online partitioner handles
    // isolation and folding of GQA blocks via the NPUW_FOLD_ONLY=attn path.
    const auto& call = recorder.only_call();
    EXPECT_GT(count_ops<ov::op::internal::GroupQueryAttention>(call.model), 0u);
}

TEST_F(GQACompiledModelTest, RunsUNQDQBeforeInnerCompilation) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::GQACompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_unqdq_model(ov::element::f16), {{"NPUW_UNQDQ", "YES"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_EQ(call.props.at("NPUW_UNQDQ").as<std::string>(), "YES");
    EXPECT_EQ(count_ops<ov::op::v1::Multiply>(call.model), 0u);
    EXPECT_EQ(count_ops<ov::op::v1::Subtract>(call.model), 0u);
    EXPECT_EQ(count_ops<ov::op::v0::FakeQuantize>(call.model), 0u);
}

TEST_F(GQACompiledModelTest, RunsConvToMatmulBeforeInnerCompilation) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::GQACompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_conv_to_matmul_model(), {{"NPUW_UNQDQ", "NO"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_EQ(call.props.at("NPUW_UNQDQ").as<std::string>(), "NO");
    EXPECT_EQ(count_ops<ov::op::v1::Convolution>(call.model), 0u);
    EXPECT_EQ(count_ops<ov::op::v0::MatMul>(call.model), 1u);
}

TEST_F(GQACompiledModelTest, RunsConvToMatmulOnDumpedGQAShapeBeforeInnerCompilation) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::GQACompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_dumped_gqa_conv_model(), {{"NPUW_UNQDQ", "YES"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_EQ(call.props.at("NPUW_UNQDQ").as<std::string>(), "YES");
    EXPECT_EQ(count_ops<ov::op::v1::Convolution>(call.model), 0u);
    EXPECT_EQ(count_ops<ov::op::v0::MatMul>(call.model), 1u);
}

TEST_F(GQACompiledModelTest, RunsConvToMatmulAndUNQDQBeforeInnerCompilation) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::GQACompiledModel> compiled;

    ASSERT_NO_THROW(
        compiled = create_compiled_model(build_conv_to_matmul_and_unqdq_model(), {{"NPUW_UNQDQ", "YES"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_EQ(call.props.at("NPUW_UNQDQ").as<std::string>(), "YES");
    EXPECT_EQ(count_ops<ov::op::v1::Convolution>(call.model), 0u);
    EXPECT_EQ(count_ops<ov::op::v0::MatMul>(call.model), 1u);
    EXPECT_EQ(count_ops<ov::op::v1::Subtract>(call.model), 0u);
    EXPECT_EQ(count_ops<ov::op::v0::FakeQuantize>(call.model), 0u);
}

TEST_F(GQACompiledModelTest, ForwardsPropertyAccessToInnerCompiledModel) {
    auto inner = std::make_shared<PropertyForwardingMockCompiledModel>(build_llm_test_model(), m_plugin);
    auto factory = [inner](const std::shared_ptr<ov::Model>&,
                           const std::shared_ptr<const ov::IPlugin>&,
                           const ov::AnyMap&) -> std::shared_ptr<ov::npuw::ICompiledModel> {
        return inner;
    };

    ov::npuw::GQACompiledModel compiled(build_llm_test_model(), m_plugin, base_props(), factory);

    compiled.set_property({{"NPUW_CWAI", "YES"}});
    EXPECT_EQ(inner->last_set_properties.at("NPUW_CWAI").as<std::string>(), "YES");
    EXPECT_TRUE(compiled.get_property("NPUW_FOLD").as<bool>());
}

}  // namespace
