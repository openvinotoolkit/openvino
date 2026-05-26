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
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

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
    EXPECT_EQ(call.props.at("NPUW_ONLINE_ISOLATE").as<std::string>(), "ATTN");
    EXPECT_EQ(call.props.at("NPUW_FOLD_ONLY").as<std::string>(), "attn");
    EXPECT_EQ(call.props.at("NPUW_ATTN").as<std::string>(), "STATIC");
    EXPECT_EQ(call.props.at("NPUW_ONLINE_KEEP_BLOCK_SIZE").as<std::string>(), "9");
}

TEST_F(GQACompiledModelTest, KeepsUserProvidedLowLevelOverrides) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::GQACompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_test_model(),
                                                     {{"NPUW_ONLINE_PIPELINE", "REG"},
                                                      {"NPUW_ONLINE_ISOLATE", "COMPUTE"},
                                                      {"NPUW_ATTN", "STATIC"},
                                                      {"NPUW_FOLD", "YES"}},
                                                     recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& call = recorder.only_call();
    EXPECT_EQ(call.props.at("NPUW_ONLINE_PIPELINE").as<std::string>(), "REG");
    EXPECT_EQ(call.props.at("NPUW_ONLINE_ISOLATE").as<std::string>(), "COMPUTE");
    EXPECT_EQ(call.props.at("NPUW_ATTN").as<std::string>(), "STATIC");
    EXPECT_EQ(call.props.at("NPUW_FOLD").as<std::string>(), "YES");
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
