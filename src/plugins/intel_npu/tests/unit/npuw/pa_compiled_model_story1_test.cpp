// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "llm_test_helpers.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "pa_compiled_model.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_icore.hpp"

namespace {

using ov::test::npuw::NullPlugin;

class LocalCompiledModel final : public ov::ICompiledModel {
public:
    LocalCompiledModel(const std::shared_ptr<const ov::Model>& model, const std::shared_ptr<const ov::IPlugin>& plugin)
        : ov::ICompiledModel(model, plugin),
          m_model(model),
          m_inputs(model->inputs()),
          m_outputs(model->outputs()) {}

    const std::vector<ov::Output<const ov::Node>>& inputs() const override {
        return m_inputs;
    }

    const std::vector<ov::Output<const ov::Node>>& outputs() const override {
        return m_outputs;
    }

    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override {
        return nullptr;
    }

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override {
        return nullptr;
    }

    std::shared_ptr<const ov::Model> get_runtime_model() const override {
        return m_model;
    }

    void export_model(std::ostream&) const override {}

    void set_property(const ov::AnyMap&) override {}

    ov::Any get_property(const std::string& name) const override {
        if (name == ov::supported_properties.name()) {
            return std::vector<ov::PropertyName>{};
        }
        return {};
    }

    ov::SoPtr<ov::IRemoteContext> get_context() const {
        return {};
    }

private:
    std::shared_ptr<const ov::Model> m_model;
    std::vector<ov::Output<const ov::Node>> m_inputs;
    std::vector<ov::Output<const ov::Node>> m_outputs;
};

std::shared_ptr<ov::Model> build_pa_test_model() {
    using ov::op::v0::Parameter;
    using ov::op::v0::Result;

    auto input_ids = std::make_shared<Parameter>(ov::element::i32, ov::PartialShape{1, ov::Dimension::dynamic()});
    input_ids->set_friendly_name("input_ids");
    input_ids->output(0).get_tensor().set_names({"input_ids"});

    auto position_ids = std::make_shared<Parameter>(ov::element::i64, ov::PartialShape{1, ov::Dimension::dynamic()});
    position_ids->set_friendly_name("position_ids");
    position_ids->output(0).get_tensor().set_names({"position_ids"});

    auto past_lens = std::make_shared<Parameter>(ov::element::i32, ov::PartialShape{ov::Dimension::dynamic()});
    past_lens->set_friendly_name("past_lens");
    past_lens->output(0).get_tensor().set_names({"past_lens"});

    auto subsequence_begins =
        std::make_shared<Parameter>(ov::element::i32, ov::PartialShape{ov::Dimension::dynamic()});
    subsequence_begins->set_friendly_name("subsequence_begins");
    subsequence_begins->output(0).get_tensor().set_names({"subsequence_begins"});

    auto block_indices = std::make_shared<Parameter>(ov::element::i32, ov::PartialShape{ov::Dimension::dynamic()});
    block_indices->set_friendly_name("block_indices");
    block_indices->output(0).get_tensor().set_names({"block_indices"});

    auto block_indices_begins =
        std::make_shared<Parameter>(ov::element::i32, ov::PartialShape{ov::Dimension::dynamic()});
    block_indices_begins->set_friendly_name("block_indices_begins");
    block_indices_begins->output(0).get_tensor().set_names({"block_indices_begins"});

    auto max_context_len = std::make_shared<Parameter>(ov::element::i32, ov::PartialShape{1});
    max_context_len->set_friendly_name("max_context_len");
    max_context_len->output(0).get_tensor().set_names({"max_context_len"});

    auto sampled_tokens_indices =
        std::make_shared<Parameter>(ov::element::i32, ov::PartialShape{ov::Dimension::dynamic()});
    sampled_tokens_indices->set_friendly_name("sampled_tokens_indices");
    sampled_tokens_indices->output(0).get_tensor().set_names({"sampled_tokens_indices"});

    auto key_cache = std::make_shared<Parameter>(ov::element::dynamic,
                                                 ov::PartialShape{ov::Dimension::dynamic(), 2, ov::Dimension::dynamic(), 16});
    key_cache->set_friendly_name("key_cache.0");
    key_cache->output(0).get_tensor().set_names({"key_cache.0"});

    auto value_cache = std::make_shared<Parameter>(ov::element::dynamic,
                                                   ov::PartialShape{ov::Dimension::dynamic(), 2, ov::Dimension::dynamic(), 16});
    value_cache->set_friendly_name("value_cache.0");
    value_cache->output(0).get_tensor().set_names({"value_cache.0"});

    auto logits = std::make_shared<ov::op::v0::Convert>(input_ids, ov::element::f32);
    logits->set_friendly_name("logits");
    logits->output(0).get_tensor().set_names({"logits"});

    auto result = std::make_shared<Result>(logits);
    result->set_friendly_name("logits");

    ov::ParameterVector params = {
        input_ids,
        position_ids,
        past_lens,
        subsequence_begins,
        block_indices,
        block_indices_begins,
        max_context_len,
        sampled_tokens_indices,
        key_cache,
        value_cache,
    };
    return std::make_shared<ov::Model>(ov::ResultVector{result}, params, "pa_test_model");
}

}  // namespace

TEST(PACompiledModelStory1Test, CompilesDynamicAndSemiStaticVariants) {
    auto plugin = std::make_shared<NullPlugin>();
    auto core = std::make_shared<testing::NiceMock<ov::MockICore>>();
    plugin->set_core(core);

    std::vector<int64_t> compiled_token_dims;

    ON_CALL(*core,
            compile_model(testing::Matcher<const std::shared_ptr<const ov::Model>&>(
                              testing::An<const std::shared_ptr<const ov::Model>&>()),
                          testing::Matcher<const std::string&>(testing::StrEq("CPU")),
                          testing::Matcher<const ov::AnyMap&>(testing::An<const ov::AnyMap&>())))
        .WillByDefault([&](const std::shared_ptr<const ov::Model>& model,
                           const std::string&,
                           const ov::AnyMap& props) -> ov::SoPtr<ov::ICompiledModel> {
            EXPECT_EQ(props.count("NPUW_PA"), 0u);
            EXPECT_EQ(props.count("NPUW_PA_DEVICE"), 0u);

            const auto shape = model->input("input_ids").get_partial_shape();
            if (shape.rank().is_static() && shape.rank().get_length() >= 2 && shape[1].is_static()) {
                compiled_token_dims.push_back(shape[1].get_length());
            } else {
                compiled_token_dims.push_back(-1);
            }

            return std::make_shared<LocalCompiledModel>(model, plugin);
        });

    auto model = build_pa_test_model();
    ov::AnyMap props = {{"NPUW_PA", "YES"}, {"NPUW_PA_DEVICE", "CPU"}, {"PERFORMANCE_HINT", "LATENCY"}};

    std::shared_ptr<ov::npuw::PACompiledModel> compiled;
    ASSERT_NO_THROW(compiled = std::make_shared<ov::npuw::PACompiledModel>(model, plugin, props));
    ASSERT_NE(compiled, nullptr);

    ASSERT_EQ(compiled_token_dims.size(), 4u);
    EXPECT_EQ(std::count(compiled_token_dims.begin(), compiled_token_dims.end(), -1), 1);

    std::set<int64_t> static_dims;
    for (const auto dim : compiled_token_dims) {
        if (dim > 0) {
            static_dims.insert(dim);
        }
    }
    EXPECT_EQ(static_dims, (std::set<int64_t>{1, 128, 1024}));
}
