// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mocks/mock_factory.hpp"
#include "model_builder.hpp"
#include "llm_compiled_model.hpp"
#include "openvino/util/shared_object.hpp"
#include "openvino/util/file_util.hpp"

#include <filesystem>

using namespace testing;
using namespace ov::npuw::tests;

namespace ov {
namespace npuw {
namespace tests {
class LLMCompiledModelTest : public ::testing::Test {
public:
    void SetUp() override {
        ov::test::npuw::ModelBuilder model_builder;
        ov::test::npuw::ModelConfig model_config;
        model_config.num_layers = 12;
        m_ov_model = model_builder.build_model(model_config);

        m_mock_core = std::make_shared<MockCore>();
        m_mock_core->create_implementation();

        m_mock_plugin = std::make_shared<MockPlugin>();
        m_mock_plugin->set_core(m_mock_core);
        m_mock_plugin->create_implementation();

        m_mock_npuw_factory = std::make_shared<MockNpuwCompiledModelFactory>();
        m_mock_npuw_factory->create_implementation();

        m_config["NPU_USE_NPUW"] = "YES";
        m_config["NPUW_LLM"] = "YES";
        m_config["NPUW_LLM_MIN_RESPONSE_LEN"] = 64;
    }

protected:
    std::shared_ptr<ov::Model> m_ov_model;
    std::shared_ptr<MockCore> m_mock_core;
    std::shared_ptr<MockPlugin> m_mock_plugin;
    std::shared_ptr<MockNpuwCompiledModelFactory> m_mock_npuw_factory;
    ov::AnyMap m_config;

private:
    std::vector<std::shared_ptr<void>> m_shared_objects;
};
}  // namespace tests
}  // namespace npuw
}  // namespace ov

using Factory = MockNpuwCompiledModelFactory;
TEST_F(LLMCompiledModelTest, PrefillGenerateAreCorrect) {
    // Set expectations first:
    EXPECT_CALL(*m_mock_npuw_factory, create(_, _, _)).Times(3);

    std::shared_ptr<ov::npuw::LLMCompiledModel> llm_compiled_model;
    EXPECT_NO_THROW(llm_compiled_model =
        std::make_shared<ov::npuw::LLMCompiledModel>(m_ov_model, m_mock_plugin, m_config, m_mock_npuw_factory));

    // Make non-GMock related checks:
    EXPECT_EQ(3u, m_mock_npuw_factory->m_ov_models.size());
    std::shared_ptr<ov::Model> ov_prefill_model = m_mock_npuw_factory->m_ov_models[Factory::kPREFILL_MODEL_INDEX];
    auto prefill_input_ids_shape = ov_prefill_model->inputs()[0].get_shape();
    auto prefill_attention_mask_shape = ov_prefill_model->inputs()[2].get_shape();
    EXPECT_EQ(ov::Shape({1, 1024}), prefill_input_ids_shape);
    EXPECT_EQ(ov::Shape({1, 1024}), prefill_attention_mask_shape);
    {
        bool npuw_output_embed_found {false};
        for (auto output : ov_prefill_model->outputs()) {
            auto names = output.get_names();
            if (names.count("npuw_output_embed")) {
                npuw_output_embed_found = true;
                break;
            }
        }
        EXPECT_TRUE(npuw_output_embed_found);
    }

    std::shared_ptr<ov::Model> ov_generate_model = m_mock_npuw_factory->m_ov_models[Factory::kGENERATE_MODEL_INDEX];
    auto generate_input_ids_shape = ov_generate_model->inputs()[0].get_shape();
    auto generate_attention_mask_shape = ov_generate_model->inputs()[2].get_shape();
    EXPECT_EQ(ov::Shape({1, 1}), generate_input_ids_shape);
    EXPECT_EQ(ov::Shape({1, 1088}), generate_attention_mask_shape);
    {
        bool npuw_output_embed_found {false};
        for (auto output : ov_generate_model->outputs()) {
            auto names = output.get_names();
            if (names.count("npuw_output_embed")) {
                npuw_output_embed_found = true;
                break;
            }
        }
        EXPECT_TRUE(npuw_output_embed_found);
    }

    EXPECT_EQ(3u, m_mock_npuw_factory->m_ov_models_properties.size());
    ov::AnyMap ov_prefill_props = m_mock_npuw_factory->m_ov_models_properties[Factory::kPREFILL_MODEL_INDEX];
    EXPECT_TRUE(ov_prefill_props.count("NPUW_SLICE_OUT"));
    ov::AnyMap ov_generate_props = m_mock_npuw_factory->m_ov_models_properties[Factory::kGENERATE_MODEL_INDEX];
    EXPECT_TRUE(ov_generate_props.count("NPUW_UNFOLD_IREQS"));
}
