// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "json_config.hpp"

#include <common_test_utils/graph_comparator.hpp>
#include <extension/json_config.hpp>
#include <extension/json_transformation.hpp>
#include <nlohmann/json.hpp>
#include <ostream>

#include "utils.hpp"

using namespace ov::frontend;

std::string FrontEndJsonConfigTest::getTestCaseName(const testing::TestParamInfo<JsonConfigFEParam>& obj) {
    std::string res = obj.param.m_frontEndName + "_" + obj.param.m_modelName;
    return FrontEndTestUtils::fileToTestName(res);
}

void FrontEndJsonConfigTest::SetUp() {
    FrontEndTestUtils::setupTestEnv();
    m_fem = FrontEndManager();  // re-initialize after setting up environment
    initParamTest();
}

void FrontEndJsonConfigTest::TearDown() {
    m_output_json.close();
    std::remove(m_file_name.c_str());
}

void FrontEndJsonConfigTest::initParamTest() {
    m_param = GetParam();
    m_param.m_modelName = FrontEndTestUtils::make_model_path(m_param.m_modelsPath + m_param.m_modelName);
}

inline std::string get_lib_path(const std::string& lib_name) {
    return ov::util::make_plugin_library_name<char>(ov::util::get_ov_lib_path(), lib_name + IE_BUILD_POSTFIX);
}

void FrontEndJsonConfigTest::generate_json_config() {
    std::string json = R"(
    [
      {
        "custom_attributes": {
          "test_attribute": true
        },
        "id": "buildin_extensions_1::TestExtension1",
        "library": "test_builtin_extensions_1",
        "match_kind": "scope"
      },
      {
        "custom_attributes": {
          "test_attribute": true
        },
        "id": "buildin_extensions_2::TestExtension1",
        "library": "test_builtin_extensions_2",
        "match_kind": "scope"
      },
      {
        "custom_attributes": {
          "test_attribute": true
        },
        "id": "buildin_extensions_2::TestExtension2",
        "library": "test_builtin_extensions_2",
        "match_kind": "scope"
      }
    ]
    )";
    nlohmann::json config_json = nlohmann::json::parse(json);

    // update lib path
    for (auto& section : config_json) {
        section["library"] = get_lib_path(section["library"]);
    }

    m_output_json.open(m_file_name);
    m_output_json << std::setw(4) << config_json << std::endl;
}

///////////////////////////////////////////////////////////////////

TEST_P(FrontEndJsonConfigTest, testJsonConfig) {
    EXPECT_NO_THROW(generate_json_config());
}

TEST_P(FrontEndJsonConfigTest, testAddJsonConfigExtension) {
    EXPECT_NO_THROW(generate_json_config());
    std::map<std::string, nlohmann::json> reference_map = {
        {"buildin_extensions_1::TestExtension1",
         nlohmann::json::parse(
             R"(
                 {
                    "custom_attributes": {
                        "test_attribute": true
                    },
                    "id": "buildin_extensions_1::TestExtension1",
                    "library": "test_builtin_extensions_1",
                    "match_kind": "scope"
                 }
                 )")},
        {"buildin_extensions_2::TestExtension1",
         nlohmann::json::parse(
             R"(
                 {
                    "custom_attributes": {
                        "test_attribute": true
                    },
                    "id": "buildin_extensions_2::TestExtension1",
                    "library": "test_builtin_extensions_2",
                    "match_kind": "scope"
                 }
                 )")},
        {"buildin_extensions_2::TestExtension2",
         nlohmann::json::parse(
             R"(
                 {
                    "custom_attributes": {
                        "test_attribute": true
                    },
                    "id": "buildin_extensions_2::TestExtension2",
                    "library": "test_builtin_extensions_2",
                    "match_kind": "scope"
                 }
                 )")},
    };

    for (auto& ref_val : reference_map) {
        ref_val.second["library"] = get_lib_path(ref_val.second["library"]);
    }

    std::shared_ptr<ov::Model> function;
    {
        ov::frontend::FrontEnd::Ptr m_frontEnd;
        ov::frontend::InputModel::Ptr m_inputModel;
        m_frontEnd = m_fem.load_by_framework(m_param.m_frontEndName);

        auto json_config_ext = std::make_shared<JsonConfigExtensionWrapper>(m_file_name);
        m_frontEnd->add_extension(json_config_ext);

        auto loaded_ext = json_config_ext->get_loaded_extensions();
        auto target_ext = json_config_ext->get_target_extensions();

        // the number of Loaded extensions can be more than the number of actually used ones.
        EXPECT_EQ(loaded_ext.size(), 10);
        EXPECT_EQ(target_ext.size(), 3);

        for (const auto& target : target_ext) {
            auto transformation = std::dynamic_pointer_cast<JsonTransformationExtension>(target.first);
            EXPECT_NE(transformation, nullptr);
            EXPECT_EQ(reference_map.at(transformation->id()).dump(), target.second);
        }
        ASSERT_NO_THROW(m_inputModel = m_frontEnd->load(m_param.m_modelName));
        ASSERT_NE(m_inputModel, nullptr);

        ASSERT_NO_THROW(function = m_frontEnd->convert(m_inputModel));
        ASSERT_NE(function, nullptr);
        EXPECT_EQ(function->get_friendly_name(), "TestFunction");
    }
}

TEST_P(FrontEndJsonConfigTest, compareFunctions) {
    EXPECT_NO_THROW(generate_json_config());

    std::shared_ptr<ov::Model> function;
    {
        ov::frontend::FrontEnd::Ptr m_frontEnd;
        ov::frontend::InputModel::Ptr m_inputModel;
        m_frontEnd = m_fem.load_by_framework(m_param.m_frontEndName);

        auto json_config_ext = std::make_shared<JsonConfigExtensionWrapper>(m_file_name);
        m_frontEnd->add_extension(json_config_ext);

        ASSERT_NO_THROW(m_inputModel = m_frontEnd->load(m_param.m_modelName));
        ASSERT_NE(m_inputModel, nullptr);

        ASSERT_NO_THROW(function = m_frontEnd->convert(m_inputModel));
        ASSERT_NE(function, nullptr);
    }

    std::shared_ptr<ov::Model> function_ref;
    {
        ov::frontend::FrontEnd::Ptr m_frontEnd;
        ov::frontend::InputModel::Ptr m_inputModel;
        m_frontEnd = m_fem.load_by_framework(m_param.m_frontEndName);

        ASSERT_NO_THROW(m_inputModel = m_frontEnd->load(m_param.m_modelName));
        ASSERT_NE(m_inputModel, nullptr);

        ASSERT_NO_THROW(function_ref = m_frontEnd->convert(m_inputModel));
        ASSERT_NE(function, nullptr);
    }
    compare_functions(function, function_ref);
}
