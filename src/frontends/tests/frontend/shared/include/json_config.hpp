// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <extension/json_config.hpp>
#include <openvino/frontend/manager.hpp>

class JsonConfigExtensionWrapper : public ov::frontend::JsonConfigExtension {
public:
    explicit JsonConfigExtensionWrapper(const std::string& config_path)
        : ov::frontend::JsonConfigExtension(config_path){};
    ~JsonConfigExtensionWrapper() override = default;

    std::vector<Extension::Ptr> get_loaded_extensions() {
        return m_loaded_extensions;
    };

    std::vector<std::pair<DecoderTransformationExtension::Ptr, std::string>> get_target_extensions() {
        return m_target_extensions;
    }
};

struct JsonConfigFEParam {
    std::string m_frontEndName;
    std::string m_modelsPath;
    std::string m_modelName;
};

class FrontEndJsonConfigTest : public ::testing::TestWithParam<JsonConfigFEParam> {
public:
    std::string m_file_name = "test_json.json";
    std::ofstream m_output_json;
    JsonConfigFEParam m_param;
    ov::frontend::FrontEndManager m_fem;

    static std::string getTestCaseName(const testing::TestParamInfo<JsonConfigFEParam>& obj);

    void SetUp() override;
    void TearDown() override;

protected:
    void generate_json_config();
    void initParamTest();
};
