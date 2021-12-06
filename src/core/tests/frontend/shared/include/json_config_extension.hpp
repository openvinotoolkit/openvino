// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <common/extensions/json_config_extension.hpp>
#include <manager.hpp>

class JsonConfigExtensionWrapper : public ov::frontend::JsonConfigExtension {
public:
    explicit JsonConfigExtensionWrapper(const std::string& config_path)
        : ov::frontend::JsonConfigExtension(config_path){};
    ~JsonConfigExtensionWrapper() override{};

    std::vector<Extension::Ptr> get_loaded_extensions() {
        return m_loaded_extensions;
    };

    std::vector<std::pair<std::shared_ptr<DecoderTransformationExtension>, nlohmann::json>> get_target_extensions() {
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
    JsonConfigFEParam m_param;
    ov::frontend::FrontEndManager m_fem;

    static std::string getTestCaseName(const testing::TestParamInfo<JsonConfigFEParam>& obj);

    void SetUp() override;

protected:
    void initParamTest();
};
