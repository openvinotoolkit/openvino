// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "openvino/frontend/manager.hpp"

using FuzzyOpTestParam = std::tuple<std::string,   // FrontEnd name
                                    std::string,   // Base path to models
                                    std::string>;  // Model name

class FrontEndFuzzyOpTest : public ::testing::TestWithParam<FuzzyOpTestParam> {
public:
    std::string m_feName;
    std::string m_pathToModels;
    std::string m_modelFile;
    ov::frontend::FrontEndManager m_fem;
    ov::frontend::FrontEnd::Ptr m_frontEnd;
    ov::frontend::InputModel::Ptr m_inputModel;

    static std::string getTestCaseName(const testing::TestParamInfo<FuzzyOpTestParam>& obj);

    void SetUp() override;

protected:
    void initParamTest();

    void doLoadFromFile();

    void runConvertedModel(const std::shared_ptr<ov::Model> function, const std::string& model_file);
};
