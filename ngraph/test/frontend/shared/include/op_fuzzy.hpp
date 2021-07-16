// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/frontend_manager.hpp>

#include <gtest/gtest.h>

using FuzzyOpTestParam = std::tuple<std::string,  // FrontEnd name
                                    std::string,  // Base path to models
                                    std::string>; // Model name

class FrontEndFuzzyOpTest : public ::testing::TestWithParam<FuzzyOpTestParam>
{
public:
    std::string m_feName;
    std::string m_pathToModels;
    std::string m_modelFile;
    ngraph::frontend::FrontEndManager m_fem;
    ngraph::frontend::FrontEnd::Ptr m_frontEnd;
    ngraph::frontend::InputModel::Ptr m_inputModel;

    static std::string getTestCaseName(const testing::TestParamInfo<FuzzyOpTestParam>& obj);

    void SetUp() override;

protected:
    void initParamTest();

    void doLoadFromFile();

    void runConvertedModel(const std::shared_ptr<ngraph::Function> function,
                           const std::string& model_file);
};
