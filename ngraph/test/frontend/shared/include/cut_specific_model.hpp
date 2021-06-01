// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/frontend_manager.hpp>

#include <gtest/gtest.h>

struct CutModelParam
{
    std::string m_frontEndName;
    std::string m_modelsPath;
    std::string m_modelName;
    std::vector<std::string> m_oldInputs;
    std::vector<std::string> m_newInputs;
    std::vector<std::string> m_oldOutputs;
    std::vector<std::string> m_newOutputs;
    std::string m_tensorValueName;
    std::vector<float> m_tensorValue;
    std::string m_op_before_name;
};

class FrontEndCutModelTest : public ::testing::TestWithParam<CutModelParam>
{
public:
    CutModelParam m_param;
    ngraph::frontend::FrontEndManager m_fem;
    ngraph::frontend::FrontEnd::Ptr m_frontEnd;
    ngraph::frontend::InputModel::Ptr m_inputModel;

    static std::string getTestCaseName(const testing::TestParamInfo<CutModelParam>& obj);

    void SetUp() override;

protected:
    void initParamTest();

    void doLoadFromFile();

    std::vector<ngraph::frontend::Place::Ptr> constructNewInputs() const;

    std::vector<ngraph::frontend::Place::Ptr> constructNewOutputs() const;
};
