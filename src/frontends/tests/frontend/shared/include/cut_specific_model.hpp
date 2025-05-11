// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "openvino/frontend/manager.hpp"

struct CutModelParam {
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

class FrontEndCutModelTest : public ::testing::TestWithParam<CutModelParam> {
public:
    CutModelParam m_param;
    ov::frontend::FrontEndManager m_fem;
    ov::frontend::FrontEnd::Ptr m_frontEnd;
    ov::frontend::InputModel::Ptr m_inputModel;

    static std::string getTestCaseName(const testing::TestParamInfo<CutModelParam>& obj);

    void SetUp() override;

protected:
    void initParamTest();

    void doLoadFromFile();

    std::vector<ov::frontend::Place::Ptr> constructNewInputs() const;

    std::vector<ov::frontend::Place::Ptr> constructNewOutputs() const;
};
