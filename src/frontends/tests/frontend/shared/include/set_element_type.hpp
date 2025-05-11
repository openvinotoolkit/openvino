// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "openvino/frontend/manager.hpp"

struct SetTypeFEParam {
    std::string m_frontEndName;
    std::string m_modelsPath;
    std::string m_modelName;
};

class FrontEndElementTypeTest : public ::testing::TestWithParam<SetTypeFEParam> {
public:
    SetTypeFEParam m_param;
    ov::frontend::FrontEndManager m_fem;
    ov::frontend::FrontEnd::Ptr m_frontEnd;
    ov::frontend::InputModel::Ptr m_inputModel;

    static std::string getTestCaseName(const testing::TestParamInfo<SetTypeFEParam>& obj);

    void SetUp() override;

protected:
    void initParamTest();
    void doLoadFromFile();
};
