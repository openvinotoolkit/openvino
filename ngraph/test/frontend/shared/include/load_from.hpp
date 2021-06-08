// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/frontend_manager.hpp>

#include <gtest/gtest.h>

struct LoadFromFEParam
{
    std::string m_frontEndName;
    std::string m_modelsPath;
    std::string m_file;
    std::vector<std::string> m_files;
    std::string m_stream;
    std::vector<std::string> m_streams;
};

class FrontEndLoadFromTest : public ::testing::TestWithParam<LoadFromFEParam>
{
public:
    LoadFromFEParam m_param;
    ngraph::frontend::FrontEndManager m_fem;
    ngraph::frontend::FrontEnd::Ptr m_frontEnd;
    ngraph::frontend::InputModel::Ptr m_inputModel;

    static std::string getTestCaseName(const testing::TestParamInfo<LoadFromFEParam>& obj);

    void SetUp() override;

protected:
    void initParamTest();
};
