// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "openvino/frontend/manager.hpp"

struct LoadFromFEParam {
    std::string m_frontEndName;
    std::string m_modelsPath;
    std::string m_file;
    std::vector<std::string> m_files;
    std::string m_stream;
    std::vector<std::string> m_streams;
};

class FrontEndLoadFromTest : public ::testing::TestWithParam<LoadFromFEParam> {
public:
    LoadFromFEParam m_param;
    ov::frontend::FrontEndManager m_fem;
    ov::frontend::FrontEnd::Ptr m_frontEnd;
    ov::frontend::InputModel::Ptr m_inputModel;

    static std::string getTestCaseName(const testing::TestParamInfo<LoadFromFEParam>& obj);

    void SetUp() override;

protected:
    void initParamTest();
};
