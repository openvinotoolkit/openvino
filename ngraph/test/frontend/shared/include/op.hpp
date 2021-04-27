// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/frontend_manager.hpp>

#include <gtest/gtest.h>
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/engine/test_engines.hpp"
#include "util/ndarray.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using Inputs = std::vector<std::vector<float>>;
using Outputs = std::vector<std::vector<float>>;

struct FrontendOpTestParam {
    std::string              m_frontEndName;
    std::string              m_modelsPath;
    std::string              m_modelName;

    Inputs inputs;
    Outputs expected_outputs;
};

class FrontendOpTest : public ::testing::TestWithParam<FrontendOpTestParam> {
public:
    FrontendOpTestParam      m_param;

    ngraph::frontend::FrontEndManager  m_fem;
    ngraph::frontend::FrontEnd::Ptr    m_frontEnd;
    ngraph::frontend::InputModel::Ptr  m_inputModel;
    
    static std::string getTestCaseName(const testing::TestParamInfo<FrontendOpTestParam> &obj);

    void SetUp() override;
    
protected:
    void initParamTest();
    void validateOp();    
};