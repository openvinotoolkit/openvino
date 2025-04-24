// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "openvino/frontend/manager.hpp"

struct FrontendLibraryExtensionTestParams {
    std::string m_frontEndName;
    std::string m_modelsPath;
    std::string m_modelName;
};

class FrontendLibraryExtensionTest : public ::testing::TestWithParam<FrontendLibraryExtensionTestParams> {
public:
    FrontendLibraryExtensionTestParams m_param;
    ov::frontend::FrontEndManager m_fem;

    static std::string getTestCaseName(const testing::TestParamInfo<FrontendLibraryExtensionTestParams>& obj);

    void SetUp() override;

protected:
    void initParamTest();
};
