// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/manager.hpp"

struct ConversionExtensionFEParam {
    std::string m_frontEndName;
    std::string m_modelsPath;
    std::string m_modelName;
    std::string m_translatorName;
    std::shared_ptr<ov::frontend::FrontEnd> m_frontend;
};

class FrontEndConversionExtensionTest : public ::testing::TestWithParam<ConversionExtensionFEParam> {
public:
    ConversionExtensionFEParam m_param;
    ov::frontend::FrontEndManager m_fem;

    static std::string getTestCaseName(const testing::TestParamInfo<ConversionExtensionFEParam>& obj);

    void SetUp() override;

protected:
    void initParamTest();
};
