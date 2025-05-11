// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_extension.hpp"

#include "openvino/runtime/core.hpp"
#include "utils.hpp"

using namespace ov::frontend;

std::string FrontEndOpExtensionTest::getTestCaseName(const testing::TestParamInfo<OpExtensionFEParam>& obj) {
    std::string res = obj.param.m_frontEndName + "_" + obj.param.m_modelName;
    return FrontEndTestUtils::fileToTestName(res);
}

void FrontEndOpExtensionTest::SetUp() {
    initParamTest();
}

void FrontEndOpExtensionTest::initParamTest() {
    m_param = GetParam();
    m_param.m_modelName = FrontEndTestUtils::make_model_path(m_param.m_modelsPath + m_param.m_modelName);
}

///////////////////////////////////////////////////////////////////

TEST_P(FrontEndOpExtensionTest, TestOpExtensionVec) {
    ov::Core core;
    core.add_extension(m_param.m_extensions);
    auto model = core.read_model(m_param.m_modelName);
}

TEST_P(FrontEndOpExtensionTest, TestOpExtension) {
    ov::Core core;
    for (size_t i = 0; i < m_param.m_extensions.size(); ++i)
        core.add_extension(m_param.m_extensions[i]);
    auto model = core.read_model(m_param.m_modelName);
}
