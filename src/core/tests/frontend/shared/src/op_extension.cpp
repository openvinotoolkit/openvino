// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_extension.hpp"

#include <openvino/frontend/extension/decoder_transformation.hpp>
#include <openvino/frontend/extension/op.hpp>
#include <openvino/op/util/framework_node.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/runtime/core.hpp>

#include "utils.hpp"

using namespace ov::frontend;

std::string FrontEndOpExtensionTest::getTestCaseName(const testing::TestParamInfo<OpExtensionFEParam>& obj) {
    std::string res = obj.param.m_frontEndName + "_" + obj.param.m_modelName;
    return FrontEndTestUtils::fileToTestName(res);
}

void FrontEndOpExtensionTest::SetUp() {
    FrontEndTestUtils::setupTestEnv();
    initParamTest();
}

void FrontEndOpExtensionTest::initParamTest() {
    m_param = GetParam();
    m_param.m_modelName = FrontEndTestUtils::make_model_path(m_param.m_modelsPath + m_param.m_modelName);
}

///////////////////////////////////////////////////////////////////

TEST_P(FrontEndOpExtensionTest, TestOpExtension) {
    auto frontend = m_param.m_frontend;
    bool invoked = false;
    ov::Core core;
    core.add_extension(m_param.m_extension);
    auto model = core.read_model(m_param.m_modelName);
}
