// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "set_element_type.hpp"

#include "utils.hpp"

using namespace ov::frontend;

std::string FrontEndElementTypeTest::getTestCaseName(const testing::TestParamInfo<SetTypeFEParam>& obj) {
    std::string res = obj.param.m_frontEndName + "_" + obj.param.m_modelName;
    return FrontEndTestUtils::fileToTestName(res);
}

void FrontEndElementTypeTest::SetUp() {
    m_fem = FrontEndManager();  // re-initialize after setting up environment
    initParamTest();
}

void FrontEndElementTypeTest::initParamTest() {
    m_param = GetParam();
    m_param.m_modelName = FrontEndTestUtils::make_model_path(m_param.m_modelsPath + m_param.m_modelName);
}

void FrontEndElementTypeTest::doLoadFromFile() {
    std::tie(m_frontEnd, m_inputModel) =
        FrontEndTestUtils::load_from_file(m_fem, m_param.m_frontEndName, m_param.m_modelName);
}

///////////////////////////////////////////////////////////////////

TEST_P(FrontEndElementTypeTest, testSetElementType) {
    OV_ASSERT_NO_THROW(doLoadFromFile());
    Place::Ptr place;
    OV_ASSERT_NO_THROW(place = m_inputModel->get_inputs()[0]);
    ASSERT_NE(place, nullptr);
    auto name = place->get_names()[0];

    OV_ASSERT_NO_THROW(m_inputModel->set_element_type(place, ov::element::f16));

    std::shared_ptr<ov::Model> model;
    model = m_frontEnd->convert(m_inputModel);
    auto ops = model->get_ordered_ops();
    auto it = std::find_if(ops.begin(), ops.end(), [&](const std::shared_ptr<ov::Node>& node) {
        return node->get_friendly_name().find(name) != std::string::npos;
    });
    ASSERT_NE(it, ops.end());
    EXPECT_EQ((*it)->get_output_element_type(0), ov::element::f16);
}
