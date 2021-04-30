// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <regex>
#include <algorithm>
#include "../include/set_element_type.hpp"
#include "../include/utils.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

std::string FrontEndElementTypeTest::getTestCaseName(const testing::TestParamInfo<SetTypeFEParam>& obj) {
    std::string res = obj.param.m_frontEndName + "_" + obj.param.m_modelName;
    return FrontEndTestUtils::fileToTestName(res);
}

void FrontEndElementTypeTest::SetUp() {
    initParamTest();
}

void FrontEndElementTypeTest::initParamTest() {
    m_param = GetParam();
    m_param.m_modelName = std::string(TEST_FILES) + m_param.m_modelsPath + m_param.m_modelName;
    std::cout << "Model: " << m_param.m_modelName << std::endl;
}

void FrontEndElementTypeTest::doLoadFromFile() {
    std::vector<std::string> frontends;
    FrontEnd::Ptr fe;
    ASSERT_NO_THROW(frontends = m_fem.availableFrontEnds());
    ASSERT_NO_THROW(m_frontEnd = m_fem.loadByFramework(m_param.m_frontEndName));
    ASSERT_NE(m_frontEnd, nullptr);
    ASSERT_NO_THROW(m_inputModel = m_frontEnd->loadFromFile(m_param.m_modelName));
    ASSERT_NE(m_inputModel, nullptr);
}

///////////////////////////////////////////////////////////////////

TEST_P(FrontEndElementTypeTest, testSetElementType)
{
    ASSERT_NO_THROW(doLoadFromFile());
    Place::Ptr place;
    ASSERT_NO_THROW(place = m_inputModel->getInputs()[0]);
    ASSERT_NE(place, nullptr);
    auto name = place->getNames()[0];

    ASSERT_NO_THROW(m_inputModel->setElementType(place, element::f16));

    std::shared_ptr<ngraph::Function> function;
    function = m_frontEnd->convert(m_inputModel);
    auto ops = function->get_ordered_ops();
    auto it = std::find_if(ops.begin(), ops.end(),
                 [&](const std::shared_ptr<ngraph::Node>& node) {
                     return node->get_friendly_name().find(name) != std::string::npos;
                 });
    ASSERT_NE(it, ops.end());
    EXPECT_EQ((*it)->get_output_element_type(0), element::f16);
}
