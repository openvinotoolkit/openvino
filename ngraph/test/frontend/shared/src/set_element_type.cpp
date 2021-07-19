// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "set_element_type.hpp"
#include "utils.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

std::string
    FrontEndElementTypeTest::getTestCaseName(const testing::TestParamInfo<SetTypeFEParam>& obj)
{
    std::string res = obj.param.m_frontEndName + "_" + obj.param.m_modelName;
    return FrontEndTestUtils::fileToTestName(res);
}

void FrontEndElementTypeTest::SetUp()
{
    FrontEndTestUtils::setupTestEnv();
    m_fem = FrontEndManager(); // re-initialize after setting up environment
    initParamTest();
}

void FrontEndElementTypeTest::initParamTest()
{
    m_param = GetParam();
    m_param.m_modelName = m_param.m_modelsPath + m_param.m_modelName;
}

void FrontEndElementTypeTest::doLoadFromFile()
{
    std::vector<std::string> frontends;
    FrontEnd::Ptr fe;
    ASSERT_NO_THROW(frontends = m_fem.get_available_front_ends());
    ASSERT_NO_THROW(m_frontEnd = m_fem.load_by_framework(m_param.m_frontEndName));
    ASSERT_NE(m_frontEnd, nullptr);
    ASSERT_NO_THROW(m_inputModel = m_frontEnd->load(m_param.m_modelName));
    ASSERT_NE(m_inputModel, nullptr);
}

///////////////////////////////////////////////////////////////////

TEST_P(FrontEndElementTypeTest, testSetElementType)
{
    ASSERT_NO_THROW(doLoadFromFile());
    Place::Ptr place;
    ASSERT_NO_THROW(place = m_inputModel->get_inputs()[0]);
    ASSERT_NE(place, nullptr);
    auto name = place->get_names()[0];

    ASSERT_NO_THROW(m_inputModel->set_element_type(place, element::f16));

    std::shared_ptr<ngraph::Function> function;
    function = m_frontEnd->convert(m_inputModel);
    auto ops = function->get_ordered_ops();
    auto it = std::find_if(ops.begin(), ops.end(), [&](const std::shared_ptr<ngraph::Node>& node) {
        return node->get_friendly_name().find(name) != std::string::npos;
    });
    ASSERT_NE(it, ops.end());
    EXPECT_EQ((*it)->get_output_element_type(0), element::f16);
}
