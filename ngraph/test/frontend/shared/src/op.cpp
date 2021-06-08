// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/op.hpp"
#include <regex>
#include "../include/utils.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

using TestEngine = test::IE_CPU_Engine;

std::string FrontendOpTest::getTestCaseName(const testing::TestParamInfo<FrontendOpTestParam>& obj)
{
    std::string res = obj.param.m_frontEndName + "_" + obj.param.m_modelName;
    return FrontEndTestUtils::fileToTestName(res);
}

void FrontendOpTest::SetUp()
{
    FrontEndTestUtils::setupTestEnv();
    m_fem = FrontEndManager(); // re-initialize after setting up environment
    initParamTest();
}

void FrontendOpTest::initParamTest()
{
    m_param = GetParam();
    m_param.m_modelName = m_param.m_modelsPath + m_param.m_modelName;
}

void FrontendOpTest::validateOp()
{
    // load
    ASSERT_NO_THROW(m_fem.get_available_front_ends());
    ASSERT_NO_THROW(m_frontEnd = m_fem.load_by_framework(m_param.m_frontEndName));
    ASSERT_NE(m_frontEnd, nullptr);
    ASSERT_NO_THROW(m_inputModel = m_frontEnd->load_from_file(m_param.m_modelName));
    ASSERT_NE(m_inputModel, nullptr);

    // convert
    std::shared_ptr<ngraph::Function> function;
    function = m_frontEnd->convert(m_inputModel);
    ASSERT_NE(function, nullptr);

    // run
    auto test_case = test::TestCase<TestEngine>(function);

    for (auto it = m_param.inputs.begin(); it != m_param.inputs.end(); it++)
    {
        test_case.add_input(*it);
    }
    for (auto it = m_param.expected_outputs.begin(); it != m_param.expected_outputs.end(); it++)
    {
        test_case.add_expected_output(*it);
    }

    test_case.run();
}

/*---------------------------------------------------------------------------------------------------------------------*/

TEST_P(FrontendOpTest, test_model_runtime)
{
    ASSERT_NO_THROW(validateOp());
}
