// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <regex>
#include "../include/op.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

using TestEngine = test::IE_CPU_Engine;

std::string FrontendOpTest::getTestCaseName(const testing::TestParamInfo<FrontendOpTestParam> &obj) {
    std::string res = obj.param.m_frontEndName + "_" + obj.param.m_modelName;
    //res += "I" + joinStrings(obj.param.m_oldInputs) + joinStrings(obj.param.m_newInputs);
    //res += "O" + joinStrings(obj.param.m_oldOutputs) + joinStrings(obj.param.m_newOutputs);
    // need to replace special characters to create valid test case name
    res = std::regex_replace(res, std::regex("[/\\.]"), "_");
    return res;
}

void FrontendOpTest::SetUp() {
    initParamTest();
}

void FrontendOpTest::initParamTest() {
    m_param = GetParam();
    m_param.m_modelName = std::string(TEST_FILES) + m_param.m_modelsPath + m_param.m_modelName;
    std::cout << "Model: " << m_param.m_modelName << std::endl;
}

void FrontendOpTest::validateOp() {
    // load
    ASSERT_NO_THROW(m_fem.availableFrontEnds());
    ASSERT_NO_THROW(m_frontEnd = m_fem.loadByFramework(m_param.m_frontEndName));
    ASSERT_NE(m_frontEnd, nullptr);
    ASSERT_NO_THROW(m_inputModel = m_frontEnd->loadFromFile(m_param.m_modelName));
    ASSERT_NE(m_inputModel, nullptr);

    // convert
    std::shared_ptr<ngraph::Function> function;
    function = m_frontEnd->convert(m_inputModel);
    ASSERT_NE(function, nullptr);

    // run
    auto test_case = test::TestCase<TestEngine>(function);

    for (auto it = m_param.inputs.begin(); it != m_param.inputs.end(); it++ ) {
        test_case.add_input(*it);        
    }   
    for (auto it = m_param.expected_outputs.begin(); it != m_param.expected_outputs.end(); it++)
    {
        test_case.add_expected_output(*it);
    }
        
    test_case.run();    
}

/*---------------------------------------------------------------------------------------------------------------------*/

TEST_P(FrontendOpTest, test_model_runtime) {
    ASSERT_NO_THROW(validateOp());
}
