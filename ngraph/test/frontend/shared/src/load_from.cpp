// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "load_from.hpp"
#include <fstream>
#include "utils.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

std::string
    FrontEndLoadFromTest::getTestCaseName(const testing::TestParamInfo<LoadFromFEParam>& obj)
{
    std::string res = obj.param.m_frontEndName;
    return FrontEndTestUtils::fileToTestName(res);
}

void FrontEndLoadFromTest::SetUp()
{
    FrontEndTestUtils::setupTestEnv();
    m_fem = FrontEndManager(); // re-initialize after setting up environment
    m_param = GetParam();
}

///////////////////load from Variants//////////////////////

TEST_P(FrontEndLoadFromTest, testLoadFromFilePath)
{
    std::string model_path = m_param.m_modelsPath + m_param.m_file;
    std::vector<std::string> frontends;
    FrontEnd::Ptr fe;
    ASSERT_NO_THROW(frontends = m_fem.get_available_front_ends());
    ASSERT_NO_THROW(m_frontEnd = m_fem.load_by_model(model_path));
    ASSERT_NE(m_frontEnd, nullptr);

    ASSERT_NO_THROW(m_inputModel = m_frontEnd->load(model_path));
    ASSERT_NE(m_inputModel, nullptr);

    std::shared_ptr<ngraph::Function> function;
    ASSERT_NO_THROW(function = m_frontEnd->convert(m_inputModel));
    ASSERT_NE(function, nullptr);
}

TEST_P(FrontEndLoadFromTest, testLoadFromTwoFiles)
{
    std::string model_path = m_param.m_modelsPath + m_param.m_files[0];
    std::string weights_path = m_param.m_modelsPath + m_param.m_files[1];
    std::vector<std::string> frontends;
    FrontEnd::Ptr fe;
    ASSERT_NO_THROW(frontends = m_fem.get_available_front_ends());
    ASSERT_NO_THROW(m_frontEnd = m_fem.load_by_model(model_path, weights_path));
    ASSERT_NE(m_frontEnd, nullptr);

    ASSERT_NO_THROW(m_inputModel = m_frontEnd->load(model_path, weights_path));
    ASSERT_NE(m_inputModel, nullptr);

    std::shared_ptr<ngraph::Function> function;
    function = m_frontEnd->convert(m_inputModel);
    ASSERT_NE(function, nullptr);
}

TEST_P(FrontEndLoadFromTest, testLoadFromStream)
{
    auto ifs = std::make_shared<std::ifstream>(m_param.m_modelsPath + m_param.m_stream,
                                               std::ios::in | std::ifstream::binary);
    auto is = std::dynamic_pointer_cast<std::istream>(ifs);
    std::vector<std::string> frontends;
    FrontEnd::Ptr fe;
    ASSERT_NO_THROW(frontends = m_fem.get_available_front_ends());
    ASSERT_NO_THROW(m_frontEnd = m_fem.load_by_model(is));
    ASSERT_NE(m_frontEnd, nullptr);

    ASSERT_NO_THROW(m_inputModel = m_frontEnd->load(is));
    ASSERT_NE(m_inputModel, nullptr);

    std::shared_ptr<ngraph::Function> function;
    ASSERT_NO_THROW(function = m_frontEnd->convert(m_inputModel));
    ASSERT_NE(function, nullptr);
}

TEST_P(FrontEndLoadFromTest, testLoadFromTwoStreams)
{
    auto model_ifs = std::make_shared<std::ifstream>(m_param.m_modelsPath + m_param.m_streams[0],
                                                     std::ios::in | std::ifstream::binary);
    auto weights_ifs = std::make_shared<std::ifstream>(m_param.m_modelsPath + m_param.m_streams[1],
                                                       std::ios::in | std::ifstream::binary);
    auto model_is = std::dynamic_pointer_cast<std::istream>(model_ifs);
    auto weights_is = std::dynamic_pointer_cast<std::istream>(weights_ifs);

    std::vector<std::string> frontends;
    FrontEnd::Ptr fe;
    ASSERT_NO_THROW(frontends = m_fem.get_available_front_ends());
    ASSERT_NO_THROW(m_frontEnd = m_fem.load_by_model(model_is, weights_is));
    ASSERT_NE(m_frontEnd, nullptr);

    ASSERT_NO_THROW(m_inputModel = m_frontEnd->load(model_is, weights_is));
    ASSERT_NE(m_inputModel, nullptr);

    std::shared_ptr<ngraph::Function> function;
    ASSERT_NO_THROW(function = m_frontEnd->convert(m_inputModel));
    ASSERT_NE(function, nullptr);
}
