// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/load_from.hpp"
#include <fstream>
#include "../include/utils.hpp"

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

///////////////////////////////////////////////////////////////////

TEST_P(FrontEndLoadFromTest, testLoadFromFile)
{
    std::vector<std::string> frontends;
    FrontEnd::Ptr fe;
    ASSERT_NO_THROW(frontends = m_fem.get_available_front_ends());
    ASSERT_NO_THROW(m_frontEnd = m_fem.load_by_framework(m_param.m_frontEndName));
    ASSERT_NE(m_frontEnd, nullptr);

    ASSERT_NO_THROW(m_inputModel =
                        m_frontEnd->load_from_file(m_param.m_modelsPath + m_param.m_file));
    ASSERT_NE(m_inputModel, nullptr);

    std::shared_ptr<ngraph::Function> function;
    ASSERT_NO_THROW(function = m_frontEnd->convert(m_inputModel));
    ASSERT_NE(function, nullptr);
}

TEST_P(FrontEndLoadFromTest, testLoadFromFiles)
{
    std::vector<std::string> frontends;
    FrontEnd::Ptr fe;
    ASSERT_NO_THROW(frontends = m_fem.get_available_front_ends());
    ASSERT_NO_THROW(m_frontEnd = m_fem.load_by_framework(m_param.m_frontEndName));
    ASSERT_NE(m_frontEnd, nullptr);

    auto dir_files = m_param.m_files;
    for (auto& file : dir_files)
    {
        file = m_param.m_modelsPath + file;
    }

    ASSERT_NO_THROW(m_inputModel = m_frontEnd->load_from_files(dir_files));
    ASSERT_NE(m_inputModel, nullptr);

    std::shared_ptr<ngraph::Function> function;
    function = m_frontEnd->convert(m_inputModel);
    ASSERT_NE(function, nullptr);
}

TEST_P(FrontEndLoadFromTest, testLoadFromStream)
{
    std::vector<std::string> frontends;
    FrontEnd::Ptr fe;
    ASSERT_NO_THROW(frontends = m_fem.get_available_front_ends());
    ASSERT_NO_THROW(m_frontEnd = m_fem.load_by_framework(m_param.m_frontEndName));
    ASSERT_NE(m_frontEnd, nullptr);

    std::ifstream is(m_param.m_modelsPath + m_param.m_stream, std::ios::in | std::ifstream::binary);
    ASSERT_NO_THROW(m_inputModel = m_frontEnd->load_from_stream(is));
    ASSERT_NE(m_inputModel, nullptr);

    std::shared_ptr<ngraph::Function> function;
    ASSERT_NO_THROW(function = m_frontEnd->convert(m_inputModel));
    ASSERT_NE(function, nullptr);
}

TEST_P(FrontEndLoadFromTest, testLoadFromStreams)
{
    std::vector<std::string> frontends;
    FrontEnd::Ptr fe;
    ASSERT_NO_THROW(frontends = m_fem.get_available_front_ends());
    ASSERT_NO_THROW(m_frontEnd = m_fem.load_by_framework(m_param.m_frontEndName));
    ASSERT_NE(m_frontEnd, nullptr);

    std::vector<std::shared_ptr<std::ifstream>> is_vec;
    std::vector<std::istream*> is_ptr_vec;
    for (auto& file : m_param.m_streams)
    {
        is_vec.push_back(std::make_shared<std::ifstream>(m_param.m_modelsPath + file,
                                                         std::ios::in | std::ifstream::binary));
        is_ptr_vec.push_back(is_vec.back().get());
    }
    ASSERT_NO_THROW(m_inputModel = m_frontEnd->load_from_streams(is_ptr_vec));
    ASSERT_NE(m_inputModel, nullptr);

    std::shared_ptr<ngraph::Function> function;
    ASSERT_NO_THROW(function = m_frontEnd->convert(m_inputModel));
    ASSERT_NE(function, nullptr);
}
