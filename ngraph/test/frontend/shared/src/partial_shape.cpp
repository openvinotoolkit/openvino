// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/partial_shape.hpp"
#include "../include/utils.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

std::string
    FrontEndPartialShapeTest::getTestCaseName(const testing::TestParamInfo<PartialShapeParam>& obj)
{
    BaseFEParam base;
    PartShape part;
    std::tie(base, part) = obj.param;
    std::string res = base.m_frontEndName + "_" + part.m_modelName + "_" + part.m_tensorName;
    for (auto s : part.m_newPartialShape)
    {
        res += "_" + std::to_string(s);
    }
    return FrontEndTestUtils::fileToTestName(res);
}

void FrontEndPartialShapeTest::SetUp()
{
    initParamTest();
}

void FrontEndPartialShapeTest::initParamTest()
{
    std::tie(m_baseParam, m_partShape) = GetParam();
    m_partShape.m_modelName =
        std::string(TEST_FILES) + m_baseParam.m_modelsPath + m_partShape.m_modelName;
    std::cout << "Model: " << m_partShape.m_modelName << std::endl;
}

void FrontEndPartialShapeTest::doLoadFromFile()
{
    std::vector<std::string> frontends;
    FrontEnd::Ptr fe;
    ASSERT_NO_THROW(frontends = m_fem.get_available_front_ends());
    ASSERT_NO_THROW(m_frontEnd = m_fem.load_by_framework(m_baseParam.m_frontEndName));
    ASSERT_NE(m_frontEnd, nullptr);
    ASSERT_NO_THROW(m_inputModel = m_frontEnd->load_from_file(m_partShape.m_modelName));
    ASSERT_NE(m_inputModel, nullptr);
}

///////////////////////////////////////////////////////////////////

TEST_P(FrontEndPartialShapeTest, testCheckOldPartialShape)
{
    ASSERT_NO_THROW(doLoadFromFile());

    std::shared_ptr<ngraph::Function> function;
    ASSERT_NO_THROW(function = m_frontEnd->convert(m_inputModel));
    auto ops = function->get_ordered_ops();
    auto it = std::find_if(ops.begin(), ops.end(), [&](const std::shared_ptr<ngraph::Node>& node) {
        return node->get_friendly_name().find(m_partShape.m_tensorName) != std::string::npos;
    });
    ASSERT_NE(it, ops.end());
    auto shape = (*it)->get_output_partial_shape(0).get_shape();
    ASSERT_EQ(shape.size(), m_partShape.m_oldPartialShape.size());
    for (std::size_t i = 0; i < shape.size(); i++)
    {
        EXPECT_EQ(shape.at(i), m_partShape.m_oldPartialShape.at(i));
    }
}

TEST_P(FrontEndPartialShapeTest, testSetNewPartialShape)
{
    ASSERT_NO_THROW(doLoadFromFile());
    Place::Ptr place;
    ASSERT_NO_THROW(place = m_inputModel->get_place_by_tensor_name(m_partShape.m_tensorName));
    ASSERT_NE(place, nullptr);
    ASSERT_NO_THROW(
        m_inputModel->set_partial_shape(place, PartialShape{m_partShape.m_newPartialShape}));

    std::shared_ptr<ngraph::Function> function;
    ASSERT_NO_THROW(function = m_frontEnd->convert(m_inputModel));
    auto ops = function->get_ordered_ops();
    auto it = std::find_if(ops.begin(), ops.end(), [&](const std::shared_ptr<ngraph::Node>& node) {
        return node->get_friendly_name().find(m_partShape.m_tensorName) != std::string::npos;
    });
    ASSERT_NE(it, ops.end());
    auto shape = (*it)->get_output_partial_shape(0).get_shape();
    ASSERT_EQ(shape.size(), m_partShape.m_newPartialShape.size());
    for (std::size_t i = 0; i < shape.size(); i++)
    {
        EXPECT_EQ(shape.at(i), m_partShape.m_newPartialShape.at(i));
    }
}
