// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <regex>
#include <algorithm>
#include "../include/cut_specific_model.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

static std::string joinStrings(const std::vector<std::string>& strings) {
    std::ostringstream res;
    std::copy(strings.begin(), strings.end(),
              std::ostream_iterator<std::string>(res, "_"));
    return res.str();
}

std::string FrontEndCutModelTest::getTestCaseName(const testing::TestParamInfo<CutModelParam> &obj) {
    std::string res = obj.param.m_frontEndName + "_" + obj.param.m_modelName;
    res += "I" + joinStrings(obj.param.m_oldInputs) + joinStrings(obj.param.m_newInputs);
    res += "O" + joinStrings(obj.param.m_oldOutputs) + joinStrings(obj.param.m_newOutputs);
    // need to replace special characters to create valid test case name
    res = std::regex_replace(res, std::regex("[/\\.]"), "_");
    return res;
}

void FrontEndCutModelTest::SetUp() {
    initParamTest();
}

void FrontEndCutModelTest::initParamTest() {
    m_param = GetParam();
    m_param.m_modelName = std::string(TEST_FILES) + m_param.m_modelsPath + m_param.m_modelName;
    std::cout << "Model: " << m_param.m_modelName << std::endl;
}

void FrontEndCutModelTest::doLoadFromFile() {
    std::vector<std::string> frontends;
    FrontEnd::Ptr fe;
    ASSERT_NO_THROW(frontends = m_fem.availableFrontEnds());
    ASSERT_NO_THROW(m_frontEnd = m_fem.loadByFramework(m_param.m_frontEndName));
    ASSERT_NE(m_frontEnd, nullptr);
    ASSERT_NO_THROW(m_inputModel = m_frontEnd->loadFromFile(m_param.m_modelName));
    ASSERT_NE(m_inputModel, nullptr);
}

std::vector<ngraph::frontend::Place::Ptr> FrontEndCutModelTest::constructNewInputs() const {
    std::vector<Place::Ptr> newInputs;
    for (const auto& name : m_param.m_newInputs) {
        newInputs.push_back(m_inputModel->getPlaceByTensorName(name));
    }
    return newInputs;
}

std::vector<ngraph::frontend::Place::Ptr> FrontEndCutModelTest::constructNewOutputs() const {
    std::vector<Place::Ptr> newOutputs;
    for (const auto& name : m_param.m_newOutputs) {
        newOutputs.push_back(m_inputModel->getPlaceByTensorName(name));
    }
    return newOutputs;
}

///////////////////////////////////////////////////////////////////

TEST_P(FrontEndCutModelTest, testOverrideInputs)
{
    ASSERT_NO_THROW(doLoadFromFile());
    std::vector<Place::Ptr> newPlaces;
    ASSERT_NO_THROW(newPlaces = constructNewInputs());
    ASSERT_NO_THROW(m_inputModel->overrideAllInputs(newPlaces));
    ASSERT_NO_THROW(m_inputModel->getInputs());
    EXPECT_EQ(m_param.m_newInputs.size(), m_inputModel->getInputs().size());
    for (auto newInput : m_inputModel->getInputs()) {
        auto names = newInput->getNames();
        bool found = false;
        for (const auto& name: m_param.m_newInputs) {
            if (std::find(names.begin(), names.begin(), name) != names.end()) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << joinStrings(names) << " were not found in new inputs";
    }
}

TEST_P(FrontEndCutModelTest, testOverrideOutputs)
{
    ASSERT_NO_THROW(doLoadFromFile());
    std::vector<Place::Ptr> newPlaces;
    ASSERT_NO_THROW(newPlaces = constructNewOutputs());
    ASSERT_NO_THROW(m_inputModel->overrideAllOutputs(newPlaces));
    ASSERT_NO_THROW(m_inputModel->getOutputs());
    EXPECT_EQ(m_param.m_newOutputs.size(), m_inputModel->getOutputs().size());
    for (auto newOutput : m_inputModel->getOutputs()) {
        auto names = newOutput->getNames();
        bool found = false;
        for (const auto& name: m_param.m_newOutputs) {
            if (std::find(names.begin(), names.begin(), name) != names.end()) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << joinStrings(names) << " were not found in new outputs";
    }
}

TEST_P(FrontEndCutModelTest, testOldInputs) {
    ASSERT_NO_THROW(doLoadFromFile());
    std::shared_ptr<ngraph::Function> function;
    ASSERT_NO_THROW(function = m_frontEnd->convert(m_inputModel));
    auto ops = function->get_ordered_ops();

    // Ensure that it contains expected old inputs
    for (const auto& name : m_param.m_oldInputs) {
        EXPECT_TRUE(std::find_if(ops.begin(), ops.end(),
                                 [&](const std::shared_ptr<ngraph::Node>& node) {
                                     return node->get_friendly_name().find(name) != std::string::npos;
                                 }) != ops.end()) << "Name not found:" << name;
    }
}

TEST_P(FrontEndCutModelTest, testOldOutputs) {
    ASSERT_NO_THROW(doLoadFromFile());
    std::shared_ptr<ngraph::Function> function;
    ASSERT_NO_THROW(function = m_frontEnd->convert(m_inputModel));
    auto ops = function->get_ordered_ops();
    // Ensure that it contains expected old outputs
    for (const auto& name : m_param.m_oldOutputs) {
        EXPECT_TRUE(std::find_if(ops.begin(), ops.end(),
                                 [&](const std::shared_ptr<ngraph::Node>& node) {
                                     return node->get_friendly_name().find(name) != std::string::npos;
                                 }) != ops.end()) << "Name not found:" << name;
    }
}

TEST_P(FrontEndCutModelTest, testNewInputs_func) {
    ASSERT_NO_THROW(doLoadFromFile());
    std::vector<Place::Ptr> newPlaces;
    ASSERT_NO_THROW(newPlaces = constructNewInputs());
    ASSERT_NO_THROW(m_inputModel->overrideAllInputs(newPlaces));

    std::shared_ptr<ngraph::Function> function;
    ASSERT_NO_THROW(function = m_frontEnd->convert(m_inputModel));
    auto ops = function->get_ordered_ops();

    // Ensure that it doesn't contain old inputs
    for (const auto &name : m_param.m_oldInputs) {
        EXPECT_TRUE(std::find_if(ops.begin(), ops.end(),
                                 [&](const std::shared_ptr<ngraph::Node>& node) {
                                     return node->get_friendly_name().find(name) != std::string::npos;
                                 }) == ops.end()) << "Name shall not exist:" << name;
    }

    // Ensure that it contains expected new inputs
    for (const auto& name : m_param.m_newInputs) {
        EXPECT_TRUE(std::find_if(ops.begin(), ops.end(),
                                 [&](const std::shared_ptr<ngraph::Node>& node) {
                                     return node->get_friendly_name().find(name) != std::string::npos;
                                 }) != ops.end()) << "Name not found:" << name;
    }
}

TEST_P(FrontEndCutModelTest, testNewOutputs_func) {
    ASSERT_NO_THROW(doLoadFromFile());
    std::vector<Place::Ptr> newPlaces;
    ASSERT_NO_THROW(newPlaces = constructNewOutputs());
    ASSERT_NO_THROW(m_inputModel->overrideAllInputs(newPlaces));

    std::shared_ptr<ngraph::Function> function;
    ASSERT_NO_THROW(function = m_frontEnd->convert(m_inputModel));
    auto ops = function->get_ordered_ops();

    // Ensure that it doesn't contain old outputs
    for (const auto& name : m_param.m_oldOutputs) {
        EXPECT_TRUE(std::find_if(ops.begin(), ops.end(),
                                 [&](const std::shared_ptr<ngraph::Node>& node) {
                                     return node->get_friendly_name().find(name) != std::string::npos;
                                 }) == ops.end()) << "Name shall not exist:" << name;
    }

    // Ensure that it contains expected new outputs
    for (const auto& name : m_param.m_newOutputs) {
        EXPECT_TRUE(std::find_if(ops.begin(), ops.end(),
                                 [&](const std::shared_ptr<ngraph::Node>& node) {
                                     return node->get_friendly_name().find(name) != std::string::npos;
                                 }) != ops.end()) << "Name not found:" << name;
    }
}

TEST_P(FrontEndCutModelTest, testExtractSubgraph) {
    ASSERT_NO_THROW(doLoadFromFile());
    std::vector<Place::Ptr> newInputs, newOutputs;
    ASSERT_NO_THROW(newInputs = constructNewOutputs());
    ASSERT_NO_THROW(newOutputs = constructNewOutputs());
    ASSERT_NO_THROW(m_inputModel->extractSubgraph(newInputs, newOutputs));

    std::shared_ptr<ngraph::Function> function;
    ASSERT_NO_THROW(function = m_frontEnd->convert(m_inputModel));
    auto ops = function->get_ordered_ops();

    // Ensure that it doesn't expected old outputs
    for (const auto& name : m_param.m_oldOutputs) {
        EXPECT_TRUE(std::find_if(ops.begin(), ops.end(),
                                 [&](const std::shared_ptr<ngraph::Node>& node) {
                                     return node->get_friendly_name().find(name) != std::string::npos;
                                 }) == ops.end()) << "Name shall not exist:" << name;
    }

    // Ensure that it contains expected new outputs
    for (const auto& name : m_param.m_newOutputs) {
        EXPECT_TRUE(std::find_if(ops.begin(), ops.end(),
                                 [&](const std::shared_ptr<ngraph::Node>& node) {
                                     return node->get_friendly_name().find(name) != std::string::npos;
                                 }) != ops.end()) << "Name not found:" << name;
    }
}
