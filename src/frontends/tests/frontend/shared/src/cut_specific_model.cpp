// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cut_specific_model.hpp"

#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "utils.hpp"

using namespace ov::frontend;

static std::string joinStrings(const std::vector<std::string>& strings) {
    std::ostringstream res;
    std::copy(strings.begin(), strings.end(), std::ostream_iterator<std::string>(res, "_"));
    return res.str();
}

std::string FrontEndCutModelTest::getTestCaseName(const testing::TestParamInfo<CutModelParam>& obj) {
    std::string res = obj.param.m_frontEndName + "_" + obj.param.m_modelName;
    res += "I" + joinStrings(obj.param.m_oldInputs) + joinStrings(obj.param.m_newInputs);
    res += "O" + joinStrings(obj.param.m_oldOutputs) + joinStrings(obj.param.m_newOutputs);
    return FrontEndTestUtils::fileToTestName(res);
}

void FrontEndCutModelTest::SetUp() {
    m_fem = FrontEndManager();  // re-initialize after setting up environment
    initParamTest();
}

void FrontEndCutModelTest::initParamTest() {
    m_param = GetParam();
    m_param.m_modelName = FrontEndTestUtils::make_model_path(m_param.m_modelsPath + m_param.m_modelName);
}

void FrontEndCutModelTest::doLoadFromFile() {
    std::tie(m_frontEnd, m_inputModel) =
        FrontEndTestUtils::load_from_file(m_fem, m_param.m_frontEndName, m_param.m_modelName);
}

std::vector<ov::frontend::Place::Ptr> FrontEndCutModelTest::constructNewInputs() const {
    std::vector<Place::Ptr> newInputs;
    for (const auto& name : m_param.m_newInputs) {
        newInputs.push_back(m_inputModel->get_place_by_tensor_name(name));
    }
    return newInputs;
}

std::vector<ov::frontend::Place::Ptr> FrontEndCutModelTest::constructNewOutputs() const {
    std::vector<Place::Ptr> newOutputs;
    for (const auto& name : m_param.m_newOutputs) {
        newOutputs.push_back(m_inputModel->get_place_by_tensor_name(name));
    }
    return newOutputs;
}

///////////////////////////////////////////////////////////////////

TEST_P(FrontEndCutModelTest, testOverrideInputs) {
    OV_ASSERT_NO_THROW(doLoadFromFile());
    std::vector<Place::Ptr> newPlaces;
    OV_ASSERT_NO_THROW(newPlaces = constructNewInputs());
    OV_ASSERT_NO_THROW(m_inputModel->override_all_inputs(newPlaces));
    OV_ASSERT_NO_THROW(m_inputModel->get_inputs());
    EXPECT_EQ(m_param.m_newInputs.size(), m_inputModel->get_inputs().size());
    for (auto newInput : m_inputModel->get_inputs()) {
        std::vector<std::string> names;
        OV_ASSERT_NO_THROW(names = newInput->get_names());
        bool found = false;
        for (const auto& name : m_param.m_newInputs) {
            if (std::find(names.begin(), names.begin(), name) != names.end()) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << joinStrings(names) << " were not found in new inputs";
    }
}

TEST_P(FrontEndCutModelTest, testOverrideOutputs) {
    OV_ASSERT_NO_THROW(doLoadFromFile());
    std::vector<Place::Ptr> newPlaces;
    OV_ASSERT_NO_THROW(newPlaces = constructNewOutputs());
    OV_ASSERT_NO_THROW(m_inputModel->override_all_outputs(newPlaces));
    OV_ASSERT_NO_THROW(m_inputModel->get_outputs());
    EXPECT_EQ(m_param.m_newOutputs.size(), m_inputModel->get_outputs().size());
    for (auto newOutput : m_inputModel->get_outputs()) {
        std::vector<std::string> names;
        OV_ASSERT_NO_THROW(names = newOutput->get_names());
        bool found = false;
        for (const auto& name : m_param.m_newOutputs) {
            if (std::find(names.begin(), names.begin(), name) != names.end()) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << joinStrings(names) << " were not found in new outputs";
    }
}

TEST_P(FrontEndCutModelTest, testOldInputs) {
    OV_ASSERT_NO_THROW(doLoadFromFile());
    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = m_frontEnd->convert(m_inputModel));
    auto ops = model->get_ordered_ops();

    // Ensure that it contains expected old inputs
    for (const auto& name : m_param.m_oldInputs) {
        EXPECT_TRUE(std::find_if(ops.begin(),
                                 ops.end(),
                                 [&](const std::shared_ptr<ov::Node>& node) {
                                     return node->get_friendly_name().find(name) != std::string::npos;
                                 }) != ops.end())
            << "Name not found:" << name;
    }
}

TEST_P(FrontEndCutModelTest, testOldOutputs) {
    OV_ASSERT_NO_THROW(doLoadFromFile());
    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = m_frontEnd->convert(m_inputModel));
    auto ops = model->get_ordered_ops();
    // Ensure that it contains expected old outputs
    for (const auto& name : m_param.m_oldOutputs) {
        EXPECT_TRUE(std::find_if(ops.begin(),
                                 ops.end(),
                                 [&](const std::shared_ptr<ov::Node>& node) {
                                     return node->get_friendly_name().find(name) != std::string::npos;
                                 }) != ops.end())
            << "Name not found:" << name;
    }
}

TEST_P(FrontEndCutModelTest, testNewInputs_func) {
    OV_ASSERT_NO_THROW(doLoadFromFile());
    std::vector<Place::Ptr> newPlaces;
    OV_ASSERT_NO_THROW(newPlaces = constructNewInputs());
    OV_ASSERT_NO_THROW(m_inputModel->override_all_inputs(newPlaces));

    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = m_frontEnd->convert(m_inputModel));
    auto ops = model->get_ordered_ops();

    // Ensure that it doesn't contain old inputs
    for (const auto& name : m_param.m_oldInputs) {
        EXPECT_TRUE(std::find_if(ops.begin(),
                                 ops.end(),
                                 [&](const std::shared_ptr<ov::Node>& node) {
                                     return node->get_friendly_name().find(name) != std::string::npos;
                                 }) == ops.end())
            << "Name shall not exist:" << name;
    }

    // Ensure that it contains expected new inputs
    for (const auto& name : m_param.m_newInputs) {
        EXPECT_TRUE(std::find_if(ops.begin(),
                                 ops.end(),
                                 [&](const std::shared_ptr<ov::Node>& node) {
                                     return node->get_friendly_name().find(name) != std::string::npos;
                                 }) != ops.end())
            << "Name not found:" << name;
    }
}

TEST_P(FrontEndCutModelTest, testNewOutputs_func) {
    OV_ASSERT_NO_THROW(doLoadFromFile());
    std::vector<Place::Ptr> newPlaces;
    OV_ASSERT_NO_THROW(newPlaces = constructNewOutputs());
    OV_ASSERT_NO_THROW(m_inputModel->override_all_outputs(newPlaces));

    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = m_frontEnd->convert(m_inputModel));
    auto ops = model->get_ordered_ops();

    // Ensure that it doesn't contain old outputs
    for (const auto& name : m_param.m_oldOutputs) {
        EXPECT_TRUE(std::find_if(ops.begin(),
                                 ops.end(),
                                 [&](const std::shared_ptr<ov::Node>& node) {
                                     return node->get_friendly_name().find(name) != std::string::npos;
                                 }) == ops.end())
            << "Name shall not exist:" << name;
    }

    // Ensure that it contains expected new outputs
    for (const auto& name : m_param.m_newOutputs) {
        EXPECT_TRUE(std::find_if(ops.begin(),
                                 ops.end(),
                                 [&](const std::shared_ptr<ov::Node>& node) {
                                     return node->get_friendly_name().find(name) != std::string::npos;
                                 }) != ops.end())
            << "Name not found:" << name;
    }
}

TEST_P(FrontEndCutModelTest, testExtractSubgraph) {
    OV_ASSERT_NO_THROW(doLoadFromFile());
    std::vector<Place::Ptr> newInputs, newOutputs;
    OV_ASSERT_NO_THROW(newInputs = constructNewInputs());
    OV_ASSERT_NO_THROW(newOutputs = constructNewOutputs());
    OV_ASSERT_NO_THROW(m_inputModel->extract_subgraph(newInputs, newOutputs));

    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = m_frontEnd->convert(m_inputModel));
    auto ops = model->get_ordered_ops();

    // Ensure that it doesn't contain expected old outputs
    for (const auto& name : m_param.m_oldOutputs) {
        EXPECT_TRUE(std::find_if(ops.begin(),
                                 ops.end(),
                                 [&](const std::shared_ptr<ov::Node>& node) {
                                     return node->get_friendly_name().find(name) != std::string::npos;
                                 }) == ops.end())
            << "Name shall not exist:" << name;
    }

    // Ensure that it contains expected new outputs
    for (const auto& name : m_param.m_newOutputs) {
        EXPECT_TRUE(std::find_if(ops.begin(),
                                 ops.end(),
                                 [&](const std::shared_ptr<ov::Node>& node) {
                                     return node->get_friendly_name().find(name) != std::string::npos;
                                 }) != ops.end())
            << "Name not found:" << name;
    }
}

TEST_P(FrontEndCutModelTest, testSetTensorValue) {
    OV_ASSERT_NO_THROW(doLoadFromFile());
    Place::Ptr place;
    OV_ASSERT_NO_THROW(place = m_inputModel->get_place_by_tensor_name(m_param.m_tensorValueName));
    OV_ASSERT_NO_THROW(m_inputModel->set_tensor_value(place, &m_param.m_tensorValue[0]));

    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = m_frontEnd->convert(m_inputModel));
    auto ops = model->get_ordered_ops();

    auto const_name = m_param.m_tensorValueName;
    auto const_node_it = std::find_if(ops.begin(), ops.end(), [&](const std::shared_ptr<ov::Node>& node) {
        return node->get_friendly_name().find(const_name) != std::string::npos;
    });
    ASSERT_TRUE(const_node_it != ops.end()) << "Name shall exist:" << const_name;
    auto data = std::dynamic_pointer_cast<ov::op::v0::Constant>(*const_node_it)->get_vector<float>();
    EXPECT_EQ(data.size(), m_param.m_tensorValue.size()) << "Data size must be equal to expected size";
    EXPECT_TRUE(std::equal(data.begin(), data.end(), m_param.m_tensorValue.begin())) << "Data must be equal";
}
