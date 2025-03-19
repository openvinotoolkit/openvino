// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "partial_shape.hpp"

#include "utils.hpp"

using namespace ov::frontend;

std::string FrontEndPartialShapeTest::getTestCaseName(const testing::TestParamInfo<PartialShapeParam>& obj) {
    BaseFEParam base;
    PartShape part;
    std::tie(base, part) = obj.param;
    std::string res = base.m_frontEndName + "_" + part.m_modelName + "_" + part.m_tensorName;
    for (auto s : part.m_newPartialShape) {
        res += "_" + (s.is_dynamic() ? "dyn" : std::to_string(s.get_length()));
    }
    return FrontEndTestUtils::fileToTestName(res);
}

void FrontEndPartialShapeTest::SetUp() {
    m_fem = FrontEndManager();  // re-initialize after setting up environment
    initParamTest();
}

void FrontEndPartialShapeTest::initParamTest() {
    std::tie(m_baseParam, m_partShape) = GetParam();
    m_partShape.m_modelName = FrontEndTestUtils::make_model_path(m_baseParam.m_modelsPath + m_partShape.m_modelName);
}

void FrontEndPartialShapeTest::doLoadFromFile() {
    std::tie(m_frontEnd, m_inputModel) =
        FrontEndTestUtils::load_from_file(m_fem, m_baseParam.m_frontEndName, m_partShape.m_modelName);
}

///////////////////////////////////////////////////////////////////

TEST_P(FrontEndPartialShapeTest, testCheckOldPartialShape) {
    OV_ASSERT_NO_THROW(doLoadFromFile());

    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = m_frontEnd->convert(m_inputModel));
    auto ops = model->get_ordered_ops();
    auto it = std::find_if(ops.begin(), ops.end(), [&](const std::shared_ptr<ov::Node>& node) {
        return node->get_friendly_name().find(m_partShape.m_tensorName) != std::string::npos;
    });
    ASSERT_NE(it, ops.end());
    auto shape = (*it)->get_output_partial_shape(0);
    ASSERT_EQ(shape, m_partShape.m_oldPartialShape);
}

TEST_P(FrontEndPartialShapeTest, testSetNewPartialShape) {
    OV_ASSERT_NO_THROW(doLoadFromFile());
    Place::Ptr place;
    OV_ASSERT_NO_THROW(place = m_inputModel->get_place_by_tensor_name(m_partShape.m_tensorName));
    ASSERT_NE(place, nullptr);
    OV_ASSERT_NO_THROW(m_inputModel->set_partial_shape(place, ov::PartialShape{m_partShape.m_newPartialShape}));

    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = m_frontEnd->convert(m_inputModel));
    auto ops = model->get_ordered_ops();
    auto it = std::find_if(ops.begin(), ops.end(), [&](const std::shared_ptr<ov::Node>& node) {
        return node->get_friendly_name().find(m_partShape.m_tensorName) != std::string::npos;
    });
    ASSERT_NE(it, ops.end());
    auto shape = (*it)->get_output_partial_shape(0);
    ASSERT_EQ(shape, m_partShape.m_newPartialShape);
}
