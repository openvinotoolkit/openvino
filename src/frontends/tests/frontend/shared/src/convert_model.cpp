// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_model.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "utils.hpp"

using namespace ov::frontend;

std::string FrontEndConvertModelTest::getTestCaseName(const testing::TestParamInfo<ConvertParam>& obj) {
    std::string fe, path, fileName;
    std::tie(fe, path, fileName) = obj.param;
    return fe + "_" + FrontEndTestUtils::fileToTestName(fileName);
}

void FrontEndConvertModelTest::SetUp() {
    m_fem = FrontEndManager();  // re-initialize after setting up environment
    initParamTest();
}

void FrontEndConvertModelTest::initParamTest() {
    std::tie(m_feName, m_pathToModels, m_modelFile) = GetParam();
    m_modelFile = FrontEndTestUtils::make_model_path(m_pathToModels + m_modelFile);
}

void FrontEndConvertModelTest::doLoadFromFile() {
    std::vector<std::string> frontends;
    OV_ASSERT_NO_THROW(frontends = m_fem.get_available_front_ends());
    OV_ASSERT_NO_THROW(m_frontEnd = m_fem.load_by_framework(m_feName));
    ASSERT_NE(m_frontEnd, nullptr);
    OV_ASSERT_NO_THROW(m_inputModel = m_frontEnd->load(m_modelFile));
    ASSERT_NE(m_inputModel, nullptr);
}

#ifdef _WIN32
// Ticket: 126320
TEST_P(FrontEndConvertModelTest, DISABLED_test_convert_partially_equal_convert) {
#else
TEST_P(FrontEndConvertModelTest, test_convert_partially_equal_convert) {
#endif
    OV_ASSERT_NO_THROW(doLoadFromFile());
    std::shared_ptr<ov::Model> model_ref;
    OV_ASSERT_NO_THROW(model_ref = m_frontEnd->convert(m_inputModel));
    ASSERT_NE(model_ref, nullptr);
    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = m_frontEnd->convert_partially(m_inputModel));
    ASSERT_NE(model, nullptr);

    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    // TODO: enable name comparison for tf when TransposeSinking is fixed, ticket 68960
    if (m_frontEnd->get_name() != "tf" && m_frontEnd->get_name() != "tflite") {
        func_comparator.enable(FunctionsComparator::NAMES);
    }
    const FunctionsComparator::Result res = func_comparator(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

#ifdef _WIN32
// Ticket: 126320
TEST_P(FrontEndConvertModelTest, DISABLED_test_decode_convert_equal_convert) {
#else
TEST_P(FrontEndConvertModelTest, test_decode_convert_equal_convert) {
#endif
    OV_ASSERT_NO_THROW(doLoadFromFile());
    std::shared_ptr<ov::Model> model_ref;
    OV_ASSERT_NO_THROW(model_ref = m_frontEnd->convert(m_inputModel));
    ASSERT_NE(model_ref, nullptr);
    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = m_frontEnd->decode(m_inputModel));
    OV_ASSERT_NO_THROW(m_frontEnd->convert(model));
    ASSERT_NE(model, nullptr);

    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    // TODO: enable name comparison for tf when TransposeSinking is fixed, ticket 68960
    if (m_frontEnd->get_name() != "tf" && m_frontEnd->get_name() != "tflite") {
        func_comparator.enable(FunctionsComparator::NAMES);
    }
    const FunctionsComparator::Result res = func_comparator(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}
