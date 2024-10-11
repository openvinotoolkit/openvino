// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_fuzzy.hpp"

#include <cnpy.h>

#include "common_test_utils/test_case.hpp"
#include "common_test_utils/test_control.hpp"
#include "utils.hpp"

using namespace ov::frontend;

std::string FrontEndFuzzyOpTest::getTestCaseName(const testing::TestParamInfo<FuzzyOpTestParam>& obj) {
    std::string fe, path, fileName;
    std::tie(fe, path, fileName) = obj.param;
    return fe + "_" + FrontEndTestUtils::fileToTestName(fileName);
}

void FrontEndFuzzyOpTest::SetUp() {
    m_fem = FrontEndManager();  // re-initialize after setting up environment
    initParamTest();
}

void FrontEndFuzzyOpTest::initParamTest() {
    std::tie(m_feName, m_pathToModels, m_modelFile) = GetParam();
    m_modelFile = FrontEndTestUtils::make_model_path(m_pathToModels + m_modelFile);
}

void FrontEndFuzzyOpTest::doLoadFromFile() {
    std::tie(m_frontEnd, m_inputModel) = FrontEndTestUtils::load_from_file(m_fem, m_feName, m_modelFile);
}

template <typename T>
inline void addInputOutput(const cnpy::NpyArray& npy_array, ov::test::TestCase& test_case, bool is_input = true) {
    const T* npy_begin = npy_array.data<T>();
    std::vector<T> data(npy_begin, npy_begin + npy_array.num_vals);
    if (is_input)
        test_case.add_input(npy_array.shape, data);
    else
        test_case.add_expected_output(npy_array.shape, data);
}

static bool ends_with(std::string const& value, std::string const& ending) {
    if (ending.size() > value.size())
        return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

static std::string getModelFolder(const std::string& modelFile) {
    if (!ends_with(modelFile, ".pdmodel"))
        return modelFile;
    size_t found = modelFile.find_last_of("/\\");
    return modelFile.substr(0, found);
};

void FrontEndFuzzyOpTest::runConvertedModel(const std::shared_ptr<ov::Model> model, const std::string& modelFile) {
    auto modelFolder = getModelFolder(modelFile);

    // run test
    auto testCase = ov::test::TestCase(model, "CPU");

    const auto parameters = model->get_parameters();
    for (size_t i = 0; i < parameters.size(); i++) {
        // read input npy file
        std::string dataFile = modelFolder + "/input" + std::to_string((parameters.size() - 1) - i) + ".npy";
        cnpy::NpyArray input = cnpy::npy_load(dataFile);
        auto input_dtype = parameters[i]->get_element_type();

        if (input_dtype == ov::element::f32) {
            addInputOutput<float>(input, testCase, true);
        } else if (input_dtype == ov::element::i32) {
            addInputOutput<int32_t>(input, testCase, true);
        } else if (input_dtype == ov::element::i64) {
            addInputOutput<int64_t>(input, testCase, true);
        } else if (input_dtype == ov::element::boolean) {
            addInputOutput<bool>(input, testCase, true);
        } else {
            throw std::runtime_error("not supported dtype in" + input_dtype.get_type_name());
        }
    }

    const auto results = model->get_results();
    bool useFloatTest = false;
    for (size_t i = 0; i < results.size(); i++) {
        // read expected output npy file
        std::string dataFile = modelFolder + "/output" + std::to_string(i) + ".npy";
        cnpy::NpyArray output = cnpy::npy_load(dataFile);
        auto outputDtype = results[i]->get_element_type();
        if (outputDtype == ov::element::f32) {
            addInputOutput<float>(output, testCase, false);
            useFloatTest = true;
        } else if (outputDtype == ov::element::i32) {
            addInputOutput<int32_t>(output, testCase, false);
        } else if (outputDtype == ov::element::i64) {
            addInputOutput<int64_t>(output, testCase, false);
        } else {
            throw std::runtime_error("not supported dtype out " + outputDtype.get_type_name());
        }
    }

    if (useFloatTest) {
        testCase.run_with_tolerance_as_fp(2e-5f);
    } else {
        testCase.run();
    }
}

#if defined OPENVINO_ARCH_ARM64 || defined OPENVINO_ARCH_ARM
// Ticket: 126830, 153158
TEST_P(FrontEndFuzzyOpTest, DISABLED_testOpFuzzy) {
#else
TEST_P(FrontEndFuzzyOpTest, testOpFuzzy) {
#endif
    // load
    OV_ASSERT_NO_THROW(doLoadFromFile());

    // convert
    std::shared_ptr<ov::Model> model;
    model = m_frontEnd->convert(m_inputModel);
    ASSERT_NE(model, nullptr);

    // run
    runConvertedModel(model, m_modelFile);
}
