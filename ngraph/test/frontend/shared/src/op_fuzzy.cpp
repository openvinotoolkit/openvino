// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cnpy.h>

#include "op_fuzzy.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "utils.hpp"

using namespace ngraph;
using namespace InferenceEngine;

using namespace ngraph;
using namespace ngraph::frontend;
using TestEngine = test::IE_CPU_Engine;

std::string
    FrontEndFuzzyOpTest::getTestCaseName(const testing::TestParamInfo<FuzzyOpTestParam>& obj)
{
    std::string fe, path, fileName;
    std::tie(fe, path, fileName) = obj.param;
    return fe + "_" + FrontEndTestUtils::fileToTestName(fileName);
}

void FrontEndFuzzyOpTest::SetUp()
{
    FrontEndTestUtils::setupTestEnv();
    m_fem = FrontEndManager(); // re-initialize after setting up environment
    initParamTest();
}

void FrontEndFuzzyOpTest::initParamTest()
{
    std::tie(m_feName, m_pathToModels, m_modelFile) = GetParam();
    m_modelFile = m_pathToModels + m_modelFile;
}

void FrontEndFuzzyOpTest::doLoadFromFile()
{
    std::vector<std::string> frontends;
    ASSERT_NO_THROW(frontends = m_fem.get_available_front_ends());
    ASSERT_NO_THROW(m_frontEnd = m_fem.load_by_framework(m_feName));
    ASSERT_NE(m_frontEnd, nullptr);
    ASSERT_NO_THROW(m_inputModel = m_frontEnd->load(m_modelFile));
    ASSERT_NE(m_inputModel, nullptr);
}

template <typename T>
inline void addInputOutput(cnpy::NpyArray& npy_array,
                           test::TestCase<TestEngine>& test_case,
                           bool is_input = true)
{
    T* npy_begin = npy_array.data<T>();
    std::vector<T> data(npy_begin, npy_begin + npy_array.num_vals);
    if (is_input)
        test_case.add_input(data);
    else
        test_case.add_expected_output(data);
}

static bool ends_with(std::string const& value, std::string const& ending)
{
    if (ending.size() > value.size())
        return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

static std::string getModelFolder(const std::string& modelFile)
{
    if (!ends_with(modelFile, ".pdmodel"))
        return modelFile;
    size_t found = modelFile.find_last_of("/\\");
    return modelFile.substr(0, found);
};

void FrontEndFuzzyOpTest::runConvertedModel(const std::shared_ptr<ngraph::Function> function,
                                            const std::string& modelFile)
{
    auto modelFolder = getModelFolder(modelFile);

    // run test
    auto testCase = test::TestCase<TestEngine>(function);

    const auto parameters = function->get_parameters();
    for (size_t i = 0; i < parameters.size(); i++)
    {
        // read input npy file
        std::string dataFile =
            modelFolder + "/input" + std::to_string((parameters.size() - 1) - i) + ".npy";
        cnpy::NpyArray input = cnpy::npy_load(dataFile);
        auto input_dtype = parameters[i]->get_element_type();

        if (input_dtype == element::f32)
        {
            addInputOutput<float>(input, testCase, true);
        }
        else if (input_dtype == element::i32)
        {
            addInputOutput<int32_t>(input, testCase, true);
        }
        else if (input_dtype == element::i64)
        {
            addInputOutput<int64_t>(input, testCase, true);
        }
        else
        {
            throw std::runtime_error("not supported dtype in" + input_dtype.get_type_name());
        }
    }

    const auto results = function->get_results();
    bool useFloatTest = false;
    for (size_t i = 0; i < results.size(); i++)
    {
        // read expected output npy file
        std::string dataFile = modelFolder + "/output" + std::to_string(i) + ".npy";
        cnpy::NpyArray output = cnpy::npy_load(dataFile);
        auto outputDtype = results[i]->get_element_type();
        if (outputDtype == element::f32)
        {
            addInputOutput<float>(output, testCase, false);
            useFloatTest = true;
        }
        else if (outputDtype == element::i32)
        {
            addInputOutput<int32_t>(output, testCase, false);
        }
        else if (outputDtype == element::i64)
        {
            addInputOutput<int64_t>(output, testCase, false);
        }
        else
        {
            throw std::runtime_error("not supported dtype out " + outputDtype.get_type_name());
        }
    }

    if (useFloatTest)
    {
        testCase.run_with_tolerance_as_fp();
    }
    else
    {
        testCase.run();
    }
}

TEST_P(FrontEndFuzzyOpTest, testOpFuzzy)
{
    // load
    ASSERT_NO_THROW(doLoadFromFile());

    // convert
    std::shared_ptr<ngraph::Function> function;
    function = m_frontEnd->convert(m_inputModel);
    ASSERT_NE(function, nullptr);

    // run
    runConvertedModel(function, m_modelFile);
}
