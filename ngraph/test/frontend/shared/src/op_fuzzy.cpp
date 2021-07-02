// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cnpy.h>

#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

#include "../include/op_fuzzy.hpp"
#include "../include/utils.hpp"


using namespace ngraph;
using namespace InferenceEngine;



using namespace ngraph;
using namespace ngraph::frontend;
using TestEngine = test::IE_CPU_Engine;


std::string FrontEndFuzzyOpTest::getTestCaseName(const testing::TestParamInfo<FuzzyOpTestParam>& obj)
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
    ASSERT_NO_THROW(m_inputModel = m_frontEnd->load_from_file(m_modelFile));
    ASSERT_NE(m_inputModel, nullptr);
}

template <typename T>
inline void add_input_output(cnpy::NpyArray& npy_array,
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

static std::string get_model_folder(std::string& modelfile)
{
    if (!ends_with(modelfile, ".pdmodel"))
        return modelfile;
    size_t found = modelfile.find_last_of("/\\");
    return modelfile.substr(0, found);
};

void FrontEndFuzzyOpTest::runConvertedModel(std::shared_ptr<ngraph::Function> function, std::string& model_file)
{
    auto model_folder = get_model_folder(model_file);

    // run test
    auto test_case = test::TestCase<TestEngine>(function);

    const auto parameters = function->get_parameters();
    for (size_t i = 0; i < parameters.size(); i++)
    {
        // read input npy file
        std::string data_file =
                model_folder + "/input" + std::to_string((parameters.size() - 1) - i) + ".npy";
        cnpy::NpyArray input = cnpy::npy_load(data_file);
        auto input_dtype = parameters[i]->get_element_type();

        if (input_dtype == element::f32)
        {
            add_input_output<float>(input, test_case, true);
        }
        else if (input_dtype == element::i32)
        {
            add_input_output<int32_t>(input, test_case, true);
        }
        else if (input_dtype == element::i64)
        {
            add_input_output<int64_t>(input, test_case, true);
        }
        else
        {
            throw std::runtime_error("not supported dtype in" + input_dtype.get_type_name());
        }
    }

    const auto results = function->get_results();
    bool use_float_test = false;
    for (size_t i = 0; i < results.size(); i++)
    {
        // read expected output npy file
        std::string data_file = model_folder + "/output" + std::to_string(i) + ".npy";
        cnpy::NpyArray output = cnpy::npy_load(data_file);
        auto output_dtype = results[i]->get_element_type();
        if (output_dtype == element::f32)
        {
            add_input_output<float>(output, test_case, false);
            use_float_test = true;
        }
        else if (output_dtype == element::i32)
        {
            add_input_output<int32_t>(output, test_case, false);
        }
        else if (output_dtype == element::i64)
        {
            add_input_output<int64_t>(output, test_case, false);
        }
        else
        {
            throw std::runtime_error("not supported dtype out " + output_dtype.get_type_name());
        }
    }

    if (use_float_test)
    {
        test_case.run_with_tolerance_as_fp();
    }
    else
    {
        test_case.run();
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

