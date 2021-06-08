// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <regex>

#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/engine/test_engines.hpp"
#include "util/ndarray.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

#include "ngraph/ngraph.hpp"

using namespace ngraph;
using namespace InferenceEngine;

#include <cnpy.h>
#include "../shared/include/basic_api.hpp"

using namespace ngraph;
using namespace ngraph::frontend;
using TestEngine = test::IE_CPU_Engine;

static const std::string PDPD = "pdpd";
static const std::string PATH_TO_MODELS = "/paddlepaddle/models/";

/* helper */
static bool ends_with(std::string const& value, std::string const& ending)
{
    if (ending.size() > value.size())
        return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

static bool starts_with(std::string const& value, std::string const& starting)
{
    if (starting.size() > value.size())
        return false;
    return std::equal(starting.begin(), starting.end(), value.begin());
}

static std::string get_model_folder(std::string& modelfile)
{
    if (!ends_with(modelfile, ".pdmodel"))
        return modelfile;
    size_t found = modelfile.find_last_of("/\\");
    return modelfile.substr(0, found);
};

static const std::string& trim_space(std::string& str) // trim leading and tailing spaces
{
    // leading
    auto it = str.begin();
    for (; it != str.end() && isspace(*it); it++)
        ;
    auto d = std::distance(str.begin(), it);
    str.erase(0, d);

    // tailing
    auto rit = str.rbegin();
    for (; rit != str.rend() && isspace(*rit); rit++)
    {
        str.pop_back();
    }

    // std::cout << "[" << str << "]" << std::endl;
    return str;
}

static std::vector<std::string> get_models(void)
{
    std::string models_csv = std::string(TEST_FILES) + PATH_TO_MODELS + "models.csv";
    std::ifstream f(models_csv);
    std::vector<std::string> models;
    std::string line;
    while (getline(f, line, ','))
    {
        auto line_trim = trim_space(line);
        if (line_trim.empty() || starts_with(line_trim, "#"))
            continue;
        // std::cout<< "line in csv: [" << line_trim<< "]" << std::endl;
        models.emplace_back(line_trim);
    }
    return models;
}

inline void visualizer(std::shared_ptr<ngraph::Function> function, std::string path)
{
    ngraph::pass::VisualizeTree("function.png").run_on_function(function);

    CNNNetwork network(function);
    network.serialize(path + ".xml", path + ".bin");
}

namespace fuzzyOp
{
    using PDPDFuzzyOpTest = FrontEndBasicTest;
    using PDPDFuzzyOpTestParam = std::tuple<std::string,  // FrontEnd name
                                            std::string,  // Base path to models
                                            std::string>; // modelname
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
    void run_fuzzy(std::shared_ptr<ngraph::Function> function, std::string& model_file)
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

    TEST_P(PDPDFuzzyOpTest, test_fuzzy)
    {
        // load
        ASSERT_NO_THROW(doLoadFromFile());

        // convert
        std::shared_ptr<ngraph::Function> function;
        function = m_frontEnd->convert(m_inputModel);
        ASSERT_NE(function, nullptr);

        // run
        run_fuzzy(function, m_modelFile);
    }

    INSTANTIATE_TEST_CASE_P(FrontendOpTest,
                            PDPDFuzzyOpTest,
                            ::testing::Combine(::testing::Values(PDPD),
                                               ::testing::Values(std::string(TEST_PDPD_MODELS)),
                                               ::testing::ValuesIn(get_models())),
                            PDPDFuzzyOpTest::getTestCaseName);

} // namespace fuzzyOp
