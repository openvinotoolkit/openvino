#include "tests_utils.h"

#include <gtest/gtest.h>
#include <pugixml.hpp>
#include <string>

#define DEBUG_MODE false

const pugi::xml_document & Environment::getTestConfig() {
    return _test_config;
}

void Environment::setTestConfig(const pugi::xml_document &test_config) {
    _test_config.reset(test_config);
}

const pugi::xml_document & Environment::getEnvConfig() {
    return _env_config;
}

void Environment::setEnvConfig(const pugi::xml_document &env_config) {
    _env_config.reset(env_config);
}

std::vector<TestCase> generateTestsParams(std::initializer_list<std::string> fields) {
    std::vector<TestCase> tests_cases;
    const pugi::xml_document & test_config = Environment::Instance().getTestConfig();
    std::string models_path = Environment::Instance().getEnvConfig()
            .child("attributes").child("irs_path").child("value").text().as_string();

    std::vector<int> processes;
    std::vector<int> threads;
    std::vector<int> iterations;
    std::vector<std::string> devices;
    std::vector<std::string> models;

    pugi::xml_node values;
    for (auto field = fields.begin(); field != fields.end(); field++) {
        if (*field == "processes") {
            values = test_config.child("attributes").child("processes");
            for (pugi::xml_node val = values.first_child(); val; val = val.next_sibling())
                processes.push_back(val.text().as_int());
        } else if (*field == "threads") {
            values = test_config.child("attributes").child("threads");
            for (pugi::xml_node val = values.first_child(); val; val = val.next_sibling())
                threads.push_back(val.text().as_int());
        } else if (*field == "iterations") {
            values = test_config.child("attributes").child("iterations");
            for (pugi::xml_node val = values.first_child(); val; val = val.next_sibling())
                iterations.push_back(val.text().as_int());
        } else if (*field == "devices") {
            values = test_config.child("attributes").child("devices");
            for (pugi::xml_node val = values.first_child(); val; val = val.next_sibling())
                devices.push_back(val.text().as_string());
        } else if (*field == "models") {
            values = test_config.child("attributes").child("models");
            for (pugi::xml_node val = values.first_child(); val; val = val.next_sibling())
                models.push_back(val.text().as_string());
        }
    }

    // Initialize variables with default value if it weren't filled
    processes = !processes.empty() ? processes: std::vector<int>{1};
    threads = !threads.empty() ? threads: std::vector<int>{1};
    iterations = !iterations.empty() ? iterations: std::vector<int>{1};
    devices = !devices.empty() ? devices : std::vector<std::string>{"NULL"};
    models = !models.empty() ? models : std::vector<std::string>{"NULL"};

    for (auto &numprocesses : processes)
        for (auto &numthreads : threads)
            for (auto &numiters : iterations)
                for (auto &device : devices)
                    for (auto &model : models)
                        tests_cases.push_back(TestCase(numprocesses, numthreads, numiters, device, OS_PATH_JOIN({models_path, model}), model));

    return tests_cases;
}

std::string getTestCaseName(const testing::TestParamInfo<TestCase> &obj) {
    return obj.param.test_case_name;
}

void test_wrapper(const std::function<void(std::string, std::string, int)> &tests_pipeline, const TestCase &params) {
    tests_pipeline(params.model, params.device, params.numiters);
}

void _runTest(const std::function<void(std::string, std::string, int)> &tests_pipeline, const TestCase &params) {
    run_in_threads(params.numthreads, test_wrapper, tests_pipeline, params);
}

void runTest(const std::function<void(std::string, std::string, int)> &tests_pipeline, const TestCase &params) {
#if DEBUG_MODE
    tests_pipeline(params.model, params.device, params.numiters);
#else
    int status = run_in_processes(params.numprocesses, _runTest, tests_pipeline, params);
    ASSERT_EQ(status, 0) << "Test failed with exitcode " << std::to_string(status);
#endif
}

