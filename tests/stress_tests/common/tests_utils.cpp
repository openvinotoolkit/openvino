// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tests_utils.h"

#include <gtest/gtest.h>
#include <map>
#include <pugixml.hpp>
#include <string>

#define DEBUG_MODE false

const pugi::xml_document &Environment::getTestConfig() { return _test_config; }

void Environment::setTestConfig(const pugi::xml_document &test_config) { _test_config.reset(test_config); }

const std::string &Environment::getCompilationConfigFile() const { return _compilation_config_file; }

void Environment::setCompilationConfigFile(const std::string &compilation_config_file) {
    _compilation_config_file = compilation_config_file;
}

std::vector<TestCase> generateTestsParams(std::initializer_list<std::string> fields) {
    std::vector<TestCase> tests_cases;
    const pugi::xml_document &test_config = Environment::Instance().getTestConfig();

    std::vector<int> processes, threads, iterations;
    std::vector<std::string> devices, models, models_names, precisions, compilation_configs;

    std::string global_compilation_config = Environment::Instance().getCompilationConfigFile();
    if (global_compilation_config.empty()) {
        pugi::xml_node global_cfg_node = test_config.child("attributes").child("compilation_config_file");
        if (!global_cfg_node) {
            global_cfg_node = test_config.child("attributes").child("compilation_config");
        }
        if (global_cfg_node) {
            global_compilation_config = global_cfg_node.text().as_string();
        }
    }
    if (!global_compilation_config.empty()) {
        global_compilation_config = expand_env_vars(global_compilation_config);
    }

    pugi::xml_node values;
    for (const auto &field: fields) {
        if (field == "processes") {
            values = test_config.child("attributes").child("processes");
            for (pugi::xml_node val = values.first_child(); val; val = val.next_sibling())
                processes.push_back(val.text().as_int());
        } else if (field == "threads") {
            values = test_config.child("attributes").child("threads");
            for (pugi::xml_node val = values.first_child(); val; val = val.next_sibling())
                threads.push_back(val.text().as_int());
        } else if (field == "iterations") {
            values = test_config.child("attributes").child("iterations");
            for (pugi::xml_node val = values.first_child(); val; val = val.next_sibling())
                iterations.push_back(val.text().as_int());
        } else if (field == "devices") {
            values = test_config.child("attributes").child("devices");
            for (pugi::xml_node val = values.first_child(); val; val = val.next_sibling())
                devices.emplace_back(val.text().as_string());
        } else if (field == "models") {
            values = test_config.child("attributes").child("models");
            for (pugi::xml_node val = values.first_child(); val; val = val.next_sibling()) {
                std::string full_path = val.attribute("full_path").as_string();
                std::string path = val.attribute("path").as_string();
                if (full_path.empty() || path.empty())
                    throw std::logic_error("One of the 'model' records from test config doesn't contain 'full_path' or "
                                           "'path' attributes");
                else {
                    models.push_back(expand_env_vars(full_path));
                    models_names.push_back(path);
                }
                std::string precision = val.attribute("precision").as_string();
                precisions.push_back(precision);

                std::string model_cfg = val.attribute("compilation_config").as_string();
                if (model_cfg.empty()) {
                    model_cfg = val.attribute("compilation_config_file").as_string();
                }
                if (model_cfg.empty()) {
                    model_cfg = global_compilation_config;
                }
                compilation_configs.push_back(!model_cfg.empty() ? expand_env_vars(model_cfg) : "");
            }
        }
    }

    // Initialize variables with default value if it weren't filled
    processes = !processes.empty() ? processes : std::vector<int>{1};
    threads = !threads.empty() ? threads : std::vector<int>{1};
    iterations = !iterations.empty() ? iterations : std::vector<int>{1};
    devices = !devices.empty() ? devices : std::vector<std::string>{"NULL"};
    models = !models.empty() ? models : std::vector<std::string>{"NULL"};
    precisions = !precisions.empty() ? precisions : std::vector<std::string>{"NULL"};
    models_names = !models_names.empty() ? models_names : std::vector<std::string>{"NULL"};

    for (auto &numprocesses: processes)
        for (auto &numthreads: threads)
            for (auto &numiters: iterations)
                for (auto &device: devices)
                    for (size_t i = 0; i < models.size(); i++)
                        tests_cases.emplace_back(numprocesses, numthreads, numiters, device, models[i],
                                                 models_names[i], precisions[i],
                                                 i < compilation_configs.size() ? compilation_configs[i] : "");
    return tests_cases;
}

// Generate multi-model test cases from config file with static test definition.
std::vector<MemLeaksTestCase> generateTestsParamsMemLeaks() {
    std::vector<MemLeaksTestCase> tests_cases;
    const pugi::xml_document &test_config = Environment::Instance().getTestConfig();

    int numprocesses, numthreads, numiterations;
    std::string device_name;

    std::string global_compilation_config = Environment::Instance().getCompilationConfigFile();
    if (global_compilation_config.empty()) {
        pugi::xml_node global_cfg_node = test_config.child("attributes").child("compilation_config_file");
        if (!global_cfg_node) {
            global_cfg_node = test_config.child("attributes").child("compilation_config");
        }
        if (global_cfg_node) {
            global_compilation_config = global_cfg_node.text().as_string();
        }
    }
    if (!global_compilation_config.empty()) {
        global_compilation_config = expand_env_vars(global_compilation_config);
    }

    pugi::xml_node cases;
    cases = test_config.child("cases");

    for (pugi::xml_node device = cases.first_child(); device; device = device.next_sibling()) {
        device_name = device.attribute("name").as_string("NULL");
        numprocesses = device.attribute("processes").as_int(1);
        numthreads = device.attribute("threads").as_int(1);
        numiterations = device.attribute("iterations").as_int(1);
        std::string dev_compilation_config = device.attribute("compilation_config").as_string();
        if (dev_compilation_config.empty()) {
            dev_compilation_config = device.attribute("compilation_config_file").as_string();
        }
        if (dev_compilation_config.empty()) {
            dev_compilation_config = global_compilation_config;
        }

        std::vector<std::map<std::string, std::string>> models;

        for (pugi::xml_node model = device.first_child(); model; model = model.next_sibling()) {
            std::string full_path = model.attribute("full_path").as_string();
            std::string path = model.attribute("path").as_string();
            if (full_path.empty() || path.empty())
                throw std::logic_error(
                        "One of the 'model' records from test config doesn't contain 'full_path' or 'path' attributes");
            std::string name = model.attribute("name").as_string();
            std::string precision = model.attribute("precision").as_string();
            std::string model_cfg = model.attribute("compilation_config").as_string();
            if (model_cfg.empty()) {
                model_cfg = model.attribute("compilation_config_file").as_string();
            }
            if (model_cfg.empty()) {
                model_cfg = dev_compilation_config;
            }
            std::map<std::string, std::string> model_map{{"name",               name},
                                                         {"path",               path},
                                                         {"full_path",          expand_env_vars(full_path)},
                                                         {"precision",          precision},
                                                         {"compilation_config", !model_cfg.empty() ? expand_env_vars(model_cfg) : ""}};
            models.push_back(model_map);
        }
        tests_cases.emplace_back(numprocesses, numthreads, numiterations, device_name, models,
                                 !dev_compilation_config.empty() ? expand_env_vars(dev_compilation_config) : "");
    }

    return tests_cases;
}

std::vector<MultiModelTestCase> generateMultiModelTestsParams() {
    std::vector<MultiModelTestCase> tests_cases;
    const pugi::xml_document &test_config = Environment::Instance().getTestConfig();

    std::vector<int> processes, threads, iterations;
    std::vector<std::string> devices, models, models_names, compilation_configs;

    std::string global_compilation_config = Environment::Instance().getCompilationConfigFile();
    if (global_compilation_config.empty()) {
        pugi::xml_node global_cfg_node = test_config.child("attributes").child("compilation_config_file");
        if (!global_cfg_node) {
            global_cfg_node = test_config.child("attributes").child("compilation_config");
        }
        if (global_cfg_node) {
            global_compilation_config = global_cfg_node.text().as_string();
        }
    }
    if (!global_compilation_config.empty()) {
        global_compilation_config = expand_env_vars(global_compilation_config);
    }

    pugi::xml_node values;
    values = test_config.child("attributes").child("processes");
    for (pugi::xml_node val = values.first_child(); val; val = val.next_sibling())
        processes.push_back(val.text().as_int());

    values = test_config.child("attributes").child("threads");
    for (pugi::xml_node val = values.first_child(); val; val = val.next_sibling())
        threads.push_back(val.text().as_int());

    values = test_config.child("attributes").child("iterations");
    for (pugi::xml_node val = values.first_child(); val; val = val.next_sibling())
        iterations.push_back(val.text().as_int());

    values = test_config.child("attributes").child("devices");
    for (pugi::xml_node val = values.first_child(); val; val = val.next_sibling())
        devices.emplace_back(val.text().as_string());

    values = test_config.child("attributes").child("models");
    for (pugi::xml_node val = values.first_child(); val; val = val.next_sibling()) {
        std::string full_path = val.attribute("full_path").as_string();
        std::string path = val.attribute("path").as_string();
        if (!full_path.empty() && !path.empty()) {
            models.push_back(expand_env_vars(full_path));
            models_names.push_back(path);
            std::string model_cfg = val.attribute("compilation_config").as_string();
            if (model_cfg.empty()) {
                model_cfg = val.attribute("compilation_config_file").as_string();
            }
            if (model_cfg.empty()) {
                model_cfg = global_compilation_config;
            }
            compilation_configs.push_back(!model_cfg.empty() ? expand_env_vars(model_cfg) : "");
        }
    }

    processes = !processes.empty() ? processes : std::vector<int>{1};
    threads = !threads.empty() ? threads : std::vector<int>{1};
    iterations = !iterations.empty() ? iterations : std::vector<int>{1};
    devices = !devices.empty() ? devices : std::vector<std::string>{"NULL"};

    if (models.empty()) {
        return tests_cases;
    }

    for (auto &numprocesses: processes) {
        for (auto &numthreads: threads) {
            for (auto &numiters: iterations) {
                for (auto &device: devices) {
                    if (models.size() >= 2) {
                        for (size_t i = 0; i < models.size(); ++i) {
                            for (size_t j = i + 1; j < models.size(); ++j) {
                                tests_cases.emplace_back(numprocesses, numthreads, numiters, device,
                                                         models[i], models_names[i],
                                                         models[j], models_names[j],
                                                         compilation_configs[i],
                                                         compilation_configs[j]);
                            }
                        }
                    } else {
                        tests_cases.emplace_back(numprocesses, numthreads, numiters, device,
                                                 models[0], models_names[0],
                                                 models[0], models_names[0],
                                                 compilation_configs.empty() ? "" : compilation_configs[0],
                                                 compilation_configs.empty() ? "" : compilation_configs[0]);
                    }
                }
            }
        }
    }
    return tests_cases;
}

std::string getTestCaseName(const testing::TestParamInfo<TestCase> &obj) {
    return obj.param.test_case_name;
}

std::string getTestCaseNameMemLeaks(const testing::TestParamInfo<MemLeaksTestCase> &obj) {
    return obj.param.test_case_name;
}

std::string getMultiModelTestCaseName(const testing::TestParamInfo<MultiModelTestCase> &obj) {
    return obj.param.test_case_name;
}

void test_wrapper(const std::function<void(std::string, std::string, int)> &tests_pipeline,
                  const TestCase &params) {
    tests_pipeline(params.model, params.device, params.numiters);
}

void test_wrapper(const std::function<void(std::string, std::string, int, std::string)> &tests_pipeline,
                  const TestCase &params) {
    tests_pipeline(params.model, params.device, params.numiters, params.compilation_config_file);
}

void _runTest(const std::function<void(std::string, std::string, int)> &tests_pipeline, const TestCase &params) {
    run_in_threads(params.numthreads, static_cast<void(*)(const std::function<void(std::string, std::string, int)>&, const TestCase&)>(test_wrapper), tests_pipeline, params);
}

void _runTest(const std::function<void(std::string, std::string, int, std::string)> &tests_pipeline, const TestCase &params) {
    run_in_threads(params.numthreads, static_cast<void(*)(const std::function<void(std::string, std::string, int, std::string)>&, const TestCase&)>(test_wrapper), tests_pipeline, params);
}

void runTest(const std::function<void(std::string, std::string, int)> &tests_pipeline, const TestCase &params) {
#if DEBUG_MODE
    tests_pipeline(params.model, params.device, params.numiters);
#else
    int status = run_in_processes(params.numprocesses, [&]() { _runTest(tests_pipeline, params); });
    ASSERT_EQ(status, 0) << "Test failed with exitcode " << std::to_string(status);
#endif
}

void runTest(const std::function<void(std::string, std::string, int, std::string)> &tests_pipeline, const TestCase &params) {
#if DEBUG_MODE
    tests_pipeline(params.model, params.device, params.numiters, params.compilation_config_file);
#else
    int status = run_in_processes(params.numprocesses, [&]() { _runTest(tests_pipeline, params); });
    ASSERT_EQ(status, 0) << "Test failed with exitcode " << std::to_string(status);
#endif
}

void runStressTest(const std::string& scenario, const TestCase& params) {
    std::vector<std::string> arguments = {get_executable_path(),
                                          "--stress_child",
                                          "--stress_scenario=" + scenario,
                                          "--stress_model=" + params.model,
                                          "--stress_device=" + params.device,
                                          "--stress_iterations=" + std::to_string(params.numiters),
                                          "--stress_threads=" + std::to_string(params.numthreads)};
    if (!params.compilation_config_file.empty()) {
        arguments.push_back("--stress_compilation_config=" + params.compilation_config_file);
    }
    const int status = run_in_processes_exec(params.numprocesses, arguments);
    ASSERT_EQ(status, 0) << "Test failed with exitcode " << std::to_string(status);
}

void runMultiModelStressTest(const std::string& scenario, const MultiModelTestCase& params) {
    std::vector<std::string> arguments = {get_executable_path(),
                                          "--stress_child",
                                          "--stress_scenario=" + scenario,
                                          "--stress_model=" + params.model1,
                                          "--stress_model2=" + params.model2,
                                          "--stress_device=" + params.device,
                                          "--stress_iterations=" + std::to_string(params.numiters),
                                          "--stress_threads=" + std::to_string(params.numthreads)};
    if (!params.compilation_config_file.empty()) {
        arguments.push_back("--stress_compilation_config=" + params.compilation_config_file);
    }
    if (!params.compilation_config_file2.empty()) {
        arguments.push_back("--stress_compilation_config2=" + params.compilation_config_file2);
    }
    const int status = run_in_processes_exec(params.numprocesses, arguments);
    ASSERT_EQ(status, 0) << "Test failed with exitcode " << std::to_string(status);
}

void runMultiModelProcessesStressTest(const MultiModelTestCase& params) {
    std::vector<std::vector<std::string>> process_arguments;
    const int procs = std::max(2, params.numprocesses);
    for (int i = 0; i < procs; ++i) {
        const std::string& model = (i % 2 == 0) ? params.model1 : params.model2;
        const std::string& config = (i % 2 == 0) ? params.compilation_config_file :
                                    (!params.compilation_config_file2.empty() ? params.compilation_config_file2 : params.compilation_config_file);
        std::vector<std::string> args = {
            get_executable_path(),
            "--stress_child",
            "--stress_scenario=stress_parallel_infer",
            "--stress_model=" + model,
            "--stress_device=" + params.device,
            "--stress_iterations=" + std::to_string(params.numiters),
            "--stress_threads=" + std::to_string(params.numthreads)
        };
        if (!config.empty()) {
            args.push_back("--stress_compilation_config=" + config);
        }
        process_arguments.push_back(args);
    }
    const int status = run_in_processes_exec_multi(process_arguments);
    ASSERT_EQ(status, 0) << "Test failed with exitcode " << std::to_string(status);
}
