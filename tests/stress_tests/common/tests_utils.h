// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "utils.h"

#include <gtest/gtest.h>
#include <pugixml.hpp>
#include <string>
#include <thread>
#include <vector>


enum TestStatus {
    TEST_NOT_STARTED = 0, TEST_FAILED, TEST_OK
};

using TestResult = std::pair<TestStatus, std::string>;

class TestCaseBase {
public:
    int numprocesses;
    int numthreads;
    int numiters;
    std::string precision;
    std::string test_case_name;
    std::string model_name;
    std::string device;
    std::string compilation_config_file;

protected:
    // Replace non-alphabetic/numeric symbols with "_" to prevent logging errors
    static std::string safe_string(const std::string &item) {
        std::string _item(item);
        for (char &ch: _item) {
            if (!isalnum(ch)) ch = '_';
        }
        return _item;
    }
};

class TestCase : public TestCaseBase {
public:
    std::string model;

    TestCase(int _numprocesses, int _numthreads, int _numiters, std::string _device,
             const std::string &_model,
             const std::string &_model_name, const std::string &_precision,
             const std::string &_compilation_config_file = "") {
        numprocesses = _numprocesses, numthreads = _numthreads, numiters = _numiters,
        device = _device, model = _model, model_name = _model_name, precision = _precision,
        compilation_config_file = _compilation_config_file;
        test_case_name = "NP" + std::to_string(numprocesses) + "_NT" + std::to_string(numthreads) +
                         "_NI" + std::to_string(numiters) + safe_string(device) +
                         safe_string(precision) + "_M" + safe_string(model_name);
        if (!compilation_config_file.empty()) {
            test_case_name += "_Config_" + safe_string(fileNameNoExt(compilation_config_file));
        }
    }
};

class MemLeaksTestCase : public TestCaseBase {
public:
    std::vector<std::map<std::string, std::string>> models;

    MemLeaksTestCase(int _numprocesses, int _numthreads, int _numiters, std::string _device,
                     std::vector<std::map<std::string, std::string>> _models,
                     const std::string &_compilation_config_file = "") {
        numprocesses = _numprocesses, numthreads = _numthreads, numiters = _numiters,
        device = _device, models = _models, compilation_config_file = _compilation_config_file;
        test_case_name = "NP" + std::to_string(numprocesses) + "_NT" + std::to_string(numthreads) +
                         "_NI" + std::to_string(numiters) + safe_string(device);
        for (size_t i = 0; i < models.size(); i++) {
            test_case_name += "_Model" + std::to_string(i + 1) + "_" + safe_string(models[i]["name"]) + "_" +
                              safe_string(models[i]["precision"]);
            model_name += "\"" + models[i]["path"] + "\"" + (i < models.size() - 1 ? ", " : "");
        }
        if (!compilation_config_file.empty()) {
            test_case_name += "_Config_" + safe_string(fileNameNoExt(compilation_config_file));
        }
    }
};

class MultiModelTestCase : public TestCaseBase {
public:
    std::string model1;
    std::string model2;
    std::string model1_name;
    std::string model2_name;
    std::string compilation_config_file2;

    MultiModelTestCase(int _numprocesses, int _numthreads, int _numiters, std::string _device,
                       const std::string &_model1, const std::string &_model1_name,
                       const std::string &_model2, const std::string &_model2_name,
                       const std::string &_compilation_config_file = "",
                       const std::string &_compilation_config_file2 = "") {
        numprocesses = _numprocesses, numthreads = _numthreads, numiters = _numiters,
        device = _device, model1 = _model1, model1_name = _model1_name,
        model2 = _model2, model2_name = _model2_name,
        compilation_config_file = _compilation_config_file,
        compilation_config_file2 = _compilation_config_file2;
        test_case_name = "NP" + std::to_string(numprocesses) + "_NT" + std::to_string(numthreads) +
                         "_NI" + std::to_string(numiters) + safe_string(device) +
                         "_M1_" + safe_string(model1_name) + "_M2_" + safe_string(model2_name);
        if (!compilation_config_file.empty()) {
            test_case_name += "_Config_" + safe_string(fileNameNoExt(compilation_config_file));
        }
    }
};

inline void PrintTo(const TestCase &param, std::ostream *os) {
    *os << "{processes: " << param.numprocesses
        << ", threads: " << param.numthreads
        << ", iterations: " << param.numiters
        << ", device: \"" << param.device << "\""
        << ", model: \"" << param.model_name << "\""
        << (param.compilation_config_file.empty() ? "" : ", compilation_config: \"" + param.compilation_config_file + "\"")
        << "}";
}

inline void PrintTo(const MemLeaksTestCase &param, std::ostream *os) {
    *os << "{processes: " << param.numprocesses
        << ", threads: " << param.numthreads
        << ", iterations: " << param.numiters
        << ", device: \"" << param.device << "\""
        << ", models: [" << param.model_name << "]"
        << (param.compilation_config_file.empty() ? "" : ", compilation_config: \"" + param.compilation_config_file + "\"")
        << "}";
}

inline void PrintTo(const MultiModelTestCase &param, std::ostream *os) {
    *os << "{processes: " << param.numprocesses
        << ", threads: " << param.numthreads
        << ", iterations: " << param.numiters
        << ", device: \"" << param.device << "\""
        << ", model1: \"" << param.model1_name << "\""
        << (param.compilation_config_file.empty() ? "" : ", compilation_config1: \"" + param.compilation_config_file + "\"")
        << ", model2: \"" << param.model2_name << "\""
        << (param.compilation_config_file2.empty() ? "" : ", compilation_config2: \"" + param.compilation_config_file2 + "\"")
        << "}";
}

class Environment {
private:
    pugi::xml_document _test_config;
    std::string _compilation_config_file;
    bool _collect_results_only = false;

    Environment() = default;

    Environment(const Environment &) = delete;

    Environment &operator=(const Environment &) = delete;

public:
    static Environment &Instance() {
        static Environment env;
        return env;
    }

    const pugi::xml_document &getTestConfig();

    void setTestConfig(const pugi::xml_document &test_config);

    const std::string &getCompilationConfigFile() const;

    void setCompilationConfigFile(const std::string &compilation_config_file);
};

std::vector<TestCase> generateTestsParams(std::initializer_list<std::string> items);

std::vector<MemLeaksTestCase> generateTestsParamsMemLeaks();

std::vector<MultiModelTestCase> generateMultiModelTestsParams();

std::string getTestCaseName(const testing::TestParamInfo<TestCase> &obj);

std::string getTestCaseNameMemLeaks(const testing::TestParamInfo<MemLeaksTestCase> &obj);

std::string getMultiModelTestCaseName(const testing::TestParamInfo<MultiModelTestCase> &obj);

void runTest(const std::function<void(std::string, std::string, int)> &tests_pipeline, const TestCase &params);

void runTest(const std::function<void(std::string, std::string, int, std::string)> &tests_pipeline, const TestCase &params);

void runStressTest(const std::string& scenario, const TestCase& params);

void runMultiModelStressTest(const std::string& scenario, const MultiModelTestCase& params);

void runMultiModelProcessesStressTest(const MultiModelTestCase& params);

void _runTest(const std::function<void(std::string, std::string, int)> &tests_pipeline, const TestCase &params);

void _runTest(const std::function<void(std::string, std::string, int, std::string)> &tests_pipeline, const TestCase &params);

void test_wrapper(const std::function<void(std::string, std::string, int)> &tests_pipeline,
                  const TestCase &params);

void test_wrapper(const std::function<void(std::string, std::string, int, std::string)> &tests_pipeline,
                  const TestCase &params);
