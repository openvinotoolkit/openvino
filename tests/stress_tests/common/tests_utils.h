// Copyright (C) 2018-2025 Intel Corporation
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

protected:
    // Replace non-alphabetic/numeric symbols with "_" to prevent logging errors
    static std::string update_item_for_name(const std::string &item) {
        std::string _item(item);
        for (char &index: _item) {
            if (!isalnum(index) && index != '_') index = '_';
        }
        return _item;
    }
};

class TestCase : public TestCaseBase {
public:
    std::string model;

    TestCase(int _numprocesses, int _numthreads, int _numiters, std::string _device,
             const std::string &_model,
             const std::string &_model_name, const std::string &_precision) {
        numprocesses = _numprocesses, numthreads = _numthreads, numiters = _numiters,
        device = _device, model = _model, model_name = _model_name, precision = _precision;
        test_case_name = "Numprocesses_" + std::to_string(numprocesses) + "_Numthreads_" + std::to_string(numthreads) +
                         "_Numiters_" + std::to_string(numiters) + "_Device_" + update_item_for_name(device) +
                         "_Precision_" + update_item_for_name(precision) + "_Model_" + update_item_for_name(model_name);
    }
};

class MemLeaksTestCase : public TestCaseBase {
public:
    std::vector<std::map<std::string, std::string>> models;

    MemLeaksTestCase(int _numprocesses, int _numthreads, int _numiters, std::string _device,
                     std::vector<std::map<std::string, std::string>> _models) {
        numprocesses = _numprocesses, numthreads = _numthreads, numiters = _numiters,
        device = _device, models = _models;
        test_case_name = "Numprocesses_" + std::to_string(numprocesses) + "_Numthreads_" + std::to_string(numthreads) +
                         "_Numiters_" + std::to_string(numiters) + "_Device_" + update_item_for_name(device);
        for (size_t i = 0; i < models.size(); i++) {
            test_case_name += "_Model" + std::to_string(i + 1) + "_" + update_item_for_name(models[i]["name"]) + "_" +
                              update_item_for_name(models[i]["precision"]);
            model_name += "\"" + models[i]["path"] + "\"" + (i < models.size() - 1 ? ", " : "");
        }
    }
};

class Environment {
private:
    pugi::xml_document _test_config;
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
};

std::vector<TestCase> generateTestsParams(std::initializer_list<std::string> items);

std::vector<MemLeaksTestCase> generateTestsParamsMemLeaks();

std::string getTestCaseName(const testing::TestParamInfo<TestCase> &obj);

std::string getTestCaseNameMemLeaks(const testing::TestParamInfo<MemLeaksTestCase> &obj);

void runTest(const std::function<void(std::string, std::string, int)> &tests_pipeline, const TestCase &params);

void _runTest(const std::function<void(std::string, std::string, int)> &tests_pipeline, const TestCase &params);

void test_wrapper(const std::function<void(std::string, std::string, int)> &tests_pipeline,
                  const TestCase &params);
