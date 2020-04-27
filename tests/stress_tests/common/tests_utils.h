// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "utils.h"

#include <gtest/gtest.h>
#include <pugixml.hpp>
#include <string>
#include <vector>
#include <thread>
#include <unistd.h>
#include <sys/wait.h>

enum TestStatus
{
    TEST_NOT_STARTED = 0,
    TEST_FAILED,
    TEST_OK
};

using TestResult = std::pair<TestStatus, std::string>;

class TestCase {
public:
    int numprocesses;
    int numthreads;
    int numiters;
    std::string device;
    std::string model_name;
    std::string model;
    std::string test_case_name;

    TestCase(int _numprocesses, int _numthreads, int _numiters, std::string _device, const std::string& _model, const std::string& _model_name) {
        numprocesses = _numprocesses, numthreads = _numthreads, numiters = _numiters, device = _device, model = _model, model_name = _model_name;
        test_case_name =
                "Numprocesses_" + std::to_string(numprocesses) + "_Numthreads_" + std::to_string(numthreads) +
                "_Numiters_" + std::to_string(numiters) + "_Device_" + update_item_for_name(device) + "_Model_" + 
                update_item_for_name(model_name);
    }

private:
    std::string update_item_for_name(const std::string &item) {
        std::string _item(item);
        for (std::string::size_type index = 0; index < _item.size(); ++index) {
            if (!isalnum(_item[index]) && _item[index] != '_')
                _item[index] = '_';
        }
        return _item;
    }
};

class Environment {
private:
    pugi::xml_document _test_config;
    pugi::xml_document _env_config;
    Environment() = default;
    Environment(const Environment&) = delete;
    Environment& operator=(const Environment&) = delete;
public:
    static Environment& Instance(){
        static Environment env;
        return env;
    }

    const pugi::xml_document & getTestConfig();
    void setTestConfig(const pugi::xml_document &test_config);
    const pugi::xml_document & getEnvConfig();
    void setEnvConfig(const pugi::xml_document &env_config);
};

std::vector<TestCase> generateTestsParams(std::initializer_list<std::string> items);
std::string getTestCaseName(const testing::TestParamInfo<TestCase> &obj);

void runTest(const std::function<void(std::string, std::string, int)> &tests_pipeline, const TestCase &params);
void _runTest(const std::function<void(std::string, std::string, int)> &tests_pipeline, const TestCase &params);
void test_wrapper(const std::function<void(std::string, std::string, int)> &tests_pipeline, const TestCase &params);
