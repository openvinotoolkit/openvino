// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../../common/tests_utils.h"
#include "../../common/utils.h"
#include "../../common/ie_pipelines/pipelines.h"

#include <string>

#include <inference_engine.hpp>

// tests_pipelines/tests_pipelines.cpp
TestResult test_load_unload_plugin(const std::string &target_device, const int &n);
TestResult test_read_network(const std::string &model, const int &n);
TestResult test_cnnnetwork_reshape_batch_x2(const std::string &model, const int &n);
TestResult test_set_input_params(const std::string &model, const int &n);
TestResult test_recreate_exenetwork(InferenceEngine::Core &ie, const std::string &model, const std::string &target_device, const int &n);
TestResult test_create_infer_request(const std::string &model, const std::string &target_device, const int &n);
TestResult test_recreate_infer_request(InferenceEngine::ExecutableNetwork& network, const std::string &model, const std::string &target_device, const int &n);
TestResult test_infer_request_inference(const std::string &model, const std::string &target_device, const int &n);
TestResult test_reinfer_request_inference(InferenceEngine::InferRequest& infer_request, InferenceEngine::OutputsDataMap& output_info, const std::string &model, const std::string &target_device, const int &n);
// tests_pipelines/tests_pipelines.cpp
