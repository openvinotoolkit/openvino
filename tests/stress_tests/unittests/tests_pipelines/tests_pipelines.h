// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../../common/utils.h"
#include "../../common/ie_pipelines/pipelines.h"

#include <string>

// tests_pipelines/tests_pipelines.cpp
void test_load_unload_plugin(const std::string &model, const std::string &target_device, const int &n);

void test_read_network(const std::string &model, const std::string &target_device, const int &n);

void test_cnnnetwork_reshape_batch_x2(const std::string &model, const std::string &target_device, const int &n);

void test_set_input_params(const std::string &model, const std::string &target_device, const int &n);

void test_create_compiled_model(const std::string &model, const std::string &target_device, const int &n);

void test_create_infer_request(const std::string &model, const std::string &target_device, const int &n);

void test_infer_request_inference(const std::string &model, const std::string &target_device, const int &n);
void test_recreate_and_infer_in_thread(const std::string &model, const std::string &target_device, const int &n);
// tests_pipelines/tests_pipelines.cpp

// tests_pipelines/tests_pipelines_full_pipeline.cpp
void test_load_unload_plugin_full_pipeline(const std::string &model, const std::string &target_device, const int &n);

void test_read_network_full_pipeline(const std::string &model, const std::string &target_device, const int &n);

void test_set_input_params_full_pipeline(const std::string &model, const std::string &target_device, const int &n);

void test_cnnnetwork_reshape_batch_x2_full_pipeline(const std::string &model, const std::string &target_device, const int &n);

void test_create_exenetwork_full_pipeline(const std::string &model, const std::string &target_device, const int &n);

void test_create_infer_request_full_pipeline(const std::string &model, const std::string &target_device, const int &n);

void test_infer_request_inference_full_pipeline(const std::string &model, const std::string &target_device, const int &n);
// tests_pipelines/tests_pipelines_full_pipeline.cpp
