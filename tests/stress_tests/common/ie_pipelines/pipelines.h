// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <functional>
#include <inference_engine.hpp>

std::function<void()> load_unload_plugin(const std::string &target_device, const int &api_version);
std::function<void()> read_cnnnetwork(const std::string &model, const int &api_version);
std::function<void()> cnnnetwork_reshape_batch_x2(const std::string &model, const int &api_version);
std::function<void()> set_input_params(const std::string &model, const int &api_version);
std::function<void()> create_compiled_model(const std::string &model, const std::string &target_device, const int &api_version);
std::function<void()> recreate_exenetwork(InferenceEngine::Core &ie, const std::string &model, const std::string &target_device);
std::function<void()> recreate_compiled_model(ov::runtime::Core &ie, const std::string &model, const std::string &target_device);
std::function<void()> create_infer_request(const std::string &model, const std::string &target_device, const int &api_version);
std::function<void()> recreate_infer_request(InferenceEngine::ExecutableNetwork& exeNetwork);
std::function<void()> recreate_infer_request(ov::runtime::CompiledModel& compiled_model);
std::function<void()> infer_request_inference(const std::string &model, const std::string &target_device, const int &api_version);
std::function<void()> reinfer_request_inference(InferenceEngine::InferRequest& infer_request, InferenceEngine::OutputsDataMap& output_info);
std::function<void()> reinfer_request_inference(ov::runtime::InferRequest& infer_request, std::vector<ov::Output<ov::Node>>& output_info);
std::function<void()> inference_with_streams(const std::string &model, const std::string &target_device, const int& nstreams, const int &api_version);
