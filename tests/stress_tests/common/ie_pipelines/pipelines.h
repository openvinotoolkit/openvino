// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <functional>
#include <inference_engine.hpp>

std::function<void()> load_unload_plugin(const std::string &target_device);
std::function<void()> create_cnnnetwork(const std::string &model);
std::function<void()> cnnnetwork_reshape_batch_x2(const std::string &model);
std::function<void()> set_input_params(const std::string &model);
std::function<void()> create_exenetwork(const std::string &model, const std::string &target_device);
std::function<void()> recreate_exenetwork(InferenceEngine::Core &ie, const std::string &model, const std::string &target_device);
std::function<void()> create_infer_request(const std::string &model, const std::string &target_device);
std::function<void()> recreate_infer_request(InferenceEngine::ExecutableNetwork& exeNetwork);
std::function<void()> infer_request_inference(const std::string &model, const std::string &target_device);
std::function<void()> infer_request_inference(const std::string &model, const std::string &target_device);
std::function<void()> reinfer_request_inference(InferenceEngine::InferRequest& infer_request, InferenceEngine::CNNNetwork& cnnNetwork);
