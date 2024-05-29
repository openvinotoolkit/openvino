// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <functional>
#include "../infer_api/infer_api.h"

std::function<void()> load_unload_plugin(const std::string &target_device);

std::function<void()> read_cnnnetwork(const std::string &model);

std::function<void()> cnnnetwork_reshape_batch_x2(const std::string &model, const int &iter);

std::function<void()> set_input_params(const std::string &model);

std::function<void()>
create_compiled_model(const std::string &model, const std::string &target_device);

std::function<void()>
create_infer_request(const std::string &model, const std::string &target_device);

std::function<void()>
infer_request_inference(const std::string &model, const std::string &target_device);

std::function<void()>
inference_with_streams(const std::string &model, const std::string &target_device, const int &nstreams);

std::function<void()>
recreate_compiled_model(std::shared_ptr<InferApiBase> &ie_wrapper, const std::string &target_device);

std::function<void()> recreate_infer_request(std::shared_ptr<InferApiBase> &ie_wrapper);

std::function<void()> reinfer_request_inference(std::shared_ptr<InferApiBase> &ie_wrapper);

std::function<void()> recreate_and_infer_in_thread(std::shared_ptr<InferApiBase> &ie_wrapper, const bool async = false);
