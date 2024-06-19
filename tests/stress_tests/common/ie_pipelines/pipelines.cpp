// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pipelines.h"
#include "../utils.h"
#include "common_utils.h"

#include <iostream>
#include <string>

#include <inference_engine.hpp>
#include <openvino/openvino.hpp>


std::function<void()> load_unload_plugin(const std::string &target_device, const int &api_version) {
    return [&] {
        auto ie_api_wrapper = create_infer_api_wrapper(api_version);
        // get_versions silently register plugin in `plugins` through `get_plugin`
        ie_api_wrapper->load_plugin(target_device);
        // Remove plugin for target_device from `plugins`
        ie_api_wrapper->unload_plugin(target_device);
    };
}

std::function<void()> read_cnnnetwork(const std::string &model, const int &api_version) {
    return [&] {
        auto ie_api_wrapper = create_infer_api_wrapper(api_version);
        ie_api_wrapper->read_network(model);
    };
}

std::function<void()> cnnnetwork_reshape_batch_x2(const std::string &model, const int &iter, const int &api_version) {
    return [&] {
        auto ie_api_wrapper = create_infer_api_wrapper(api_version);
        ie_api_wrapper->read_network(model);
        ie_api_wrapper->change_batch_size(2, iter);
    };
}

std::function<void()> set_input_params(const std::string &model, const int &api_version) {
    return [&] {
        auto ie_api_wrapper = create_infer_api_wrapper(api_version);
        ie_api_wrapper->read_network(model);
        ie_api_wrapper->set_input_params(model);
    };
}

std::function<void()>
create_compiled_model(const std::string &model, const std::string &target_device, const int &api_version) {
    return [&] {
        auto ie_api_wrapper = create_infer_api_wrapper(api_version);
        ie_api_wrapper->read_network(model);
        ie_api_wrapper->load_network(target_device);
    };
}

std::function<void()> recreate_compiled_model(std::shared_ptr<InferApiBase> &ie_wrapper,
                                              const std::string &target_device, const int &api_version) {
    return [=] {
        ie_wrapper->load_network(target_device);
    };
}


std::function<void()>
create_infer_request(const std::string &model, const std::string &target_device, const int &api_version) {
    return [&] {
        auto ie_api_wrapper = create_infer_api_wrapper(api_version);
        ie_api_wrapper->read_network(model);
        ie_api_wrapper->load_network(target_device);
        ie_api_wrapper->create_infer_request();
    };
}


std::function<void()> recreate_infer_request(std::shared_ptr<InferApiBase> &ie_wrapper) {
    return [=] {
        ie_wrapper->create_infer_request();
    };
}


std::function<void()>
infer_request_inference(const std::string &model, const std::string &target_device, const int &api_version) {
    return [&] {
        auto ie_api_wrapper = create_infer_api_wrapper(api_version);
        ie_api_wrapper->read_network(model);
        ie_api_wrapper->load_network(target_device);
        ie_api_wrapper->create_infer_request();
        ie_api_wrapper->prepare_input();
        ie_api_wrapper->infer();
    };
}


std::function<void()> reinfer_request_inference(std::shared_ptr<InferApiBase> &ie_wrapper) {
    return [=] {
        ie_wrapper->infer();
    };
}

std::function<void()> recreate_and_infer_in_thread(std::shared_ptr<InferApiBase> &ie_wrapper, const bool async) {
    return [async, &ie_wrapper] {
        auto func = [&ie_wrapper, &async] {
            ie_wrapper->create_and_infer(async);
        };
        std::thread t(func);
        t.join();
    };
}

std::function<void()>
inference_with_streams(const std::string &model, const std::string &target_device, const int &nstreams,
                       const int &api_version) {
    return [&] {
        unsigned int nireq = nstreams;
        auto ie_api_wrapper = create_infer_api_wrapper(api_version);
        ie_api_wrapper->load_plugin(target_device);
        ie_api_wrapper->set_config(target_device, "THROUGHPUT_STREAMS", nstreams);
        ie_api_wrapper->read_network(model);
        ie_api_wrapper->load_network(target_device);
        try {
            nireq = ie_api_wrapper->get_property(ov::optimal_number_of_infer_requests.name());
        } catch (const std::exception &ex) {
            log_err("Failed to query OPTIMAL_NUMBER_OF_INFER_REQUESTS");
        }

        for (int counter = 0; counter < nireq; counter++) {
            ie_api_wrapper->create_infer_request();
            ie_api_wrapper->prepare_input();
            ie_api_wrapper->infer();
        }
    };
}
