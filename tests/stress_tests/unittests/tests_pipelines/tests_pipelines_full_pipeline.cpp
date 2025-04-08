// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tests_pipelines.h"

#include <string>
#include <openvino/core/preprocess/pre_post_process.hpp>


#define batchIndex 0

#define setInputParameters()                                                        \
    input.second->getPreProcess().setResizeAlgorithm(NO_RESIZE);                    \
    input.second->setPrecision(Precision::U8);                                      \
    if (input.second->getInputData()->getTensorDesc().getDims().size() == 4)        \
        input.second->setLayout(Layout::NCHW);                                      \
    else if (input.second->getInputData()->getTensorDesc().getDims().size() == 2)   \
        input.second->setLayout(Layout::NC);

#define computeShapesToReshape()                                \
    auto layout = input.second->getTensorDesc().getLayout();    \
    if ((layout == Layout::NCHW) || (layout == Layout::NC)) {   \
        shapes[input.first][batchIndex] *= 2;                   \
        doReshape = true;                                       \
    }

#define reshapeCNNNetwork()                                             \
    if (doReshape)                                                      \
        cnnNetwork.reshape(shapes);                                     \
    else                                                                \
        throw std::logic_error("Reshape wasn't applied for a model.");

void test_load_unload_plugin_full_pipeline(const std::string &model, const std::string &target_device, const int &n) {
    log_info("Load/unload plugin for device: " << target_device << " for " << n << " times");
    auto ie_api_wrapper = create_infer_api_wrapper();
    for (int i = 0; i < n; i++) {
        if (i == n / 2) {
            log_info("Half of the test have already passed");
        }
        // get_versions silently register plugin in `plugins` through `get_plugin`
        ie_api_wrapper->load_plugin(target_device);
        // Remove plugin for target_device from `plugins`
        ie_api_wrapper->unload_plugin(target_device);
    }
    ie_api_wrapper->read_network(model);
    ie_api_wrapper->load_network(target_device);
    ie_api_wrapper->create_infer_request();
    ie_api_wrapper->prepare_input();
    ie_api_wrapper->infer();
}

void test_read_network_full_pipeline(const std::string &model, const std::string &target_device, const int &n) {
    auto ie_api_wrapper = create_infer_api_wrapper();
    log_info("Read network: \"" << model << "\" for " << n << " times");
    for (int i = 0; i < n; i++) {
        if (i == n / 2) {
            log_info("Half of the test have already passed");
        }
        ie_api_wrapper->read_network(model);
    }
    ie_api_wrapper->load_network(target_device);
    ie_api_wrapper->create_infer_request();
    ie_api_wrapper->prepare_input();
    ie_api_wrapper->infer();
}

void test_set_input_params_full_pipeline(const std::string &model, const std::string &target_device, const int &n) {
    auto ie_api_wrapper = create_infer_api_wrapper();
    log_info("Apply preprocessing for CNNNetwork from network: \"" << model << "\" for " << n << " times");
    for (int i = 0; i < n; i++) {
        if (i == n / 2) {
            log_info("Half of the test have already passed");
        }
        ie_api_wrapper->set_input_params(model);
    }
    ie_api_wrapper->load_network(target_device);
    ie_api_wrapper->create_infer_request();
    ie_api_wrapper->prepare_input();
    ie_api_wrapper->infer();
}

void test_cnnnetwork_reshape_batch_x2_full_pipeline(const std::string &model, const std::string &target_device,
                                                    const int &n) {
    auto ie_api_wrapper = create_infer_api_wrapper();
    log_info("Reshape to batch*=2 of CNNNetwork created from network: \"" << model << "\" for " << n << " times");
    ie_api_wrapper->read_network(model);
    for (int i = 0; i < n; i++) {
        if (i == n / 2) {
            log_info("Half of the test have already passed");
        }
        ie_api_wrapper->change_batch_size(2, i);
    }
    ie_api_wrapper->load_network(target_device);
    ie_api_wrapper->create_infer_request();
    ie_api_wrapper->prepare_input();
    ie_api_wrapper->infer();
}

void test_create_exenetwork_full_pipeline(const std::string &model, const std::string &target_device, const int &n) {
    auto ie_api_wrapper = create_infer_api_wrapper();
    log_info("Create ExecutableNetwork from network: \"" << model
                                                         << "\" for device: \"" << target_device << "\" for " << n
                                                         << " times");
    ie_api_wrapper->read_network(model);
    for (int i = 0; i < n; i++) {
        if (i == n / 2) {
            log_info("Half of the test have already passed");
        }
        ie_api_wrapper->load_network(target_device);
    }
    ie_api_wrapper->create_infer_request();
    ie_api_wrapper->prepare_input();
    ie_api_wrapper->infer();
}

void test_create_infer_request_full_pipeline(const std::string &model, const std::string &target_device, const int &n) {
    auto ie_api_wrapper = create_infer_api_wrapper();
    log_info("Create InferRequest from network: \"" << model
                                                    << "\" for device: \"" << target_device << "\" for " << n
                                                    << " times");
    ie_api_wrapper->read_network(model);
    ie_api_wrapper->load_network(target_device);
    for (int i = 0; i < n; i++) {
        if (i == n / 2) {
            log_info("Half of the test have already passed");
        }
        ie_api_wrapper->create_infer_request();
        ie_api_wrapper->prepare_input();
    }
    ie_api_wrapper->infer();
}


void test_infer_request_inference_full_pipeline(const std::string &model, const std::string &target_device,
                                                const int &n) {
    auto ie_api_wrapper = create_infer_api_wrapper();
    log_info("Inference of InferRequest from network: \"" << model
                                                          << "\" for device: \"" << target_device << "\" for " << n
                                                          << " times");
    ie_api_wrapper->read_network(model);
    ie_api_wrapper->load_network(target_device);
    ie_api_wrapper->create_infer_request();
    ie_api_wrapper->prepare_input();
    for (int i = 0; i < n; i++) {
        if (i == n / 2) {
            log_info("Half of the test have already passed");
        }
        ie_api_wrapper->infer();
    }
}
