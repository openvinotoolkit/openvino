// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tests_pipelines.h"

#include <string>

#include <inference_engine.hpp>


using namespace InferenceEngine;

void test_load_unload_plugin(const std::string &model, const std::string &target_device, const int &n) {
    log_info("Load/unload plugin for device: " << target_device << " for " << n << " times");
    for (int i = 0; i < n; i++) {
        if (i == n / 2) {
            log_info("Half of the test have already passed");
        }
        load_unload_plugin(target_device)();
    }
}

void test_read_network(const std::string &model, const std::string &target_device, const int &n) {
    log_info("Read network: \"" << model << "\" for " << n << " times");
    for (int i = 0; i < n; i++) {
        if (i == n / 2) {
            log_info("Half of the test have already passed");
        }
        read_network(model)();
    }
}

void test_create_cnnnetwork(const std::string &model, const std::string &target_device, const int &n) {
    log_info("Create CNNNetwork from network: \"" << model << "\" for " << n << " times");
    for (int i = 0; i < n; i++) {
        if (i == n / 2) {
            log_info("Half of the test have already passed");
        }
        create_cnnnetwork(model)();
    }
}

void test_cnnnetwork_reshape_batch_x2(const std::string &model, const std::string &target_device, const int &n) {
    log_info("Reshape to batch*=2 of CNNNetwork created from network: \"" << model << "\" for " << n << " times");
    for (int i = 0; i < n; i++) {
        if (i == n / 2) {
            log_info("Half of the test have already passed");
        }
        cnnnetwork_reshape_batch_x2(model)();
    }
}

void test_set_input_params(const std::string &model, const std::string &target_device, const int &n) {
    log_info("Apply preprocessing for CNNNetwork from network: \"" << model << "\" for " << n << " times");
    for (int i = 0; i < n; i++) {
        if (i == n / 2) {
            log_info("Half of the test have already passed");
        }
        set_input_params(model)();
    }
}

void test_create_exenetwork(const std::string &model, const std::string &target_device, const int &n) {
    log_info("Create ExecutableNetwork from network: \"" << model
             << "\" for device: \"" << target_device << "\" for " << n << " times");
    for (int i = 0; i < n; i++) {
        if (i == n / 2) {
            log_info("Half of the test have already passed");
        }
        create_exenetwork(model, target_device)();
    }
}

void test_create_infer_request(const std::string &model, const std::string &target_device, const int &n) {
    log_info("Create InferRequest from network: \"" << model
             << "\" for device: \"" << target_device << "\" for " << n << " times");
    for (int i = 0; i < n; i++) {
        if (i == n / 2) {
            log_info("Half of the test have already passed");
        }
        create_infer_request(model, target_device)();
    }
}

void test_infer_request_inference(const std::string &model, const std::string &target_device, const int &n) {
    log_info("Inference of InferRequest from network: \"" << model
             << "\" for device: \"" << target_device << "\" for " << n << " times");
    for (int i = 0; i < n; i++) {
        if (i == n / 2) {
            log_info("Half of the test have already passed");
        }
        infer_request_inference(model, target_device)();
    }
}
