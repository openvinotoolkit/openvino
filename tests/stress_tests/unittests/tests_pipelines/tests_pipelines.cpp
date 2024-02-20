// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tests_pipelines.h"

#include <string>

void test_load_unload_plugin(const std::string &model, const std::string &target_device, const int &n,
                             const int &api_version) {
    log_info("Load/unload plugin for device: " << target_device << " for " << n << " times");
    for (int i = 0; i < n; i++) {
        if (i == n / 2) {
            log_info("Half of the test have already passed");
        }
        load_unload_plugin(target_device, api_version)();
    }
}

void test_read_network(const std::string &model, const std::string &target_device, const int &n, const int &api_version) {
    log_info("Read network: \"" << model << "\" for " << n << " times");
    for (int i = 0; i < n; i++) {
        if (i == n / 2) {
            log_info("Half of the test have already passed");
        }
        read_cnnnetwork(model, api_version)();
    }
}

void test_cnnnetwork_reshape_batch_x2(const std::string &model, const std::string &target_device, const int &n,
                                      const int &api_version) {
    log_info("Reshape to batch*=2 of CNNNetwork created from network: \"" << model << "\" for " << n << " times");
    for (int i = 0; i < n; i++) {
        if (i == n / 2) {
            log_info("Half of the test have already passed");
        }
        cnnnetwork_reshape_batch_x2(model, i, api_version)();
    }
}

void test_set_input_params(const std::string &model, const std::string &target_device, const int &n,
                           const int &api_version) {
    log_info("Apply preprocessing for CNNNetwork from network: \"" << model << "\" for " << n << " times");
    for (int i = 0; i < n; i++) {
        if (i == n / 2) {
            log_info("Half of the test have already passed");
        }
        set_input_params(model, api_version)();
    }
}

void test_create_compiled_model(const std::string &model, const std::string &target_device, const int &n,
                                const int &api_version) {
    log_info("Create ExecutableNetwork from network: \"" << model
                                                         << "\" for device: \"" << target_device << "\" for " << n
                                                         << " times");
    for (int i = 0; i < n; i++) {
        if (i == n / 2) {
            log_info("Half of the test have already passed");
        }
        create_compiled_model(model, target_device, api_version)();
    }
}

void test_create_infer_request(const std::string &model, const std::string &target_device, const int &n,
                               const int &api_version) {
    log_info("Create InferRequest from network: \"" << model
                                                    << "\" for device: \"" << target_device << "\" for " << n
                                                    << " times");
    for (int i = 0; i < n; i++) {
        if (i == n / 2) {
            log_info("Half of the test have already passed");
        }
        create_infer_request(model, target_device, api_version)();
    }
}

void test_infer_request_inference(const std::string &model, const std::string &target_device, const int &n,
                                  const int &api_version) {
    log_info("Inference of InferRequest from network: \"" << model
                                                          << "\" for device: \"" << target_device << "\" for " << n
                                                          << " times");
    for (int i = 0; i < n; i++) {
        if (i == n / 2) {
            log_info("Half of the test have already passed");
        }
        infer_request_inference(model, target_device, api_version)();
    }
}
