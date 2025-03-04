// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tests_pipelines.h"

#include <string>

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
        read_cnnnetwork(model)();
    }
}

void test_cnnnetwork_reshape_batch_x2(const std::string &model, const std::string &target_device, const int &n) {
    log_info("Reshape to batch*=2 of CNNNetwork created from network: \"" << model << "\" for " << n << " times");
    for (int i = 0; i < n; i++) {
        if (i == n / 2) {
            log_info("Half of the test have already passed");
        }
        cnnnetwork_reshape_batch_x2(model, i)();
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

void test_create_compiled_model(const std::string &model, const std::string &target_device, const int &n) {
    log_info("Create ExecutableNetwork from network: \"" << model
                                                         << "\" for device: \"" << target_device << "\" for " << n
                                                         << " times");
    for (int i = 0; i < n; i++) {
        if (i == n / 2) {
            log_info("Half of the test have already passed");
        }
        create_compiled_model(model, target_device)();
    }
}

void test_create_infer_request(const std::string &model, const std::string &target_device, const int &n) {
    log_info("Create InferRequest from network: \"" << model
                                                    << "\" for device: \"" << target_device << "\" for " << n
                                                    << " times");
    for (int i = 0; i < n; i++) {
        if (i == n / 2) {
            log_info("Half of the test have already passed");
        }
        create_infer_request(model, target_device)();
    }
}

void test_infer_request_inference(const std::string &model, const std::string &target_device, const int &n) {
    log_info("Inference of InferRequest from network: \"" << model
                                                          << "\" for device: \"" << target_device << "\" for " << n
                                                          << " times");
    for (int i = 0; i < n; i++) {
        if (i == n / 2) {
            log_info("Half of the test have already passed");
        }
        infer_request_inference(model, target_device)();
    }
}

static void test_recreate_and_infer_in_thread_one_model(const std::string &model, const std::string &target_device, const int &n, const bool &async) {
    auto ie_wrapper = create_infer_api_wrapper();
    ie_wrapper->read_network(model);
    ie_wrapper->set_config(target_device, ov::AnyMap{ov::inference_num_threads(2)});
    ie_wrapper->load_network(target_device);
    auto fun = recreate_and_infer_in_thread(ie_wrapper, async);
    for(int y = 0; y < n; y++) {
        std::vector<std::thread> threads;
        for (int i = 0; i < 4; i++) {
            ie_wrapper->create_and_infer(async);
            threads.emplace_back(fun);
        }
        for (int i = 0; i < 4; i++) {
            threads[i].join();
        }
    }
}

static void test_recreate_and_infer_in_thread_two_model(const std::string &model, const std::string &target_device, const int &n) {
    std::vector<std::shared_ptr<InferApiBase>> ie_wrapper_vector;
    for(int i = 0; i < 2; i++) {
        auto ie_wrapper = create_infer_api_wrapper();
        ie_wrapper->read_network(model);
        ie_wrapper->set_config(target_device, ov::AnyMap{ov::inference_num_threads(2)});
        ie_wrapper->load_network(target_device);
        ie_wrapper_vector.push_back(ie_wrapper);
    }
    auto async_func0 = recreate_and_infer_in_thread(ie_wrapper_vector[0], true);
    auto async_func1 = recreate_and_infer_in_thread(ie_wrapper_vector[1], true);
    auto sync_func0 = recreate_and_infer_in_thread(ie_wrapper_vector[0], false);
    auto sync_func1 = recreate_and_infer_in_thread(ie_wrapper_vector[1], false);

    for(int y = 0; y < n; y++) {
        std::vector<std::thread> threads;
        for (int i = 0; i < 4; i++) {
            if ( i % 2 == 0) {
                ie_wrapper_vector[0]->create_and_infer(true);
                threads.emplace_back(async_func0);
            } else {
                ie_wrapper_vector[0]->create_and_infer(false);
                threads.emplace_back(sync_func0);
            }
        }
        for (int i = 0; i < 4; i++) {
            if ( i % 2 == 0) {
                ie_wrapper_vector[1]->create_and_infer(true);
                threads.emplace_back(async_func1);
            } else {
                ie_wrapper_vector[1]->create_and_infer(false);
                threads.emplace_back(sync_func1);
            }
        }
        for (size_t i = 0; i < threads.size(); i++) {
            threads[i].join();
        }
    }
}

void test_recreate_and_infer_in_thread(const std::string &model, const std::string &target_device, const int &n) {
    test_recreate_and_infer_in_thread_one_model(model, target_device, n, false);
    test_recreate_and_infer_in_thread_one_model(model, target_device, n, true);
    test_recreate_and_infer_in_thread_two_model(model, target_device, n);
}

