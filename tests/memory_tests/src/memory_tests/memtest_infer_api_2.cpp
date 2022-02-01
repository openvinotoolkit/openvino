// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/runtime/infer_request.hpp>
#include <iostream>
#include <fstream>

#include "common_utils.h"
#include "memory_tests_helper/memory_counter.h"
#include "memory_tests_helper/utils.h"
#include "openvino/runtime/core.hpp"


/**
 * @brief Function that contain executable pipeline which will be called from
 * main(). The function should not throw any exceptions and responsible for
 * handling it by itself.
 */
int runPipeline(const std::string &model, const std::string &device) {
    auto pipeline = [](const std::string &model, const std::string &device) {
        ov::Core ie;
        std::shared_ptr<ov::Model> network;
        ov::CompiledModel compiled_model;
        ov::InferRequest infer_request;

        ie.get_versions(device);
        MEMORY_SNAPSHOT(load_plugin);

        if (MemoryTest::fileExt(model) == "blob") {
            std::ifstream streamModel{model};
            compiled_model = ie.import_model(streamModel, device);
            MEMORY_SNAPSHOT(import_network);
        } else {
            network = ie.read_model(model);
            MEMORY_SNAPSHOT(read_network);

            compiled_model = ie.compile_model(network, device);

            MEMORY_SNAPSHOT(load_network);
        }
        MEMORY_SNAPSHOT(create_exenetwork);

        infer_request = compiled_model.create_infer_request();

        auto inputs = network->inputs();
        fillTensors(infer_request, inputs);
        MEMORY_SNAPSHOT(fill_inputs)

        infer_request.infer();
        MEMORY_SNAPSHOT(first_inference);
        MEMORY_SNAPSHOT(full_run);
    };

    try {
        pipeline(model, device);
    } catch (const InferenceEngine::Exception &iex) {
        std::cerr
                << "Inference Engine pipeline failed with Inference Engine exception:\n"
                << iex.what();
        return 1;
    } catch (const std::exception &ex) {
        std::cerr << "Inference Engine pipeline failed with exception:\n"
                  << ex.what();
        return 2;
    } catch (...) {
        std::cerr << "Inference Engine pipeline failed\n";
        return 3;
    }
    return 0;
}
