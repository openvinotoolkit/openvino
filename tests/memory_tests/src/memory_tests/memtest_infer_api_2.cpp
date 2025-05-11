// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <openvino/runtime/core.hpp>

#include <fstream>

#include "common_utils.h"
#include "reshape_utils.h"
#include "memory_tests_helper/memory_counter.h"
#include "memory_tests_helper/utils.h"



/**
 * @brief Function that contain executable pipeline which will be called from
 * main(). The function should not throw any exceptions and responsible for
 * handling it by itself.
 */
int runPipeline(const std::string &model, const std::string &device,
                std::map<std::string, ov::PartialShape> reshapeShapes,
                std::map<std::string, std::vector<size_t>> dataShapes) {
    auto pipeline = [](const std::string &model, const std::string &device,
                       std::map<std::string, ov::PartialShape> reshapeShapes,
                       std::map<std::string, std::vector<size_t>> dataShapes) {
        ov::Core ie;
        std::shared_ptr<ov::Model> cnnNetwork;
        ov::CompiledModel exeNetwork;
        ov::InferRequest inferRequest;

        std::vector<ov::Output<ov::Node>> defaultInputs;

        bool reshape = false;
        if (!reshapeShapes.empty()) {
            reshape = true;
        }

        ie.get_versions(device);
        MEMORY_SNAPSHOT(load_plugin);

        if (MemoryTest::fileExt(model) == "blob") {
            std::ifstream streamModel{model};
            exeNetwork = ie.import_model(streamModel, device);
            MEMORY_SNAPSHOT(import_network);
        } else {
            cnnNetwork = ie.read_model(model);
            MEMORY_SNAPSHOT(read_network);

            if (reshape) {
                defaultInputs = getCopyOfDefaultInputs(cnnNetwork->inputs());
                cnnNetwork->reshape(reshapeShapes);
                MEMORY_SNAPSHOT(reshape);
            }

            exeNetwork = ie.compile_model(cnnNetwork, device);

            MEMORY_SNAPSHOT(load_network);
        }
        MEMORY_SNAPSHOT(create_exenetwork);

        inferRequest = exeNetwork.create_infer_request();

        std::vector<ov::Output<const ov::Node>> inputs = exeNetwork.inputs();
        if (reshape && dataShapes.empty()) {
            fillTensors(inferRequest, defaultInputs);
        } else if (reshape && !dataShapes.empty()) {
            fillTensorsWithSpecifiedShape(inferRequest, inputs, dataShapes);
        } else {
            fillTensors(inferRequest, inputs);
        }
        MEMORY_SNAPSHOT(fill_inputs);

        inferRequest.infer();
        MEMORY_SNAPSHOT(first_inference);
        MEMORY_SNAPSHOT(full_run);
    };

    try {
        pipeline(model, device, reshapeShapes, dataShapes);
    } catch (const ov::Exception &iex) {
        std::cerr
                << "OpenVINO pipeline failed with OpenVINO exception:\n"
                << iex.what();
        return 1;
    } catch (const std::exception &ex) {
        std::cerr << "OpenVINO pipeline failed with exception:\n"
                  << ex.what();
        return 2;
    } catch (...) {
        std::cerr << "OpenVINO pipeline failed\n";
        return 3;
    }
    return 0;
}
