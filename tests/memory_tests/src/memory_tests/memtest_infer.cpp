// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <inference_engine.hpp>
#include <iostream>

#include "common_utils.h"
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
        InferenceEngine::Core ie;
        InferenceEngine::CNNNetwork cnnNetwork;
        InferenceEngine::ExecutableNetwork exeNetwork;
        InferenceEngine::InferRequest inferRequest;
        size_t batchSize = 0;

        ie.GetVersions(device);
        MEMORY_SNAPSHOT(load_plugin);

        if (MemoryTest::fileExt(model) == "blob") {
            exeNetwork = ie.ImportNetwork(model, device);
            MEMORY_SNAPSHOT(import_network);
        } else {
            cnnNetwork = ie.ReadNetwork(model);
            MEMORY_SNAPSHOT(read_network);

            exeNetwork = ie.LoadNetwork(cnnNetwork, device);

            MEMORY_SNAPSHOT(load_network);
            batchSize = cnnNetwork.getBatchSize();
        }
        MEMORY_SNAPSHOT(create_exenetwork);

        inferRequest = exeNetwork.CreateInferRequest();

        batchSize = batchSize != 0 ? batchSize : 1;
        const InferenceEngine::ConstInputsDataMap inputsInfo(exeNetwork.GetInputsInfo());
        fillBlobs(inferRequest, inputsInfo, batchSize);
        MEMORY_SNAPSHOT(fill_inputs)

        inferRequest.Infer();
        MEMORY_SNAPSHOT(first_inference);
        MEMORY_SNAPSHOT(full_run);
    };

    try {
        pipeline(model, device, reshapeShapes, dataShapes);
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
