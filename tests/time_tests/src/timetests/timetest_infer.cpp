// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>

#include "common_utils.h"
#include "timetests_helper/timer.h"
#include "timetests_helper/utils.h"


/**
 * @brief Function that contain executable pipeline which will be called from
 * main(). The function should not throw any exceptions and responsible for
 * handling it by itself.
 */
int runPipeline(const std::string &model, const std::string &device, const bool isCacheEnabled,
                const std::string &, const std::string &,
                std::map<std::string, ov::PartialShape> reshapeShapes,
                std::map<std::string, std::vector<size_t>> dataShapes) {
    auto pipeline = [](const std::string &model, const std::string &device, const bool isCacheEnabled,
                       std::map<std::string, ov::PartialShape> reshapeShapes,
                       std::map<std::string, std::vector<size_t>> dataShapes) {
        InferenceEngine::Core ie;
        InferenceEngine::CNNNetwork cnnNetwork;
        InferenceEngine::ExecutableNetwork exeNetwork;
        InferenceEngine::InferRequest inferRequest;
        size_t batchSize = 0;

        std::string device_prefix = device.substr(0, device.find(':'));

        // first_inference_latency = time_to_inference + first_inference
        {
            SCOPED_TIMER(first_inference_latency);
            {
                SCOPED_TIMER(time_to_inference);
                {
                    SCOPED_TIMER(load_plugin);
                    TimeTest::setPerformanceConfig(ie, device_prefix);
                    ie.GetVersions(device_prefix);

                    if (isCacheEnabled)
                        ie.SetConfig({ {CONFIG_KEY(CACHE_DIR), "models_cache"} });
                }
                {
                    SCOPED_TIMER(create_exenetwork);
                    if (!isCacheEnabled) {
                        if (TimeTest::fileExt(model) == "blob") {
                            SCOPED_TIMER(import_network);
                            exeNetwork = ie.ImportNetwork(model, device);
                        }
                        else {
                            {
                                SCOPED_TIMER(read_network);
                                cnnNetwork = ie.ReadNetwork(model);
                                batchSize = cnnNetwork.getBatchSize();
                            }
                            {
                                SCOPED_TIMER(load_network);
                                exeNetwork = ie.LoadNetwork(cnnNetwork, device);
                            }
                        }
                    }
                    else {
                        SCOPED_TIMER(load_network_cache);
                        exeNetwork = ie.LoadNetwork(model, device);
                    }
                }
                inferRequest = exeNetwork.CreateInferRequest();
            }
            {
                SCOPED_TIMER(first_inference);
                {
                    SCOPED_TIMER(fill_inputs);
                    const InferenceEngine::ConstInputsDataMap inputsInfo(exeNetwork.GetInputsInfo());
                    batchSize = batchSize != 0 ? batchSize : 1;
                    fillBlobs(inferRequest, inputsInfo, batchSize);
                }
                inferRequest.Infer();
            }
        }
    };

    try {
        pipeline(model, device, isCacheEnabled, reshapeShapes, dataShapes);
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
