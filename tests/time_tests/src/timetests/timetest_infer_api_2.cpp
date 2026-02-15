// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <openvino/runtime/core.hpp>

#include <fstream>

#include "common_utils.h"
#include "reshape_utils.h"
#include "timetests_helper/timer.h"
#include "timetests_helper/utils.h"


/**
 * @brief Function that contain executable pipeline which will be called from
 * main(). The function should not throw any exceptions and responsible for
 * handling it by itself.
 */
int runPipeline(const std::string &model, const std::string &device, const bool isCacheEnabled,
                const std::string &inputPrecision, const std::string &outputPrecision,
                std::map<std::string, ov::PartialShape> reshapeShapes,
                std::map<std::string, std::vector<size_t>> dataShapes) {
    auto pipeline = [](const std::string &model, const std::string &device, const bool isCacheEnabled,
                       const std::string &inputPrecision, const std::string &outputPrecision,
                       std::map<std::string, ov::PartialShape> reshapeShapes,
                       std::map<std::string, std::vector<size_t>> dataShapes) {
        ov::Core ie;
        std::shared_ptr<ov::Model> cnnNetwork;
        ov::CompiledModel exeNetwork;
        ov::InferRequest inferRequest;

        std::vector<ov::Output<ov::Node>> defaultInputs;

        ie.set_property("AUTO", ov::log::level(ov::log::Level::DEBUG));
        std::string device_prefix = device.substr(0, device.find(':'));

        bool reshape = false;
        if (!reshapeShapes.empty()) {
            reshape = true;
        }
        bool ip = false;
        if (!inputPrecision.empty()) {
            ip = true;
        }
        bool op = false;
        if (!outputPrecision.empty()) {
            op = true;
        }

         // first_inference_latency = time_to_inference + first_inference
        {
            SCOPED_TIMER(first_inference_latency);
            {
                SCOPED_TIMER(time_to_inference);
                {
                    SCOPED_TIMER(load_plugin);
                    TimeTest::setPerformanceConfig(ie, device_prefix);
                    ie.get_versions(device_prefix);

                    if (isCacheEnabled)
                        ie.set_property({ov::cache_dir("models_cache")});
                }
                {
                    SCOPED_TIMER(create_exenetwork);
                    if (!isCacheEnabled) {
                        if (TimeTest::fileExt(model) == "blob") {
                            SCOPED_TIMER(import_network);
                            std::ifstream streamModel{model};
                            exeNetwork = ie.import_model(streamModel, device);
                        }
                        else {
                            {
                                SCOPED_TIMER(read_network);
                                cnnNetwork = ie.read_model(model);
                            }
                            if (reshape) {
                                {
                                    SCOPED_TIMER(reshape);
                                    defaultInputs = getCopyOfDefaultInputs(cnnNetwork->inputs());
                                    cnnNetwork->reshape(reshapeShapes);
                                }
                            }
                            if (ip || op) {
                                auto preprocessor = ov::preprocess::PrePostProcessor(cnnNetwork);
                                if (ip) {
                                    const auto inputs = cnnNetwork->inputs();
                                    for (size_t i = 0; i < inputs.size(); i++) {
                                        preprocessor.input(i).tensor().set_element_type(ov::element::Type(inputPrecision));
                                    }
                                }
                                if (op) {
                                    const auto outputs = cnnNetwork->outputs();
                                    for (size_t i = 0; i < outputs.size(); i++) {
                                        preprocessor.output(i).tensor().set_element_type(ov::element::Type(outputPrecision));
                                    }
                                }
                                cnnNetwork = preprocessor.build();
                            }
                            {
                                SCOPED_TIMER(load_network);
                                exeNetwork = ie.compile_model(cnnNetwork, device);
                            }
                        }
                    }
                    else {
                        SCOPED_TIMER(load_network_cache);
                        exeNetwork = ie.compile_model(model, device);
                    }
                }
                inferRequest = exeNetwork.create_infer_request();
            }
            {
                SCOPED_TIMER(first_inference);
                {
                    SCOPED_TIMER(fill_inputs);
                    std::vector<ov::Output<const ov::Node>> inputs = exeNetwork.inputs();
                    if (reshape && dataShapes.empty()) {
                        fillTensors(inferRequest, defaultInputs);
                    } else if (reshape && !dataShapes.empty()) {
                        fillTensorsWithSpecifiedShape(inferRequest, inputs, dataShapes);
                    } else {
                        fillTensors(inferRequest, inputs);
                    }
                }
                inferRequest.infer();
            }
        }
    };

    try {
        pipeline(model, device, isCacheEnabled, inputPrecision, outputPrecision, reshapeShapes, dataShapes);
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
