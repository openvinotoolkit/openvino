// Copyright (C) 2018-2022 Intel Corporation
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
    if (api_version == 1) {
        return [&] {
            InferenceEngine::Core ie;
            // GetVersions silently register plugin in `plugins` through `GetCPPPluginByName`
            ie.GetVersions(target_device);
            // Remove plugin for target_device from `plugins`
            ie.UnregisterPlugin(target_device);
        };
    }
    else {
        return [&] {
            ov::Core ie;
            // get_versions silently register plugin in `plugins` through `GetCPPPluginByName`
            ie.get_versions(target_device);
            // Remove plugin for target_device from `plugins`
            ie.unload_plugin(target_device);
        };
    }
}

std::function<void()> read_cnnnetwork(const std::string &model, const int &api_version) {
    if (api_version == 1) {
        return [&] {
            InferenceEngine::Core ie;
            InferenceEngine::CNNNetwork cnnNetwork = ie.ReadNetwork(model);
        };
    }
    else {
        return [&] {
            ov::Core ie;
            std::shared_ptr<ov::Model> network = ie.read_model(model);
        };
    }
}

std::function<void()> cnnnetwork_reshape_batch_x2(const std::string &model, const int &api_version) {
    if (api_version == 1) {
        return [&] {
            InferenceEngine::Core ie;
            InferenceEngine::CNNNetwork cnnNetwork = ie.ReadNetwork(model);
            const InferenceEngine::InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
            InferenceEngine::ICNNNetwork::InputShapes shapes = cnnNetwork.getInputShapes();
            bool doReshape = false;
            for (const InferenceEngine::InputsDataMap::value_type &input: inputInfo) {
                int batchIndex = -1;
                auto layout = input.second->getTensorDesc().getLayout();
                if ((layout == Layout::NCHW) || (layout == Layout::NCDHW) ||
                    (layout == Layout::NHWC) || (layout == Layout::NDHWC) ||
                    (layout == Layout::NC)) {
                    batchIndex = 0;
                } else if (layout == CN) {
                    batchIndex = 1;
                }
                if (batchIndex != -1) {
                    shapes[input.first][batchIndex] *= 2;
                    doReshape = true;
                }
            }
            if (doReshape)
                cnnNetwork.reshape(shapes);
            else
                throw std::logic_error("Reshape wasn't applied for a model.");
        };
    }
    else {
        return [&] {
            ov::Core ie;
            auto network = ie.read_model(model);
            auto inputs = network->inputs();

            for (auto &input: inputs) {
                auto tensor_shape = input.get_shape();
                auto model_layout = ov::layout::get_layout(input);
                tensor_shape[0] *= 2;
                network->reshape({{input.get_any_name(), tensor_shape}});
            }
        };
    }
}

std::function<void()> set_input_params(const std::string &model, const int &api_version) {
    if (api_version == 1) {
        return [&] {
            InferenceEngine::Core ie;
            InferenceEngine::CNNNetwork cnnNetwork = ie.ReadNetwork(model);
            InferenceEngine::InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
            for (auto &input: inputInfo) {
                input.second->getPreProcess().setResizeAlgorithm(NO_RESIZE);
                input.second->setPrecision(Precision::U8);
                if (input.second->getInputData()->getTensorDesc().getDims().size() == 4)
                    input.second->setLayout(Layout::NCHW);
                else if (input.second->getInputData()->getTensorDesc().getDims().size() == 2)
                    input.second->setLayout(Layout::NC);
                else
                    throw std::logic_error("Setting of input parameters wasn't applied for a model.");
            }
        };
    }
    else {
        return [&] {
            ov::Core ie;
            auto network = ie.read_model(model);
            auto inputs = network->inputs();
            auto ppp = ov::preprocess::PrePostProcessor(network);
            for (size_t i = 0; i < inputs.size(); ++i) {
                auto &input_info = ppp.input(i);
                if (inputs[i].get_shape().size() == 4) {
                    input_info.tensor().set_element_type(ov::element::u8).set_layout("NCHW");
                    input_info.model().set_layout("NCHW");
                }
                else if (inputs[i].get_shape().size() == 2) {
                    input_info.tensor().set_element_type(ov::element::u8).set_layout("NCHW");
                    input_info.model().set_layout("NC");
                }
                else {
                    throw std::logic_error("Setting of input parameters wasn't applied for a model.");
                }
                ppp.input(i).preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
            }
            network = ppp.build();
        };
    }
}

std::function<void()> create_compiled_model(const std::string &model, const std::string &target_device, const int &api_version) {
    if (api_version == 1) {
        return [&] {
            InferenceEngine::Core ie;
            InferenceEngine::CNNNetwork cnnNetwork = ie.ReadNetwork(model);
            InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, target_device);
        };
    }
    else {
        return [&] {
            ov::Core ie;
            std::shared_ptr<ov::Model> network = ie.read_model(model);
            auto compiled_model = ie.compile_model(network, target_device);
        };
    }
}

std::function<void()> recreate_exenetwork(InferenceEngine::Core &ie, const std::string &model, const std::string &target_device) {
        return [&] {
            InferenceEngine::CNNNetwork cnnNetwork = ie.ReadNetwork(model);
            InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, target_device);
        };
}

std::function<void()> recreate_compiled_model(ov::Core &ie, const std::string &model, const std::string &target_device) {
        return [&] {
            ie.get_versions(target_device);
            auto network = ie.read_model(model);
            auto compiled_model = ie.compile_model(network, target_device);
        };
}

std::function<void()> create_infer_request(const std::string &model, const std::string &target_device, const int &api_version) {
    if (api_version == 1) {
        return [&] {
            InferenceEngine::Core ie;
            InferenceEngine::CNNNetwork cnnNetwork = ie.ReadNetwork(model);
            InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, target_device);
            InferenceEngine::InferRequest infer_request = exeNetwork.CreateInferRequest();
        };
    }
    else {
        return [&] {
            ov::Core ie;
            auto network = ie.read_model(model);
            auto compiled_model = ie.compile_model(network, target_device);
            auto infer_request = compiled_model.create_infer_request();
        };
    }
}

std::function<void()> recreate_infer_request(InferenceEngine::ExecutableNetwork& exeNetwork) {
    return [&] {
        InferenceEngine::InferRequest infer_request = exeNetwork.CreateInferRequest();
    };
}

std::function<void()> recreate_infer_request(ov::CompiledModel& compiled_model) {
    return [&] {
        auto infer_request = compiled_model.create_infer_request();
    };
}

std::function<void()> infer_request_inference(const std::string &model, const std::string &target_device, const int &api_version) {
    if (api_version == 1) {
        return [&] {
            InferenceEngine::Core ie;
            InferenceEngine::CNNNetwork cnnNetwork = ie.ReadNetwork(model);
            InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, target_device);
            InferenceEngine::InferRequest infer_request = exeNetwork.CreateInferRequest();

            auto batchSize = cnnNetwork.getBatchSize();
            batchSize = batchSize != 0 ? batchSize : 1;
            const InferenceEngine::ConstInputsDataMap inputsInfo(exeNetwork.GetInputsInfo());
            fillBlobs(infer_request, inputsInfo, batchSize);

            infer_request.Infer();
            OutputsDataMap output_info(cnnNetwork.getOutputsInfo());
            for (auto &output: output_info)
                InferenceEngine::Blob::Ptr outputBlob = infer_request.GetBlob(output.first);
        };
    }
    else {
        return [&] {
            ov::Core ie;
            auto network = ie.read_model(model);
            auto compiled_model = ie.compile_model(network, target_device);
            auto infer_request = compiled_model.create_infer_request();
            auto inputs = network->inputs();

            fillTensors(infer_request, inputs);
            infer_request.infer();
            auto outputs = network->outputs();
            for (size_t i = 0; i < outputs.size(); ++i) {
                const auto &output_tensor = infer_request.get_output_tensor(i);
            }
        };
    }
}

std::function<void()> reinfer_request_inference(InferenceEngine::InferRequest& infer_request, InferenceEngine::OutputsDataMap& output_info) {
    return [&] {
        infer_request.Infer();
        for (auto &output : output_info)
            Blob::Ptr outputBlob = infer_request.GetBlob(output.first);
    };
}

std::function<void()> reinfer_request_inference(ov::InferRequest& infer_request, std::vector<ov::Output<ov::Node>>& output_info) {
    return [&] {
        infer_request.infer();
        for (size_t i = 0; i < output_info.size(); ++i)
            const auto &output_tensor = infer_request.get_output_tensor(i);
    };
}

std::function<void()> inference_with_streams(const std::string &model, const std::string &target_device, const int& nstreams, const int &api_version) {
    if (api_version == 1) {
        return [&] {
            std::map<std::string, std::string> config;
            config[target_device + "_THROUGHPUT_STREAMS"] = std::to_string(nstreams);

            InferenceEngine::Core ie;
            ie.GetVersions(target_device);
            ie.SetConfig(config, target_device);

            InferenceEngine::InferRequest inferRequest;

            InferenceEngine::CNNNetwork cnnNetwork = ie.ReadNetwork(model);
            InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, target_device);
            auto batchSize = cnnNetwork.getBatchSize();
            batchSize = batchSize != 0 ? batchSize : 1;
            const InferenceEngine::ConstInputsDataMap inputsInfo(exeNetwork.GetInputsInfo());

            unsigned int nireq = nstreams;
            try {
                nireq = exeNetwork.GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
            } catch (const std::exception &ex) {
                log_err("Failed to query OPTIMAL_NUMBER_OF_INFER_REQUESTS");
            }
            for (int counter = 0; counter < nireq; counter++) {
                inferRequest = exeNetwork.CreateInferRequest();
                fillBlobs(inferRequest, inputsInfo, batchSize);

                inferRequest.Infer();
                InferenceEngine::OutputsDataMap output_info(cnnNetwork.getOutputsInfo());
                for (auto &output: output_info)
                    InferenceEngine::Blob::Ptr outputBlob = inferRequest.GetBlob(output.first);
            }
        };
    }
    else {
        return [&] {
            std::map<std::string, std::string> config;
            config[target_device + "_THROUGHPUT_STREAMS"] = std::to_string(nstreams);
            ov::Core ie;
            ie.get_versions(target_device);
            ie.set_config(config, target_device);
            std::shared_ptr<ov::Model> network = ie.read_model(model);
            auto compiled_model = ie.compile_model(network, target_device);
            ov::InferRequest infer_request;
            auto inputs = network->inputs();

            unsigned int nireq = nstreams;
            try {
                nireq = compiled_model.get_metric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
            } catch (const std::exception &ex) {
                log_err("Failed to query OPTIMAL_NUMBER_OF_INFER_REQUESTS");
            }
            for (int counter = 0; counter < nireq; counter++) {
                infer_request = compiled_model.create_infer_request();
                fillTensors(infer_request, inputs);

                infer_request.infer();
                auto outputs = network->outputs();
                for (size_t i = 0; i < outputs.size(); ++i) {
                    const auto &output_tensor = infer_request.get_output_tensor(i);
                }
            }
        };
    }
}
