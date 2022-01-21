// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tests_pipelines.h"
#include "common_utils.h"

#include <string>

#include <inference_engine.hpp>
#include <openvino/runtime/core.hpp>
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

void test_load_unload_plugin_full_pipeline(const std::string &model, const std::string &target_device, const int &n, const int &api_version) {
    if (api_version == 1) {
        log_info("Load/unload plugin for device: " << target_device << " for " << n << " times");
        InferenceEngine::Core ie;
        for (int i = 0; i < n; i++) {
            if (i == n / 2) {
                log_info("Half of the test have already passed");
            }
            // GetVersions silently register plugin in `plugins` through `GetCPPPluginByName`
            ie.GetVersions(target_device);
            // Remove plugin for target_device from `plugins`
            ie.UnregisterPlugin(target_device);
        }
        InferenceEngine::CNNNetwork cnnNetwork = ie.ReadNetwork(model);
        InferenceEngine::InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
        InferenceEngine::ICNNNetwork::InputShapes shapes = cnnNetwork.getInputShapes();
        bool doReshape = false;
        for (auto &input: inputInfo) {
            setInputParameters();
            computeShapesToReshape();
        }
        reshapeCNNNetwork();
        InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, target_device);
        InferenceEngine::InferRequest infer_request = exeNetwork.CreateInferRequest();

        auto batchSize = cnnNetwork.getBatchSize();
        batchSize = batchSize != 0 ? batchSize : 1;
        const InferenceEngine::ConstInputsDataMap inputsInfo(exeNetwork.GetInputsInfo());
        fillBlobs(infer_request, inputsInfo, batchSize);

        infer_request.Infer();
        InferenceEngine::OutputsDataMap output_info(cnnNetwork.getOutputsInfo());
        for (auto &output: output_info) {
            InferenceEngine::Blob::Ptr outputBlob = infer_request.GetBlob(output.first);
        }
    }
    else {
        log_info("Load/unload plugin for device: " << target_device << " for " << n << " times");
        ov::runtime::Core ie;
        for (int i = 0; i < n; i++) {
            if (i == n / 2) {
                log_info("Half of the test have already passed");
            }
            // get_versions silently register plugin in `plugins` through `GetCPPPluginByName`
            ie.get_versions(target_device);
            // Remove plugin for target_device from `plugins`
            ie.unload_plugin(target_device);
        }
        std::shared_ptr<ov::Model> network = ie.read_model(model);
        ov::runtime::CompiledModel compiled_model = ie.compile_model(network, target_device);
        ov::runtime::InferRequest infer_request = compiled_model.create_infer_request();
        std::vector<ov::Output<ov::Node>> inputs = network->inputs();

        fillTensors(infer_request, inputs);
        infer_request.infer();
        auto outputs = network->outputs();
        for (size_t i = 0; i < outputs.size(); ++i) {
            const auto &output_tensor = infer_request.get_output_tensor(i);
        }
    }
}

void test_read_network_full_pipeline(const std::string &model, const std::string &target_device, const int &n,
                                     const int &api_version) {
    if (api_version == 1) {
        log_info("Read network: \"" << model << "\" for " << n << " times");
        InferenceEngine::Core ie;
        InferenceEngine::CNNNetwork cnnNetwork;
        for (int i = 0; i < n; i++) {
            if (i == n / 2) {
                log_info("Half of the test have already passed");
            }
            cnnNetwork = ie.ReadNetwork(model);
        }
        InferenceEngine::InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
        InferenceEngine::ICNNNetwork::InputShapes shapes = cnnNetwork.getInputShapes();
        bool doReshape = false;
        for (auto &input: inputInfo) {
            setInputParameters();
            computeShapesToReshape();
        }
        reshapeCNNNetwork();
        InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, target_device);
        InferenceEngine::InferRequest infer_request = exeNetwork.CreateInferRequest();

        auto batchSize = cnnNetwork.getBatchSize();
        batchSize = batchSize != 0 ? batchSize : 1;
        const InferenceEngine::ConstInputsDataMap inputsInfo(exeNetwork.GetInputsInfo());
        fillBlobs(infer_request, inputsInfo, batchSize);

        infer_request.Infer();
        InferenceEngine::OutputsDataMap output_info(cnnNetwork.getOutputsInfo());
        for (auto &output: output_info)
            InferenceEngine::Blob::Ptr outputBlob = infer_request.GetBlob(output.first);
    }
    else {
        log_info("Read network: \"" << model << "\" for " << n << " times");
        ov::runtime::Core ie;
        std::shared_ptr<ov::Model> network;
        for (int i = 0; i < n; i++) {
            if (i == n / 2) {
                log_info("Half of the test have already passed");
            }
            network = ie.read_model(model);
        }
        ov::runtime::CompiledModel compiled_model = ie.compile_model(network, target_device);
        ov::runtime::InferRequest infer_request = compiled_model.create_infer_request();
        std::vector<ov::Output<ov::Node>> inputs = network->inputs();

        fillTensors(infer_request, inputs);
        infer_request.infer();
        auto outputs = network->outputs();
        for (size_t i = 0; i < outputs.size(); ++i) {
            const auto &output_tensor = infer_request.get_output_tensor(i);
        }
    }
}

void test_set_input_params_full_pipeline(const std::string &model, const std::string &target_device, const int &n, const int &api_version) {
    if (api_version == 1) {
        log_info("Apply preprocessing for CNNNetwork from network: \"" << model << "\" for " << n << " times");
        InferenceEngine::Core ie;
        InferenceEngine::CNNNetwork cnnNetwork = ie.ReadNetwork(model);
        InferenceEngine::InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
        for (int i = 0; i < n; i++) {
            if (i == n / 2) {
                log_info("Half of the test have already passed");
            }
            for (auto &input: inputInfo) {
                setInputParameters();
            }
        }
        InferenceEngine::ICNNNetwork::InputShapes shapes = cnnNetwork.getInputShapes();
        bool doReshape = false;
        for (auto &input: inputInfo) {
            computeShapesToReshape();
        }
        reshapeCNNNetwork();
        InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, target_device);
        InferenceEngine::InferRequest infer_request = exeNetwork.CreateInferRequest();

        auto batchSize = cnnNetwork.getBatchSize();
        batchSize = batchSize != 0 ? batchSize : 1;
        const InferenceEngine::ConstInputsDataMap inputsInfo(exeNetwork.GetInputsInfo());
        fillBlobs(infer_request, inputsInfo, batchSize);

        infer_request.Infer();
        InferenceEngine::OutputsDataMap output_info(cnnNetwork.getOutputsInfo());
        for (auto &output: output_info)
            InferenceEngine::Blob::Ptr outputBlob = infer_request.GetBlob(output.first);
    }
    else {
        log_info("Apply preprocessing for CNNNetwork from network: \"" << model << "\" for " << n << " times");
        ov::runtime::Core ie;
        std::shared_ptr<ov::Model> network = ie.read_model(model);

        std::vector<ov::Output<ov::Node>> inputs = network->inputs();
        ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(network);
        for (int i = 0; i < n; i++) {
            if (i == n / 2) {
                log_info("Half of the test have already passed");
            }
            for (size_t input_index = 0; input_index < inputs.size(); ++input_index) {
                ov::preprocess::InputInfo& input_info = ppp.input(input_index);
                if (inputs[input_index].get_shape().size() == 4) {
                    input_info.tensor().set_element_type(ov::element::u8).set_layout("NCHW");
                    input_info.model().set_layout("NCHW");
                }
                else if (inputs[input_index].get_shape().size() == 2) {
                    input_info.tensor().set_element_type(ov::element::u8).set_layout("NCHW");
                    input_info.model().set_layout("NC");
                }
                else {
                    throw std::logic_error("Setting of input parameters wasn't applied for a model.");
                }
                ppp.input(input_index).preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
            }
        }
        network = ppp.build();
        ov::runtime::CompiledModel compiled_model = ie.compile_model(network, target_device);
        ov::runtime::InferRequest infer_request = compiled_model.create_infer_request();

        fillTensors(infer_request, network->inputs());
        infer_request.infer();
        auto outputs = network->outputs();
        for (size_t i = 0; i < outputs.size(); ++i) {
            const auto &output_tensor = infer_request.get_output_tensor(i);
        }
    }
}

void test_cnnnetwork_reshape_batch_x2_full_pipeline(const std::string &model, const std::string &target_device,
                                                    const int &n, const int &api_version) {
    if (api_version == 1) {
        log_info("Reshape to batch*=2 of CNNNetwork created from network: \"" << model << "\" for " << n << " times");
        InferenceEngine::Core ie;
        InferenceEngine::CNNNetwork cnnNetwork = ie.ReadNetwork(model);
        InferenceEngine::InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
        for (auto &input: inputInfo) {
            setInputParameters();
        }
        InferenceEngine::ICNNNetwork::InputShapes shapes = cnnNetwork.getInputShapes();
        bool doReshape = false;
        int prev_batch = -1, new_batch;
        for (auto &input: inputInfo) {
            auto layout = input.second->getTensorDesc().getLayout();
            if ((layout == Layout::NCHW) || (layout == Layout::NC))
                prev_batch = shapes[input.first][batchIndex];
        }
        if (prev_batch == -1)
            throw std::logic_error("Reshape wasn't applied for a model.");

        for (int i = 0; i < n; i++) {
            if (i == n / 2) {
                log_info("Half of the test have already passed");
            }

            new_batch = ((i % 2) == 0) ? prev_batch * 2 : prev_batch;
            for (auto &input: inputInfo) {
                auto layout = input.second->getTensorDesc().getLayout();
                if ((layout == Layout::NCHW) || (layout == Layout::NC)) {
                    shapes[input.first][batchIndex] = new_batch;
                    doReshape = true;
                }
            }
            reshapeCNNNetwork();
        }
        InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, target_device);
        InferenceEngine::InferRequest infer_request = exeNetwork.CreateInferRequest();

        auto batchSize = cnnNetwork.getBatchSize();
        batchSize = batchSize != 0 ? batchSize : 1;
        const InferenceEngine::ConstInputsDataMap inputsInfo(exeNetwork.GetInputsInfo());
        fillBlobs(infer_request, inputsInfo, batchSize);

        infer_request.Infer();
        InferenceEngine::OutputsDataMap output_info(cnnNetwork.getOutputsInfo());
        for (auto &output: output_info)
            InferenceEngine::Blob::Ptr outputBlob = infer_request.GetBlob(output.first);
    }
    else {
        log_info("Reshape to batch*=2 of CNNNetwork created from network: \"" << model << "\" for " << n << " times");
        ov::runtime::Core ie;
        std::shared_ptr<ov::Model> network = ie.read_model(model);
        std::vector<ov::Output<ov::Node>> inputs = network->inputs();
        size_t prev_batch;
        for (int i = 0; i < n; i++) {
            if (i == n / 2) {
                log_info("Half of the test have already passed");
            }
            for (size_t input_index = 0; input_index < inputs.size(); ++input_index) {
                ov::Shape tensor_shape = inputs[input_index].get_shape();
                if (i == 0) {
                    prev_batch = tensor_shape[0];
                }
                size_t new_batch = ((i % 2) == 0) ? prev_batch * 2 : prev_batch;
                tensor_shape[0] = new_batch;
                network->reshape({{input_index, tensor_shape}});
            }
        }
        ov::runtime::CompiledModel compiled_model = ie.compile_model(network, target_device);
        ov::runtime::InferRequest infer_request = compiled_model.create_infer_request();

        fillTensors(infer_request, network->inputs());
        infer_request.infer();
        auto outputs = network->outputs();
        for (size_t i = 0; i < outputs.size(); ++i) {
            const auto &output_tensor = infer_request.get_output_tensor(i);
        }
    }
}

void test_create_exenetwork_full_pipeline(const std::string &model, const std::string &target_device, const int &n,
                                          const int &api_version) {
    if (api_version == 1) {
        log_info("Create ExecutableNetwork from network: \"" << model
                                                             << "\" for device: \"" << target_device << "\" for " << n
                                                             << " times");
        InferenceEngine::Core ie;
        InferenceEngine::CNNNetwork cnnNetwork = ie.ReadNetwork(model);
        InferenceEngine::InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
        InferenceEngine::ICNNNetwork::InputShapes shapes = cnnNetwork.getInputShapes();
        bool doReshape = false;
        for (auto &input: inputInfo) {
            setInputParameters();
            computeShapesToReshape();
        }
        reshapeCNNNetwork();
        InferenceEngine::ExecutableNetwork exeNetwork;
        for (int i = 0; i < n; i++) {
            if (i == n / 2) {
                log_info("Half of the test have already passed");
            }
            exeNetwork = ie.LoadNetwork(cnnNetwork, target_device);
        }
        InferenceEngine::InferRequest infer_request = exeNetwork.CreateInferRequest();

        auto batchSize = cnnNetwork.getBatchSize();
        batchSize = batchSize != 0 ? batchSize : 1;
        const InferenceEngine::ConstInputsDataMap inputsInfo(exeNetwork.GetInputsInfo());
        fillBlobs(infer_request, inputsInfo, batchSize);

        infer_request.Infer();
        InferenceEngine::OutputsDataMap output_info(cnnNetwork.getOutputsInfo());
        for (auto &output: output_info)
            InferenceEngine::Blob::Ptr outputBlob = infer_request.GetBlob(output.first);
    }
    else {
        log_info("Create ExecutableNetwork from network: \"" << model
                                                             << "\" for device: \"" << target_device << "\" for " << n
                                                             << " times");
        ov::runtime::Core ie;
        std::shared_ptr<ov::Model> network = ie.read_model(model);
        std::vector<ov::Output<ov::Node>> inputs = network->inputs();
        ov::runtime::CompiledModel compiled_model;
        for (int i = 0; i < n; i++) {
            if (i == n / 2) {
                log_info("Half of the test have already passed");
            }
            compiled_model = ie.compile_model(network, target_device);
        }
        ov::runtime::InferRequest infer_request = compiled_model.create_infer_request();

        fillTensors(infer_request, network->inputs());
        infer_request.infer();
        auto outputs = network->outputs();
        for (size_t i = 0; i < outputs.size(); ++i) {
            const auto &output_tensor = infer_request.get_output_tensor(i);
        }
    }
}

void test_create_infer_request_full_pipeline(const std::string &model, const std::string &target_device, const int &n, const int &api_version) {
    if (api_version == 1) {
        log_info("Create InferRequest from network: \"" << model
                                                        << "\" for device: \"" << target_device << "\" for " << n
                                                        << " times");
        InferenceEngine::Core ie;
        InferenceEngine::CNNNetwork cnnNetwork = ie.ReadNetwork(model);
        InferenceEngine::InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
        InferenceEngine::ICNNNetwork::InputShapes shapes = cnnNetwork.getInputShapes();
        bool doReshape = false;
        for (auto &input: inputInfo) {
            setInputParameters();
            computeShapesToReshape();
        }
        reshapeCNNNetwork();
        InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, target_device);
        InferenceEngine::InferRequest infer_request;

        auto batchSize = cnnNetwork.getBatchSize();
        batchSize = batchSize != 0 ? batchSize : 1;
        const InferenceEngine::ConstInputsDataMap inputsInfo(exeNetwork.GetInputsInfo());
        for (int i = 0; i < n; i++) {
            if (i == n / 2) {
                log_info("Half of the test have already passed");
            }
            infer_request = exeNetwork.CreateInferRequest();
            fillBlobs(infer_request, inputsInfo, batchSize);
        }
        infer_request.Infer();
        InferenceEngine::OutputsDataMap output_info(cnnNetwork.getOutputsInfo());
        for (auto &output: output_info)
            InferenceEngine::Blob::Ptr outputBlob = infer_request.GetBlob(output.first);
    }
    else {
        log_info("Create InferRequest from network: \"" << model
                                                        << "\" for device: \"" << target_device << "\" for " << n
                                                        << " times");
        ov::runtime::Core ie;
        std::shared_ptr<ov::Model> network = ie.read_model(model);
        std::vector<ov::Output<ov::Node>> inputs = network->inputs();
        ov::runtime::CompiledModel compiled_model = ie.compile_model(network, target_device);
        ov::runtime::InferRequest infer_request;
        for (int i = 0; i < n; i++) {
            if (i == n / 2) {
                log_info("Half of the test have already passed");
            }
            infer_request = compiled_model.create_infer_request();

        }
        fillTensors(infer_request, network->inputs());
        infer_request.infer();
        auto outputs = network->outputs();
        for (size_t i = 0; i < outputs.size(); ++i) {
            const auto &output_tensor = infer_request.get_output_tensor(i);
        }
    }
}


void test_infer_request_inference_full_pipeline(const std::string &model, const std::string &target_device,
                                                const int &n, const int &api_version) {
    if (api_version == 1) {
        log_info("Inference of InferRequest from network: \"" << model
                                                              << "\" for device: \"" << target_device << "\" for " << n
                                                              << " times");
        InferenceEngine::Core ie;
        InferenceEngine::CNNNetwork cnnNetwork = ie.ReadNetwork(model);
        InferenceEngine::InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
        InferenceEngine::ICNNNetwork::InputShapes shapes = cnnNetwork.getInputShapes();
        bool doReshape = false;
        for (auto &input: inputInfo) {
            setInputParameters();
            computeShapesToReshape();
        }
        reshapeCNNNetwork();
        InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, target_device);
        InferenceEngine::InferRequest infer_request = exeNetwork.CreateInferRequest();

        auto batchSize = cnnNetwork.getBatchSize();
        batchSize = batchSize != 0 ? batchSize : 1;
        const InferenceEngine::ConstInputsDataMap inputsInfo(exeNetwork.GetInputsInfo());
        fillBlobs(infer_request, inputsInfo, batchSize);

        for (int i = 0; i < n; i++) {
            if (i == n / 2) {
                log_info("Half of the test have already passed");
            }
            infer_request.Infer();
            InferenceEngine::OutputsDataMap output_info(cnnNetwork.getOutputsInfo());
            for (auto &output: output_info)
                InferenceEngine::Blob::Ptr outputBlob = infer_request.GetBlob(output.first);
        }
    }
    else {
        log_info("Inference of InferRequest from network: \"" << model
                                                              << "\" for device: \"" << target_device << "\" for " << n
                                                              << " times");
        ov::runtime::Core ie;
        std::shared_ptr<ov::Model> network = ie.read_model(model);
        std::vector<ov::Output<ov::Node>> inputs = network->inputs();
        ov::runtime::CompiledModel compiled_model = ie.compile_model(network, target_device);
        ov::runtime::InferRequest infer_request = compiled_model.create_infer_request();
        fillTensors(infer_request, network->inputs());
        for (int i = 0; i < n; i++) {
            if (i == n / 2) {
                log_info("Half of the test have already passed");
            }
            infer_request.infer();
            auto outputs = network->outputs();
            for (size_t output_index = 0; output_index < outputs.size(); ++output_index) {
                const auto &output_tensor = infer_request.get_output_tensor(output_index);
            }
        }
    }
}
