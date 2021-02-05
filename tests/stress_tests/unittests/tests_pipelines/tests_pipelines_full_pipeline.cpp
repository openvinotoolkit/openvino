// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tests_pipelines.h"
#include "../common/ie_utils.h"

#include <string>

#include <inference_engine.hpp>

using namespace InferenceEngine;

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

void test_load_unload_plugin_full_pipeline(const std::string &model, const std::string &target_device, const int &n) {
    log_info("Load/unload plugin for device: " << target_device << " for " << n << " times");
    Core ie;
    for (int i = 0; i < n; i++) {
        if (i == n / 2) {
            log_info("Half of the test have already passed");
        }
        // GetVersions silently register plugin in `plugins` through `GetCPPPluginByName`
        ie.GetVersions(target_device);
        // Remove plugin for target_device from `plugins`
        ie.UnregisterPlugin(target_device);
    }
    CNNNetwork cnnNetwork = ie.ReadNetwork(model);
    InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    ICNNNetwork::InputShapes shapes = cnnNetwork.getInputShapes();
    bool doReshape = false;
    for (auto &input : inputInfo) {
        setInputParameters();
        computeShapesToReshape();
    }
    reshapeCNNNetwork();
    ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, target_device);
    InferRequest infer_request = exeNetwork.CreateInferRequest();

    auto batchSize = cnnNetwork.getBatchSize();
    batchSize = batchSize != 0 ? batchSize : 1;
    const ConstInputsDataMap inputsInfo(exeNetwork.GetInputsInfo());
    fillBlobs(infer_request, inputsInfo, batchSize);

    infer_request.Infer();
    OutputsDataMap output_info(cnnNetwork.getOutputsInfo());
    for (auto &output : output_info)
        Blob::Ptr outputBlob = infer_request.GetBlob(output.first);
}

void test_read_network_full_pipeline(const std::string &model, const std::string &target_device, const int &n) {
    log_info("Read network: \"" << model << "\" for " << n << " times");
    Core ie;
    CNNNetwork cnnNetwork;
    for (int i = 0; i < n; i++) {
        if (i == n / 2) {
            log_info("Half of the test have already passed");
        }
        cnnNetwork = ie.ReadNetwork(model);
    }
    InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    ICNNNetwork::InputShapes shapes = cnnNetwork.getInputShapes();
    bool doReshape = false;
    for (auto &input : inputInfo) {
        setInputParameters();
        computeShapesToReshape();
    }
    reshapeCNNNetwork();
    ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, target_device);
    InferRequest infer_request = exeNetwork.CreateInferRequest();

    auto batchSize = cnnNetwork.getBatchSize();
    batchSize = batchSize != 0 ? batchSize : 1;
    const ConstInputsDataMap inputsInfo(exeNetwork.GetInputsInfo());
    fillBlobs(infer_request, inputsInfo, batchSize);

    infer_request.Infer();
    OutputsDataMap output_info(cnnNetwork.getOutputsInfo());
    for (auto &output : output_info)
        Blob::Ptr outputBlob = infer_request.GetBlob(output.first);
}

void test_set_input_params_full_pipeline(const std::string &model, const std::string &target_device, const int &n) {
    log_info("Apply preprocessing for CNNNetwork from network: \"" << model << "\" for " << n << " times");
    Core ie;
    CNNNetwork cnnNetwork = ie.ReadNetwork(model);
    InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    for (int i = 0; i < n; i++) {
        if (i == n / 2) {
            log_info("Half of the test have already passed");
        }
        for (auto &input : inputInfo) {
            setInputParameters();
        }
    }
    ICNNNetwork::InputShapes shapes = cnnNetwork.getInputShapes();
    bool doReshape = false;
    for (auto &input : inputInfo) {
        computeShapesToReshape();
    }
    reshapeCNNNetwork();
    ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, target_device);
    InferRequest infer_request = exeNetwork.CreateInferRequest();

    auto batchSize = cnnNetwork.getBatchSize();
    batchSize = batchSize != 0 ? batchSize : 1;
    const ConstInputsDataMap inputsInfo(exeNetwork.GetInputsInfo());
    fillBlobs(infer_request, inputsInfo, batchSize);

    infer_request.Infer();
    OutputsDataMap output_info(cnnNetwork.getOutputsInfo());
    for (auto &output : output_info)
        Blob::Ptr outputBlob = infer_request.GetBlob(output.first);
}

void test_cnnnetwork_reshape_batch_x2_full_pipeline(const std::string &model, const std::string &target_device, const int &n) {
    log_info("Reshape to batch*=2 of CNNNetwork created from network: \"" << model << "\" for " << n << " times");
    Core ie;
    CNNNetwork cnnNetwork = ie.ReadNetwork(model);
    InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    for (auto &input : inputInfo) {
        setInputParameters();
    }
    ICNNNetwork::InputShapes shapes = cnnNetwork.getInputShapes();
    bool doReshape = false;
    int prev_batch = -1, new_batch;
    for (auto &input : inputInfo) {
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
        for (auto &input : inputInfo) {
            auto layout = input.second->getTensorDesc().getLayout();
            if ((layout == Layout::NCHW) || (layout == Layout::NC)) {
                shapes[input.first][batchIndex] = new_batch;
                doReshape = true;
            }
        }
        reshapeCNNNetwork();
    }
    ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, target_device);
    InferRequest infer_request = exeNetwork.CreateInferRequest();

    auto batchSize = cnnNetwork.getBatchSize();
    batchSize = batchSize != 0 ? batchSize : 1;
    const ConstInputsDataMap inputsInfo(exeNetwork.GetInputsInfo());
    fillBlobs(infer_request, inputsInfo, batchSize);

    infer_request.Infer();
    OutputsDataMap output_info(cnnNetwork.getOutputsInfo());
    for (auto &output : output_info)
        Blob::Ptr outputBlob = infer_request.GetBlob(output.first);
}

void test_create_exenetwork_full_pipeline(const std::string &model, const std::string &target_device, const int &n) {
    log_info("Create ExecutableNetwork from network: \"" << model
             << "\" for device: \"" << target_device << "\" for " << n << " times");
    Core ie;
    CNNNetwork cnnNetwork = ie.ReadNetwork(model);
    InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    ICNNNetwork::InputShapes shapes = cnnNetwork.getInputShapes();
    bool doReshape = false;
    for (auto &input : inputInfo) {
        setInputParameters();
        computeShapesToReshape();
    }
    reshapeCNNNetwork();
    ExecutableNetwork exeNetwork;
    for (int i = 0; i < n; i++) {
        if (i == n / 2) {
            log_info("Half of the test have already passed");
        }
        exeNetwork = ie.LoadNetwork(cnnNetwork, target_device);
    }
    InferRequest infer_request = exeNetwork.CreateInferRequest();

    auto batchSize = cnnNetwork.getBatchSize();
    batchSize = batchSize != 0 ? batchSize : 1;
    const ConstInputsDataMap inputsInfo(exeNetwork.GetInputsInfo());
    fillBlobs(infer_request, inputsInfo, batchSize);

    infer_request.Infer();
    OutputsDataMap output_info(cnnNetwork.getOutputsInfo());
    for (auto &output : output_info)
        Blob::Ptr outputBlob = infer_request.GetBlob(output.first);
}

void test_create_infer_request_full_pipeline(const std::string &model, const std::string &target_device, const int &n) {
    log_info("Create InferRequest from network: \"" << model
             << "\" for device: \"" << target_device << "\" for " << n << " times");
    Core ie;
    CNNNetwork cnnNetwork = ie.ReadNetwork(model);
    InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    ICNNNetwork::InputShapes shapes = cnnNetwork.getInputShapes();
    bool doReshape = false;
    for (auto &input : inputInfo) {
        setInputParameters();
        computeShapesToReshape();
    }
    reshapeCNNNetwork();
    ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, target_device);
    InferRequest infer_request;

    auto batchSize = cnnNetwork.getBatchSize();
    batchSize = batchSize != 0 ? batchSize : 1;
    const ConstInputsDataMap inputsInfo(exeNetwork.GetInputsInfo());
    for (int i = 0; i < n; i++) {
        if (i == n / 2) {
            log_info("Half of the test have already passed");
        }
        infer_request = exeNetwork.CreateInferRequest();
        fillBlobs(infer_request, inputsInfo, batchSize);
    }
    infer_request.Infer();
    OutputsDataMap output_info(cnnNetwork.getOutputsInfo());
    for (auto &output : output_info)
        Blob::Ptr outputBlob = infer_request.GetBlob(output.first);
}

void test_infer_request_inference_full_pipeline(const std::string &model, const std::string &target_device, const int &n) {
    log_info("Inference of InferRequest from network: \"" << model
             << "\" for device: \"" << target_device << "\" for " << n << " times");
    Core ie;
    CNNNetwork cnnNetwork = ie.ReadNetwork(model);
    InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    ICNNNetwork::InputShapes shapes = cnnNetwork.getInputShapes();
    bool doReshape = false;
    for (auto &input : inputInfo) {
        setInputParameters();
        computeShapesToReshape();
    }
    reshapeCNNNetwork();
    ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, target_device);
    InferRequest infer_request = exeNetwork.CreateInferRequest();

    auto batchSize = cnnNetwork.getBatchSize();
    batchSize = batchSize != 0 ? batchSize : 1;
    const ConstInputsDataMap inputsInfo(exeNetwork.GetInputsInfo());
    fillBlobs(infer_request, inputsInfo, batchSize);

    for (int i = 0; i < n; i++) {
        if (i == n / 2) {
            log_info("Half of the test have already passed");
        }
        infer_request.Infer();
        OutputsDataMap output_info(cnnNetwork.getOutputsInfo());
        for (auto &output : output_info)
            Blob::Ptr outputBlob = infer_request.GetBlob(output.first);
    }
}
