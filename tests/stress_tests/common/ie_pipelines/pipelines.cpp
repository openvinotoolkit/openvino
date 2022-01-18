// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pipelines.h"
#include "../utils.h"
#include "common_utils.h"

#include <iostream>
#include <string>

#include <inference_engine.hpp>

using namespace InferenceEngine;

std::function<void()> load_unload_plugin(const std::string &target_device) {
    return [&] {
        Core ie;
        // GetVersions silently register plugin in `plugins` through `GetCPPPluginByName`
        ie.GetVersions(target_device);
        // Remove plugin for target_device from `plugins`
        ie.UnregisterPlugin(target_device);
    };
}

std::function<void()> read_cnnnetwork(const std::string &model) {
    return [&] {
        Core ie;
        CNNNetwork cnnNetwork = ie.ReadNetwork(model);
    };
}

std::function<void()> cnnnetwork_reshape_batch_x2(const std::string &model) {
    return [&] {
        Core ie;
        CNNNetwork cnnNetwork = ie.ReadNetwork(model);
        const InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
        ICNNNetwork::InputShapes shapes = cnnNetwork.getInputShapes();
        bool doReshape = false;
        for (const InputsDataMap::value_type& input : inputInfo) {
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

std::function<void()> set_input_params(const std::string &model) {
    return [&] {
        Core ie;
        CNNNetwork cnnNetwork = ie.ReadNetwork(model);
        InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
        for (auto &input : inputInfo) {
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

std::function<void()> create_exenetwork(const std::string &model, const std::string &target_device) {
    return [&] {
        Core ie;
        CNNNetwork cnnNetwork = ie.ReadNetwork(model);
        ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, target_device);
    };
}

std::function<void()> recreate_exenetwork(Core &ie, const std::string &model, const std::string &target_device) {
    return [&] {
        CNNNetwork cnnNetwork = ie.ReadNetwork(model);
        ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, target_device);
    };
}

std::function<void()> create_infer_request(const std::string &model, const std::string &target_device) {
    return [&] {
        Core ie;
        CNNNetwork cnnNetwork = ie.ReadNetwork(model);
        ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, target_device);
        InferRequest infer_request = exeNetwork.CreateInferRequest();
    };
}


std::function<void()> recreate_infer_request(InferenceEngine::ExecutableNetwork& exeNetwork) {
    return [&] {
        InferRequest infer_request = exeNetwork.CreateInferRequest();
    };
}

std::function<void()> infer_request_inference(const std::string &model, const std::string &target_device) {
    return [&] {
        Core ie;
        CNNNetwork cnnNetwork = ie.ReadNetwork(model);
        ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, target_device);
        InferRequest infer_request = exeNetwork.CreateInferRequest();

        auto batchSize = cnnNetwork.getBatchSize();
        batchSize = batchSize != 0 ? batchSize : 1;
        const InferenceEngine::ConstInputsDataMap inputsInfo(exeNetwork.GetInputsInfo());
        fillBlobs(infer_request, inputsInfo, batchSize);

        infer_request.Infer();
        OutputsDataMap output_info(cnnNetwork.getOutputsInfo());
        for (auto &output : output_info)
            Blob::Ptr outputBlob = infer_request.GetBlob(output.first);
    };
}

std::function<void()> reinfer_request_inference(InferenceEngine::InferRequest& infer_request, InferenceEngine::OutputsDataMap& output_info) {
    return [&] {
        infer_request.Infer();
        for (auto &output : output_info)
            Blob::Ptr outputBlob = infer_request.GetBlob(output.first);
    };
}

std::function<void()> inference_with_streams(const std::string &model, const std::string &target_device, const int& nstreams) {
    return [&] {
        std::map<std::string, std::string> config;
        config[target_device + "_THROUGHPUT_STREAMS"] = std::to_string(nstreams);

        Core ie;
        ie.GetVersions(target_device);
        ie.SetConfig(config, target_device);

        InferRequest inferRequest;

        CNNNetwork cnnNetwork = ie.ReadNetwork(model);
        ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, target_device);
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
            OutputsDataMap output_info(cnnNetwork.getOutputsInfo());
            for (auto &output : output_info)
                Blob::Ptr outputBlob = inferRequest.GetBlob(output.first);
        }
    };
}
