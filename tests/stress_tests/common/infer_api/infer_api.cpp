// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "infer_api.h"
#include "openvino/core/preprocess/pre_post_process.hpp"


InferAPI1::InferAPI1() = default;

void InferAPI1::load_plugin(const std::string &device) {
    ie.GetVersions(device);
}

void InferAPI1::unload_plugin(const std::string &device) {
    ie.UnregisterPlugin(device);
}

void InferAPI1::read_network(const std::string &model) {
    cnnNetwork = ie.ReadNetwork(model);
    inputsInfo = cnnNetwork.getInputsInfo();
    InferenceEngine::ICNNNetwork::InputShapes shapes = cnnNetwork.getInputShapes();
    for (const auto &input: inputsInfo) {
        original_batch_size = shapes[input.first][0];

    }
    original_batch_size = original_batch_size ? original_batch_size : 1;
}

void InferAPI1::load_network(const std::string &device) {
    exeNetwork = ie.LoadNetwork(cnnNetwork, device);
}

void InferAPI1::create_infer_request() {
    inferRequest = exeNetwork.CreateInferRequest();
}

void InferAPI1::prepare_input() {
    auto batchSize = cnnNetwork.getBatchSize();
    batchSize = batchSize != 0 ? batchSize : 1;
    fillBlobs(inferRequest, exeNetwork.GetInputsInfo(), batchSize);
}

void InferAPI1::infer() {
    inferRequest.Infer();
    for (auto &output: outputInfo) {
        InferenceEngine::Blob::Ptr outputBlob = inferRequest.GetBlob(output.first);
    }
}

void InferAPI1::change_batch_size(int multiplier, int cur_iter) {
    bool doReshape = false;
    auto shapes = cnnNetwork.getInputShapes();
    int new_batch_size = ((cur_iter % 2) == 0) ? original_batch_size * multiplier : original_batch_size;
    for (const auto &input: inputsInfo) {
        int batchIndex = -1;
        auto layout = input.second->getTensorDesc().getLayout();
        if ((layout == InferenceEngine::Layout::NCHW) || (layout == InferenceEngine::Layout::NCDHW) ||
            (layout == InferenceEngine::Layout::NHWC) || (layout == InferenceEngine::Layout::NDHWC) ||
            (layout == InferenceEngine::Layout::NC)) {
            batchIndex = 0;
        } else if (layout == InferenceEngine::CN) {
            batchIndex = 1;
        }
        if (batchIndex != -1) {
            shapes[input.first][batchIndex] = new_batch_size;
            doReshape = true;
        }
    }
    if (doReshape)
        cnnNetwork.reshape(shapes);
    else
        throw std::logic_error("Reshape wasn't applied for a model.");
}

void InferAPI1::set_input_params(const std::string &model) {
    cnnNetwork = ie.ReadNetwork(model);
    InferenceEngine::InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    for (auto &input: inputInfo) {
        input.second->getPreProcess().setResizeAlgorithm(InferenceEngine::NO_RESIZE);
        input.second->setPrecision(InferenceEngine::Precision::U8);
        if (input.second->getInputData()->getTensorDesc().getDims().size() == 4)
            input.second->setLayout(InferenceEngine::Layout::NCHW);
        else if (input.second->getInputData()->getTensorDesc().getDims().size() == 2)
            input.second->setLayout(InferenceEngine::Layout::NC);
        else
            throw std::logic_error("Setting of input parameters wasn't applied for a model.");
    }
}

void InferAPI1::set_config(const std::string &device, const std::string &property, int nstreams) {
    config[device + "_" + property] = std::to_string(nstreams);
    ie.SetConfig(config, device);
}

unsigned int InferAPI1::get_property(const std::string &name) {
    return exeNetwork.GetMetric(name).as<unsigned int>();
}


InferAPI2::InferAPI2() = default;

void InferAPI2::load_plugin(const std::string &device) {
    ie.get_versions(device);
}

void InferAPI2::unload_plugin(const std::string &device) {
    ie.unload_plugin(device);
}

void InferAPI2::read_network(const std::string &model) {
    network = ie.read_model(model);
    inputs = network->inputs();

    for (const auto &input: inputs) {
        auto tensor_shape = input.get_shape();
        original_batch_size = tensor_shape[0];
        original_batch_size = original_batch_size ? original_batch_size : 1;
    }
}

void InferAPI2::load_network(const std::string &device) {
    compiled_model = ie.compile_model(network, device);
}

void InferAPI2::create_infer_request() {
    infer_request = compiled_model.create_infer_request();
}

void InferAPI2::prepare_input() {
    fillTensors(infer_request, inputs);
}

void InferAPI2::infer() {
    infer_request.infer();
    for (size_t i = 0; i < outputs.size(); ++i) {
        const auto &output_tensor = infer_request.get_output_tensor(i);
    }
}

void InferAPI2::change_batch_size(int multiplier, int cur_iter) {
    int new_batch_size = ((cur_iter % 2) == 0) ? original_batch_size * multiplier : original_batch_size;
    for (auto &input: inputs) {
        auto tensor_shape = input.get_shape();
        tensor_shape[0] = new_batch_size;
        network->reshape({{input.get_any_name(), tensor_shape}});
    }
}

void InferAPI2::set_config(const std::string &device, const std::string &property, int nstreams) {
    config[device + "_" + property] = std::to_string(nstreams);
    ie.set_property(device, config);
}

unsigned int InferAPI2::get_property(const std::string &name) {
    return compiled_model.get_property(name).as<unsigned int>();
}

void InferAPI2::set_input_params(const std::string &model) {
    network = ie.read_model(model);
    inputs = network->inputs();
    auto ppp = ov::preprocess::PrePostProcessor(network);
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto &input_info = ppp.input(i);
        if (inputs[i].get_shape().size() == 4) {
            input_info.tensor().set_element_type(ov::element::u8).set_layout("NCHW");
            input_info.model().set_layout("NCHW");
            ppp.input(i).preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
        } else if (inputs[i].get_shape().size() == 2) {
            input_info.tensor().set_element_type(ov::element::u8).set_layout("NC");
            input_info.model().set_layout("NC");
        } else {
            throw std::logic_error("Setting of input parameters wasn't applied for a model.");
        }
    }
    network = ppp.build();
    inputs = network->inputs();
}

std::shared_ptr<InferApiBase> create_infer_api_wrapper(const int &api_version) {
    if (api_version == 1) {
        return std::make_shared<InferAPI1>(InferAPI1());
    } else if (api_version == 2) {
        return std::make_shared<InferAPI2>(InferAPI2());
    } else {
        throw std::logic_error("Unsupported API version");
    }
}
