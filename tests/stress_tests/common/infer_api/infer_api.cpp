// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "infer_api.h"
#include "openvino/core/preprocess/pre_post_process.hpp"


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

void InferAPI2::create_and_infer(const bool &async) {
    auto new_infer_request = compiled_model.create_infer_request();
    fillTensors(new_infer_request, inputs);
    if (async) {
        new_infer_request.start_async();
        new_infer_request.wait();
    } else {
        new_infer_request.infer();
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        const auto &output_tensor = new_infer_request.get_output_tensor(i);
    }
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

void InferAPI2::set_config(const std::string &device, const ov::AnyMap& properties) {
    ie.set_property(device, properties);
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

std::shared_ptr<InferApiBase> create_infer_api_wrapper() {
    return std::make_shared<InferAPI2>(InferAPI2());
}
