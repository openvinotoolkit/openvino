// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>

#ifdef __EMSCRIPTEN__
#include <emscripten/bind.h>
#endif

#include "openvino/openvino.hpp"

#ifdef __EMSCRIPTEN__
using namespace emscripten;
#endif

ov::Layout TENSOR_LAYOUT = ov::Layout("NHWC");

std::string get_version_string()
{
    ov::Version version = ov::get_openvino_version();
    std::string str;

    return str.assign(version.buildNumber);
}

std::string get_description_string()
{
    ov::Version version = ov::get_openvino_version();
    std::string str;

    return str.assign(version.description);
}

std::string read_model(std::string xml_path, std::string bin_path)
{
    ov::Core core;
    std::string errorMessage;

    try {
        auto model = core.read_model(xml_path, bin_path);

        std::cout << "Name: " << model->get_friendly_name() << std::endl;

        const ov::Layout tensor_layout{"NHWC"};
        ov::element::Type input_type = ov::element::u8;

        ov::preprocess::PrePostProcessor ppp(model);
        ppp.input().tensor().set_shape({ 1, 224, 224, 3}).set_element_type(input_type).set_layout(tensor_layout);
        ppp.input().preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
        ppp.output().tensor().set_element_type(ov::element::f32);
        ppp.input().model().set_layout(TENSOR_LAYOUT);
        ppp.build();

        ov::CompiledModel compiled_model = core.compile_model(model, "TEMPLATE");

        auto v = compiled_model.inputs();
        std::cout << "Inputs: " << v.size() << std::endl;
    } catch(ov::Exception e) {
        errorMessage = e.what();
    }
    catch(std::exception e) {
        errorMessage = e.what();
    }
    
    if (!errorMessage.empty()) {
        std::cout << "Was error: " << errorMessage << std::endl;
    }
    
    return "Was processed: " + xml_path + " and " + bin_path;
}

#ifndef __EMSCRIPTEN__
int main() {
    const std::string xml_path = "./data/models/v3-small_224_1.0_float.xml";
    const std::string bin_path = "./data/models/v3-small_224_1.0_float.bin";

    std::cout << "== start" << std::endl;

    std::cout << get_version_string() << std::endl;
    std::cout << get_description_string() << std::endl;

    read_model(xml_path, bin_path);

    std::cout << "== end" << std::endl;
}
#endif

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_BINDINGS(my_module) {
    function("getVersionString", &get_version_string);
    function("getDescriptionString", &get_description_string);
    function("readModel", &read_model);
}
#endif
