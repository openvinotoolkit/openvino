// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <filesystem>

#ifdef __EMSCRIPTEN__
#include <emscripten/bind.h>
#endif

#include "openvino/openvino.hpp"

#ifdef __EMSCRIPTEN__
using namespace emscripten;
#endif

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

    try {
        std::shared_ptr<ov::Model> model = core.read_model(xml_path, bin_path);
    } catch(ov::Exception e) {
        std::cout << "Was error: " << e.what() << std::endl;
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
