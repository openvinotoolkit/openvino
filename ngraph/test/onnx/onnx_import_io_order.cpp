// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/runtime/runtime.hpp>

#include "gtest/gtest.h"
#include "onnx_import/onnx.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/util/file_util.hpp"
#include "util/test_control.hpp"

using namespace ngraph;

static std::string s_manifest = "${MANIFEST}";

NGRAPH_TEST(onnx, check_inputs) {
    std::string xml_path{"model.xml"};
    std::string bin_path{"model.bin"};

    ov::runtime::Core core;
    const std::string model_path = ov::util::path_join({SERIALIZED_ZOO, "onnx/io_order/few_inputs.onnx"});
    auto base_function = core.read_model(model_path);
    EXPECT_NE(base_function, nullptr);

    std::cout << "Imported model=" << model_path << std::endl;
    for (auto param : base_function->get_parameters()) {
        std::cout << "index: " << base_function->get_parameter_index(param) << " name: " << param->get_friendly_name()
                  << std::endl;
    }

    std::cout << "-------" << std::endl;

    auto serialize = ov::pass::Serialize(xml_path, bin_path);
    serialize.run_on_function(base_function);

    auto serialized_func = core.read_model(xml_path);
    std::cout << "Serialized model=" << xml_path << std::endl;

    for (auto param : serialized_func->get_parameters()) {
        std::cout << "index: " << serialized_func->get_parameter_index(param) << " name: " << param->get_friendly_name()
                  << std::endl;
    }
}

NGRAPH_TEST(onnx, check_outputs) {
    ov::runtime::Core core;

    const std::string model_path = ov::util::path_join({SERIALIZED_ZOO, "onnx/io_order/few_outputs.onnx"});
    auto base_function = core.read_model(model_path);
    EXPECT_NE(base_function, nullptr);

    std::cout << "Imported model=" << model_path << std::endl;
    for (const auto& result : base_function->get_results()) {
        std::cout << "index: " << base_function->get_result_index(result) << " name: " << result->get_friendly_name()
                  << std::endl;
    }

    std::cout << "-------" << std::endl;

    const std::string xml_path{"model_few_outputs.xml"};
    const std::string bin_path{"model_few_outputs.bin"};
    auto serialize = ov::pass::Serialize(xml_path, bin_path);
    serialize.run_on_function(base_function);

    auto serialized_func = core.read_model(xml_path);
    std::cout << "Serialized model=" << xml_path << std::endl;

    for (const auto& result : serialized_func->get_results()) {
        std::cout << "index: " << serialized_func->get_result_index(result) << " name: " << result->get_friendly_name()
                  << std::endl;
    }
}