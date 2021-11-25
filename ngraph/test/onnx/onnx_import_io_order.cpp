// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <experimental/filesystem>
#include <openvino/runtime/runtime.hpp>

#include "gtest/gtest.h"
#include "onnx_import/onnx.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/util/file_util.hpp"
#include "util/test_control.hpp"

using namespace ngraph;
namespace fs = std::experimental::filesystem;

static std::string s_manifest = "${MANIFEST}";

NGRAPH_TEST(onnx, check_io_orders) {
    std::string xml_path{"model.xml"};
    std::string bin_path{"model.bin"};

    const std::string model_path = ov::util::path_join({SERIALIZED_ZOO, "onnx/io_order/model.onnx"});
    auto base_function = onnx_import::import_onnx_model(model_path);
    EXPECT_NE(base_function, nullptr);
    std::cout << "Imported model=" << model_path << std::endl;
    for (auto param : base_function->get_parameters()) {
        std::cout << "index: " << base_function->get_parameter_index(param) << " name: " << param->get_friendly_name()
                  << std::endl;
    }

    std::cout << "-------" << std::endl;

    auto serialize = ov::pass::Serialize(xml_path, bin_path);
    serialize.run_on_function(base_function);

    ov::runtime::Core core;
    auto serialized_func = core.read_model(xml_path);
    std::cout << "Serialized model=" << xml_path;

    for (auto param : serialized_func->get_parameters()) {
        std::cout << "index: " << serialized_func->get_parameter_index(param) << " name: " << param->get_friendly_name()
                  << std::endl;
    }
}