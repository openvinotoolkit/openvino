// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "openvino/runtime/core.hpp"
#include "common_test_utils/test_constants.hpp"

static const std::string ONNX_FE = "onnx";

inline std::string backend_name_to_device(const std::string& backend_name) {
    if (backend_name == "INTERPRETER")
        return ov::test::utils::DEVICE_TEMPLATE;
    if (backend_name == "IE_CPU")
        return ov::test::utils::DEVICE_CPU;
    if (backend_name == "IE_GPU")
        return ov::test::utils::DEVICE_GPU;
    OPENVINO_THROW("Unsupported backend name");
}

inline std::shared_ptr<ov::Model> function_from_ir(const std::string& xml_path, const std::string& bin_path = {}) {
    ov::Core c;
    return c.read_model(xml_path, bin_path);
}
