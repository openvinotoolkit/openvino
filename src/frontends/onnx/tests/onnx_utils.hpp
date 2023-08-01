// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "openvino/runtime/core.hpp"

static const std::string ONNX_FE = "onnx";

inline std::string backend_name_to_device(const std::string& backend_name) {
    if (backend_name == "INTERPRETER")
        return "TEMPLATE";
    if (backend_name == "IE_CPU")
        return "CPU";
    if (backend_name == "IE_GPU")
        return "GPU";
    OPENVINO_THROW("Unsupported backend name");
}

inline std::shared_ptr<ov::Model> function_from_ir(const std::string& xml_path, const std::string& bin_path = {}) {
    ov::Core c;
    return c.read_model(xml_path, bin_path);
}
