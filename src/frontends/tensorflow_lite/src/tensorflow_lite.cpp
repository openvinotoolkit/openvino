// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/manager.hpp"
#include "openvino/frontend/tensorflow_lite/frontend.hpp"
#include "openvino/frontend/tensorflow_lite/visibility.hpp"

TENSORFLOW_LITE_FRONTEND_C_API ov::frontend::FrontEndVersion get_api_version() {
    return OV_FRONTEND_API_VERSION;
}

TENSORFLOW_LITE_FRONTEND_C_API void* get_front_end_data() {
    auto res = new ov::frontend::FrontEndPluginInfo();
    res->m_name = "tflite";
    res->m_creator = []() {
        return std::make_shared<ov::frontend::tensorflow_lite::FrontEnd>();
    };
    return res;
}
