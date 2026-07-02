// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/gguf/frontend.hpp"
#include "openvino/frontend/gguf/visibility.hpp"
#include "openvino/frontend/manager.hpp"

GGUF_FRONTEND_C_API ov::frontend::FrontEndVersion get_api_version() {
    return OV_FRONTEND_API_VERSION;
}

GGUF_FRONTEND_C_API void* get_front_end_data() {
    auto res = new ov::frontend::FrontEndPluginInfo();
    res->m_name = "gguf";
    res->m_creator = []() {
        return std::make_shared<ov::frontend::gguf::FrontEnd>();
    };
    return res;
}
