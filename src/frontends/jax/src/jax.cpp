// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/jax/frontend.hpp"
#include "openvino/frontend/jax/visibility.hpp"
#include "openvino/frontend/manager.hpp"

JAX_FRONTEND_C_API ov::frontend::FrontEndVersion get_api_version() {
    return OV_FRONTEND_API_VERSION;
}

JAX_FRONTEND_C_API void* get_front_end_data() {
    auto res = new ov::frontend::FrontEndPluginInfo();
    res->m_name = "jax";
    res->m_creator = []() {
        return std::make_shared<ov::frontend::jax::FrontEnd>();
    };
    return res;
}
