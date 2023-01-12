// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/manager.hpp"
#include "openvino/frontend/pytorch/frontend.hpp"
#include "openvino/frontend/pytorch/visibility.hpp"

PYTORCH_C_API ov::frontend::FrontEndVersion GetAPIVersion() {
    return OV_FRONTEND_API_VERSION;
}

PYTORCH_C_API void* GetFrontEndData() {
    auto res = new ov::frontend::FrontEndPluginInfo();
    res->m_name = "pytorch";
    res->m_creator = []() {
        return std::make_shared<ov::frontend::pytorch::FrontEnd>();
    };
    return res;
}
