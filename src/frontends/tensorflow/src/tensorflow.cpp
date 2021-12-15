// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/common/manager.hpp"
#include "tensorflow_frontend/frontend.hpp"

TF_C_API ov::frontend::FrontEndVersion GetAPIVersion() {
    return OV_FRONTEND_API_VERSION;
}

TF_C_API void* GetFrontEndData() {
    auto res = new ov::frontend::FrontEndPluginInfo();
    res->m_name = "tf";
    res->m_creator = []() {
        return std::make_shared<ov::frontend::FrontEndTF>();
    };
    return res;
}
