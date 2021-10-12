// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <frontend_manager/frontend_manager.hpp>
#include <tensorflow_frontend/frontend.hpp>

extern "C" NGRAPH_HELPER_DLL_EXPORT ngraph::frontend::FrontEndVersion GetAPIVersion() {
    return OV_FRONTEND_API_VERSION;
}

extern "C" NGRAPH_HELPER_DLL_EXPORT void* GetFrontEndData() {
    auto res = new ngraph::frontend::FrontEndPluginInfo();
    res->m_name = "tf";
    res->m_creator = []() {
        return std::make_shared<ov::frontend::FrontEndTF>();
    };
    return res;
}
