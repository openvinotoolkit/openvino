// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mock_py_frontend.hpp"
#include "frontend_manager/frontend_manager.hpp"
#include "frontend_manager/frontend_manager_defs.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

extern "C" MOCK_API FrontEndVersion GetAPIVersion()
{
    return OV_FRONTEND_API_VERSION;
}

extern "C" MOCK_API void* GetFrontEndData()
{
    FrontEndPluginInfo* res = new FrontEndPluginInfo();
    res->m_name = "mock_py";
    res->m_creator = []() { return std::make_shared<FrontEndMockPy>(); };

    return res;
}
