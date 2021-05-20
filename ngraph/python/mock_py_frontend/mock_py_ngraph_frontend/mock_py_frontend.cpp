// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "frontend_manager/frontend_manager.hpp"
#include "frontend_manager/frontend_manager_defs.hpp"
#include "ngraph/visibility.hpp"
#include "mock_py_frontend.hpp"

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
    res->m_creator = [](FrontEndCapFlags flags) {
        std::cout << "MOCK PY creator called\n";
        return std::make_shared<FrontEndMockPy>(flags);
    };

    return res;
}