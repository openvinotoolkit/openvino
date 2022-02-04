// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mock_py_frontend/mock_py_frontend.hpp"

#include "openvino/frontend/manager.hpp"
#include "openvino/frontend/visibility.hpp"

FeStat FrontEndMockPy::m_stat = {};
ModelStat InputModelMockPy::m_stat = {};
PlaceStat PlaceMockPy::m_stat = {};

using namespace ngraph;
using namespace ov::frontend;

extern "C" MOCK_API FrontEndVersion GetAPIVersion() {
    return OV_FRONTEND_API_VERSION;
}

extern "C" MOCK_API void* GetFrontEndData() {
    FrontEndPluginInfo* res = new FrontEndPluginInfo();
    res->m_name = "mock_py";
    res->m_creator = []() {
        return std::make_shared<FrontEndMockPy>();
    };

    return res;
}
