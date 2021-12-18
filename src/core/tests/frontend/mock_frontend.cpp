// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/visibility.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/frontend/visibility.hpp"

// Defined if we are building the plugin DLL (instead of using it)
#ifdef ov_mock1_frontend_EXPORTS
#    define MOCK_API OPENVINO_CORE_EXPORTS
#else
#    define MOCK_API OPENVINO_CORE_IMPORTS
#endif  // ov_mock1_frontend_EXPORTS

using namespace ngraph;
using namespace ov::frontend;

class FrontEndMock : public FrontEnd {
public:
    std::string get_name() const override {
        return "mock1";
    }
};

extern "C" MOCK_API FrontEndVersion GetAPIVersion() {
    return OV_FRONTEND_API_VERSION;
}

extern "C" MOCK_API void* GetFrontEndData() {
    FrontEndPluginInfo* res = new FrontEndPluginInfo();
    res->m_name = "mock1";
    res->m_creator = []() {
        return std::make_shared<FrontEndMock>();
    };
    return res;
}
