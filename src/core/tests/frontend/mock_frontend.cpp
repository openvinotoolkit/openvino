// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/visibility.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/frontend/visibility.hpp"
#include "openvino/opsets/opset8.hpp"

// Defined if we are building the plugin DLL (instead of using it)
#ifdef ov_mock1_frontend_EXPORTS
#    define MOCK_API OPENVINO_CORE_EXPORTS
#else
#    define MOCK_API OPENVINO_CORE_IMPORTS
#endif  // ov_mock1_frontend_EXPORTS

using namespace ngraph;
using namespace ov::frontend;

class InputModelMock : public InputModel {};

class FrontEndMock : public FrontEnd {
public:
    std::string get_name() const override {
        return "mock1";
    }

    InputModel::Ptr load_impl(const std::vector<ov::Any>& variants) const override {
        return std::make_shared<InputModelMock>();
    }

    std::shared_ptr<ov::Model> convert(const InputModel::Ptr& model) const override {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, Shape{1, 2, 3, 4});
        auto op = std::make_shared<ov::opset8::Relu>(param);
        auto res = std::make_shared<ov::opset8::Result>(op);
        return std::make_shared<ov::Model>(ResultVector({res}), ParameterVector({param}), "mock1_model");
    }
};

extern "C" MOCK_API FrontEndVersion GetAPIVersion() {
    return OV_FRONTEND_API_VERSION;
}

extern "C" MOCK_API void* GetFrontEndData() {
    auto* res = new FrontEndPluginInfo();
    res->m_name = "mock1";
    res->m_creator = []() {
        return std::make_shared<FrontEndMock>();
    };
    return res;
}
