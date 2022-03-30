// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/visibility.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/frontend/visibility.hpp"
#include "openvino/opsets/opset8.hpp"

// Defined if we are building the plugin DLL (instead of using it)
#ifdef openvino_mock1_frontend_EXPORTS
#    define MOCK_API OPENVINO_CORE_EXPORTS
#else
#    define MOCK_API OPENVINO_CORE_IMPORTS
#endif  // openvino_mock1_frontend_EXPORTS

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
        auto shape = Shape{1, 2, 300, 300};
        auto param = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);
        param->set_friendly_name("mock_param");
        param->set_layout("NCHW");
        std::vector<float> data(ov::shape_size(shape), 1.f);
        auto aligned_weights_buffer =
            std::make_shared<ngraph::runtime::AlignedBuffer>(shape_size(shape) * ::element::f32.size());
        auto weights = std::make_shared<ngraph::runtime::SharedBuffer<std::shared_ptr<ngraph::runtime::AlignedBuffer>>>(
            aligned_weights_buffer->get_ptr<char>(),
            aligned_weights_buffer->size(),
            aligned_weights_buffer);
        auto constant = std::make_shared<ov::opset8::Constant>(ov::element::f32, shape, weights);
        constant->set_friendly_name("mock_const");
        auto op = std::make_shared<ov::opset8::Add>(param, constant);
        op->set_friendly_name("mock_add");
        auto op1 = std::make_shared<ov::opset8::Abs>(op);
        op1->set_friendly_name("mock_abs");
        auto res = std::make_shared<ov::opset8::Result>(op1);
        res->set_friendly_name("mock_result");
        auto ov_model = std::make_shared<ov::Model>(ResultVector({res}), ParameterVector({param}), "mock1_model");
        ov_model->get_rt_info()["mock_test"] = std::string(1024, 't');
        ov_model->input(0).set_names({"mock_input"});
        ov_model->output(0).set_names({"mock_output"});
        return ov_model;
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
