// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <frontend_manager/frontend_exceptions.hpp>
#include <frontend_manager/frontend_manager.hpp>
#include <input_model.hpp>
#include <onnx_frontend/frontend.hpp>
#include <onnx_import/onnx.hpp>
#include <utils/onnx_internal.hpp>

using namespace ngraph;
using namespace ngraph::frontend;

extern "C" ONNX_FRONTEND_API FrontEndVersion GetAPIVersion() {
    return OV_FRONTEND_API_VERSION;
}

extern "C" ONNX_FRONTEND_API void* GetFrontEndData() {
    FrontEndPluginInfo* res = new FrontEndPluginInfo();
    res->m_name = "onnx_experimental";
    res->m_creator = []() {
        return std::make_shared<FrontEndONNX>();
    };
    return res;
}

InputModel::Ptr FrontEndONNX::load_impl(const std::vector<std::shared_ptr<Variant>>& variants) const {
    NGRAPH_CHECK(variants.size() == 1,
                 "Only one parameter to load function is expected. Got " + std::to_string(variants.size()));
    NGRAPH_CHECK(ov::is_type<VariantWrapper<std::string>>(variants[0]),
                 "Parameter to load function need to be a std::string");
    auto path = ov::as_type_ptr<VariantWrapper<std::string>>(variants[0])->get();
    return std::make_shared<InputModelONNX>(path);
}

std::shared_ptr<ngraph::Function> FrontEndONNX::convert(InputModel::Ptr model) const {
    auto model_onnx = std::dynamic_pointer_cast<InputModelONNX>(model);
    NGRAPH_CHECK(model_onnx != nullptr, "Invalid input model");
    return model_onnx->convert();
}

void FrontEndONNX::convert(std::shared_ptr<ngraph::Function> partially_converted) const {
    onnx_import::detail::convert_decoded_function(partially_converted);
}

std::shared_ptr<ngraph::Function> FrontEndONNX::decode(InputModel::Ptr model) const {
    auto model_onnx = std::dynamic_pointer_cast<InputModelONNX>(model);
    NGRAPH_CHECK(model_onnx != nullptr, "Invalid input model");
    return model_onnx->decode();
}
