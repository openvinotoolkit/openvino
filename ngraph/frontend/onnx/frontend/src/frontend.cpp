// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <frontend_manager/frontend_exceptions.hpp>
#include <frontend_manager/frontend_manager.hpp>
#include <fstream>
#include <input_model.hpp>
#include <ngraph/file_util.hpp>
#include <onnx_frontend/frontend.hpp>
#include <onnx_import/onnx.hpp>
#include <utils/onnx_internal.hpp>

#include "onnx_common/onnx_model_validator.hpp"

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
    if (variants.size() > 0 && is_type<VariantWrapper<std::string>>(variants[0])) {
        const auto path = as_type_ptr<VariantWrapper<std::string>>(variants[0])->get();
        return std::make_shared<InputModelONNX>(path);
    }
    if (variants.size() > 0 && is_type<VariantWrapper<std::istream*>>(variants[0])) {
        auto stream = as_type_ptr<VariantWrapper<std::istream*>>(variants[0])->get();
        if (variants.size() > 1 && is_type<VariantWrapper<std::string>>(variants[1])) {
            const auto path = as_type_ptr<VariantWrapper<std::string>>(variants[1])->get();
            return std::make_shared<InputModelONNX>(*stream, path);
        }
        return std::make_shared<InputModelONNX>(*stream);
    }
    return nullptr;
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

std::string FrontEndONNX::get_name() const {
    return "onnx";
}

bool FrontEndONNX::supported_impl(const std::vector<std::shared_ptr<Variant>>& variants) const {
    if (variants.size() > 0 && is_type<VariantWrapper<std::string>>(variants[0])) {
        const auto path = as_type_ptr<VariantWrapper<std::string>>(variants[0])->get();
        std::ifstream model_stream(path, std::ios::in | std::ifstream::binary);
        model_stream.seekg(0, model_stream.beg);
        const bool is_valid_model = onnx_common::is_valid_model(model_stream);
        model_stream.close();
        return is_valid_model;
    }
    if (variants.size() > 0 && is_type<VariantWrapper<std::istream*>>(variants[0])) {
        auto stream = as_type_ptr<VariantWrapper<std::istream*>>(variants[0])->get();
        return onnx_common::is_valid_model(*stream);
    }
    return false;
}
