// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <onnx_import/onnx.hpp>
#include <onnx_frontend/frontend.hpp>
#include <onnx_frontend/input_model.hpp>
#include <frontend_manager/frontend_manager.hpp>
#include <frontend_manager/frontend_exceptions.hpp>


using namespace ngraph;
using namespace ngraph::frontend;


extern "C" FRONTEND_API FrontEndVersion GetAPIVersion()
{
    return OV_FRONTEND_API_VERSION;
}

extern "C" FRONTEND_API void* GetFrontEndData()
{
    FrontEndPluginInfo* res = new FrontEndPluginInfo();
    res->m_name = "onnx";
    res->m_creator = [] (FrontEndCapFlags) { return std::make_shared<FrontEndONNX>(); };
    return res;
}

InputModel::Ptr FrontEndONNX::load_from_file(const std::string& path) const
{
    return std::make_shared<InputModelONNX>(path);
}

std::shared_ptr<ngraph::Function> FrontEndONNX::convert(InputModel::Ptr model) const
{
    auto model_onnx = std::dynamic_pointer_cast<InputModelONNX>(model);
    NGRAPH_CHECK(model_onnx != nullptr, "Invalid input model");
    return model_onnx->convert();
}

std::shared_ptr<ngraph::Function> FrontEndONNX::convert(std::shared_ptr<ngraph::Function> partially_converted) const
{
    return onnx_import::convert_decoded_function(partially_converted);
}

std::shared_ptr<ngraph::Function> FrontEndONNX::decode(InputModel::Ptr model) const
{
    auto model_onnx = std::dynamic_pointer_cast<InputModelONNX>(model);
    NGRAPH_CHECK(model_onnx != nullptr, "Invalid input model");
    return model_onnx->decode();
}
