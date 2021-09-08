// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <frontend_manager/frontend_exceptions.hpp>
#include <frontend_manager/frontend_manager.hpp>
#include <fstream>
#include <input_model.hpp>
#include <onnx_frontend/frontend.hpp>
#include <onnx_import/onnx.hpp>
#include <sstream>
#include <utils/onnx_internal.hpp>

#include "onnx_common/onnx_model_validator.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

using VariantString = VariantWrapper<std::string>;
using VariantWString = VariantWrapper<std::wstring>;
using VariantIstreamPtr = VariantWrapper<std::istream*>;

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
    if (variants.size() == 0) {
        return nullptr;
    }
    if (ov::is_type<VariantString>(variants[0])) {
        const auto path = ov::as_type_ptr<VariantString>(variants[0])->get();
        return std::make_shared<InputModelONNX>(path);
    }
#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    if (ov::is_type<VariantWString>(variants[0])) {
        const auto path = ov::as_type_ptr<VariantWString>(variants[0])->get();
        return std::make_shared<InputModelONNX>(path);
    }
#endif
    if (ov::is_type<VariantIstreamPtr>(variants[0])) {
        const auto stream = ov::as_type_ptr<VariantIstreamPtr>(variants[0])->get();
        if (variants.size() > 1 && ov::is_type<VariantString>(variants[1])) {
            const auto path = ov::as_type_ptr<VariantString>(variants[1])->get();
            return std::make_shared<InputModelONNX>(*stream, path);
        }
#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        if (variants.size() > 1 && ov::is_type<VariantWString>(variants[1])) {
            const auto path = ov::as_type_ptr<VariantWString>(variants[1])->get();
            return std::make_shared<InputModelONNX>(*stream, path);
        }
#endif
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

namespace {
/**
 * This helper struct uses RAII to rewind/reset the stream so that it points to the beginning
 * of the underlying resource (string, file, and so on). It works similarly to std::lock_guard,
 * which releases a mutex upon destruction.
 *
 * This ensures that the stream is always reset (exception, successful and unsuccessful
 * model validation).
 */
struct StreamRewinder {
    StreamRewinder(std::istream& stream) : m_stream(stream) {
        m_stream.seekg(0, m_stream.beg);
    }
    ~StreamRewinder() {
        m_stream.seekg(0, m_stream.beg);
    }

private:
    std::istream& m_stream;
};
}  // namespace

bool FrontEndONNX::supported_impl(const std::vector<std::shared_ptr<Variant>>& variants) const {
    if (variants.size() == 0) {
        return false;
    }
    std::ifstream model_stream;
    if (ov::is_type<VariantString>(variants[0])) {
        const auto path = ov::as_type_ptr<VariantString>(variants[0])->get();
        model_stream.open(path, std::ios::in | std::ifstream::binary);
    }
#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    else if (ov::is_type<VariantWString>(variants[0])) {
        const auto path = ov::as_type_ptr<VariantWString>(variants[0])->get();
        model_stream.open(path, std::ios::in | std::ifstream::binary);
    }
#endif
    if (model_stream.is_open()) {
        model_stream.seekg(0, model_stream.beg);
        const bool is_valid_model = onnx_common::is_valid_model(model_stream);
        model_stream.close();
        return is_valid_model;
    }
    if (ov::is_type<VariantIstreamPtr>(variants[0])) {
        const auto stream = ov::as_type_ptr<VariantIstreamPtr>(variants[0])->get();
        StreamRewinder rwd{*stream};
        return onnx_common::is_valid_model(*stream);
    }
    return false;
}
