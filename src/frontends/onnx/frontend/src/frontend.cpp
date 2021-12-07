// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common/conversion_extension.hpp>
#include <common/frontend_exceptions.hpp>
#include <common/telemetry_extension.hpp>
#include <fstream>
#include <input_model.hpp>
#include <manager.hpp>
#include <onnx_frontend/frontend.hpp>
#include <onnx_import/onnx.hpp>
#include <sstream>
#include <utils/onnx_internal.hpp>
#include <onnx_import/onnx_utils.hpp>

#include "onnx_common/onnx_model_validator.hpp"
#include "conversion_extension.hpp"
#include "node_context.hpp"

using namespace ov;
using namespace ov::frontend;

ONNX_FRONTEND_C_API FrontEndVersion GetAPIVersion() {
    return OV_FRONTEND_API_VERSION;
}

ONNX_FRONTEND_C_API void* GetFrontEndData() {
    FrontEndPluginInfo* res = new FrontEndPluginInfo();
    res->m_name = "onnx";
    res->m_creator = []() {
        return std::make_shared<FrontEndONNX>();
    };
    return res;
}

InputModel::Ptr FrontEndONNX::load_impl(const std::vector<ov::Any>& variants) const {
    if (variants.size() == 0) {
        return nullptr;
    }
    if (variants[0].is<std::string>()) {
        const auto path = variants[0].as<std::string>();
        return std::make_shared<InputModelONNX>(path, m_telemetry);
    }
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    if (variants[0].is<std::wstring>()) {
        const auto path = variants[0].as<std::wstring>();
        return std::make_shared<InputModelONNX>(path, m_telemetry);
    }
#endif
    if (variants[0].is<std::istream*>()) {
        const auto stream = variants[0].as<std::istream*>();
        if (variants.size() > 1 && variants[1].is<std::string>()) {
            const auto path = variants[0].as<std::string>();
            return std::make_shared<InputModelONNX>(*stream, path, m_telemetry);
        }
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        if (variants.size() > 1 && variants[1].is<std::wstring>()) {
            const auto path = variants[1].as<std::wstring>();
            return std::make_shared<InputModelONNX>(*stream, path, m_telemetry);
        }
#endif
        return std::make_shared<InputModelONNX>(*stream, m_telemetry);
    }
    return nullptr;
}

std::shared_ptr<ngraph::Function> FrontEndONNX::convert(InputModel::Ptr model) const {
    auto model_onnx = std::dynamic_pointer_cast<InputModelONNX>(model);
    NGRAPH_CHECK(model_onnx != nullptr, "Invalid input model");
    return model_onnx->convert();
}

void FrontEndONNX::convert(std::shared_ptr<ngraph::Function> partially_converted) const {
    ngraph::onnx_import::detail::convert_decoded_function(partially_converted);
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

bool FrontEndONNX::supported_impl(const std::vector<ov::Any>& variants) const {
    if (variants.size() == 0) {
        return false;
    }
    std::ifstream model_stream;
    if (variants[0].is<std::string>()) {
        const auto path = variants[0].as<std::string>();
        model_stream.open(path, std::ios::in | std::ifstream::binary);
    }
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    else if (variants[0].is<std::wstring>()) {
        const auto path = variants[0].as<std::wstring>();
        model_stream.open(path, std::ios::in | std::ifstream::binary);
    }
#endif
    if (model_stream.is_open()) {
        model_stream.seekg(0, model_stream.beg);
        const bool is_valid_model = ngraph::onnx_common::is_valid_model(model_stream);
        model_stream.close();
        return is_valid_model;
    }
    if (variants[0].is<std::istream*>()) {
        const auto stream = variants[0].as<std::istream*>();
        StreamRewinder rwd{*stream};
        return ngraph::onnx_common::is_valid_model(*stream);
    }

    return false;
}

void FrontEndONNX::add_extension(const std::shared_ptr<ov::Extension>& extension) {
    if (auto telemetry = std::dynamic_pointer_cast<TelemetryExtension>(extension)) {
        m_telemetry = telemetry;
    } else if (const auto& so_ext = std::dynamic_pointer_cast<ov::detail::SOExtension>(extension)) {
        add_extension(so_ext->extension());
        m_extensions.push_back(so_ext);
    } else if (const auto& conv_ext = std::dynamic_pointer_cast<ov::frontend::ConversionExtensionBase>(extension)) {
        if (conv_ext) {
            // TODO: register for all opsets
            for (int i = 1; i < 13; ++i)
                ngraph::onnx_import::register_operator(
                        conv_ext->get_op_type(),
                    i,
                    "",
                    [=](const ngraph::onnx_import::Node& context) -> OutputVector {
                        return conv_ext->get_converter()(ov::frontend::onnx::NodeContext(context));
                    });
        }
        m_conversion_extensions.push_back(conv_ext);
    }
}
