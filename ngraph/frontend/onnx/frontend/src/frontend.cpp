// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <frontend_manager/frontend_exceptions.hpp>
#include <frontend_manager/frontend_manager.hpp>
#include <fstream>
#include <input_model.hpp>
#include <onnx_frontend/frontend.hpp>
#include <frontend_manager/extension.hpp>
#include <onnx_import/onnx.hpp>
#include <onnx_import/onnx_utils.hpp>
#include <openvino/pass/manager.hpp>
#include <sstream>
#include <utils/onnx_internal.hpp>

#include "onnx_common/onnx_model_validator.hpp"
#include <openvino/op/util/framework_node.hpp>

using namespace ov;
using namespace ov::frontend;

using VariantString = VariantWrapper<std::string>;
using VariantWString = VariantWrapper<std::wstring>;
using VariantIstreamPtr = VariantWrapper<std::istream*>;

extern "C" ONNX_FRONTEND_API FrontEndVersion GetAPIVersion() {
    return OV_FRONTEND_API_VERSION;
}

extern "C" ONNX_FRONTEND_API void* GetFrontEndData() {
    FrontEndPluginInfo* res = new FrontEndPluginInfo();
    res->m_name = "onnx";
    res->m_creator = []() {
        auto frontend = std::make_shared<FrontEndONNX>();
        // TODO: Remove this, added for debugging purposes
#if 0
        frontend->add_extension(std::make_shared<DecoderTransformationExtension>([](std::shared_ptr<ov::Function> f){
            auto ops = f->get_ordered_ops();
            std::cerr << "HELLO! Run on function with " << ops.size() << " nodes\n";
            for(size_t i = 0; i < 10; ++i) {
                if(auto fwop = std::dynamic_pointer_cast<ov::op::util::FrameworkNode>(ops[i])) {
                    std::cerr << "op[" << i << "]: " << fwop->get_type_name() << ", attrs = {" << fwop->get_attrs() << "}\n";
                } else {
                    std::cerr << "Not a framework node\n";
                }
            }
            return true;
        }));
#endif

#if 0
        frontend->add_extension(std::make_shared<JsonConfigExtension>("/localdisk/slyalin/openvino_github/openvino_2/model-optimizer/extensions/front/onnx/mask_rcnn.json"));
#endif
        return frontend;
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
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
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
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
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
    auto telemetry = std::dynamic_pointer_cast<TelemetryExtension>(m_telemetry);
    if (!m_extensions.empty() || telemetry) {
        // The list of extension may contain not only decoder extensions
        // TODO: sort the extension initially in add_extension, avoid double checking of extension type
        ov::pass::Manager manager;
        bool activated = false;
        for (auto extension: m_extensions) {
            if (auto decoder_extension = std::dynamic_pointer_cast<DecoderTransformationExtension>(extension)) {
                decoder_extension->register_pass(manager);
                activated = true;
            }
        }

        if(activated || telemetry) {
            // at least one decoder transformation registered trigger alternative path with separate passes
            auto function = decode(model);
            telemetry->send("Number of nodes in original graph: " +std::to_string(function->get_ops().size()));
            manager.run_passes(function);
            convert(function);
            telemetry->send("Number of nodes in converted graph: " +std::to_string(function->get_ops().size()));
            return function;
        }
    }

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

bool FrontEndONNX::supported_impl(const std::vector<std::shared_ptr<Variant>>& variants) const {
    if (variants.size() == 0) {
        return false;
    }
    std::ifstream model_stream;
    if (ov::is_type<VariantString>(variants[0])) {
        const auto path = ov::as_type_ptr<VariantString>(variants[0])->get();
        model_stream.open(path, std::ios::in | std::ifstream::binary);
    }
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    else if (ov::is_type<VariantWString>(variants[0])) {
        const auto path = ov::as_type_ptr<VariantWString>(variants[0])->get();
        model_stream.open(path, std::ios::in | std::ifstream::binary);
    }
#endif
    if (model_stream.is_open()) {
        model_stream.seekg(0, model_stream.beg);
        const bool is_valid_model = ngraph::onnx_common::is_valid_model(model_stream);
        model_stream.close();
        return is_valid_model;
    }
    if (ov::is_type<VariantIstreamPtr>(variants[0])) {
        const auto stream = ov::as_type_ptr<VariantIstreamPtr>(variants[0])->get();
        StreamRewinder rwd{*stream};
        return ngraph::onnx_common::is_valid_model(*stream);
    }
    return false;
}

void FrontEndONNX::add_extension(const std::shared_ptr<ov::Extension>& extension) {
    if (std::dynamic_pointer_cast<DecoderTransformationExtension>(extension)) {
        m_extensions.push_back(extension);
    }

    if (auto telemetry = std::dynamic_pointer_cast<TelemetryExtension>(extension)) {
        m_telemetry = telemetry;
    }

    if (auto newop = std::dynamic_pointer_cast<ConversionExtension>(extension)) {
        std::cerr << "++++++++++++++++REGISTER NEW OP+++++++++: " << newop->m_optype << '\n';
        for (int i = 1; i < 13; ++i)
            onnx_import::register_operator(newop->m_optype, i, "", [=](const onnx_import::Node &context) {
                return newop->m_converter(
                        std::make_shared<NodeContext>(context.op_type(), context.get_ng_inputs()));
            });
    }
}