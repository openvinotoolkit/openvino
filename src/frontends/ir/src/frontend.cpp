// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/ir/frontend.hpp"

#include <array>
#include <vector>

#include "input_model.hpp"
#include "mmap_object.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/shared_buffer.hpp"
#include "openvino/core/any.hpp"
#include "openvino/util/file_util.hpp"
#include "so_extension.hpp"
#include "xml_parse_utils.h"

using namespace ov;

namespace ov {
namespace frontend {
namespace ir {
namespace {

inline size_t GetIRVersion(pugi::xml_node& root) {
    return XMLParseUtils::GetUIntAttr(root, "version", 0);
}

/**
 * @brief Extracts IR version from model stream
 * @param model Models stream
 * @return IR version, 0 if model does represent IR
 */
size_t GetIRVersion(std::istream& model) {
    std::array<char, 512> header{};

    model.seekg(0, model.beg);
    model.read(header.data(), header.size());
    model.clear();
    model.seekg(0, model.beg);

    pugi::xml_document doc;
    auto res =
        doc.load_buffer(header.data(), header.size(), pugi::parse_default | pugi::parse_fragment, pugi::encoding_utf8);

    if (res == pugi::status_ok) {
        pugi::xml_node root = doc.document_element();

        std::string node_name = root.name();
        std::transform(node_name.begin(), node_name.end(), node_name.begin(), ::tolower);

        if (node_name == "net") {
            return GetIRVersion(root);
        }
    }

    return 0;
}

}  // namespace

bool FrontEnd::supported_impl(const std::vector<ov::Any>& variants) const {
    std::ifstream local_model_stream;
    std::istream* provided_model_stream = nullptr;

    if (variants.empty() || variants.size() > 3) {
        return false;
    }

    const auto& model_variant = variants[0];
    if (model_variant.is<std::string>()) {
        const auto& path = model_variant.as<std::string>();
        local_model_stream.open(path, std::ios::in | std::ifstream::binary);
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    } else if (model_variant.is<std::wstring>()) {
        const auto& path = model_variant.as<std::wstring>();
        local_model_stream.open(path, std::ios::in | std::ifstream::binary);
#endif
    } else if (model_variant.is<std::istream*>()) {
        provided_model_stream = model_variant.as<std::istream*>();
    } else if (model_variant.is<std::istringstream*>()) {
        provided_model_stream = model_variant.as<std::istringstream*>();
    }

    if (provided_model_stream && local_model_stream.is_open()) {
        return false;
    }

    size_t version;
    if (provided_model_stream) {
        version = GetIRVersion(*provided_model_stream);
    } else if (local_model_stream.is_open()) {
        version = GetIRVersion(local_model_stream);
        local_model_stream.close();
    } else {
        return false;
    }

    return version >= 10 && version <= 11;
}

void FrontEnd::add_extension(const ov::Extension::Ptr& ext) {
    if (auto telemetry = std::dynamic_pointer_cast<TelemetryExtension>(ext)) {
        m_telemetry = telemetry;
    } else if (auto so_ext = std::dynamic_pointer_cast<ov::detail::SOExtension>(ext)) {
        if (std::dynamic_pointer_cast<ov::BaseOpExtension>(so_ext->extension())) {
            m_extensions.emplace_back(so_ext->extension());
        }
    } else if (std::dynamic_pointer_cast<ov::BaseOpExtension>(ext))
        m_extensions.emplace_back(ext);
}

InputModel::Ptr FrontEnd::load_impl(const std::vector<ov::Any>& variants) const {
    std::ifstream local_model_stream;
    std::istream* provided_model_stream = nullptr;
    std::shared_ptr<ngraph::runtime::AlignedBuffer> weights;

    auto create_extensions_map = [&]() -> std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr> {
        std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr> exts;
        for (const auto& ext : m_extensions) {
            if (auto base_ext = std::dynamic_pointer_cast<ov::BaseOpExtension>(ext))
                exts.insert({base_ext->get_type_info(), base_ext});
        }
        return exts;
    };

    auto create_input_model = [&]() -> std::shared_ptr<InputModel> {
        if (provided_model_stream) {
            return std::make_shared<InputModel>(*provided_model_stream, weights, create_extensions_map());
        } else if (local_model_stream.is_open()) {
            auto input_model = std::make_shared<InputModel>(local_model_stream, weights, create_extensions_map());
            local_model_stream.close();
            return input_model;
        }
        return nullptr;
    };

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    std::wstring weights_path, model_path;
#else
    std::string weights_path, model_path;
#endif

    const auto& model_variant = variants.at(0);

    if (model_variant.is<std::string>()) {
        const auto& tmp_path = model_variant.as<std::string>();
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        model_path = ov::util::string_to_wstring(tmp_path.c_str());
#else
        model_path = tmp_path;
#endif
        local_model_stream.open(model_path, std::ios::in | std::ifstream::binary);
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    } else if (model_variant.is<std::wstring>()) {
        model_path = model_variant.as<std::wstring>();
        local_model_stream.open(model_path, std::ios::in | std::ifstream::binary);
#endif
    } else if (model_variant.is<std::istream*>()) {
        provided_model_stream = model_variant.as<std::istream*>();
    } else if (model_variant.is<std::istringstream*>()) {
        provided_model_stream = model_variant.as<std::istringstream*>();
    }

    // Check weights and extensions
    for (size_t variant_id = 1; variant_id < variants.size(); ++variant_id) {
        const auto& variant = variants.at(variant_id);
        if (variant.is<std::string>()) {
            const auto& tmp_path = variant.as<std::string>();
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
            weights_path = ov::util::string_to_wstring(tmp_path.c_str());
#else
            weights_path = tmp_path;
#endif
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        } else if (variant.is<std::wstring>()) {
            weights_path = variant.as<std::wstring>();
#endif
        } else if (variant.is<std::shared_ptr<ngraph::runtime::AlignedBuffer>>()) {
            weights = variant.as<std::shared_ptr<ngraph::runtime::AlignedBuffer>>();
        }
    }

    // Find weights if only path to xml was provided
    if (weights_path.empty()) {
        auto pos = model_path.rfind('.');
        if (pos != model_path.npos)
            weights_path = model_path.substr(0, pos);

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        weights_path += L".bin";
#else
        weights_path += ".bin";
#endif
        if (!FileUtils::fileExist(weights_path)) {
            weights_path.clear();
        }
    }
    if (!weights_path.empty()) {
        std::ifstream bin_stream;
        bin_stream.open(weights_path, std::ios::binary);
        if (!bin_stream.is_open())
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
            IE_THROW() << "Weights file " + ov::util::wstring_to_string(weights_path) + " cannot be opened!";
#else
            IE_THROW() << "Weights file " + weights_path + " cannot be opened!";
#endif

        bin_stream.seekg(0, std::ios::end);
        size_t file_size = bin_stream.tellg();
        bin_stream.seekg(0, std::ios::beg);

        auto aligned_weights_buffer = std::make_shared<ngraph::runtime::AlignedBuffer>(file_size);
        bin_stream.read(aligned_weights_buffer->get_ptr<char>(), aligned_weights_buffer->size());
        bin_stream.close();

        weights = std::make_shared<ngraph::runtime::SharedBuffer<std::shared_ptr<ngraph::runtime::AlignedBuffer>>>(
            aligned_weights_buffer->get_ptr<char>(),
            aligned_weights_buffer->size(),
            aligned_weights_buffer);
    }

    return create_input_model();
}

std::shared_ptr<ov::Model> FrontEnd::convert(const InputModel::Ptr& model) const {
    auto ir_model = std::dynamic_pointer_cast<InputModel>(model);
    OPENVINO_ASSERT(ir_model != nullptr);
    return ir_model->convert();
}

std::string FrontEnd::get_name() const {
    return "ir";
}

}  // namespace ir
}  // namespace frontend
}  // namespace ov

IR_C_API ov::frontend::FrontEndVersion GetAPIVersion() {
    return OV_FRONTEND_API_VERSION;
}

IR_C_API void* GetFrontEndData() {
    frontend::FrontEndPluginInfo* res = new frontend::FrontEndPluginInfo();
    res->m_name = "ir";
    res->m_creator = []() {
        return std::make_shared<ov::frontend::ir::FrontEnd>();
    };
    return res;
}
