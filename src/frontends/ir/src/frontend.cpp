// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/ir/frontend.hpp"

#include <array>
#include <pugixml.hpp>
#include <vector>

#include "input_model.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/so_extension.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/mmap_object.hpp"
#include "openvino/util/xml_parse_utils.hpp"
#include "transformations/resolve_names_collisions.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace ir {
namespace {

inline size_t get_ir_version(pugi::xml_node& root) {
    return static_cast<size_t>(ov::util::pugixml::get_uint64_attr(root, "version", 0));
}

/**
 * @brief Extracts IR version from model stream
 * @param model Models stream
 * @return IR version, 0 if model does represent IR
 */
size_t get_ir_version(std::istream& model) {
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
            return get_ir_version(root);
        }
    }

    return 0;
}

}  // namespace

bool FrontEnd::supported_impl(const std::vector<ov::Any>& variants) const {
    // Last boolean flag in `variants` (if presented) is reserved for FE configuration
    size_t extra_variants_num = variants.size() > 0 && variants[variants.size() - 1].is<bool>() ? 1 : 0;
    std::ifstream local_model_stream;
    std::istream* provided_model_stream = nullptr;

    if (variants.empty() || variants.size() > 3 + extra_variants_num) {
        return false;
    }

    const auto& model_variant = variants[0];
    if (model_variant.is<std::string>()) {
        const auto& path = model_variant.as<std::string>();
        validate_path(path);
        local_model_stream.open(path, std::ios::in | std::ifstream::binary);
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    } else if (model_variant.is<std::wstring>()) {
        const auto& path = model_variant.as<std::wstring>();
        validate_path(path);
        local_model_stream.open(path.c_str(), std::ios::in | std::ifstream::binary);
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
        version = get_ir_version(*provided_model_stream);
    } else if (local_model_stream.is_open()) {
        version = get_ir_version(local_model_stream);
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
    std::shared_ptr<ov::AlignedBuffer> weights;

    auto create_extensions_map = [&]() -> std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr> {
        std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr> exts;
        for (const auto& ext : m_extensions) {
            if (auto base_ext = std::dynamic_pointer_cast<ov::BaseOpExtension>(ext))
                exts.insert({base_ext->get_type_info(), base_ext});
        }
        return exts;
    };

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    auto create_input_model = [&](std::wstring& weights_path) -> std::shared_ptr<InputModel> {
#else
    auto create_input_model = [&](std::string& weights_path) -> std::shared_ptr<InputModel> {
#endif
        if (provided_model_stream) {
            return std::make_shared<InputModel>(*provided_model_stream, weights, create_extensions_map(), weights_path);
        } else if (local_model_stream.is_open()) {
            auto input_model = std::make_shared<InputModel>(local_model_stream, weights, create_extensions_map(), weights_path);
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
        validate_path(tmp_path);
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        model_path = ov::util::string_to_wstring(tmp_path.c_str());
#else
        model_path = tmp_path;
#endif
        local_model_stream.open(model_path.c_str(), std::ios::in | std::ifstream::binary);
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    } else if (model_variant.is<std::wstring>()) {
        model_path = model_variant.as<std::wstring>();
        validate_path(model_path);
        local_model_stream.open(model_path.c_str(), std::ios::in | std::ifstream::binary);
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
        } else if (variant.is<std::shared_ptr<ov::AlignedBuffer>>()) {
            weights = variant.as<std::shared_ptr<ov::AlignedBuffer>>();
        }
    }
    bool enable_mmap = variants[variants.size() - 1].is<bool>() ? variants[variants.size() - 1].as<bool>() : false;

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
        if (!ov::util::file_exists(weights_path)) {
            weights_path.clear();
        }
    }
    if (!weights_path.empty()) {
        if (enable_mmap) {
            auto mapped_memory = ov::load_mmap_object(weights_path);
            weights = std::make_shared<ov::SharedBuffer<std::shared_ptr<MappedMemory>>>(mapped_memory->data(),
                                                                                        mapped_memory->size(),
                                                                                        mapped_memory);
        } else {
            std::ifstream bin_stream;
            bin_stream.open(weights_path.c_str(), std::ios::binary);
            if (!bin_stream.is_open())
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
                OPENVINO_THROW("Weights file ", ov::util::wstring_to_string(weights_path), " cannot be opened!");
#else
                OPENVINO_THROW("Weights file ", weights_path, " cannot be opened!");
#endif

            bin_stream.seekg(0, std::ios::end);
            size_t file_size = bin_stream.tellg();
            bin_stream.seekg(0, std::ios::beg);

            auto aligned_weights_buffer = std::make_shared<ov::AlignedBuffer>(file_size);
            bin_stream.read(aligned_weights_buffer->get_ptr<char>(), aligned_weights_buffer->size());
            bin_stream.close();

            weights = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(
                aligned_weights_buffer->get_ptr<char>(),
                aligned_weights_buffer->size(),
                aligned_weights_buffer);
        }
    }

    return create_input_model(weights_path);
}

std::shared_ptr<ov::Model> FrontEnd::convert(const InputModel::Ptr& model) const {
    auto ir_model = std::dynamic_pointer_cast<InputModel>(model);
    OPENVINO_ASSERT(ir_model != nullptr);
    const auto& converted_model = ir_model->convert();
    normalize(converted_model);
    return converted_model;
}

std::string FrontEnd::get_name() const {
    return "ir";
}

void FrontEnd::normalize(const std::shared_ptr<ov::Model>& model) const {
    ov::pass::Manager manager("Frontend:IR:normalize");
    manager.register_pass<pass::ResolveNameCollisions>();
    manager.run_passes(model);
}

}  // namespace ir
}  // namespace frontend
}  // namespace ov

IR_C_API ov::frontend::FrontEndVersion get_api_version() {
    return OV_FRONTEND_API_VERSION;
}

IR_C_API void* get_front_end_data() {
    ov::frontend::FrontEndPluginInfo* res = new ov::frontend::FrontEndPluginInfo();
    res->m_name = "ir";
    res->m_creator = []() {
        return std::make_shared<ov::frontend::ir::FrontEnd>();
    };
    return res;
}
