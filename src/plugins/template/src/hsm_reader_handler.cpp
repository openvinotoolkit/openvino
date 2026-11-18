// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "hsm_reader_handler.hpp"

#include <algorithm>
#include <string_view>

#include "openvino/core/except.hpp"
#include "openvino/runtime/hsm_model_id_codec.hpp"
#include "openvino/runtime/shared_buffer.hpp"

namespace ov::template_plugin {

namespace {
std::optional<std::string> decode_string(const runtime::hsm::Section& section) {
    std::string value(section.size(), '\0');
    return section.read(value.data()) ? std::optional(std::move(value)) : std::nullopt;
}

std::optional<hsm::BlobInfo> decode_blob_info(const runtime::hsm::Section& section) {
    hsm::BlobInfo info{};
    return section.size() >= sizeof(info) && section.read(0, sizeof(info), &info) ? std::optional(info) : std::nullopt;
}

// Zero-copy only when the section's bytes are addressable AND something keeps them alive past this call
// (m_source_blob) - a stream-backed import has neither, so it always copies into an owned buffer.
std::shared_ptr<ov::AlignedBuffer> decode_buffer(const HsmReaderHandler& handler,
                                                 const runtime::hsm::Section& section) {
    const auto section_view = section.view();
    if (handler.m_source_blob && !section_view.empty()) {
        return std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(
            const_cast<std::byte*>(section_view.data()),
            section_view.size(),
            handler.m_source_blob);
    } else {
        auto buffer = std::make_shared<ov::AlignedBuffer>(section.size());
        return section.read(buffer->get_ptr()) ? buffer : nullptr;
    }
}

// Inverse of encode_compiled_options() in hsm_writer_handler.cpp - one `key=value` per line, split on the first '='.
ov::AnyMap decode_compiled_options(std::string_view encoded) {
    ov::AnyMap options;
    while (!encoded.empty()) {
        const auto line_end = std::min(encoded.find('\n'), encoded.size());
        const auto line = encoded.substr(0, line_end);
        if (const auto separator = line.find('='); separator != std::string_view::npos) {
            options.emplace(std::string(line.substr(0, separator)), std::string(line.substr(separator + 1)));
        }
        encoded.remove_prefix(std::min(line_end + 1, encoded.size()));
    }
    return options;
}

void set_model_id(HsmReaderHandler& handler, const runtime::hsm::Section& section) {
    handler.m_model_id = runtime::hsm::decode_model_id(section);
}

void set_runtime_requirements(HsmReaderHandler& handler, const runtime::hsm::Section& section) {
    handler.m_runtime_requirements = decode_string(section);
}

void set_model(HsmReaderHandler& handler, const runtime::hsm::Section& section) {
    handler.m_model = decode_buffer(handler, section);
}

void set_weights(HsmReaderHandler& handler, const runtime::hsm::Section& section) {
    // An empty section means "no weights embedded" (weightless) - a null buffer represents that, matching
    // check_mandatory_sections()/read_hsm_blob()'s "falsy means absent" convention.
    handler.m_weights = section.size() == 0 ? nullptr : decode_buffer(handler, section);
}

void set_compiled_options(HsmReaderHandler& handler, const runtime::hsm::Section& section) {
    handler.m_compiled_options = decode_compiled_options(decode_string(section).value_or(""));
}

void set_blob_info(HsmReaderHandler& handler, const runtime::hsm::Section& section) {
    handler.m_blob_info = decode_blob_info(section);
}
}  // namespace

const std::map<uint32_t, HsmReaderHandler::Decoder> HsmReaderHandler::s_decoders = {
    // model id and weights are each registered under BOTH device scopes - a blob may carry either as a
    // common (any_device_id) tag or scoped to this device, decoded identically either way.
    {hsm::section_key(runtime::hsm::any_device_id, hsm::weights.tag), set_weights},
    {hsm::section_key(hsm::model_id), set_model_id},
    {hsm::section_key(hsm::device_id, hsm::model_id.tag), set_model_id},
    {hsm::section_key(hsm::runtime_requirements), set_runtime_requirements},
    {hsm::section_key(hsm::model_struct), set_model},
    {hsm::section_key(hsm::weights), set_weights},
    {hsm::section_key(hsm::compiled_options), set_compiled_options},
    {hsm::section_key(hsm::blob_info), set_blob_info},
};

bool HsmReaderHandler::handle_section(const runtime::hsm::ManifestEntry& entry, const runtime::hsm::Section& section) {
    if (const auto decoder = s_decoders.find(hsm::section_key(entry.device, entry.tag)); decoder != s_decoders.end()) {
        decoder->second(*this, section);
        return true;
    } else {
        return false;  // unrecognized (device, tag) pair - skip, per the unknown-tag rule.
    }
}

void HsmReaderHandler::check_mandatory_sections() const {
    OPENVINO_ASSERT(m_blob_info && m_blob_info->layout_version <= hsm::BlobInfo::current_layout_version,
                    "TEMPLATE: missing or unsupported blob info section");
    OPENVINO_ASSERT(m_model, "TEMPLATE: missing mandatory model struct section");
}

}  // namespace ov::template_plugin
