// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "hsm_reader_handler.hpp"

#include <algorithm>
#include <string_view>

#include "openvino/core/except.hpp"
#include "openvino/runtime/shared_buffer.hpp"

namespace ov::template_plugin {
namespace hsm = ov::runtime::hsm;

namespace {
constexpr uint32_t section_key(hsm::DeviceId device, uint32_t tag_id) noexcept {
    return (static_cast<uint32_t>(device) << 24) | tag_id;
}

std::optional<uint64_t> decode_model_id(const hsm::Section& section) {
    uint64_t id = 0;
    return section.size() >= sizeof(id) && section.read(0, sizeof(id), &id) ? std::optional(id) : std::nullopt;
}

std::optional<std::string> decode_string(const hsm::Section& section) {
    std::string value(section.size(), '\0');
    return section.read(value.data()) ? std::optional(std::move(value)) : std::nullopt;
}

std::optional<hsm_codec::BlobInfo> decode_blob_info(const hsm::Section& section) {
    hsm_codec::BlobInfo info{};
    return section.size() >= sizeof(info) && section.read(0, sizeof(info), &info) ? std::optional(info) : std::nullopt;
}

// Zero-copy only when the section's bytes are addressable AND something keeps them alive past this call
// (source_blob) - a stream-backed import has neither, so it always copies into an owned buffer.
std::shared_ptr<ov::AlignedBuffer> decode_buffer(const HsmReaderHandler& handler, const hsm::Section& section) {
    const auto section_view = section.view();
    if (handler.source_blob && !section_view.empty()) {
        return std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(
            const_cast<std::byte*>(section_view.data()),
            section_view.size(),
            handler.source_blob);
    } else {
        auto buffer = std::make_shared<ov::AlignedBuffer>(section.size());
        return section.read(buffer->get_ptr()) ? buffer : nullptr;
    }
}

// Inverse of hsm_codec::encode_compiled_options() - one `key=value` per line, split on the first '='.
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

void set_model_id(HsmReaderHandler& handler, const hsm::Section& section) {
    handler.model_id = decode_model_id(section);
}

void set_runtime_requirements(HsmReaderHandler& handler, const hsm::Section& section) {
    handler.runtime_requirements = decode_string(section);
}

void set_model(HsmReaderHandler& handler, const hsm::Section& section) {
    handler.model = decode_buffer(handler, section);
}

void set_weights(HsmReaderHandler& handler, const hsm::Section& section) {
    // An empty section means "no weights embedded" (weightless) - a null buffer represents that, matching
    // check_mandatory_sections()/read_hsm_blob()'s "falsy means absent" convention.
    handler.weights = section.size() == 0 ? nullptr : decode_buffer(handler, section);
}

void set_compiled_options(HsmReaderHandler& handler, const hsm::Section& section) {
    handler.compiled_options = decode_compiled_options(decode_string(section).value_or(""));
}

void set_blob_info(HsmReaderHandler& handler, const hsm::Section& section) {
    handler.blob_info = decode_blob_info(section);
}
}  // namespace

const std::map<uint32_t, HsmReaderHandler::Decoder> HsmReaderHandler::s_decoders = {
    // model id and weights are each registered under BOTH device scopes - a blob may carry either as a
    // common (any_device_id) tag or scoped to this device, decoded identically either way.
    {section_key(hsm::any_device_id, hsm_codec::weights.tag.id()), set_weights},
    {section_key(hsm_codec::model_id.device, hsm_codec::model_id.tag.id()), set_model_id},
    {section_key(hsm_codec::device_id, hsm_codec::model_id.tag.id()), set_model_id},
    {section_key(hsm_codec::runtime_requirements.device, hsm_codec::runtime_requirements.tag.id()),
     set_runtime_requirements},
    {section_key(hsm_codec::model_struct.device, hsm_codec::model_struct.tag.id()), set_model},
    {section_key(hsm_codec::weights.device, hsm_codec::weights.tag.id()), set_weights},
    {section_key(hsm_codec::compiled_options.device, hsm_codec::compiled_options.tag.id()), set_compiled_options},
    {section_key(hsm_codec::blob_info.device, hsm_codec::blob_info.tag.id()), set_blob_info},
};

bool HsmReaderHandler::handle_section(const hsm::ManifestEntry& entry, const hsm::Section& section) {
    if (const auto decoder = s_decoders.find(section_key(entry.device, entry.tag.id())); decoder != s_decoders.end()) {
        decoder->second(*this, section);
        return true;
    } else {
        return false;  // unrecognized (device, tag) pair - skip, per the unknown-tag rule.
    }
}

void HsmReaderHandler::check_mandatory_sections() const {
    OPENVINO_ASSERT(blob_info && blob_info->layout_version <= hsm_codec::BlobInfo::current_layout_version,
                    "TEMPLATE: missing or unsupported blob info section");
    OPENVINO_ASSERT(model, "TEMPLATE: missing mandatory model struct section");
}

}  // namespace ov::template_plugin
