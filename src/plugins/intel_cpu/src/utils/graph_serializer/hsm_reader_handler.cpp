// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/graph_serializer/hsm_reader_handler.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>

#include "openvino/core/except.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/hsm_format.hpp"
#include "openvino/runtime/hsm_model_id_codec.hpp"
#include "openvino/runtime/hsm_reader.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "utils/graph_serializer/hsm_common.hpp"
#include "utils/graph_serializer/serializer.hpp"

namespace ov::intel_cpu {

namespace {

std::string read_string(const runtime::hsm::Section& section) {
    std::string value(section.size(), '\0');
    OPENVINO_ASSERT(section.read(value.data()), "[CPU] Cannot import compiled blob: failed to read a section.");
    return value;
}

// Maps the section from the source blob when the bytes are addressable and kept alive by it, otherwise copies.
std::shared_ptr<ov::AlignedBuffer> read_buffer(const std::shared_ptr<ov::AlignedBuffer>& source_blob,
                                               const runtime::hsm::Section& section) {
    if (section.size() == 0) {
        return nullptr;
    }
    if (const auto view = section.view(); source_blob && !view.empty()) {
        return std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(
            const_cast<std::byte*>(view.data()),
            view.size(),
            source_blob);
    }
    auto buffer = std::make_shared<ov::AlignedBuffer>(section.size());
    OPENVINO_ASSERT(section.read(buffer->get_ptr()), "[CPU] Cannot import compiled blob: failed to read a section.");
    return buffer;
}

// Rejected before the payload sections that follow are read.
std::string read_compatible_requirements(const runtime::hsm::Section& section) {
    // CVS-196454: decode with the common Runtime Requirements codec.
    OPENVINO_ASSERT(section.size() <= runtime_requirements_max_size,
                    "[CPU] Cannot import compiled blob: runtime requirements block is too large.");
    auto requirements = read_string(section);
    OPENVINO_ASSERT(is_runtime_requirements_compatible(requirements),
                    "[CPU] Cannot import compiled blob: incompatible runtime requirements, blob: ",
                    requirements,
                    ", current: ",
                    build_runtime_requirements());
    return requirements;
}

constexpr uint32_t requirements_key = hsm::section_key(hsm::runtime_requirements);
constexpr uint32_t model_struct_key = hsm::section_key(hsm::model_struct);

}  // namespace

const std::map<uint32_t, HsmReaderHandler::Decoder> HsmReaderHandler::s_decoders = {
    {hsm::section_key(hsm::model_id),
     [](HsmReaderHandler& handler, const runtime::hsm::Section& section) {
         handler.model_id = runtime::hsm::decode_model_id(section);
     }},
    {model_struct_key,
     [](HsmReaderHandler& handler, const runtime::hsm::Section& section) {
         OPENVINO_ASSERT(section.size() > 0, "[CPU] Cannot import compiled blob: empty model section.");
         handler.model_struct = read_buffer(handler.m_source_blob, section);
     }},
    {hsm::section_key(hsm::weights),
     [](HsmReaderHandler& handler, const runtime::hsm::Section& section) {
         // TODO: a mapped blob can prefetch these pages here so that reading a cold file overlaps the rest of the
         // import.
         handler.weights = read_buffer(handler.m_source_blob, section);
     }},
};

HsmReaderHandler::HsmReaderHandler(std::shared_ptr<ov::AlignedBuffer> source_blob)
    : m_source_blob(std::move(source_blob)) {}

bool HsmReaderHandler::handle_section(const runtime::hsm::ManifestEntry& entry, const runtime::hsm::Section& section) {
    if (const auto decoder = s_decoders.find(hsm::section_key(entry.device, entry.tag)); decoder != s_decoders.end()) {
        decoder->second(*this, section);
        return true;
    }
    return false;
}

void HsmReaderHandler::read_sections(const runtime::hsm::Reader& reader) {
    // Not left to the manifest order: an incompatible blob is rejected before any payload is read.
    const auto requirements = reader.section(hsm::runtime_requirements.device, hsm::runtime_requirements.tag.id());
    OPENVINO_ASSERT(requirements, "[CPU] Cannot import compiled blob: missing runtime requirements section.");
    runtime_requirements = read_compatible_requirements(*requirements);
    reader.read_sections({this});
}

void HsmReaderHandler::check_manifest(const runtime::hsm::Reader& reader) {
    std::unordered_set<uint32_t> seen;
    for (const auto& entry : reader.entries()) {
        const auto key = hsm::section_key(entry.device, entry.tag);
        // Sections this plugin does not use are ignored and may repeat.
        if (key == requirements_key || s_decoders.count(key) != 0) {
            OPENVINO_ASSERT(seen.insert(key).second, "[CPU] Cannot import compiled blob: duplicated section ", key);
        }
    }
    OPENVINO_ASSERT(seen.count(requirements_key) != 0,
                    "[CPU] Cannot import compiled blob: missing runtime requirements section.");
    OPENVINO_ASSERT(seen.count(model_struct_key) != 0, "[CPU] Cannot import compiled blob: missing model section.");
}

}  // namespace ov::intel_cpu
