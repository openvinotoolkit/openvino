// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/graph_serializer/hsm_reader_handler.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
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
    } else if (const auto view = section.view(); source_blob && !view.empty()) {
        return std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(
            const_cast<std::byte*>(view.data()),
            view.size(),
            source_blob);
    } else {
        auto buffer = std::make_shared<ov::AlignedBuffer>(section.size());
        OPENVINO_ASSERT(section.read(buffer->get_ptr()),
                        "[CPU] Cannot import compiled blob: failed to read a section.");
        return buffer;
    }
}

// Stores a decoded section; a second section of the same kind makes the blob invalid.
template <class Slot, class T>
void store_unique_section(Slot& slot, T value, const char* section_name) {
    OPENVINO_ASSERT(!slot, "[CPU] Cannot import compiled blob: duplicated ", section_name, " section.");
    slot = std::move(value);
}

}  // namespace

const std::map<uint32_t, HsmReaderHandler::Decoder> HsmReaderHandler::s_decoders = {
    {hsm::section_key(hsm::runtime_requirements),
     [](HsmReaderHandler& handler, const runtime::hsm::Section& section) {
         // TODO CVS-196454: decode with the common Runtime Requirements codec.
         auto requirements = read_string(section);
         // Rejected here, before the payload sections that follow are read.
         OPENVINO_ASSERT(is_runtime_requirements_compatible(requirements),
                         "[CPU] Cannot import compiled blob: incompatible runtime requirements, blob: ",
                         requirements,
                         ", current: ",
                         build_runtime_requirements());
         store_unique_section(handler.runtime_requirements, std::move(requirements), "runtime requirements");
     }},
    {hsm::section_key(hsm::model_id),
     [](HsmReaderHandler& handler, const runtime::hsm::Section& section) {
         if (const auto id = runtime::hsm::decode_model_id(section)) {
             store_unique_section(handler.model_id, *id, "model id");
         }
     }},
    {hsm::section_key(hsm::model_struct),
     [](HsmReaderHandler& handler, const runtime::hsm::Section& section) {
         store_unique_section(handler.model_struct, read_buffer(handler.m_source_blob, section), "model");
     }},
    {hsm::section_key(hsm::weights),
     [](HsmReaderHandler& handler, const runtime::hsm::Section& section) {
         // TODO: a mapped blob can prefetch these pages here (ov::MappedMemory::hint_prefetch_async, or the constants
         // prefetch scheduler of PR #38993) so that reading a cold file overlaps the rest of the import.
         store_unique_section(handler.weights, read_buffer(handler.m_source_blob, section), "weights");
     }},
};

HsmReaderHandler::HsmReaderHandler(std::shared_ptr<ov::AlignedBuffer> source_blob)
    : m_source_blob(std::move(source_blob)) {}

bool HsmReaderHandler::handle_section(const runtime::hsm::ManifestEntry& entry, const runtime::hsm::Section& section) {
    if (const auto decoder = s_decoders.find(hsm::section_key(entry.device, entry.tag.id()));
        decoder != s_decoders.end()) {
        decoder->second(*this, section);
        return true;
    } else {
        return false;
    }
}

void HsmReaderHandler::check_mandatory_sections() const {
    OPENVINO_ASSERT(runtime_requirements, "[CPU] Cannot import compiled blob: missing runtime requirements section.");
    OPENVINO_ASSERT(model_struct, "[CPU] Cannot import compiled blob: missing model section.");
}

}  // namespace ov::intel_cpu
