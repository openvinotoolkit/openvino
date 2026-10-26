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
#include "openvino/runtime/hsm_reader.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "utils/graph_serializer/hsm_common.hpp"

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
    if (const auto view = section.view(); source_blob && !view.empty()) {
        return std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(
            const_cast<std::byte*>(view.data()),
            view.size(),
            source_blob);
    } else {
        auto buffer = std::make_shared<ov::AlignedBuffer>(section.size());
        OPENVINO_ASSERT(section.read(buffer->get_ptr()),
                        "[CPU] Cannot import compiled blob: failed to read the model.");
        return buffer;
    }
}

template <class Slot, class T>
void set_once(Slot& slot, T value, const char* section_name) {
    OPENVINO_ASSERT(!slot, "[CPU] Cannot import compiled blob: duplicated ", section_name, " section.");
    slot = std::move(value);
}

}  // namespace

const std::map<uint32_t, HsmReaderHandler::Decoder> HsmReaderHandler::s_decoders = {
    {hsm::section_key(hsm::runtime_requirements),
     [](HsmReaderHandler& handler, const runtime::hsm::Section& section) {
         // TODO CVS-196454: decode with the common Runtime Requirements codec.
         set_once(handler.runtime_requirements, read_string(section), "runtime requirements");
     }},
    {hsm::section_key(hsm::serialized_model),
     [](HsmReaderHandler& handler, const runtime::hsm::Section& section) {
         set_once(handler.serialized_model, read_buffer(handler.m_source_blob, section), "model");
     }},
};

HsmReaderHandler::HsmReaderHandler(std::shared_ptr<ov::AlignedBuffer> source_blob)
    : m_source_blob(std::move(source_blob)) {}

bool HsmReaderHandler::handle_section(const runtime::hsm::ManifestEntry& entry, const runtime::hsm::Section& section) {
    const auto decoder = s_decoders.find(hsm::section_key(entry.device, entry.tag.id()));
    if (decoder == s_decoders.end()) {
        return false;
    }
    decoder->second(*this, section);
    return true;
}

void HsmReaderHandler::check_mandatory_sections() const {
    OPENVINO_ASSERT(runtime_requirements, "[CPU] Cannot import compiled blob: missing runtime requirements section.");
    OPENVINO_ASSERT(serialized_model, "[CPU] Cannot import compiled blob: missing model section.");
}

}  // namespace ov::intel_cpu
