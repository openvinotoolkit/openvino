// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/blob_reader.hpp"

namespace intel_npu {

BlobReader::BlobReader(std::istream& source) : m_source(source) {}

BlobReader::BlobReader(const ov::Tensor& source) : m_source(source), m_cursor_offset(0) {}

void BlobReader::register_reader(
    const SectionID section_id,
    std::function<std::shared_ptr<ISection>(const BlobSource&, const std::unordered_map<SectionID, uint64_t>&)>
        reader) {
    m_readers[section_id] = reader;
}

std::shared_ptr<ISection> BlobReader::retrieve_section(const SectionID section_id) {
    auto search_result = m_parsed_sections.find(section_id);
    if (search_result != m_parsed_sections.end()) {
        return search_result->second;
    }
    return nullptr;
}

// TODO test the windows debug build works properly if using the "better" implementation
void BlobReader::read_data_from_source(char* destination, const size_t size) {
    if (const std::reference_wrapper<std::istream>* stream =
            std::get_if<std::reference_wrapper<std::istream>>(&m_source)) {
        stream->get().read(destination, size);
    } else if (const std::reference_wrapper<const ov::Tensor>* tensor =
                   std::get_if<std::reference_wrapper<const ov::Tensor>>(&m_source)) {
        std::memcpy(destination, tensor->get().data<const char>() + m_cursor_offset, size);
        m_cursor_offset += size;
    }
}

void BlobReader::read() {}

}  // namespace intel_npu
