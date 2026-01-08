// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cinttypes>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cre.hpp"

namespace intel_npu {

class BlobReader {
public:
    BlobReader(std::istream& source);

    BlobReader(const ov::Tensor& source);

    void read();

    void register_reader(const SectionID section_id,
                         std::function<std::shared_ptr<ISection>(BlobReader*, const size_t)> reader);

    std::shared_ptr<ISection> retrieve_section(const SectionID section_id);

private:
    friend class BlobWriter;

    void read_data_from_source(char* destination, const size_t size);

    size_t get_cursor_position();

    void move_cursor(const size_t offset);

    BlobSource m_source;
    std::shared_ptr<std::unordered_map<SectionID, uint64_t>> m_offsets_table;
    std::unordered_map<SectionID, std::shared_ptr<ISection>> m_parsed_sections;
    std::unordered_map<SectionID, std::function<std::shared_ptr<ISection>(BlobReader*, const size_t)>> m_readers;

    size_t m_cursor;
    std::streampos m_stream_base;
};

}  // namespace intel_npu
