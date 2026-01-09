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
    BlobReader(const ov::Tensor& source);

    void read(const std::unordered_set<CRE::Token>& plugin_capabilities_ids);

    void register_reader(const SectionID section_id,
                         std::function<std::shared_ptr<ISection>(BlobReader*, const size_t)> reader);

    std::shared_ptr<ISection> retrieve_section(const SectionID section_id);

    void copy_data_from_source(char* destination, const size_t size);

    template <class T>
    void interpret_data_from_source(T& destination);

    static size_t get_npu_region_size(std::istream& stream);

    static size_t get_npu_region_size(const ov::Tensor& tensor);

private:
    friend class BlobWriter;

    size_t get_cursor_position();

    void move_cursor(const size_t offset);

    std::reference_wrapper<const ov::Tensor> m_source;
    std::shared_ptr<std::unordered_map<SectionID, uint64_t>> m_offsets_table;
    std::unordered_map<SectionID, std::shared_ptr<ISection>> m_parsed_sections;
    std::unordered_map<SectionID, std::function<std::shared_ptr<ISection>(BlobReader*, const size_t)>> m_readers;

    size_t m_cursor;
};

}  // namespace intel_npu
