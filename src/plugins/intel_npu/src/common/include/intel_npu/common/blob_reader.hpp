// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cinttypes>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "intel_npu/common/icapability.hpp"
#include "intel_npu/common/offsets_table.hpp"

namespace intel_npu {

class BlobReader {
public:
    BlobReader(const ov::Tensor& source);

    void read(const std::unordered_map<CRE::Token, std::shared_ptr<ICapability>>& plugin_capabilities);

    void register_reader(const SectionType type,
                         std::function<std::shared_ptr<ISection>(BlobReader*, const size_t)> reader);

    std::shared_ptr<ISection> retrieve_section(const SectionID& id);

    std::shared_ptr<ISection> retrieve_first_section(const SectionType section_type);

    std::optional<std::unordered_map<SectionTypeInstance, std::shared_ptr<ISection>>> retrieve_sections_same_type(
        const SectionType type);

    void copy_data_from_source(char* destination, const size_t size);

    const void* interpret_data_from_source(const size_t size);

    ov::Tensor get_roi_tensor(const size_t size);

    size_t get_cursor_relative_position();

    void move_cursor_to_relative_position(const size_t offset);

    static size_t get_npu_region_size(std::istream& stream);

    static size_t get_npu_region_size(const ov::Tensor& tensor);

private:
    friend class BlobWriter;

    std::reference_wrapper<const ov::Tensor> m_source;
    size_t m_npu_region_size;
    OffsetsTable m_offsets_table;
    std::unordered_map<SectionType, std::unordered_map<SectionTypeInstance, std::shared_ptr<ISection>>>
        m_parsed_sections;
    std::unordered_map<SectionType, std::function<std::shared_ptr<ISection>(BlobReader*, const size_t)>> m_readers;

    size_t m_cursor;
};

}  // namespace intel_npu
