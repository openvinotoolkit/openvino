// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "io_layouts_section.hpp"

#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"

namespace {

constexpr intel_npu::ISection::SectionID IO_LAYOUTS_SECTION_ID = 104;

}  // namespace

namespace intel_npu {

IOLayoutsSection::IOLayoutsSection(const std::vector<ov::Layout>& input_layouts,
                                   const std::vector<ov::Layout>& output_layouts)
    : ISection(IO_LAYOUTS_SECTION_ID),
      m_input_layouts(std::move(input_layouts)),
      m_output_layouts(std::move(output_layouts)) {}

void IOLayoutsSection::write(std::ostream& stream, BlobWriter* writer) {
    const uint64_t number_of_input_layouts = m_input_layouts.size();
    const uint64_t number_of_output_layouts = m_output_layouts.size();
    stream.write(reinterpret_cast<const char*>(&number_of_input_layouts), sizeof(number_of_input_layouts));
    stream.write(reinterpret_cast<const char*>(&number_of_output_layouts), sizeof(number_of_output_layouts));
    writer->cursor += sizeof(number_of_input_layouts) + sizeof(number_of_output_layouts);

    const auto write_layouts = [&](const std::vector<ov::Layout>& layouts) {
        for (const ov::Layout& layout : layouts) {
            const std::string layout_string = layout.to_string();
            const uint16_t string_length = static_cast<uint16_t>(layout_string.size());
            stream.write(reinterpret_cast<const char*>(&string_length), sizeof(string_length));
            stream.write(layout_string.c_str(), string_length);
            writer->cursor += sizeof(string_length) + string_length;
        }
    };

    write_layouts(m_input_layouts);
    write_layouts(m_output_layouts);
}

std::optional<uint64_t> IOLayoutsSection::get_length() const {
    const auto get_layouts_length = [&](const std::vector<ov::Layout>& layouts) -> uint64_t {
        uint64_t accumulator = 0;
        for (const ov::Layout& layout : layouts) {
            const uint16_t string_length = static_cast<uint16_t>(layout.to_string().size());
            accumulator += sizeof(string_length) + string_length;
        }
        return accumulator;
    };

    const uint64_t SIZE_OF_COUNTS = 2 * sizeof(uint64_t);  // number of input layouts + number of output layouts
    return SIZE_OF_COUNTS + get_layouts_length(m_input_layouts) + get_layouts_length(m_output_layouts);
}

}  // namespace intel_npu
