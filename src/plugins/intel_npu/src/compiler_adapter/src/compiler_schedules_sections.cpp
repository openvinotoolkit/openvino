// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler_schedules_sections.hpp"

#include <iterator>

#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"
#include "intel_npu/utils/utils.hpp"

namespace {

constexpr intel_npu::ISection::SectionID ELF_MAIN_SCHEDULE_SECTION_ID = 102;
constexpr intel_npu::ISection::SectionID ELF_INIT_SCHEDULES_SECTION_ID = 103;

}  // namespace

namespace intel_npu {

ELFMainScheduleSection::ELFMainScheduleSection(const std::shared_ptr<Graph>& graph)
    : ISection(ELF_MAIN_SCHEDULE_SECTION_ID),
      m_graph(graph) {}

void ELFMainScheduleSection::write(std::ostream& stream, BlobWriter* writer) {
    // At import time, position "cursor = 0" is guaranteed to be aligned to the standard page size (4096). Therefore, we
    // only need to make sure the value of the cursor is a multiple of 4096 before writting any schedule.
    const auto cursor = writer->get_stream_relative_position(stream);
    size_t cursor_and_padding = utils::align_size_to_standard_page_size(cursor);
    size_t padding_size = cursor_and_padding - cursor;
    if (padding_size > 0) {
        std::fill_n(std::ostream_iterator<char>(stream), padding_size, 0);
    }
    m_graph->export_main_blob(stream);
}

ELFInitSchedulesSection::ELFInitSchedulesSection(const std::shared_ptr<WeightlessGraph>& weightless_graph)
    : ISection(ELF_INIT_SCHEDULES_SECTION_ID),
      m_weightless_graph(weightless_graph) {}

void ELFInitSchedulesSection::write(std::ostream& stream, BlobWriter* writer) {
    const uint64_t number_of_inits = m_weightless_graph->get_number_of_inits();
    stream.write(reinterpret_cast<const char*>(&number_of_inits), sizeof(number_of_inits));

    // Placeholder until we get the sizes written in the stream
    const auto will_get_to_this_later = stream.tellp();
    std::fill_n(std::ostream_iterator<char>(stream), number_of_inits * sizeof(uint64_t), 0);

    // At import time, position "cursor = 0" is guaranteed to be aligned to the standard page size (4096). Therefore, we
    // only need to make sure the value of the cursor is a multiple of 4096 before writting any schedule.
    const auto cursor = writer->get_stream_relative_position(stream);
    size_t cursor_and_padding = utils::align_size_to_standard_page_size(cursor);
    size_t padding_size = cursor_and_padding - cursor;
    if (padding_size > 0) {
        std::fill_n(std::ostream_iterator<char>(stream), padding_size, 0);
    }

    const std::vector<uint64_t> init_sizes = m_weightless_graph->export_init_blobs(stream);

    // Go back and write the sizes of the init schedules
    stream.seekp(will_get_to_this_later);
    for (const uint64_t init_size : init_sizes) {
        stream.write(reinterpret_cast<const char*>(&init_size), sizeof(init_size));
    }
}

}  // namespace intel_npu
