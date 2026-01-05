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
    size_t cursorAndPadding = utils::align_size_to_standard_page_size(writer->cursor);
    size_t paddingSize = cursorAndPadding - writer->cursor;
    if (paddingSize > 0) {
        std::fill_n(std::ostream_iterator<char>(stream), paddingSize, 0);
    }
    writer->cursor += paddingSize;
    writer->cursor += m_graph->export_main_blob(stream);
}

ELFInitSchedulesSection::ELFInitSchedulesSection(const std::shared_ptr<WeightlessGraph>& weightless_graph)
    : ISection(ELF_INIT_SCHEDULES_SECTION_ID),
      m_weightless_graph(weightless_graph) {}

void ELFInitSchedulesSection::write(std::ostream& stream, BlobWriter* writer) {
    const uint64_t numberOfInits = m_weightless_graph->get_number_of_inits();
    stream.write(reinterpret_cast<const char*>(&numberOfInits), sizeof(numberOfInits));

    // Placeholder until we get the sizes written in the stream
    const auto willGetToThisLater = stream.tellp();
    std::fill_n(std::ostream_iterator<char>(stream), numberOfInits * sizeof(uint64_t), 0);
    writer->cursor += sizeof(numberOfInits) + numberOfInits * sizeof(uint64_t);

    // At import time, position "cursor = 0" is guaranteed to be aligned to the standard page size (4096). Therefore, we
    // only need to make sure the value of the cursor is a multiple of 4096 before writting any schedule.
    size_t cursorAndPadding = utils::align_size_to_standard_page_size(writer->cursor);
    size_t paddingSize = cursorAndPadding - writer->cursor;
    if (paddingSize > 0) {
        std::fill_n(std::ostream_iterator<char>(stream), paddingSize, 0);
    }
    writer->cursor += paddingSize;

    const std::vector<uint64_t> initSizes = m_weightless_graph->export_init_blobs(stream);
    writer->cursor += std::accumulate(initSizes.begin(), initSizes.end(), 0);

    // Go back and write the sizes of the init schedules
    stream.seekp(willGetToThisLater);
    for (const uint64_t initSize : initSizes) {
        stream.write(reinterpret_cast<const char*>(&initSize), sizeof(initSize));
    }

    // Leave the cursor at the end of the section
    stream.seekp(writer->stream_base + writer->cursor);
}

}  // namespace intel_npu
