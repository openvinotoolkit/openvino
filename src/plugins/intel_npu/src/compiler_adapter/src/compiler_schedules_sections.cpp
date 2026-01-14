// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler_schedules_sections.hpp"

#include <iterator>

#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"
#include "intel_npu/utils/utils.hpp"

namespace intel_npu {

ELFMainScheduleSection::ELFMainScheduleSection(const std::shared_ptr<Graph>& graph)
    : ISection(PredefinedSectionID::ELF_MAIN_SCHEDULE),
      m_graph(graph) {}

ELFMainScheduleSection::ELFMainScheduleSection(ov::Tensor main_schedule)
    : ISection(PredefinedSectionID::ELF_MAIN_SCHEDULE),
      m_main_schedule(main_schedule) {}

void ELFMainScheduleSection::write(std::ostream& stream, BlobWriter* writer) {
    // At import time, position "cursor = 0" is guaranteed to be aligned to the standard page size (4096). Therefore, we
    // only need to make sure the value of the cursor is a multiple of 4096 before writting any schedule.

    // Also take the padding size into account, we'll write that first
    const auto cursor = writer->get_stream_relative_position(stream) + sizeof(uint64_t);
    size_t cursor_and_padding = utils::align_size_to_standard_page_size(cursor);
    uint64_t padding_size = cursor_and_padding - cursor;
    stream.write(reinterpret_cast<const char*>(&padding_size), sizeof(padding_size));
    if (padding_size > 0) {
        std::fill_n(std::ostream_iterator<char>(stream), padding_size, 0);
    }
    m_graph->export_main_blob(stream);
}

void ELFMainScheduleSection::set_graph(const std::shared_ptr<Graph>& graph) {
    m_graph = graph;
    m_main_schedule = ov::Tensor();  // Don't need this anymore
}

ov::Tensor ELFMainScheduleSection::get_schedule() const {
    return m_main_schedule;
}

std::shared_ptr<ISection> ELFMainScheduleSection::read(BlobReader* blob_reader, const size_t section_length) {
    // Skip the first padding
    uint64_t padding_size;
    blob_reader->copy_data_from_source(reinterpret_cast<char*>(&padding_size), sizeof(padding_size));
    blob_reader->interpret_data_from_source(padding_size);

    return std::make_shared<ELFMainScheduleSection>(
        blob_reader->get_roi_tensor(section_length - sizeof(uint64_t) - padding_size));
}

ELFInitSchedulesSection::ELFInitSchedulesSection(const std::shared_ptr<WeightlessGraph>& weightless_graph)
    : ISection(PredefinedSectionID::ELF_INIT_SCHEDULES),
      m_weightless_graph(weightless_graph) {}

ELFInitSchedulesSection::ELFInitSchedulesSection(std::vector<ov::Tensor>& init_schedules)
    : ISection(PredefinedSectionID::ELF_INIT_SCHEDULES),
      m_init_schedules(std::move(init_schedules)) {}

void ELFInitSchedulesSection::write(std::ostream& stream, BlobWriter* writer) {
    const uint64_t number_of_inits = m_weightless_graph->get_number_of_inits();
    stream.write(reinterpret_cast<const char*>(&number_of_inits), sizeof(number_of_inits));

    // Placeholder until we get the sizes written in the stream
    const auto will_get_to_this_later = stream.tellp();
    std::fill_n(std::ostream_iterator<char>(stream), number_of_inits * sizeof(uint64_t), 0);

    // At import time, position "cursor = 0" is guaranteed to be aligned to the standard page size (4096). Therefore, we
    // only need to make sure the value of the cursor is a multiple of 4096 before writting any schedule.

    // Also take the padding size into account, we'll write that next
    const auto cursor = writer->get_stream_relative_position(stream) + sizeof(uint64_t);
    size_t cursor_and_padding = utils::align_size_to_standard_page_size(cursor);
    size_t padding_size = cursor_and_padding - cursor;
    stream.write(reinterpret_cast<const char*>(&padding_size), sizeof(padding_size));
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

void ELFInitSchedulesSection::set_graph(const std::shared_ptr<WeightlessGraph>& weightless_graph) {
    m_weightless_graph = weightless_graph;
    m_init_schedules = std::vector<ov::Tensor>();  // Don't need this anymore
}

std::vector<ov::Tensor> ELFInitSchedulesSection::get_schedules() const {
    return m_init_schedules;
}

std::shared_ptr<ISection> ELFInitSchedulesSection::read(BlobReader* blob_reader, const size_t section_length) {
    uint64_t number_of_inits;
    blob_reader->copy_data_from_source(reinterpret_cast<char*>(&number_of_inits), sizeof(number_of_inits));

    std::vector<uint64_t> init_sizes;
    uint64_t value;
    while (number_of_inits--) {
        blob_reader->copy_data_from_source(reinterpret_cast<char*>(&value), sizeof(value));
        init_sizes.push_back(value);
    }

    // Skip the first padding
    uint64_t padding_size;
    blob_reader->copy_data_from_source(reinterpret_cast<char*>(&padding_size), sizeof(padding_size));
    blob_reader->interpret_data_from_source(padding_size);

    std::vector<ov::Tensor> init_schedules;
    for (const auto& init_size : init_sizes) {
        init_schedules.push_back(blob_reader->get_roi_tensor(init_size));
    }

    return std::make_shared<ELFInitSchedulesSection>(init_schedules);
}

}  // namespace intel_npu
