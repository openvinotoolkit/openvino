// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler_schedules_sections.hpp"

#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/utils/utils.hpp"

namespace {

constexpr std::string_view INVALID_STATE_MESSAGE = "Invalid state";

}

namespace intel_npu {

ELFMainScheduleSection::ELFMainScheduleSection(const std::shared_ptr<Graph>& graph, const ov::log::Level log_level)
    : ISection(PredefinedSectionType::ELF_MAIN_SCHEDULE),
      m_graph_or_schedule(graph),
      m_logger("ELFMainScheduleSection", log_level) {}

ELFMainScheduleSection::ELFMainScheduleSection(ov::Tensor main_schedule, const ov::log::Level log_level)
    : ISection(PredefinedSectionType::ELF_MAIN_SCHEDULE),
      m_graph_or_schedule(main_schedule),
      m_logger("ELFMainScheduleSection", log_level) {}

std::vector<CREToken> ELFMainScheduleSection::get_compatibility_requirements_subexpression(
    const std::unordered_map<SectionType, std::unordered_map<SectionTypeInstance, std::shared_ptr<ISection>>>&
    /*all_registered_sections*/) const {
    m_logger.debug("Added the ELF_MAIN_SCHEDULE section type to the CRE");
    return {get_section_type()};
}

void ELFMainScheduleSection::write(BlobWriterInterface& writer) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "ELFMainScheduleSection::write");
    const auto* graph = std::get_if<std::shared_ptr<Graph>>(&m_graph_or_schedule);
    OPENVINO_ASSERT(graph, INVALID_STATE_MESSAGE);

    // At import time, position "cursor = 0" is guaranteed to be aligned to the standard page size (4096). Therefore, we
    // only need to make sure the value of the cursor is a multiple of 4096 before writting any schedule.

    // Also take the padding size into account, we'll write that first
    const size_t offset = writer.get_offset_relative_to_npu_region();
    const size_t padding_size = utils::align_size_to_standard_page_size(offset) - offset;
    writer.add_padding(padding_size);

    m_logger.debug("Added %lu padding to offset %lu", padding_size, offset);

    (*graph)->export_main_blob(writer.m_stream.get());
}

void ELFMainScheduleSection::set_graph(const std::shared_ptr<Graph>& graph) {
    OPENVINO_ASSERT(std::holds_alternative<ov::Tensor>(m_graph_or_schedule), INVALID_STATE_MESSAGE);
    m_graph_or_schedule = graph;
}

ov::Tensor ELFMainScheduleSection::get_schedule() const {
    const auto* schedule = std::get_if<ov::Tensor>(&m_graph_or_schedule);
    OPENVINO_ASSERT(schedule, INVALID_STATE_MESSAGE);
    return *schedule;
}

std::shared_ptr<ISection> ELFMainScheduleSection::read(BlobReaderInterface& blob_reader) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "ELFMainScheduleSection::read");
    const Logger logger("ELFMainScheduleSection", blob_reader.get_log_level());

    // Skip the first padding region
    const size_t offset = blob_reader.get_offset_relative_to_npu_region();
    const size_t padding_size = utils::align_size_to_standard_page_size(offset) - offset;
    blob_reader.interpret_from_source(padding_size);  // moves the cursor

    logger.debug("Skipped %lu padding from offset %lu", padding_size, offset);

    return std::make_shared<ELFMainScheduleSection>(
        blob_reader.get_roi_tensor(blob_reader.get_section_length() - padding_size),
        logger.level());
}

ELFInitSchedulesSection::ELFInitSchedulesSection(const std::shared_ptr<WeightlessGraph>& weightless_graph,
                                                 const ov::log::Level log_level)
    : ISection(PredefinedSectionType::ELF_INIT_SCHEDULES),
      m_graph_or_schedules(weightless_graph),
      m_logger("ELFInitSchedulesSection", log_level) {}

ELFInitSchedulesSection::ELFInitSchedulesSection(std::vector<ov::Tensor>& init_schedules,
                                                 const ov::log::Level log_level)
    : ISection(PredefinedSectionType::ELF_INIT_SCHEDULES),
      m_graph_or_schedules(std::move(init_schedules)),
      m_logger("ELFInitSchedulesSection", log_level) {}

std::vector<CREToken> ELFInitSchedulesSection::get_compatibility_requirements_subexpression(
    const std::unordered_map<SectionType, std::unordered_map<SectionTypeInstance, std::shared_ptr<ISection>>>&
    /*all_registered_sections*/) const {
    m_logger.debug("Added the ELF_INIT_SCHEDULES section type to the CRE");
    return {get_section_type()};
}

void ELFInitSchedulesSection::write(BlobWriterInterface& writer) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "ELFInitSchedulesSection::write");
    const auto* weightless_graph = std::get_if<std::shared_ptr<WeightlessGraph>>(&m_graph_or_schedules);
    OPENVINO_ASSERT(weightless_graph, INVALID_STATE_MESSAGE);

    const uint64_t number_of_inits = (*weightless_graph)->get_number_of_inits();
    writer.write_from(&number_of_inits, sizeof(number_of_inits));

    m_logger.debug("Writting %lu init schedules", number_of_inits);

    // Placeholder until we get the sizes written in the stream
    const auto will_get_to_this_later = writer.get_offset_relative_to_current_section();
    writer.add_padding(number_of_inits * sizeof(uint64_t));

    // At import time, position "cursor = 0" is guaranteed to be aligned to the standard page size (4096). Therefore, we
    // only need to make sure the value of the cursor is a multiple of 4096 before writting any schedule.

    // Also take the padding size into account, we'll write that next
    const size_t offset = writer.get_offset_relative_to_npu_region();
    const size_t padding_size = utils::align_size_to_standard_page_size(offset) - offset;
    writer.add_padding(padding_size);

    const std::vector<uint64_t> init_sizes = (*weightless_graph)->export_init_blobs(writer.m_stream.get());

    // Go back and write the sizes of the init schedules
    writer.move_cursor_relative_to_current_section(will_get_to_this_later);
    for (const uint64_t init_size : init_sizes) {
        writer.write_from(&init_size, sizeof(init_size));
        m_logger.debug("Init size %lu written", init_size);
    }
}

void ELFInitSchedulesSection::set_graph(const std::shared_ptr<WeightlessGraph>& weightless_graph) {
    OPENVINO_ASSERT(std::holds_alternative<std::vector<ov::Tensor>>(m_graph_or_schedules), INVALID_STATE_MESSAGE);
    m_graph_or_schedules = weightless_graph;
}

std::vector<ov::Tensor> ELFInitSchedulesSection::get_schedules() const {
    const auto* schedules = std::get_if<std::vector<ov::Tensor>>(&m_graph_or_schedules);
    OPENVINO_ASSERT(schedules, INVALID_STATE_MESSAGE);
    return *schedules;
}

std::shared_ptr<ISection> ELFInitSchedulesSection::read(BlobReaderInterface& blob_reader) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "ELFInitSchedulesSection::read");
    Logger logger("ELFInitSchedulesSection", blob_reader.get_log_level());

    const size_t section_length = blob_reader.get_section_length();

    uint64_t number_of_inits;
    blob_reader.copy_from_source(reinterpret_cast<char*>(&number_of_inits), sizeof(number_of_inits));
    OPENVINO_ASSERT(
        number_of_inits * sizeof(uint64_t) < section_length,
        "The parsed number of init schedules is too big for the current section size. Number of init schedules: ",
        number_of_inits,
        ". Section length: ",
        section_length);

    logger.debug("Parsed number of init schedules: %lu", number_of_inits);

    size_t total_init_sizes = 0;
    std::vector<uint64_t> init_sizes;
    uint64_t value;
    while (number_of_inits--) {
        blob_reader.copy_from_source(reinterpret_cast<char*>(&value), sizeof(value));
        init_sizes.push_back(value);
        total_init_sizes += value;

        logger.debug("Init schedule parsed size: %lu", value);
    }

    OPENVINO_ASSERT(total_init_sizes < blob_reader.get_section_length(),
                    "The sum of the parsed init schedule sizes is too big for the current section size. Sum: ",
                    total_init_sizes,
                    ". Section length: ",
                    section_length);

    // Skip the first padding
    const size_t offset = blob_reader.get_offset_relative_to_npu_region();
    const size_t padding_size = utils::align_size_to_standard_page_size(offset) - offset;
    blob_reader.interpret_from_source(padding_size);

    std::vector<ov::Tensor> init_schedules;
    for (const auto& init_size : init_sizes) {
        init_schedules.push_back(blob_reader.get_roi_tensor(init_size));
    }

    return std::make_shared<ELFInitSchedulesSection>(init_schedules, logger.level());
}

}  // namespace intel_npu
