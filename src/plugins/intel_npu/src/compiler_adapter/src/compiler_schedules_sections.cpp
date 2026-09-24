// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler_schedules_sections.hpp"

#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/tensor_decryption.hpp"
#include "intel_npu/utils/utils.hpp"

namespace {

using namespace intel_npu;

constexpr std::string_view INVALID_STATE_MESSAGE = "Invalid state";
constexpr std::string_view NEW_PAGE_ALIGNED_BUFFER_MESSAGE =
    "A new, page aligned buffer of size %zu has been allocated to host a compiled model";
constexpr std::string_view NULL_GRAPH_MESSAGE = "The section's code cannot operate on a null \"graph\" object";
constexpr std::string_view SECTION_TOO_SHORT_MESSAGE = "The given section is too short";
constexpr size_t SIZE_OF_INIT_SCHEDULE_SIZE = sizeof(uint64_t);
constexpr char LIST_START_DELIMITER = '[';
constexpr char LIST_END_DELIMITER = ']';

constexpr size_t SIZE_OF_PADDING_SIZE = sizeof(uint16_t);
constexpr size_t MINIMUM_MAIN_SCHEDULE_SECTION_SIZE = SIZE_OF_PADDING_SIZE;
constexpr size_t NUMBER_OF_INITS_SIZE = sizeof(uint16_t);
constexpr size_t MINIMUM_INIT_SCHEDULES_SECTION_SIZE = NUMBER_OF_INITS_SIZE;

std::function<std::string(const std::string&)> get_encryption_callback_from_config(
    const std::optional<FilteredConfig>& config) {
    OPENVINO_ASSERT(config.has_value(), "A config object is required to query the encryption callback");

    if (config->has(CACHE_ENCRYPTION_CALLBACKS::key().data()) &&
        config->get<CACHE_ENCRYPTION_CALLBACKS>().encrypt != nullptr) {
        return config->get<CACHE_ENCRYPTION_CALLBACKS>().encrypt;
    }
    return nullptr;
}

}  // namespace

namespace intel_npu {

ELFMainScheduleSection::ELFMainScheduleSection(
    const std::shared_ptr<Graph>& graph,
    const std::function<std::string(const std::string&)>& encryption_callback,
    const ov::log::Level log_level)
    : ISection(SectionTypeCode::ELF_MAIN_SCHEDULE),
      m_graph_or_schedule(graph),
      m_encryption_callback(encryption_callback),
      m_logger("ELFMainScheduleSection", log_level) {
    OPENVINO_ASSERT(graph, NULL_GRAPH_MESSAGE);
}

ELFMainScheduleSection::ELFMainScheduleSection(
    ov::Tensor&& main_schedule,
    const std::function<std::string(const std::string&)>& encryption_callback,
    const ov::log::Level log_level)
    : ISection(SectionTypeCode::ELF_MAIN_SCHEDULE),
      m_graph_or_schedule(std::move(main_schedule)),
      m_encryption_callback(encryption_callback),
      m_logger("ELFMainScheduleSection", log_level) {}

std::vector<std::shared_ptr<CREToken>> ELFMainScheduleSection::get_compatibility_requirements_subexpression(
    const std::unordered_map<SectionID, std::shared_ptr<ISection>>&
    /*all_registered_sections*/) const {
    OPENVINO_ASSERT(get_id().has_value());
    m_logger.debug("Added the ELF_MAIN_SCHEDULE_<ID> to the CRE");
    return {std::make_shared<SectionType>(get_type()), std::make_shared<SectionID>(get_id().value())};
}

void ELFMainScheduleSection::write(BlobWriterInterface& writer) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "ELFMainScheduleSection::write");
    const auto* graph = std::get_if<std::shared_ptr<Graph>>(&m_graph_or_schedule);
    OPENVINO_ASSERT(graph, INVALID_STATE_MESSAGE);

    // Also take the padding size into account, we'll write that first
    const size_t offset = writer.get_offset_relative_to_npu_region() + sizeof(uint16_t);
    const uint16_t padding_size = utils::align_size_to_standard_page_size(offset) - offset;
    writer.write_from(&padding_size, sizeof(padding_size));
    // TODO add method that adds padding until page aligned relative to NPU region start
    writer.add_padding(padding_size);

    m_logger.debug("Added %lu padding to offset %lu", padding_size, offset);

    if (!m_encryption_callback) {
        (*graph)->export_main_blob(writer.m_stream.get());
        return;
    }

    // Encrypt the compiler payload, then write it
    std::string encrypted_payload;
    {
        std::string tmp_plain_payload;
        {
            std::stringstream tmp_stream;
            (*graph)->export_main_blob(tmp_stream);  // +1x blob size
            tmp_plain_payload = tmp_stream.str();    // +2x blob size
        }  // -1x blob size when deallocating temporary stringstream
        encrypted_payload = m_encryption_callback(tmp_plain_payload);  // +2x blob size
    }  // -1x blob size when deallocating temporary blob string

    writer.write_from(encrypted_payload.c_str(), encrypted_payload.size());
}

void ELFMainScheduleSection::set_graph(const std::shared_ptr<Graph>& graph) {
    OPENVINO_ASSERT(graph, NULL_GRAPH_MESSAGE);
    OPENVINO_ASSERT(std::holds_alternative<ov::Tensor>(m_graph_or_schedule), INVALID_STATE_MESSAGE);
    m_graph_or_schedule = graph;
}

ov::Tensor ELFMainScheduleSection::get_schedule() const {
    const auto* schedule = std::get_if<ov::Tensor>(&m_graph_or_schedule);
    OPENVINO_ASSERT(schedule, INVALID_STATE_MESSAGE);
    return *schedule;
}

void ELFMainScheduleSection::decrypt(const std::function<std::string(const std::string&)>& decryption_callback) {
    auto* schedule = std::get_if<ov::Tensor>(&m_graph_or_schedule);
    OPENVINO_ASSERT(schedule, INVALID_STATE_MESSAGE);

    utils::decrypt_payload(*schedule, decryption_callback, m_logger);
}

std::shared_ptr<ISection> ELFMainScheduleSection::read(BlobReaderInterface& blob_reader) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "ELFMainScheduleSection::read");
    const Logger logger("ELFMainScheduleSection", blob_reader.get_log_level());
    OPENVINO_ASSERT(blob_reader.get_total_section_size() >= MINIMUM_MAIN_SCHEDULE_SECTION_SIZE,
                    SECTION_TOO_SHORT_MESSAGE);

    // Skip the first padding region
    // TODO double check no "size_t" used for r/w sizes
    uint16_t padding_size;
    blob_reader.read_into_buffer(&padding_size, sizeof(padding_size));
    OPENVINO_ASSERT(padding_size <= blob_reader.get_remaining_section_size(),
                    "The read padding size is greater than the length of the blob section");
    blob_reader.move_cursor_relative_to_current_section(blob_reader.get_offset_relative_to_current_section() +
                                                        padding_size);

    logger.debug("Skipped %lu padding from offset %lu", padding_size, blob_reader.get_offset_relative_to_npu_region());

    const size_t main_schedule_size = blob_reader.get_remaining_section_size();

    if (!blob_reader.source_is_contiguous()) {
        ov::Tensor main_schedule = utils::allocate_aligned_tensor(main_schedule_size);
        blob_reader.read_into_buffer(main_schedule.data(), main_schedule_size);

        logger.info(NEW_PAGE_ALIGNED_BUFFER_MESSAGE.data(), main_schedule_size);
        return std::make_shared<ELFMainScheduleSection>(std::move(main_schedule),
                                                        get_encryption_callback_from_config(blob_reader.get_config()),
                                                        logger.level());
    }

    OPENVINO_ASSERT(blob_reader.get_remaining_section_size() == 0, "Failed to read the whole content of the section");

    return std::make_shared<ELFMainScheduleSection>(blob_reader.create_roi_tensor(main_schedule_size),
                                                    get_encryption_callback_from_config(blob_reader.get_config()),
                                                    logger.level());
}

std::optional<std::string> ELFMainScheduleSection::get_individual_compatibility_requirements() const {
    const auto* graph = std::get_if<std::shared_ptr<Graph>>(&m_graph_or_schedule);
    OPENVINO_ASSERT(graph, INVALID_STATE_MESSAGE);
    std::optional<std::string_view> requirements = (*graph)->get_compatibility_descriptor();
    return requirements.has_value()
               ? std::make_optional(LIST_START_DELIMITER + std::string(requirements.value()) + LIST_END_DELIMITER)
               : std::nullopt;
}

ELFInitSchedulesSection::ELFInitSchedulesSection(
    const std::shared_ptr<WeightlessGraph>& weightless_graph,
    const std::function<std::string(const std::string&)>& encryption_callback,
    const ov::log::Level log_level)
    : ISection(SectionTypeCode::ELF_INIT_SCHEDULES),
      m_graph_or_schedules(weightless_graph),
      m_encryption_callback(encryption_callback),
      m_logger("ELFInitSchedulesSection", log_level) {
    OPENVINO_ASSERT(weightless_graph, NULL_GRAPH_MESSAGE);
}

ELFInitSchedulesSection::ELFInitSchedulesSection(
    std::vector<ov::Tensor>&& init_schedules,
    const std::function<std::string(const std::string&)>& encryption_callback,
    const ov::log::Level log_level)
    : ISection(SectionTypeCode::ELF_INIT_SCHEDULES),
      m_graph_or_schedules(std::move(init_schedules)),
      m_encryption_callback(encryption_callback),
      m_logger("ELFInitSchedulesSection", log_level) {}

std::vector<std::shared_ptr<CREToken>> ELFInitSchedulesSection::get_compatibility_requirements_subexpression(
    const std::unordered_map<SectionID, std::shared_ptr<ISection>>&
    /*all_registered_sections*/) const {
    m_logger.debug("Added the ELF_INIT_SCHEDULES section type to the CRE");
    return {std::make_shared<SectionType>(get_type())};
}

void ELFInitSchedulesSection::write(BlobWriterInterface& writer) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "ELFInitSchedulesSection::write");
    const auto* weightless_graph = std::get_if<std::shared_ptr<WeightlessGraph>>(&m_graph_or_schedules);
    OPENVINO_ASSERT(weightless_graph, INVALID_STATE_MESSAGE);

    // TODO check all written data types to eliminate some redundancy in size
    const uint16_t number_of_inits = (*weightless_graph)->get_number_of_inits();
    writer.write_from(&number_of_inits, sizeof(number_of_inits));

    m_logger.debug("Writting %lu init schedules", number_of_inits);

    // Placeholder until we get the sizes written in the stream
    const auto will_get_to_this_later = writer.get_offset_relative_to_current_section();
    writer.add_padding(number_of_inits * sizeof(uint64_t));

    // Also take the padding size into account, we'll write that next
    const size_t offset = writer.get_offset_relative_to_npu_region() + sizeof(uint16_t);
    const uint16_t padding_size = utils::align_size_to_standard_page_size(offset) - offset;
    writer.write_from(&padding_size, sizeof(padding_size));
    writer.add_padding(padding_size);

    std::vector<uint64_t> init_sizes = (*weightless_graph)->export_init_blobs(writer.m_stream.get());

    if (!m_encryption_callback) {
        init_sizes = (*weightless_graph)->export_init_blobs(writer.m_stream.get());
    } else {
        // Encrypt the compiler payload, then write it
        std::string encrypted_payload;
        {
            std::string tmp_plain_payload;
            {
                std::stringstream tmp_stream;
                init_sizes = (*weightless_graph)->export_init_blobs(tmp_stream);
                tmp_plain_payload = tmp_stream.str();  // +2x blob size
            }  // -1x blob size when deallocating temporary stringstream
            encrypted_payload = m_encryption_callback(tmp_plain_payload);  // +2x blob size
        }  // -1x blob size when deallocating temporary blob string

        writer.write_from(encrypted_payload.c_str(), encrypted_payload.size());
    }

    // Go back and write the sizes of the init schedules
    writer.move_cursor_relative_to_current_section(will_get_to_this_later);
    for (const uint64_t init_size : init_sizes) {
        writer.write_from(&init_size, sizeof(init_size));
        m_logger.debug("Init size %lu written", init_size);
    }
}

void ELFInitSchedulesSection::set_graph(const std::shared_ptr<WeightlessGraph>& weightless_graph) {
    OPENVINO_ASSERT(weightless_graph, NULL_GRAPH_MESSAGE);
    OPENVINO_ASSERT(std::holds_alternative<std::vector<ov::Tensor>>(m_graph_or_schedules), INVALID_STATE_MESSAGE);
    m_graph_or_schedules = weightless_graph;
}

std::vector<ov::Tensor> ELFInitSchedulesSection::get_schedules() const {
    const auto* schedules = std::get_if<std::vector<ov::Tensor>>(&m_graph_or_schedules);
    OPENVINO_ASSERT(schedules, INVALID_STATE_MESSAGE);
    return *schedules;
}

void ELFInitSchedulesSection::decrypt(const std::function<std::string(const std::string&)>& decryption_callback) {
    auto* schedules = std::get_if<std::vector<ov::Tensor>>(&m_graph_or_schedules);
    OPENVINO_ASSERT(schedules, INVALID_STATE_MESSAGE);

    for (ov::Tensor& schedule : *schedules) {
        utils::decrypt_payload(schedule, decryption_callback, m_logger);
    }
}

std::shared_ptr<ISection> ELFInitSchedulesSection::read(BlobReaderInterface& blob_reader) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "ELFInitSchedulesSection::read");
    Logger logger("ELFInitSchedulesSection", blob_reader.get_log_level());
    OPENVINO_ASSERT(blob_reader.get_total_section_size() >= MINIMUM_INIT_SCHEDULES_SECTION_SIZE,
                    SECTION_TOO_SHORT_MESSAGE);

    uint16_t number_of_inits;
    blob_reader.read_into_buffer(&number_of_inits, sizeof(number_of_inits));
    OPENVINO_ASSERT(number_of_inits <= blob_reader.get_remaining_section_size() / SIZE_OF_INIT_SCHEDULE_SIZE,
                    "The number of init schedules read from the blob is too great relative to the size of the section");

    logger.debug("Parsed number of init schedules: %lu", number_of_inits);

    size_t total_init_sizes = 0;
    std::vector<uint64_t> init_sizes;
    uint64_t value;
    while (number_of_inits--) {
        blob_reader.read_into_buffer(&value, sizeof(value));
        init_sizes.push_back(value);

        OPENVINO_ASSERT(total_init_sizes <= total_init_sizes + value, "Integer overflow");
        total_init_sizes += value;

        logger.debug("Init schedule parsed size: %lu", value);
    }

    OPENVINO_ASSERT(total_init_sizes < blob_reader.get_remaining_section_size(),
                    "The sum of the parsed init schedule sizes is too big for the current section size");

    // Skip the first padding
    uint16_t padding_size;
    blob_reader.read_into_buffer(&padding_size, sizeof(padding_size));
    blob_reader.move_cursor_relative_to_current_section(blob_reader.get_offset_relative_to_current_section() +
                                                        padding_size);

    std::vector<ov::Tensor> init_schedules;
    for (const auto& init_size : init_sizes) {
        ov::Tensor init_schedule;

        if (!blob_reader.source_is_contiguous()) {
            init_schedule = utils::allocate_aligned_tensor(init_size);
            blob_reader.read_into_buffer(init_schedule.data(), init_size);

            logger.info(NEW_PAGE_ALIGNED_BUFFER_MESSAGE.data(), init_size);
        } else {
            init_schedule = blob_reader.create_roi_tensor(init_size);
        }

        init_schedules.push_back(std::move(init_schedule));
    }

    OPENVINO_ASSERT(blob_reader.get_remaining_section_size() == 0, "Failed to read the whole content of the section");

    return std::make_shared<ELFInitSchedulesSection>(std::move(init_schedules),
                                                     get_encryption_callback_from_config(blob_reader.get_config()),
                                                     logger.level());
}

DynamicScheduleSection::DynamicScheduleSection(
    const std::shared_ptr<DynamicGraph>& graph,
    const std::function<std::string(const std::string&)>& encryption_callback,
    const ov::log::Level log_level)
    : ISection(SectionTypeCode::DYNAMIC_SCHEDULE),
      m_impl(std::dynamic_pointer_cast<Graph>(graph), encryption_callback, log_level),
      m_blob_type(graph->get_blob_type()),
      m_logger("DynamicScheduleSection", log_level) {}

DynamicScheduleSection::DynamicScheduleSection(
    ov::Tensor&& main_schedule,
    const BlobType blob_type,
    const std::function<std::string(const std::string&)>& encryption_callback,
    const ov::log::Level log_level)
    : ISection(SectionTypeCode::DYNAMIC_SCHEDULE),
      m_impl(std::move(main_schedule), encryption_callback, log_level),
      m_blob_type(blob_type),
      m_logger("DynamicScheduleSection", log_level) {}

std::vector<std::shared_ptr<CREToken>> DynamicScheduleSection::get_compatibility_requirements_subexpression(
    const std::unordered_map<SectionID, std::shared_ptr<ISection>>&
    /*all_registered_sections*/) const {
    OPENVINO_ASSERT(get_id().has_value());
    m_logger.debug("Added the DYNAMIC_SCHEDULE_<ID> to the CRE");
    return {std::make_shared<SectionType>(get_type()), std::make_shared<SectionID>(get_id().value())};
}

void DynamicScheduleSection::write(BlobWriterInterface& writer) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "DynamicScheduleSection::write");
    // TODO might need casting to uint8_t
    writer.write_from(&m_blob_type, sizeof(m_blob_type));
    m_impl.write(writer);
}

void DynamicScheduleSection::set_graph(const std::shared_ptr<DynamicGraph>& graph) {
    m_impl.set_graph(std::dynamic_pointer_cast<Graph>(graph));
}

ov::Tensor DynamicScheduleSection::get_schedule() const {
    return m_impl.get_schedule();
}

BlobType DynamicScheduleSection::get_blob_type() const {
    return m_blob_type;
}

void DynamicScheduleSection::decrypt(const std::function<std::string(const std::string&)>& decryption_callback) {
    m_impl.decrypt(decryption_callback);
}

std::shared_ptr<ISection> DynamicScheduleSection::read(BlobReaderInterface& blob_reader) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "DynamicScheduleSection::read");
    // TODO more logs

    BlobType blob_type;
    blob_reader.read_into_buffer(&blob_type, sizeof(blob_type));

    return std::make_shared<DynamicScheduleSection>(
        std::dynamic_pointer_cast<ELFMainScheduleSection>(ELFMainScheduleSection::read(blob_reader))->get_schedule(),
        blob_type,
        get_encryption_callback_from_config(blob_reader.get_config()),
        blob_reader.get_log_level());
}

std::optional<std::string> DynamicScheduleSection::get_individual_compatibility_requirements() const {
    // TODO is this correct?
    return m_impl.get_individual_compatibility_requirements();
}

}  // namespace intel_npu
