// Copyright (C) 2018-2026 Intel Corporation
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

constexpr size_t NUMBER_OF_INITS_SIZE = sizeof(uint16_t);
constexpr size_t MINIMUM_INIT_SCHEDULES_SECTION_SIZE = NUMBER_OF_INITS_SIZE;

const ov::runtime::hsm::SectionAlignment PAGE_ALIGNED_BEGINNING{utils::STANDARD_PAGE_SIZE, 1};

}  // namespace

namespace intel_npu {

ELFMainScheduleSection::ELFMainScheduleSection(
    const std::shared_ptr<Graph>& graph,
    const std::function<std::string(const std::string&)>& encryption_callback,
    const ov::log::Level log_level)
    : ISection(SectionTypeCode::ELF_MAIN_SCHEDULE, PAGE_ALIGNED_BEGINNING),
      m_graph_or_schedule(graph),
      m_encryption_callback(encryption_callback),
      m_logger("ELFMainScheduleSection", log_level) {
    OPENVINO_ASSERT(graph, NULL_GRAPH_MESSAGE);
}

ELFMainScheduleSection::ELFMainScheduleSection(
    ov::Tensor&& main_schedule,
    const std::function<std::string(const std::string&)>& encryption_callback,
    const ov::log::Level log_level)
    : ISection(SectionTypeCode::ELF_MAIN_SCHEDULE, PAGE_ALIGNED_BEGINNING),
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

// TODO remember to register this with page alignment
void ELFMainScheduleSection::encode(const ov::runtime::hsm::SectionSink& sink) const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "ELFMainScheduleSection::write");
    const auto* graph = std::get_if<std::shared_ptr<Graph>>(&m_graph_or_schedule);
    OPENVINO_ASSERT(graph, INVALID_STATE_MESSAGE);

    if (!m_encryption_callback) {
        (*graph)->export_main_blob(sink);
        return;
    }

    // Encrypt the compiler payload, then write it
    std::string encrypted_payload;
    {
        std::string tmp_plain_payload;
        {
            std::stringstream tmp_stream;
            (*graph)->export_main_blob([&tmp_stream](ov::util::MemoryView data) {
                tmp_stream.write(reinterpret_cast<const char*>(data.data()), data.size());
                OPENVINO_ASSERT(tmp_stream, "Stream invalidated during blob writing");
            });                                    // +1x blob size
            tmp_plain_payload = tmp_stream.str();  // +2x blob size
        }  // -1x blob size when deallocating temporary stringstream
        encrypted_payload = m_encryption_callback(tmp_plain_payload);  // +2x blob size
    }  // -1x blob size when deallocating temporary blob string

    sink({reinterpret_cast<const std::byte*>(encrypted_payload.c_str()), encrypted_payload.size()});
}

void ELFMainScheduleSection::set_graph(const std::shared_ptr<Graph>& graph) {
    OPENVINO_ASSERT(graph, NULL_GRAPH_MESSAGE);
    OPENVINO_ASSERT(std::holds_alternative<ov::Tensor>(m_graph_or_schedule), INVALID_STATE_MESSAGE);
    m_graph_or_schedule = graph;
}

// TODO remember to call this after parsing. Maybe a test could be useful for this
void ELFMainScheduleSection::set_encryption_callback(
    const std::function<std::string(const std::string&)> encryption_callback) {
    m_encryption_callback = encryption_callback;
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

ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>> ELFMainScheduleSection::decode =
    [](const ov::runtime::hsm::Section& section) -> std::optional<std::shared_ptr<ISection>> {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "ELFMainScheduleSection::decode");
    const Logger logger("ELFMainScheduleSection", Logger::global().level());
    // TODO test these
    OPENVINO_ASSERT(section.tag().id() == static_cast<uint32_t>(SectionTypeCode::ELF_MAIN_SCHEDULE),
                    "Logic error: the decoder received a section of the wrong type");

    // TODO update the tests, there is a minimum size no more
    // TODO double check no "size_t" used for r/w sizes
    const size_t main_schedule_size = section.size();
    std::shared_ptr<ELFMainScheduleSection> result;
    const ov::util::MemoryView content_view = section.view();

    if (content_view.empty()) {
        ov::Tensor main_schedule = utils::allocate_aligned_tensor(main_schedule_size);
        section.read(main_schedule.data());

        logger.info(NEW_PAGE_ALIGNED_BUFFER_MESSAGE.data(), main_schedule_size);
        result = std::make_shared<ELFMainScheduleSection>(std::move(main_schedule), nullptr, logger.level());
    } else {
        result = std::make_shared<ELFMainScheduleSection>(
            ov::Tensor(ov::element::Type_t::u8, ov::Shape({content_view.size()}), content_view.data()),
            nullptr,
            logger.level());
    }

    return result;
};

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
    : ISection(SectionTypeCode::ELF_INIT_SCHEDULES, PAGE_ALIGNED_BEGINNING),
      m_graph_or_schedules(weightless_graph),
      m_encryption_callback(encryption_callback),
      m_logger("ELFInitSchedulesSection", log_level) {
    OPENVINO_ASSERT(weightless_graph, NULL_GRAPH_MESSAGE);
}

ELFInitSchedulesSection::ELFInitSchedulesSection(
    std::vector<ov::Tensor>&& init_schedules,
    const std::function<std::string(const std::string&)>& encryption_callback,
    const ov::log::Level log_level)
    : ISection(SectionTypeCode::ELF_INIT_SCHEDULES, PAGE_ALIGNED_BEGINNING),
      m_graph_or_schedules(std::move(init_schedules)),
      m_encryption_callback(encryption_callback),
      m_logger("ELFInitSchedulesSection", log_level) {}

std::vector<std::shared_ptr<CREToken>> ELFInitSchedulesSection::get_compatibility_requirements_subexpression(
    const std::unordered_map<SectionID, std::shared_ptr<ISection>>&
    /*all_registered_sections*/) const {
    OPENVINO_ASSERT(get_id().has_value());
    m_logger.debug("Added the ELF_INIT_SCHEDULES section type to the CRE");
    return {std::make_shared<SectionType>(get_type()), std::make_shared<SectionID>(get_id().value())};
}

// TODO document the form of the section on disk
void ELFInitSchedulesSection::encode(const ov::runtime::hsm::SectionSink& sink) const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "ELFInitSchedulesSection::write");
    const auto* weightless_graph = std::get_if<std::shared_ptr<WeightlessGraph>>(&m_graph_or_schedules);
    OPENVINO_ASSERT(weightless_graph, INVALID_STATE_MESSAGE);

    // TODO check all written data types to eliminate some redundancy in size
    const uint16_t number_of_inits = (*weightless_graph)->get_number_of_inits();
    m_logger.debug("Writting %lu init schedules", number_of_inits);

    std::vector<uint64_t> init_sizes;
    if (!m_encryption_callback) {
        init_sizes = (*weightless_graph)->export_init_blobs(sink);
    } else {
        // Encrypt the compiler payload, then write it
        std::string encrypted_payload;
        {
            std::string tmp_plain_payload;
            {
                std::stringstream tmp_stream;
                init_sizes = (*weightless_graph)->export_init_blobs([&tmp_stream](ov::util::MemoryView data) {
                    tmp_stream.write(reinterpret_cast<const char*>(data.data()), data.size());
                    OPENVINO_ASSERT(tmp_stream, "Stream invalidated during blob writing");
                });
                tmp_plain_payload = tmp_stream.str();  // +2x blob size
            }  // -1x blob size when deallocating temporary stringstream
            encrypted_payload = m_encryption_callback(tmp_plain_payload);  // +2x blob size
        }  // -1x blob size when deallocating temporary blob string

        sink({reinterpret_cast<const std::byte*>(encrypted_payload.c_str()), encrypted_payload.size()});
    }

    // Go back and write the sizes of the init schedules
    for (const uint64_t init_size : init_sizes) {
        sink({reinterpret_cast<const std::byte*>(&init_size), sizeof(init_size)});
        m_logger.debug("Init size %lu written", init_size);
    }

    sink({reinterpret_cast<const std::byte*>(&number_of_inits), sizeof(number_of_inits)});
}

void ELFInitSchedulesSection::set_graph(const std::shared_ptr<WeightlessGraph>& weightless_graph) {
    OPENVINO_ASSERT(weightless_graph, NULL_GRAPH_MESSAGE);
    OPENVINO_ASSERT(std::holds_alternative<std::vector<ov::Tensor>>(m_graph_or_schedules), INVALID_STATE_MESSAGE);
    m_graph_or_schedules = weightless_graph;
}

void ELFInitSchedulesSection::set_encryption_callback(
    const std::function<std::string(const std::string&)> encryption_callback) {
    m_encryption_callback = encryption_callback;
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

ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>> ELFInitSchedulesSection::decode =
    [](const ov::runtime::hsm::Section& section) -> std::optional<std::shared_ptr<ISection>> {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "ELFInitSchedulesSection::decode");
    Logger logger("ELFInitSchedulesSection", Logger::global().level());

    OPENVINO_ASSERT(section.tag().id() == static_cast<uint32_t>(SectionTypeCode::ELF_INIT_SCHEDULES),
                    "Logic error: the decoder received a section of the wrong type");
    if (section.size() < MINIMUM_INIT_SCHEDULES_SECTION_SIZE) {
        logger.error(SECTION_TOO_SHORT_MESSAGE.data());
        return std::nullopt;
    }

    uint16_t number_of_inits;
    size_t offset = section.size() - sizeof(number_of_inits);
    section.read(offset, sizeof(number_of_inits), &number_of_inits);
    if (number_of_inits > (section.size() - sizeof(number_of_inits)) / SIZE_OF_INIT_SCHEDULE_SIZE) {
        logger.error(
            "The number of init schedules read from the blob is too great relative to the size of the section");
        return std::nullopt;
    }

    logger.debug("Parsed number of init schedules: %lu", number_of_inits);

    std::vector<ov::Tensor> init_schedules;
    if (number_of_inits == 0) {
        if (section.size() != MINIMUM_INIT_SCHEDULES_SECTION_SIZE) {
            logger.error("Failed to read the whole content of the section");
            return std::nullopt;
        }
        return std::make_shared<ELFInitSchedulesSection>(std::move(init_schedules), nullptr, logger.level());
    }

    // TODO comment
    offset = section.size() - sizeof(number_of_inits) - number_of_inits * SIZE_OF_INIT_SCHEDULE_SIZE;
    size_t total_init_sizes = 0;
    std::vector<uint64_t> init_sizes;
    uint64_t value;
    while (number_of_inits--) {
        section.read(offset, sizeof(value), &value);
        offset += sizeof(value);
        init_sizes.push_back(value);

        if (total_init_sizes > total_init_sizes + value) {
            logger.error("Integer overflow");
            return std::nullopt;
        }
        total_init_sizes += value;

        logger.debug("Init schedule parsed size: %lu", value);
    }

    // TODO check the checks and their tests
    if (total_init_sizes != section.size() - sizeof(number_of_inits) - number_of_inits * SIZE_OF_INIT_SCHEDULE_SIZE) {
        logger.error("The sum of the parsed init schedule sizes does not correspond to the current section size");
        return std::nullopt;
    }

    const ov::util::MemoryView content_view = section.view();
    offset = 0;

    for (const auto& init_size : init_sizes) {
        ov::Tensor init_schedule;

        if (content_view.empty()) {
            init_schedule = utils::allocate_aligned_tensor(init_size);
            section.read(offset, init_size, init_schedule.data());
            offset += init_size;

            logger.info(NEW_PAGE_ALIGNED_BUFFER_MESSAGE.data(), init_size);
        } else {
            init_schedule = ov::Tensor(ov::element::Type_t::u8, ov::Shape({init_size}), content_view.data() + offset);
            offset += init_size;
        }

        init_schedules.push_back(std::move(init_schedule));
    }

    return std::make_shared<ELFInitSchedulesSection>(std::move(init_schedules), nullptr, logger.level());
};

DynamicScheduleSection::DynamicScheduleSection(
    const std::shared_ptr<DynamicGraph>& graph,
    const std::function<std::string(const std::string&)>& encryption_callback,
    const ov::log::Level log_level)
    : ISection(SectionTypeCode::DYNAMIC_SCHEDULE, PAGE_ALIGNED_BEGINNING),
      m_impl(std::dynamic_pointer_cast<Graph>(graph), encryption_callback, log_level),
      m_logger("DynamicScheduleSection", log_level) {}

DynamicScheduleSection::DynamicScheduleSection(
    ov::Tensor&& main_schedule,
    const std::function<std::string(const std::string&)>& encryption_callback,
    const ov::log::Level log_level)
    : ISection(SectionTypeCode::DYNAMIC_SCHEDULE, PAGE_ALIGNED_BEGINNING),
      m_impl(std::move(main_schedule), encryption_callback, log_level),
      m_logger("DynamicScheduleSection", log_level) {}

std::vector<std::shared_ptr<CREToken>> DynamicScheduleSection::get_compatibility_requirements_subexpression(
    const std::unordered_map<SectionID, std::shared_ptr<ISection>>&
    /*all_registered_sections*/) const {
    OPENVINO_ASSERT(get_id().has_value());
    m_logger.debug("Added the DYNAMIC_SCHEDULE_<ID> to the CRE");
    return {std::make_shared<SectionType>(get_type()), std::make_shared<SectionID>(get_id().value())};
}

void DynamicScheduleSection::encode(const ov::runtime::hsm::SectionSink& sink) const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "DynamicScheduleSection::write");
    m_impl.encode(sink);
}

void DynamicScheduleSection::set_graph(const std::shared_ptr<DynamicGraph>& graph) {
    m_impl.set_graph(std::dynamic_pointer_cast<Graph>(graph));
}

void DynamicScheduleSection::set_encryption_callback(
    const std::function<std::string(const std::string&)> encryption_callback) {
    m_impl.set_encryption_callback(encryption_callback);
}

ov::Tensor DynamicScheduleSection::get_schedule() const {
    return m_impl.get_schedule();
}

void DynamicScheduleSection::decrypt(const std::function<std::string(const std::string&)>& decryption_callback) {
    m_impl.decrypt(decryption_callback);
}

ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>> DynamicScheduleSection::decode =
    [](const ov::runtime::hsm::Section& section) -> std::optional<std::shared_ptr<ISection>> {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "DynamicScheduleSection::decode");
    OPENVINO_ASSERT(section.tag().id() == static_cast<uint32_t>(SectionTypeCode::DYNAMIC_SCHEDULE),
                    "Logic error: the decoder received a section of the wrong type");
    // TODO more logs

    const auto main_schedule_section = ELFMainScheduleSection::decode(section);
    if (!main_schedule_section.has_value()) {
        return std::nullopt;
    }

    return std::make_shared<DynamicScheduleSection>(
        std::dynamic_pointer_cast<ELFMainScheduleSection>(main_schedule_section.value())->get_schedule(),
        nullptr,
        Logger::global().level());
};

std::optional<std::string> DynamicScheduleSection::get_individual_compatibility_requirements() const {
    // TODO is this correct?
    return m_impl.get_individual_compatibility_requirements();
}

}  // namespace intel_npu
