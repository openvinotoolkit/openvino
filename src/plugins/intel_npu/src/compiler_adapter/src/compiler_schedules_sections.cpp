// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler_schedules_sections.hpp"

#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/utils.hpp"

namespace {

using namespace intel_npu;

constexpr std::string_view INVALID_STATE_MESSAGE = "Invalid state";
constexpr std::string_view NEW_PAGE_ALIGNED_BUFFER_MESSAGE =
    "A new, page aligned buffer of size %zu has been allocated to host a compiled model";

ov::Tensor allocate_aligned_tensor(size_t blobSize) {
    ov::Allocator customAllocator{utils::AlignedAllocator{utils::STANDARD_PAGE_SIZE}};
    if (blobSize > static_cast<decltype(blobSize)>(std::numeric_limits<std::streamsize>::max())) {
        OPENVINO_THROW("Blob size is too large to be represented on a std::streamsize!");
    }

    return ov::Tensor(ov::element::u8, ov::Shape{blobSize}, customAllocator);
}

/**
 * @brief Uses the provided decryption callback to decrypt the given payload.
 */
void decrypt_payload(ov::Tensor& payload, const ov::EncryptionCallbacks& encryption_callbacks, const Logger& logger) {
    OPENVINO_ASSERT(encryption_callbacks.decrypt, "Decryption requested without providing a decryption callback");

    std::string decryptedBlobStr;
    {
        std::string encryptedBlobStr(payload.data<const char>(), payload.get_byte_size());  // +1x blob size
        decryptedBlobStr = encryption_callbacks.decrypt(encryptedBlobStr);                  // +1x blob size
    }  // -1x blob size when deallocating temporary encrypted blob string
    ov::Allocator customAllocator{utils::AlignedAllocator{utils::STANDARD_PAGE_SIZE}};
    size_t alignedSize = utils::align_size_to_standard_page_size(decryptedBlobStr.size());
    size_t paddingSize = alignedSize - decryptedBlobStr.size();
    payload = ov::Tensor(ov::element::u8, ov::Shape{alignedSize},
                         customAllocator);  // +1x blob size
    std::memcpy(payload.data<char>(), decryptedBlobStr.c_str(), decryptedBlobStr.size());
    if (paddingSize > 0) {
        // The blob obtained after decryption is expected to be the same as the blob we had before encryption.
        // That means blobs compiled with the current plugin version are expected to be already aligned.
        // However, the alignment might not be mandatory in a future plugin version. For this scenario, the
        // padding is added here in order to make use of this "non-copy optimization".
        logger.warning("Decrypted blob size was not page aligned, additional %zu bytes padding will be added",
                       paddingSize);
        std::memset(payload.data<char>() + decryptedBlobStr.size(), 0, paddingSize);
    }
}  // -1x blob size when deallocating decrypted blob string

std::optional<ov::EncryptionCallbacks> get_encryption_callbacks_from_config(const FilteredConfig& config) {
    if (config.has(CACHE_ENCRYPTION_CALLBACKS::key().data()) &&
        config.get<CACHE_ENCRYPTION_CALLBACKS>().encrypt != nullptr) {
        return config.get<CACHE_ENCRYPTION_CALLBACKS>();
    }
    return std::nullopt;
}

}  // namespace

namespace intel_npu {

ELFMainScheduleSection::ELFMainScheduleSection(const std::shared_ptr<Graph>& graph,
                                               const std::optional<ov::EncryptionCallbacks>& encryption_callbacks,
                                               const ov::log::Level log_level)
    : ISection(PredefinedSectionType::ELF_MAIN_SCHEDULE),
      m_graph_or_schedule(graph),
      m_encryption_callbacks(encryption_callbacks),
      m_logger("ELFMainScheduleSection", log_level) {}

ELFMainScheduleSection::ELFMainScheduleSection(ov::Tensor&& main_schedule,
                                               const std::optional<ov::EncryptionCallbacks>& encryption_callbacks,
                                               const ov::log::Level log_level)
    : ISection(PredefinedSectionType::ELF_MAIN_SCHEDULE),
      m_graph_or_schedule(std::move(main_schedule)),
      m_encryption_callbacks(encryption_callbacks),
      m_logger("ELFMainScheduleSection", log_level) {}

std::vector<CREToken> ELFMainScheduleSection::get_compatibility_requirements_subexpression(
    const std::unordered_map<SectionID, std::shared_ptr<ISection>>&
    /*all_registered_sections*/) const {
    m_logger.debug("Added the ELF_MAIN_SCHEDULE section type to the CRE");
    return {get_type()};
}

void ELFMainScheduleSection::write(BlobWriterInterface& writer) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "ELFMainScheduleSection::write");
    const auto* graph = std::get_if<std::shared_ptr<Graph>>(&m_graph_or_schedule);
    OPENVINO_ASSERT(graph, INVALID_STATE_MESSAGE);

    // Also take the padding size into account, we'll write that first
    const size_t offset = writer.get_offset_relative_to_npu_region();
    const size_t padding_size = utils::align_size_to_standard_page_size(offset) - offset;
    writer.write_from(&padding_size, sizeof(padding_size));
    // TODO add method that adds padding until page aligned relative to NPU region start
    writer.add_padding(padding_size);

    m_logger.debug("Added %lu padding to offset %lu", padding_size, offset);

    if (!m_encryption_callbacks.has_value()) {
        (*graph)->export_main_blob(writer.m_stream.get());
        return;
    }

    // Encrypt the compiler payload, then write it
    OPENVINO_ASSERT(m_encryption_callbacks->encrypt, "Missing encryption callback");

    std::string encrypted_payload;
    {
        std::string tmp_plain_payload;
        {
            std::stringstream tmp_stream;
            (*graph)->export_main_blob(tmp_stream);  // +1x blob size
            tmp_plain_payload = tmp_stream.str();    // +2x blob size
        }  // -1x blob size when deallocating temporary stringstream
        encrypted_payload = m_encryption_callbacks->encrypt(tmp_plain_payload);  // +2x blob size
    }  // -1x blob size when deallocating temporary blob string

    writer.write_from(encrypted_payload.c_str(), encrypted_payload.size());
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

void ELFMainScheduleSection::decrypt(const ov::EncryptionCallbacks& encryption_callbacks) {
    auto* schedule = std::get_if<ov::Tensor>(&m_graph_or_schedule);
    OPENVINO_ASSERT(schedule, INVALID_STATE_MESSAGE);

    decrypt_payload(*schedule, encryption_callbacks, m_logger);
}

std::shared_ptr<ISection> ELFMainScheduleSection::read(BlobReaderInterface& blob_reader) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "ELFMainScheduleSection::read");
    const Logger logger("ELFMainScheduleSection", blob_reader.get_log_level());

    // Skip the first padding region
    const size_t offset = blob_reader.get_offset_relative_to_npu_region();

    size_t padding_size;
    blob_reader.read_into_buffer(&padding_size, sizeof(padding_size));
    OPENVINO_ASSERT(padding_size <= blob_reader.get_section_length(),
                    "The read padding size is greater than the length of the blob section");
    blob_reader.move_cursor_relative_to_current_section(blob_reader.get_offset_relative_to_current_section() +
                                                        padding_size);

    logger.debug("Skipped %lu padding from offset %lu", padding_size, offset);

    const size_t main_schedule_size = blob_reader.get_section_length() - padding_size;

    if (!blob_reader.source_is_contiguous()) {
        ov::Tensor main_schedule = allocate_aligned_tensor(main_schedule_size);
        blob_reader.read_into_buffer(main_schedule.data(), main_schedule_size);

        logger.info(NEW_PAGE_ALIGNED_BUFFER_MESSAGE.data(), main_schedule_size);
        return std::make_shared<ELFMainScheduleSection>(std::move(main_schedule), logger.level());
    }

    return std::make_shared<ELFMainScheduleSection>(blob_reader.create_roi_tensor(main_schedule_size),
                                                    get_encryption_callbacks_from_config(blob_reader.get_config()),
                                                    logger.level());
}

std::optional<std::string> ELFMainScheduleSection::get_inidividual_compatibility_requirements() const {
    const auto* graph = std::get_if<std::shared_ptr<Graph>>(&m_graph_or_schedule);
    OPENVINO_ASSERT(graph, INVALID_STATE_MESSAGE);
    std::optional<std::string_view> requirements = (*graph)->get_compatibility_descriptor();
    return requirements.has_value() ? std::make_optional<>(std::string(requirements.value())) : std::nullopt;
}

ELFInitSchedulesSection::ELFInitSchedulesSection(const std::shared_ptr<WeightlessGraph>& weightless_graph,
                                                 const std::optional<ov::EncryptionCallbacks>& encryption_callbacks,
                                                 const ov::log::Level log_level)
    : ISection(PredefinedSectionType::ELF_INIT_SCHEDULES),
      m_graph_or_schedules(weightless_graph),
      m_encryption_callbacks(encryption_callbacks),
      m_logger("ELFInitSchedulesSection", log_level) {}

ELFInitSchedulesSection::ELFInitSchedulesSection(std::vector<ov::Tensor>&& init_schedules,
                                                 const std::optional<ov::EncryptionCallbacks>& encryption_callbacks,
                                                 const ov::log::Level log_level)
    : ISection(PredefinedSectionType::ELF_INIT_SCHEDULES),
      m_graph_or_schedules(std::move(init_schedules)),
      m_encryption_callbacks(encryption_callbacks),
      m_logger("ELFInitSchedulesSection", log_level) {}

std::vector<CREToken> ELFInitSchedulesSection::get_compatibility_requirements_subexpression(
    const std::unordered_map<SectionID, std::shared_ptr<ISection>>&
    /*all_registered_sections*/) const {
    m_logger.debug("Added the ELF_INIT_SCHEDULES section type to the CRE");
    return {get_type()};
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

    // Also take the padding size into account, we'll write that next
    const size_t offset = writer.get_offset_relative_to_npu_region();
    const size_t padding_size = utils::align_size_to_standard_page_size(offset) - offset;
    writer.write_from(&padding_size, sizeof(padding_size));
    writer.add_padding(padding_size);

    std::vector<uint64_t> init_sizes = (*weightless_graph)->export_init_blobs(writer.m_stream.get());

    if (!m_encryption_callbacks.has_value()) {
        init_sizes = (*weightless_graph)->export_init_blobs(writer.m_stream.get());
    } else {
        // Encrypt the compiler payload, then write it
        OPENVINO_ASSERT(m_encryption_callbacks->encrypt, "Missing encryption callback");

        std::string encrypted_payload;
        {
            std::string tmp_plain_payload;
            {
                std::stringstream tmp_stream;
                init_sizes = (*weightless_graph)->export_init_blobs(tmp_stream);
                tmp_plain_payload = tmp_stream.str();  // +2x blob size
            }  // -1x blob size when deallocating temporary stringstream
            encrypted_payload = m_encryption_callbacks->encrypt(tmp_plain_payload);  // +2x blob size
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
    OPENVINO_ASSERT(std::holds_alternative<std::vector<ov::Tensor>>(m_graph_or_schedules), INVALID_STATE_MESSAGE);
    m_graph_or_schedules = weightless_graph;
}

std::vector<ov::Tensor> ELFInitSchedulesSection::get_schedules() const {
    const auto* schedules = std::get_if<std::vector<ov::Tensor>>(&m_graph_or_schedules);
    OPENVINO_ASSERT(schedules, INVALID_STATE_MESSAGE);
    return *schedules;
}

void ELFInitSchedulesSection::decrypt(const ov::EncryptionCallbacks& encryption_callbacks) {
    auto* schedules = std::get_if<std::vector<ov::Tensor>>(&m_graph_or_schedules);
    OPENVINO_ASSERT(schedules, INVALID_STATE_MESSAGE);

    for (ov::Tensor& schedule : *schedules) {
        decrypt_payload(schedule, encryption_callbacks, m_logger);
    }
}

std::shared_ptr<ISection> ELFInitSchedulesSection::read(BlobReaderInterface& blob_reader) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "ELFInitSchedulesSection::read");
    Logger logger("ELFInitSchedulesSection", blob_reader.get_log_level());

    const size_t section_length = blob_reader.get_section_length();

    uint64_t number_of_inits;
    blob_reader.read_into_buffer(&number_of_inits, sizeof(number_of_inits));
    // TODO tighter constraints
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
        blob_reader.read_into_buffer(&value, sizeof(value));
        init_sizes.push_back(value);

        OPENVINO_ASSERT(total_init_sizes <= total_init_sizes + value, "Integer overflow");
        total_init_sizes += value;

        logger.debug("Init schedule parsed size: %lu", value);
    }

    OPENVINO_ASSERT(total_init_sizes < section_length,
                    "The sum of the parsed init schedule sizes is too big for the current section size. Sum: ",
                    total_init_sizes,
                    ". Section length: ",
                    section_length);

    // Skip the first padding
    const size_t offset = blob_reader.get_offset_relative_to_npu_region();
    size_t padding_size;
    blob_reader.read_into_buffer(&padding_size, sizeof(padding_size));
    blob_reader.move_cursor_relative_to_current_section(blob_reader.get_offset_relative_to_current_section() +
                                                        padding_size);

    std::vector<ov::Tensor> init_schedules;
    for (const auto& init_size : init_sizes) {
        ov::Tensor init_schedule;

        if (!blob_reader.source_is_contiguous()) {
            init_schedule = allocate_aligned_tensor(init_size);
            blob_reader.read_into_buffer(init_schedule.data(), init_size);

            logger.info(NEW_PAGE_ALIGNED_BUFFER_MESSAGE.data(), init_size);
        } else {
            init_schedule = blob_reader.create_roi_tensor(init_size);
        }

        init_schedules.push_back(std::move(init_schedule));
    }

    return std::make_shared<ELFInitSchedulesSection>(std::move(init_schedules),
                                                     get_encryption_callbacks_from_config(blob_reader.get_config()),
                                                     logger.level());
}

DynamicScheduleSection::DynamicScheduleSection(const std::shared_ptr<DynamicGraph>& graph,
                                               const std::optional<ov::EncryptionCallbacks>& encryption_callbacks,
                                               const ov::log::Level log_level)
    : ISection(PredefinedSectionType::DYNAMIC_SCHEDULE),
      m_impl(std::dynamic_pointer_cast<Graph>(graph), encryption_callbacks, log_level),
      m_blob_type(graph->get_blob_type()),
      m_logger("DynamicScheduleSection", log_level) {}

DynamicScheduleSection::DynamicScheduleSection(ov::Tensor&& main_schedule,
                                               const BlobType blob_type,
                                               const std::optional<ov::EncryptionCallbacks>& encryption_callbacks,
                                               const ov::log::Level log_level)
    : ISection(PredefinedSectionType::DYNAMIC_SCHEDULE),
      m_impl(std::move(main_schedule), encryption_callbacks, log_level),
      m_blob_type(blob_type),
      m_logger("DynamicScheduleSection", log_level) {}

std::vector<CREToken> DynamicScheduleSection::get_compatibility_requirements_subexpression(
    const std::unordered_map<SectionID, std::shared_ptr<ISection>>&
    /*all_registered_sections*/) const {
    m_logger.debug("Added the DYNAMIC_SCHEDULE section type to the CRE");
    return {get_type()};
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

void DynamicScheduleSection::decrypt(const ov::EncryptionCallbacks& encryption_callbacks) {
    m_impl.decrypt(encryption_callbacks);
}

std::shared_ptr<ISection> DynamicScheduleSection::read(BlobReaderInterface& blob_reader) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "DynamicScheduleSection::read");
    // TODO more logs

    BlobType blob_type;
    blob_reader.read_into_buffer(&blob_type, sizeof(blob_type));

    return std::make_shared<DynamicScheduleSection>(
        std::dynamic_pointer_cast<ELFMainScheduleSection>(ELFMainScheduleSection::read(blob_reader))->get_schedule(),
        blob_type,
        get_encryption_callbacks_from_config(blob_reader.get_config()),
        blob_reader.get_log_level());
}

std::optional<std::string> DynamicScheduleSection::get_inidividual_compatibility_requirements() const {
    // TODO is this correct?
    return m_impl.get_inidividual_compatibility_requirements();
}

}  // namespace intel_npu
