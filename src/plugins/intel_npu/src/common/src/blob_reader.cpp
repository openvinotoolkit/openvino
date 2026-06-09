// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/blob_reader.hpp"

#include "intel_npu/common/cre_section.hpp"
#include "intel_npu/common/itt.hpp"

namespace {

constexpr std::string_view MAGIC_BYTES = "OVNPU";
constexpr uint32_t FORMAT_VERSION = 0x30000;  // 3.0;

constexpr intel_npu::SectionTypeInstance FIRST_INSTANCE_ID = 0;

size_t move_cursor_with_bound_checking(const size_t destination, const size_t npu_region_size) {
    OPENVINO_ASSERT(destination <= npu_region_size,
                    "Attempted to move the cursor beyond the NPU region. Destination: ",
                    destination,
                    ". Limit: ",
                    npu_region_size);
    return destination;
}

}  // namespace

namespace intel_npu {

BlobReader::BlobReader(const ov::log::Level log_level) : m_logger("BlobReader", log_level) {
    // Register the core sections
    register_reader(PredefinedSectionType::CRE, CRESection::read);
    register_reader(PredefinedSectionType::OFFSETS_TABLE, OffsetsTableSection::read);
}

void BlobReader::register_reader(const SectionType type,
                                 std::function<std::shared_ptr<ISection>(BlobReaderInterface&)> reader) {
    m_readers[type] = reader;
    m_logger.debug("Registered a reader for section type %lu", type);
}

void BlobReader::register_section_type_evaluator(const std::shared_ptr<ISectionTypeEvaluator>& evaluator) {
    m_section_type_evaluators[evaluator->get_section_type()] = evaluator;
    m_logger.debug("Registered a section type evaluator for section type %lu", evaluator->get_section_type());
}

void BlobReader::register_section_type_instance_evaluate_fn(const SectionType type,
                                                            std::function<bool(BlobReaderInterface&)> function) {
    m_section_type_instance_evaluate_fn[type] = function;
    m_logger.debug("Registered a section type instance evaluation function for section type %lu", type);
}

std::shared_ptr<ISection> BlobReader::retrieve_section(const SectionID& id) {
    auto type_search_result = m_parsed_sections.find(id.type);
    if (type_search_result != m_parsed_sections.end()) {
        auto instance_search_result = type_search_result->second.find(id.type_instance);
        if (instance_search_result != type_search_result->second.end()) {
            return instance_search_result->second;
        }
    }
    return nullptr;
}

std::shared_ptr<ISection> BlobReader::retrieve_first_section(const SectionType section_type) {
    return retrieve_section(SectionID(section_type, FIRST_INSTANCE_ID));
}

std::optional<std::unordered_map<SectionTypeInstance, std::shared_ptr<ISection>>>
BlobReader::retrieve_sections_same_type(const SectionType type) {
    auto type_search_result = m_parsed_sections.find(type);
    if (type_search_result != m_parsed_sections.end()) {
        return type_search_result->second;
    }
    return std::nullopt;
}

std::unordered_map<SectionID, SectionTypeInstanceEvaluator> BlobReader::build_section_type_instance_evaluators(
    const ov::Tensor& source,
    const OffsetsTable& offsets_table,
    const size_t npu_region_size) const {
    std::unordered_map<SectionID, SectionTypeInstanceEvaluator> instance_evaluators;
    const std::unordered_set<SectionID> all_section_ids = offsets_table.get_all_registered_section_ids();

    for (const SectionID& section_id : all_section_ids) {
        BlobReaderInterface reader(source,
                                   offsets_table.lookup_offset(section_id).value(),
                                   offsets_table.lookup_length(section_id).value(),
                                   npu_region_size,
                                   m_section_type_evaluators,
                                   m_logger.level());

        // Do not create any evaluator if no function has been provided. The CRE code will treat such cases as supported
        // by default
        if (m_section_type_instance_evaluate_fn.count(section_id.type)) {
            instance_evaluators.emplace(
                section_id,
                SectionTypeInstanceEvaluator(m_section_type_instance_evaluate_fn.at(section_id.type),
                                             std::move(reader)));
        }
    }

    return instance_evaluators;
}

void BlobReader::read(const ov::Tensor& source) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "BlobReader::read");
    m_logger.debug("Starting to parse a blob");

    if (!m_parsed_sections.empty()) {
        m_logger.warning("The same BlobReader object was used to read a blob more than once. This operation is "
                         "supported, but it has little use. Disregard this message if this is a test that validates "
                         "this functionality. Otherwise, this may indicate a bug.");
        // Discard all parsed section and parse the given arguments. If the given blob and capabilities are the same
        // as the ones received during the previous "read" call, then this operation should be idempotent.
        // Otherwise, the result may differ.
        m_parsed_sections = {};
    }

    // Read the size of the NPU region
    size_t npu_region_size = get_npu_region_size(source);
    m_logger.trace("NPU region size: %lu", npu_region_size);
    size_t cursor = MAGIC_BYTES.size() + sizeof(FORMAT_VERSION) + sizeof(npu_region_size);

    // Step 1: Read the table of offsets. First, get the location and size of the table from the region of
    // persistent format. Then, use this information to parse the table.
    uint64_t offsets_table_location;
    uint64_t offsets_table_size;

    const size_t where_the_region_of_persistent_format_starts =
        cursor + sizeof(offsets_table_location) + sizeof(offsets_table_size);
    OPENVINO_ASSERT(where_the_region_of_persistent_format_starts <= npu_region_size,
                    "The parsed NPU region size is too small. Found: ",
                    npu_region_size,
                    ". Minimum required: ",
                    where_the_region_of_persistent_format_starts);
    std::memcpy(reinterpret_cast<char*>(&offsets_table_location),
                source.data<const char>() + cursor,
                sizeof(offsets_table_location));
    cursor = move_cursor_with_bound_checking(cursor + sizeof(offsets_table_location), npu_region_size);
    std::memcpy(reinterpret_cast<char*>(&offsets_table_size),
                source.data<const char>() + cursor,
                sizeof(offsets_table_size));
    m_logger.trace("Offsets table location %lu; size %lu", offsets_table_location, offsets_table_size);

    cursor = move_cursor_with_bound_checking(offsets_table_location, npu_region_size);

    BlobReaderInterface interface(source,
                                  cursor,
                                  offsets_table_size,
                                  npu_region_size,
                                  m_section_type_evaluators,
                                  m_logger.level());
    m_parsed_sections[PredefinedSectionType::OFFSETS_TABLE][FIRST_INSTANCE_ID] = OffsetsTableSection::read(interface);
    m_parsed_sections[PredefinedSectionType::OFFSETS_TABLE][FIRST_INSTANCE_ID]->set_section_type_instance(
        FIRST_INSTANCE_ID);
    // The offset table is required only within the scope of the read method
    OffsetsTable offsets_table = std::dynamic_pointer_cast<OffsetsTableSection>(
                                     m_parsed_sections.at(PredefinedSectionType::OFFSETS_TABLE).at(FIRST_INSTANCE_ID))
                                     ->get_table();
    m_logger.debug("Parsed the table of offsets");

    // Step 2: Look for the CRE and evaluate it
    std::optional<uint64_t> cre_location = offsets_table.lookup_offset(CRE_SECTION_ID);
    std::optional<uint64_t> cre_length = offsets_table.lookup_length(CRE_SECTION_ID);
    OPENVINO_ASSERT(cre_location.has_value(), "The CRE was not found within the table of offsets");

    // TODO test the negative branch as well
    if (cre_location.has_value()) {
        cursor = move_cursor_with_bound_checking(cre_location.value(), npu_region_size);

        interface = BlobReaderInterface(source,
                                        cursor,
                                        cre_length.value(),
                                        npu_region_size,
                                        m_section_type_evaluators,
                                        m_logger.level());
        m_parsed_sections[PredefinedSectionType::CRE][FIRST_INSTANCE_ID] = CRESection::read(interface);
        m_parsed_sections[PredefinedSectionType::CRE][FIRST_INSTANCE_ID]->set_section_type_instance(FIRST_INSTANCE_ID);
        const bool is_compatible =
            std::dynamic_pointer_cast<CRESection>(
                m_parsed_sections.at(PredefinedSectionType::CRE).at(FIRST_INSTANCE_ID))
                ->get_cre()
                .check_compatibility(m_section_type_evaluators,
                                     build_section_type_instance_evaluators(source, offsets_table, npu_region_size));
        OPENVINO_ASSERT(is_compatible, "The imported model is not compatible");
        m_logger.debug("CRE evaluation passed");
    } else {
        m_logger.warning("The CRE section was not found within the table of offsets. Proceeding without performing any "
                         "compatibility checks");
    }

    // Step 3: Parse all known sections
    size_t number_of_sections_encountered = 0;
    cursor = move_cursor_with_bound_checking(where_the_region_of_persistent_format_starts, npu_region_size);
    while (cursor < npu_region_size) {
        // The table of offsets & CRE have already been parsed
        if (cursor == offsets_table_location) {
            cursor = move_cursor_with_bound_checking(cursor + offsets_table_size, npu_region_size);
            continue;
        }
        if (cursor == cre_location.value()) {
            cursor = move_cursor_with_bound_checking(cursor + cre_length.value(), npu_region_size);
            ++number_of_sections_encountered;
            continue;
        }

        const std::optional<SectionID> section_id = offsets_table.lookup_section_id(cursor);
        OPENVINO_ASSERT(section_id.has_value(),
                        "Did not find any section corresponding to the relative offset ",
                        cursor);
        const std::optional<uint64_t> section_length = offsets_table.lookup_length(section_id.value());
        ++number_of_sections_encountered;

        const size_t next_section_location = cursor + section_length.value();

        m_logger.trace("Found section ID %s at offset %lu, length %lu",
                       section_id->to_string(),
                       cursor,
                       section_length.value());

        // Read the section if we have a reader for it. Otherwise, skip it.
        if (m_readers.count(section_id->type)) {
            m_logger.debug("Parsing section ID ", section_id);

            interface = BlobReaderInterface(source,
                                            cursor,
                                            section_length.value(),
                                            npu_region_size,
                                            m_section_type_evaluators,
                                            m_logger.level());
            m_parsed_sections[section_id->type][section_id->type_instance] = m_readers.at(section_id->type)(interface);
            m_parsed_sections[section_id->type][section_id->type_instance]->set_section_type_instance(
                section_id->type_instance);
            m_parsed_sections_order.push_back(section_id.value());
        } else {
            m_logger.debug("No section reader found for section type %lu. Skipping", section_id->type);
        }

        cursor = move_cursor_with_bound_checking(next_section_location, npu_region_size);
    }

    OPENVINO_ASSERT(
        number_of_sections_encountered == offsets_table.get_number_of_entries(),
        "The number of sections encountered doesn't match the number of offsets table entries. Sections encountered: ",
        number_of_sections_encountered,
        ". Offsets table entries: ",
        offsets_table.get_number_of_entries());
}

size_t BlobReader::get_npu_region_size(std::istream& stream) {
    const auto cursor_before_reading = stream.tellg();

    std::string magic_bytes(MAGIC_BYTES.size(), 0);
    stream.read(const_cast<char*>(magic_bytes.c_str()), MAGIC_BYTES.size());
    OPENVINO_ASSERT(magic_bytes == MAGIC_BYTES,
                    "Invalid magic bytes. Found: ",
                    magic_bytes,
                    ". Expected: ",
                    MAGIC_BYTES);

    uint32_t format_version;
    stream.read(reinterpret_cast<char*>(&format_version), sizeof(format_version));
    OPENVINO_ASSERT(format_version == FORMAT_VERSION,
                    "Invalid blob format version. Found: ",
                    format_version,
                    ". Expected: ",
                    FORMAT_VERSION);

    uint64_t npu_region_size;
    stream.read(reinterpret_cast<char*>(&npu_region_size), sizeof(npu_region_size));
    stream.seekg(cursor_before_reading);

    return npu_region_size;
}

size_t BlobReader::get_npu_region_size(const ov::Tensor& tensor) {
    std::string magic_bytes(MAGIC_BYTES.size(), 0);
    std::memcpy(const_cast<char*>(magic_bytes.c_str()), tensor.data<const char>(), MAGIC_BYTES.size());
    OPENVINO_ASSERT(magic_bytes == MAGIC_BYTES,
                    "Invalid magic bytes. Found: ",
                    magic_bytes,
                    ". Expected: ",
                    MAGIC_BYTES);

    uint32_t format_version;
    std::memcpy(reinterpret_cast<char*>(&format_version),
                tensor.data<const char>() + MAGIC_BYTES.size(),
                sizeof(format_version));
    OPENVINO_ASSERT(format_version == FORMAT_VERSION,
                    "Invalid blob format version. Found: ",
                    format_version,
                    ". Expected: ",
                    FORMAT_VERSION);

    uint64_t npu_region_size;
    std::memcpy(reinterpret_cast<char*>(&npu_region_size),
                tensor.data<const char>() + MAGIC_BYTES.size() + sizeof(FORMAT_VERSION),
                sizeof(npu_region_size));

    return npu_region_size;
}

}  // namespace intel_npu
