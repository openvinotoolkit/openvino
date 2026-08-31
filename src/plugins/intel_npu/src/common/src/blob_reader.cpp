// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/blob_reader.hpp"

#include "intel_npu/common/itt.hpp"
#include "intel_npu/common/runtime_requirements_section.hpp"
#include "intel_npu/config/options.hpp"

namespace {

constexpr std::string_view MAGIC_BYTES = "OVNPU";
constexpr uint32_t FORMAT_VERSION = 0x30000;  // 3.0;

// The header: magic, format version, NPU region size, manifest location, manifest size
constexpr size_t MINIMUM_BLOB_SIZE = MAGIC_BYTES.size() + sizeof(FORMAT_VERSION) + 3 * sizeof(uint64_t);

void seekg_with_bound_checking(intel_npu::BlobSource& source,
                               const size_t destination,
                               const size_t npu_region_start,
                               const size_t npu_region_size) {
    OPENVINO_ASSERT(npu_region_start <= destination && destination <= npu_region_size,
                    "Attempted to move the cursor outside the NPU region. Destination: ",
                    destination,
                    ". Limits: [",
                    npu_region_start,
                    ", ",
                    npu_region_size,
                    "]");
    source.seekg(destination, std::ios::beg);
}

}  // namespace

namespace intel_npu {

BlobReader::BlobReader(const FilteredConfig& config)
    : m_config(config),
      m_logger("BlobReader", config.get<LOG_LEVEL>()) {
    // Register the core sections
    register_reader(PredefinedSectionType::CRE, RuntimeRequirementsSection::read);
    register_reader(PredefinedSectionType::MANIFEST, ManifestSection::read);
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
    m_section_instance_evaluate_fn[type] = function;
    m_logger.debug("Registered a section type instance evaluation function for section type %lu", type);
}

bool BlobReader::has_section_of_type(const SectionType section_type) const {
    return m_type_to_parsed_sections.count(section_type) ? m_type_to_parsed_sections.at(section_type).size() > 0
                                                         : false;
}

size_t BlobReader::count_sections_of_type(const SectionType section_type) const {
    return m_type_to_parsed_sections.count(section_type) ? m_type_to_parsed_sections.at(section_type).size() : 0;
}

std::unordered_map<SectionType, size_t> BlobReader::get_content_summary() const {
    std::unordered_map<SectionType, size_t> summary;
    for (const auto& [section_type, section_instances] : m_type_to_parsed_sections) {
        summary[section_type] = section_instances.size();
    }

    return summary;
}

std::shared_ptr<ISection> BlobReader::retrieve_section(const SectionID& id) const {
    auto search_result = m_id_to_parsed_sections.find(id);
    if (search_result != m_id_to_parsed_sections.end()) {
        return search_result->second;
    }
    return nullptr;
}

std::shared_ptr<ISection> BlobReader::retrieve_first_section(const SectionType section_type) const {
    if (!m_type_to_parsed_sections.count(section_type) || m_type_to_parsed_sections.at(section_type).empty()) {
        return nullptr;
    }

    return *m_type_to_parsed_sections.at(section_type).begin();
}

std::optional<std::unordered_set<std::shared_ptr<ISection>>> BlobReader::retrieve_sections_same_type(
    const SectionType type) const {
    auto search_result = m_type_to_parsed_sections.find(type);
    if (search_result != m_type_to_parsed_sections.end()) {
        return search_result->second;
    }
    return std::nullopt;
}

// TODO refactor
std::unordered_map<SectionID, SectionInstanceEvaluator> BlobReader::build_section_type_instance_evaluators(
    BlobSource& source,
    const Manifest& manifest,
    const size_t npu_region_start,
    const size_t npu_region_size) const {
    std::unordered_map<SectionID, SectionInstanceEvaluator> instance_evaluators;
    const std::unordered_set<SectionID> all_section_ids = manifest.get_all_registered_section_ids();

    for (const SectionID& section_id : all_section_ids) {
        BlobReaderInterface reader(source,
                                   npu_region_start,
                                   npu_region_size,
                                   manifest.lookup_offset(section_id).value(),
                                   manifest.lookup_length(section_id).value(),
                                   m_config);

        // Do not create any evaluator if no function has been provided. The CRE code will treat such cases as supported
        // by default
        if (m_section_instance_evaluate_fn.count(section_id.type)) {
            instance_evaluators.emplace(
                section_id,
                SectionInstanceEvaluator(m_section_instance_evaluate_fn.at(section_id.type), std::move(reader)));
        }
    }

    return instance_evaluators;
}

void BlobReader::parse_section(BlobSource& source,
                               const SectionType type,
                               const SectionID id,
                               const size_t length,
                               const size_t npu_region_start,
                               const size_t npu_region_size,
                               const bool include_in_sections_order) {
    BlobReaderInterface interface(source, npu_region_start, npu_region_size, source.tellg(), length, m_config);

    m_id_to_parsed_sections[id] = m_readers.at(type)(interface);
    m_id_to_parsed_sections[id]->set_id(id);
    m_type_to_parsed_sections[id].insert(m_id_to_parsed_sections.at(id));

    // TODO can include_in_sections_order be avoided?
    if (include_in_sections_order) {
        m_parsed_sections_order.push_back(id);
    }
}

void BlobReader::read(BlobSource& source) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "BlobReader::read");
    m_logger.debug("Starting to parse a blob");

    OPENVINO_ASSERT(
        m_id_to_parsed_sections.empty() && m_type_to_parsed_sections.empty() && m_parsed_sections_order.empty(),
        "Invalid state. There should be no parsed sections before attempting to read a blob. This may have "
        "happened if `read()` was called twice; `read()` must be called at most once.");

    const size_t npu_region_start = source.tellg();

    // Read the size of the NPU region
    const size_t npu_region_size = get_npu_region_size(source);
    m_logger.trace("NPU region size: %lu", npu_region_size);
    // The magic and format version have been already checked within "get_npu_region_size"
    source.seekg(MAGIC_BYTES.size() + sizeof(FORMAT_VERSION) + sizeof(npu_region_size), std::ios::cur);

    // Step 1: Read the manifest. First, get the location and size of the table from the header.
    // Then, use this information to parse the table.
    uint64_t manifest_location;
    uint64_t manifest_size;

    const size_t dynamic_format_region_start = source.tellg() + sizeof(manifest_location) + sizeof(manifest_size);

    source.read_into_buffer(&manifest_location, sizeof(manifest_location));
    seekg_with_bound_checking(source, source.tellg() + sizeof(manifest_location), npu_region_start, npu_region_size);
    source.read_into_buffer(&manifest_size, sizeof(manifest_size));
    m_logger.trace("Manifest location %lu; size %lu", manifest_location, manifest_size);

    seekg_with_bound_checking(source, manifest_location, npu_region_start, npu_region_size);

    OPENVINO_ASSERT(m_readers.count(PredefinedSectionType::MANIFEST), "No reader found for the manifest");
    parse_section(source,
                  PredefinedSectionType::MANIFEST,
                  MANIFEST_SECTION_ID,
                  manifest_size,
                  npu_region_start,
                  npu_region_size,
                  /*include_in_sections_order*/ false);

    // The offset table is required only within the scope of the read method
    Manifest manifest =
        std::dynamic_pointer_cast<ManifestSection>(m_id_to_parsed_sections.at(MANIFEST_SECTION_ID))->get_table();
    m_logger.debug("Parsed the manifest");

    // Step 2: Look for the CRE and evaluate it
    std::optional<uint64_t> cre_location = manifest.lookup_offset(CRE_SECTION_ID);
    std::optional<uint64_t> cre_length = manifest.lookup_length(CRE_SECTION_ID);

    std::unordered_map<SectionID, SectionInstanceEvaluator> section_instance_evaluators =
        build_section_type_instance_evaluators(source, manifest, npu_region_start, npu_region_size);

    // TODO test the negative branch as well
    // TODO safeguards for multiple manifests/CREs?
    if (cre_location.has_value()) {
        seekg_with_bound_checking(source, cre_location.value(), npu_region_start, npu_region_size);

        OPENVINO_ASSERT(m_readers.count(PredefinedSectionType::CRE), "No reader found for the manifest");
        parse_section(source,
                      PredefinedSectionType::CRE,
                      CRE_SECTION_ID,
                      cre_length.value(),
                      npu_region_start,
                      npu_region_size,
                      /*include_in_sections_order*/ false);

        const bool is_compatible =
            std::dynamic_pointer_cast<RuntimeRequirementsSection>(m_id_to_parsed_sections.at(CRE_ID))
                ->get_cre()
                .check_compatibility(m_section_type_evaluators, section_instance_evaluators);
        OPENVINO_ASSERT(is_compatible, "The imported model is not compatible");
        m_logger.debug("CRE evaluation passed");
    } else {
        m_logger.warning("The CRE section was not found within the manifest. Proceeding without performing any "
                         "compatibility checks");
    }

    // Step 3: Parse all known sections
    size_t number_of_sections_encountered = 0;
    seekg_with_bound_checking(source, dynamic_format_region_start, npu_region_start, npu_region_size);
    while (source.tellg() < npu_region_start + npu_region_size) {
        // The manifest & CRE have already been parsed
        if (source.tellg() == manifest_location) {
            seekg_with_bound_checking(source, source.tellg() + manifest_size, npu_region_start, npu_region_size);
            continue;
        }
        if (source.tellg() == cre_location.value()) {
            seekg_with_bound_checking(source, source.tellg() + cre_length.value(), npu_region_start, npu_region_size);
            ++number_of_sections_encountered;
            continue;
        }

        const std::optional<SectionID> section_id = manifest.lookup_section_id(source.tellg());
        OPENVINO_ASSERT(section_id.has_value(),
                        "Did not find any section corresponding to the relative offset ",
                        source.tellg());
        const std::optional<SectionType> section_type = manifest.lookup_type(section_id.value());
        const std::optional<uint64_t> section_length = manifest.lookup_length(section_id.value());
        OPENVINO_ASSERT(section_type.has_value() && section_length.has_value(), "Incomplete manifest");
        ++number_of_sections_encountered;

        const size_t next_section_location = source.tellg() + section_length.value();

        m_logger.trace("Found section %s at offset %lu, length %lu",
                       section_type_and_id_to_string(section_type.value(), section_id.value()),
                       source.tellg(),
                       section_length.value());

        // The section is considered for parsing only if the BlobReader has a reader registered for its type. Then, how
        // the section wil be handled depends on the type & type instance evaluators:
        //  * Case 1: evaluated & supported type, evaluated & supported instance - the section is mandatory. It is read
        //  without a try-catch block.
        //  * Case 2: evaluated & supported type, evaluated & unsupported instance - the section is unsupported. It's
        //  parsing is skipped.
        //  * Case 3: evaluated & supported type, unevaluated instance - the section is optional. It is read within a
        //  try-catch block.
        //  * Case 4: evaluated & unsupported type - the section is unsupported. It's parsing is skipped.
        //  * Case 5: unevaluated type, unevaluated instance - the section is optional. It is read within a try-catch
        //  block.
        if (!m_readers.count(section_type.value())) {
            m_logger.debug("No section reader found for section %s. Skipping", section_id);
            seekg_with_bound_checking(source, next_section_location, npu_region_start, npu_region_size);
            continue;
        }

        m_logger.trace("Found a reader for section ", section_id);

        const std::shared_ptr<ISectionTypeEvaluator> type_evaluator = m_section_type_evaluators.at(section_id->type);
        const SectionInstanceEvaluator& instance_evaluator =
            section_instance_evaluators.at(section_id.value());  // TODO fix this invalid "at"

        if (type_evaluator->evaluated() && type_evaluator->get_result()) {
            if (instance_evaluator.evaluated() && instance_evaluator.get_result()) {
                // Case 1
                m_logger.trace("Parsing mandatory section ", section_id);
                parse_section(source,
                              section_type.value(),
                              section_id.value(),
                              section_length.value(),
                              npu_region_start,
                              npu_region_size);
            } else if (instance_evaluator.evaluated() && !instance_evaluator.get_result()) {
                // Case 2
                m_logger.debug("The parsing of section ID ",
                               section_id,
                               " has been skipped. The section type instance is not supported");
            } else {
                // Case 3
                m_logger.trace("The section type instance corresponding to section ID ",
                               section_id,
                               " was not evaluated");

                try {
                    parse_section(source,
                                  section_type.value(),
                                  section_id.value(),
                                  section_length.value(),
                                  npu_region_start,
                                  npu_region_size);
                } catch (std::exception& e) {
                    m_logger.warning("The parsing of optional section ",
                                     section_id.value(),
                                     " has failed. Error message: ",
                                     e.what());
                }
            }
        } else if (type_evaluator->evaluated() && !type_evaluator->get_result()) {
            // Case 4
            m_logger.debug("The parsing of section ID ",
                           section_id,
                           " has been skipped. The section type is not supported");
        } else {
            // Case 5
            m_logger.trace("Section type ", section_type.value(), " not evaluated");
            OPENVINO_ASSERT(
                !instance_evaluator.evaluated(),
                "Found a section type instance that was evaluated without evaluating the section type first");

            try {
                parse_section(source,
                              section_type.value(),
                              section_id.value(),
                              section_length.value(),
                              npu_region_start,
                              npu_region_size);
            } catch (std::exception& e) {
                m_logger.warning("The parsing of optional section ",
                                 section_id.value(),
                                 " has failed. Error message: ",
                                 e.what());
            }
        }

        seekg_with_bound_checking(source, next_section_location, npu_region_start, npu_region_size);
    }

    OPENVINO_ASSERT(
        number_of_sections_encountered == manifest.get_number_of_entries(),
        "The number of sections encountered doesn't match the number of manifest entries. Sections encountered: ",
        number_of_sections_encountered,
        ". Manifest entries: ",
        manifest.get_number_of_entries());
}

size_t BlobReader::get_npu_region_size(BlobSource& npu_formatted_blob) {
    OPENVINO_ASSERT(
        npu_formatted_blob.get_remaining_size() >= MINIMUM_BLOB_SIZE,
        "The remaining size of the blob is too small to contain all mandatory information. Remaining size: ",
        npu_formatted_blob.get_remaining_size());
    const size_t cursor_before_reading = npu_formatted_blob.tellg();

    std::string magic_bytes(MAGIC_BYTES.size(), 0);
    npu_formatted_blob.read_into_buffer(const_cast<char*>(magic_bytes.c_str()), MAGIC_BYTES.size());
    OPENVINO_ASSERT(magic_bytes == MAGIC_BYTES,
                    "Invalid magic bytes. Found: ",
                    magic_bytes,
                    ". Expected: ",
                    MAGIC_BYTES);

    uint32_t format_version;
    npu_formatted_blob.read_into_buffer(&format_version, sizeof(format_version));
    OPENVINO_ASSERT(format_version == FORMAT_VERSION,
                    "Invalid blob format version. Found: ",
                    format_version,
                    ". Expected: ",
                    FORMAT_VERSION);

    uint64_t npu_region_size;
    npu_formatted_blob.read_into_buffer(&npu_region_size, sizeof(npu_region_size));
    npu_formatted_blob.seekg(cursor_before_reading);

    OPENVINO_ASSERT(
        npu_region_size <= npu_formatted_blob.get_remaining_size(),
        "The size of the NPU blob region is too great compared to the remaining size of the blob. NPU region size: ",
        npu_region_size);

    return npu_region_size;
}

}  // namespace intel_npu
