// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "io_layouts_section.hpp"

#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"
#include "intel_npu/common/itt.hpp"

namespace {

// The size of the number of input + output layouts, assuming these values are "0"
constexpr size_t MINIMUM_LAYOUTS_SECTION_SIZE = 2 * sizeof(uint32_t);
constexpr size_t SIZE_OF_LAYOUT_SIZE = sizeof(uint16_t);

}  // namespace

namespace intel_npu {

IOLayoutsSection::IOLayoutsSection(const std::vector<ov::Layout>& input_layouts,
                                   const std::vector<ov::Layout>& output_layouts,
                                   const ov::log::Level log_level)
    : ISection(SectionTypeCode::IO_LAYOUTS),
      m_input_layouts(std::move(input_layouts)),
      m_output_layouts(std::move(output_layouts)),
      m_logger("IOLayoutsSection", log_level) {}

void IOLayoutsSection::write(BlobWriterInterface& writer) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "IOLayoutsSection::write");

    const uint32_t number_of_input_layouts = m_input_layouts.size();
    const uint32_t number_of_output_layouts = m_output_layouts.size();
    writer.write_from(&number_of_input_layouts, sizeof(number_of_input_layouts));
    writer.write_from(&number_of_output_layouts, sizeof(number_of_output_layouts));

    m_logger.debug("Writting %lu input layouts and %lu output layouts",
                   number_of_input_layouts,
                   number_of_output_layouts);

    const auto write_layouts = [&](const std::vector<ov::Layout>& layouts) {
        for (const ov::Layout& layout : layouts) {
            const std::string layout_string = layout.to_string();
            const uint16_t string_length = static_cast<uint16_t>(layout_string.size());
            writer.write_from(&string_length, sizeof(string_length));
            writer.write_from(layout_string.c_str(), string_length);

            m_logger.trace("Layout %s written", layout_string.data());
        }
    };

    write_layouts(m_input_layouts);
    write_layouts(m_output_layouts);
}

std::vector<ov::Layout> IOLayoutsSection::get_input_layouts() const {
    return m_input_layouts;
}

std::vector<ov::Layout> IOLayoutsSection::get_output_layouts() const {
    return m_output_layouts;
}

std::shared_ptr<ISection> IOLayoutsSection::read(BlobReaderInterface& blob_reader) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "IOLayoutsSection::read");
    const Logger logger("IOLayoutsSection", blob_reader.get_log_level());

    const size_t section_length = blob_reader.get_total_section_size();
    OPENVINO_ASSERT(section_length >= MINIMUM_LAYOUTS_SECTION_SIZE,
                    "The length of the IOLayouts section is too small. Received: ",
                    section_length,
                    ". Minimum expected: ",
                    MINIMUM_LAYOUTS_SECTION_SIZE);

    uint32_t number_of_input_layouts;
    uint32_t number_of_output_layouts;
    blob_reader.read_into_buffer(&number_of_input_layouts, sizeof(number_of_input_layouts));
    blob_reader.read_into_buffer(&number_of_output_layouts, sizeof(number_of_output_layouts));

    const size_t max_layouts =
        (section_length - sizeof(number_of_input_layouts) - sizeof(number_of_output_layouts)) / SIZE_OF_LAYOUT_SIZE;
    OPENVINO_ASSERT(
        number_of_input_layouts <= max_layouts && number_of_output_layouts <= max_layouts - number_of_input_layouts,
        "The number of I/O layouts exceeds the limit of the section");

    logger.debug("Reading %lu input layouts and %lu output layouts", number_of_input_layouts, number_of_output_layouts);

    const auto read_n_layouts = [&](const uint32_t number_of_layouts, const char* logger_addition) {
        std::vector<ov::Layout> layouts;
        if (!number_of_layouts) {
            return layouts;
        }

        uint16_t string_length;
        layouts.reserve(number_of_layouts);
        for (uint32_t layout_index = 0; layout_index < number_of_layouts; ++layout_index) {
            blob_reader.read_into_buffer(&string_length, sizeof(string_length));
            OPENVINO_ASSERT(string_length <= blob_reader.get_remaining_section_size(),
                            "The size of at least one layout exceeds the limit of the section");

            std::string layout_string(string_length, 0);
            blob_reader.read_into_buffer(const_cast<char*>(layout_string.c_str()), string_length);

            try {
                layouts.push_back(ov::Layout(std::move(layout_string)));
                logger.trace("Read layout %s", layout_string);
            } catch (const ov::Exception&) {
                logger.warning("Error encountered while constructing an ov::Layout object. %s index: %d. Value "
                               "read from blob: %s. A default value will be used instead.",
                               logger_addition,
                               layout_index,
                               layout_string.c_str());
                layouts.push_back(ov::Layout());
            }
        }
        return layouts;
    };

    const std::vector<ov::Layout> input_layouts = read_n_layouts(number_of_input_layouts, "Input");
    const std::vector<ov::Layout> output_layouts = read_n_layouts(number_of_output_layouts, "Output");
    OPENVINO_ASSERT(blob_reader.get_remaining_section_size() == 0, "Failed to read the whole content of the section");
    return std::make_shared<IOLayoutsSection>(input_layouts, output_layouts, blob_reader.get_log_level());
}

}  // namespace intel_npu
