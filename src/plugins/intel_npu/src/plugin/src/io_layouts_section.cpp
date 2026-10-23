// Copyright (C) 2018-2026 Intel Corporation
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

void IOLayoutsSection::encode(const ov::runtime::hsm::SectionSink& sink) const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "IOLayoutsSection::write");

    const uint32_t number_of_input_layouts = m_input_layouts.size();
    const uint32_t number_of_output_layouts = m_output_layouts.size();
    sink({reinterpret_cast<const std::byte*>(&number_of_input_layouts), sizeof(number_of_input_layouts)});
    sink({reinterpret_cast<const std::byte*>(&number_of_output_layouts), sizeof(number_of_output_layouts)});

    m_logger.debug("Writting %lu input layouts and %lu output layouts",
                   number_of_input_layouts,
                   number_of_output_layouts);

    const auto write_layouts = [&](const std::vector<ov::Layout>& layouts) {
        for (const ov::Layout& layout : layouts) {
            const std::string layout_string = layout.to_string();
            const uint16_t string_length = static_cast<uint16_t>(layout_string.size());
            sink({reinterpret_cast<const std::byte*>(&string_length), sizeof(string_length)});
            sink({reinterpret_cast<const std::byte*>(layout_string.c_str()), string_length});

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

ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>> IOLayoutsSection::decode =
    [](const ov::runtime::hsm::Section& section) -> std::optional<std::shared_ptr<ISection>> {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "IOLayoutsSection::decode");
    const Logger logger("IOLayoutsSection", Logger::global().level());
    OPENVINO_ASSERT(section.tag().id() == static_cast<uint32_t>(SectionTypeCode::IO_LAYOUTS),
                    "Logic error: the decoder received a section of the wrong type");

    const size_t section_length = section.size();
    if (section_length < MINIMUM_LAYOUTS_SECTION_SIZE) {
        logger.error("The length of the IOLayouts section is too small. Received: %zu. Minimum expected: %zu",
                     section_length,
                     MINIMUM_LAYOUTS_SECTION_SIZE);
        return std::nullopt;
    }

    size_t offset = 0;
    uint32_t number_of_input_layouts;
    uint32_t number_of_output_layouts;
    section.read(offset, sizeof(number_of_input_layouts), &number_of_input_layouts);
    offset += sizeof(number_of_input_layouts);
    section.read(offset, sizeof(number_of_output_layouts), &number_of_output_layouts);
    offset += sizeof(number_of_output_layouts);

    const size_t max_layouts =
        (section_length - sizeof(number_of_input_layouts) - sizeof(number_of_output_layouts)) / SIZE_OF_LAYOUT_SIZE;
    if (number_of_input_layouts > max_layouts || number_of_output_layouts > max_layouts - number_of_input_layouts) {
        logger.error("The number of I/O layouts exceeds the limit of the section");
        return std::nullopt;
    }

    logger.debug("Reading %lu input layouts and %lu output layouts", number_of_input_layouts, number_of_output_layouts);

    const auto read_n_layouts = [&](const uint32_t number_of_layouts, const char* logger_addition) {
        std::vector<ov::Layout> layouts;
        if (!number_of_layouts) {
            return layouts;
        }

        uint16_t string_length;
        layouts.reserve(number_of_layouts);
        for (uint32_t layout_index = 0; layout_index < number_of_layouts; ++layout_index) {
            section.read(offset, sizeof(string_length), &string_length);
            offset += sizeof(string_length);
            if (string_length > section_length - offset) {
                logger.error("The size of at least one layout exceeds the limit of the section");
                OPENVINO_THROW("");
            }

            std::string layout_string(string_length, 0);
            section.read(offset, string_length, const_cast<char*>(layout_string.c_str()));
            offset += sizeof(number_of_input_layouts);

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

    std::vector<ov::Layout> input_layouts, output_layouts;
    try {
        input_layouts = read_n_layouts(number_of_input_layouts, "Input");
        output_layouts = read_n_layouts(number_of_output_layouts, "Output");
    } catch (const std::exception&) {
        return std::nullopt;
    }

    if (section_length - offset != 0) {
        logger.error("Failed to read the whole content of the section");
        return std::nullopt;
    }
    return std::make_shared<IOLayoutsSection>(input_layouts, output_layouts, Logger::global().level());
};

}  // namespace intel_npu
