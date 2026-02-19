// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "io_layouts_section.hpp"

#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"

namespace intel_npu {

IOLayoutsSection::IOLayoutsSection(const std::vector<ov::Layout>& input_layouts,
                                   const std::vector<ov::Layout>& output_layouts)
    : ISection(PredefinedSectionType::IO_LAYOUTS),
      m_input_layouts(std::move(input_layouts)),
      m_output_layouts(std::move(output_layouts)) {}

void IOLayoutsSection::write(std::ostream& stream, BlobWriter* writer) {
    const uint64_t number_of_input_layouts = m_input_layouts.size();
    const uint64_t number_of_output_layouts = m_output_layouts.size();
    stream.write(reinterpret_cast<const char*>(&number_of_input_layouts), sizeof(number_of_input_layouts));
    stream.write(reinterpret_cast<const char*>(&number_of_output_layouts), sizeof(number_of_output_layouts));

    const auto write_layouts = [&](const std::vector<ov::Layout>& layouts) {
        for (const ov::Layout& layout : layouts) {
            const std::string layout_string = layout.to_string();
            const uint16_t string_length = static_cast<uint16_t>(layout_string.size());
            stream.write(reinterpret_cast<const char*>(&string_length), sizeof(string_length));
            stream.write(layout_string.c_str(), string_length);
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

std::shared_ptr<ISection> IOLayoutsSection::read(BlobReader* blob_reader, const size_t section_length) {
    OPENVINO_ASSERT(section_length >= 2 * sizeof(uint64_t),
                    "The length of the IOLayouts section is too small. Received: ",
                    section_length,
                    ". Minimum expected: ",
                    2 * sizeof(uint64_t));

    uint64_t number_of_input_layouts;
    uint64_t number_of_output_layouts;
    blob_reader->copy_data_from_source(reinterpret_cast<char*>(&number_of_input_layouts),
                                       sizeof(number_of_input_layouts));
    blob_reader->copy_data_from_source(reinterpret_cast<char*>(&number_of_output_layouts),
                                       sizeof(number_of_output_layouts));

    const Logger logger("IOLayoutsSection", Logger::global().level());  // TODO is the verbosity correct?

    const auto read_n_layouts = [&](const uint64_t number_of_layouts, const char* logger_addition) {
        std::vector<ov::Layout> layouts;
        if (!number_of_layouts) {
            return layouts;
        }

        uint16_t string_length;
        layouts.reserve(number_of_layouts);
        for (uint64_t layout_index = 0; layout_index < number_of_layouts; ++layout_index) {
            blob_reader->copy_data_from_source(reinterpret_cast<char*>(&string_length), sizeof(string_length));

            std::string layoutString(string_length, 0);
            blob_reader->copy_data_from_source(const_cast<char*>(layoutString.c_str()), string_length);

            try {
                layouts.push_back(ov::Layout(std::move(layoutString)));
            } catch (const ov::Exception&) {
                logger.warning("Error encountered while constructing an ov::Layout object. %s index: %d. Value "
                               "read from blob: %s. A default value will be used instead.",
                               logger_addition,
                               layout_index,
                               layoutString.c_str());
                layouts.push_back(ov::Layout());
            }
        }
        return layouts;
    };

    const std::vector<ov::Layout> input_layouts = read_n_layouts(number_of_input_layouts, "Input");
    const std::vector<ov::Layout> output_layouts = read_n_layouts(number_of_output_layouts, "Output");
    return std::make_shared<IOLayoutsSection>(input_layouts, output_layouts);
}

}  // namespace intel_npu
