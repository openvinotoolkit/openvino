#ifndef _LAYOUTS_HPP_
 #define _LAYOUTS_HPP_

#include "isection.hpp"

// Faking an ov::Layout
namespace ov{

    struct Layout{
        Layout() {};
        Layout(std::string s) : layout(s) {};

        const std::string to_string() const{
            return layout;
        }

        std::string layout;
    };

}

// | HEADER | uint64_t uint64_t 
struct IOLayoutsSection : ISection {
    std::vector<ov::Layout> input_layouts;
    std::vector<ov::Layout> output_layouts;

    explicit IOLayoutsSection(std::vector<ov::Layout>& input_layouts, std::vector<ov::Layout>& output_layouts) : 
            input_layouts(input_layouts),
            output_layouts(output_layouts) 
    {
        const uint64_t num_input_layouts = input_layouts.size();
        const uint64_t num_output_layouts = output_layouts.size();

        auto get_length = [](const std::vector<ov::Layout>& layouts) {
                int64_t length = 0;
                for (ov::Layout layout : layouts) {
                    const std::string layout_string = layout.to_string();
                    const uint16_t string_length = static_cast<uint16_t>(layout_string.size());
                    length += string_length;
                }
                return length;
        };
        
        header.type = SectionType::IO_LAYOUTS;
        header.length = sizeof(num_input_layouts) + sizeof(num_output_layouts) + get_length(input_layouts) + get_length(output_layouts);
        // std::cout << "IOLayoutsSection() header.length: " << header.length << std::endl;
    }

    explicit IOLayoutsSection(SectionHeader header) : ISection(header) { };

    // some constructor here

    // TODO: discuss about having a standardized format to serialize container elements such as
    // std::vector<int> or std::vector<std::string>
    // why? probably because having methods serializePOD and serializeVector to delegate
    void serialize(std::ostream& stream) override {
        header.serialize(stream);

        const uint64_t num_input_layouts = input_layouts.size();
        const uint64_t num_output_layouts = output_layouts.size();
        stream.write(reinterpret_cast<const char*>(&num_input_layouts), sizeof(num_input_layouts));
        stream.write(reinterpret_cast<const char*>(&num_output_layouts), sizeof(num_output_layouts));

        const auto write_layouts = [&](const std::vector<ov::Layout>& layouts) {
                for (const ov::Layout& layout : layouts) {
                    const std::string layout_string = layout.to_string();
                    const uint16_t string_length = static_cast<uint16_t>(layout_string.size());
                    stream.write(reinterpret_cast<const char*>(&string_length), sizeof(string_length));
                    stream.write(layout_string.c_str(), string_length);
                }
        };
        write_layouts(input_layouts);
        write_layouts(output_layouts);
    }

    void read_value(std::istream& stream) override {
        uint64_t num_input_layouts, num_output_layouts;

        stream.read(reinterpret_cast<char*>(&num_input_layouts), sizeof(num_input_layouts));
        stream.read(reinterpret_cast<char*>(&num_output_layouts), sizeof(num_output_layouts));
        // std::cout << "IOLayouts::read_value num_inputs: " << num_input_layouts << " num_outputs: " << num_output_layouts << std::endl;

        auto read_n_layouts = [&](const uint64_t num_layouts, std::vector<ov::Layout>& layouts) {
            uint16_t string_length;
            layouts.reserve(num_layouts);
            for (uint64_t index = 0; index < num_layouts; ++index) {
                stream.read(reinterpret_cast<char*>(&string_length), sizeof(string_length));

                std::string layout_string(string_length, 0);
                stream.read(const_cast<char*>(layout_string.c_str()), string_length);

                try {
                    layouts.push_back(ov::Layout(std::move(layout_string)));
                } catch (...) {
                    // _logger.warning("Error encountered while constructing an ov::Layout object. %s index: %d. Value "
                    //                 "read from blob: %s. A default value will be used instead.",
                    //                 loggerAddition,
                    //                 layoutIndex,
                    //                 layoutString.c_str());
                    layouts.push_back(ov::Layout());
                }
            }
        };
        
        read_n_layouts(num_input_layouts, input_layouts);
        read_n_layouts(num_output_layouts, output_layouts);
    }

    void read_value(const uint8_t* data) override {
        uint64_t num_input_layouts, num_output_layouts;

        uint64_t curr = 0;
        num_input_layouts = *(uint64_t*)(&data[curr]);
        curr += sizeof(num_input_layouts);
        num_output_layouts = *(uint64_t*)(&data[curr]);
        curr += sizeof(num_output_layouts);
        // std::cout << "IOLayouts::read_value num_inputs: " << num_input_layouts << " num_outputs: " << num_output_layouts << std::endl;

        auto read_n_layouts = [&](const uint64_t num_layouts, std::vector<ov::Layout>& layouts) {
            uint16_t string_length;
            layouts.reserve(num_layouts);
            for (uint64_t index = 0; index < num_layouts; ++index) {
                string_length = *(uint16_t*)(&data[curr]);
                curr += sizeof(string_length);

                std::string layout_string(reinterpret_cast<const char*>(&data[curr]), string_length);
                curr += string_length;

                try {
                    layouts.push_back(ov::Layout(std::move(layout_string)));
                } catch (...) {
                    // _logger.warning("Error encountered while constructing an ov::Layout object. %s index: %d. Value "
                    //                 "read from blob: %s. A default value will be used instead.",
                    //                 loggerAddition,
                    //                 layoutIndex,
                    //                 layoutString.c_str());
                    layouts.push_back(ov::Layout());
                }
            }
        };
        

        read_n_layouts(num_input_layouts, input_layouts);
        read_n_layouts(num_output_layouts, output_layouts);
    }
};

// Self-registration
namespace sections::layouts
{
    const bool registered = []()
    {
        SectionFactory::instance().registerSection(
            SectionType::IO_LAYOUTS,
            [](SectionHeader& header)
            {
                // std::cout << "returning a shared ptr of IOLayoutsSection" << std::endl;
                return std::make_shared<IOLayoutsSection>(header);
            }
        );
        return true;
    }();
}

#endif // _LAYOUTS_HPP_