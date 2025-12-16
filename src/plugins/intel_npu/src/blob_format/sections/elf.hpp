#ifndef _ELF_HPP_
 #define _ELF_HPP_
#include <cstdint>
#include <ostream>
#include <vector>
#include <span>

#include "isection.hpp"

struct ELFSection : ISection {
    std::vector<uint8_t> blob;
    std::span<const uint8_t> blob_view;

    explicit ELFSection(std::vector<uint8_t> data) : blob(data), blob_view(data) {
        header.type = SectionType::ELF;
        header.length = data.size();
        // std::cout << "header.length: " << header.length << std::endl;
        // for (auto elem : view)
        //     std::cout << (int)elem << " ";
        // std::cout << std::endl;
    }
    
    explicit ELFSection(SectionHeader header) : ISection(header) { };

    void serialize(std::ostream& stream) override {
        header.serialize(stream);
        stream.write(reinterpret_cast<const char*>(blob.data()), header.length);
    }

    void read_value(std::istream& stream) override {
        blob.resize(header.length);
        stream.read(reinterpret_cast<char*>(blob.data()), header.length);
        // std::cout << "blob.size(): " << blob.size() << std::endl;
        // for (auto elem : blob)
        //     std::cout << (int)elem << " ";
        // std::cout << std::endl;
        blob_view = blob;
    }

    void read_value(const uint8_t* data) override {
        blob_view = std::span<const uint8_t>{data, static_cast<size_t>(header.length)};
        // for (auto elem : view)
        //     std::cout << (int)elem << " ";
        // std::cout << std::endl;
        // view = blob;
    }
};

// Self-registration
namespace sections::elf
{
    const bool registered = []()
    {
        SectionFactory::instance().registerSection(
            SectionType::ELF,
            [](SectionHeader& header)
            {
                // std::cout << "returning a shared ptr of ELFSection" << std::endl;
                return std::make_shared<ELFSection>(header);
            }
        );
        return true;
    }();
}

#endif // _ELF_HPP_