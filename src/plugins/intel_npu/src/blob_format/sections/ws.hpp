#ifndef _WS_HPP_
 #define _WS_HPP_

#include <iostream>
#include <vector>

#include "isection.hpp"

// | Header | N ELF1 ELF2 ... ELFn |
struct WSSection : ISection {
    std::vector<std::shared_ptr<ELFSection>> elf_blobs;
    uint64_t num_subgraphs; // (N-1) x Inits + Main

    explicit WSSection(std::vector<std::shared_ptr<ELFSection>>& elf_blobs) : elf_blobs(elf_blobs) {
        num_subgraphs = elf_blobs.size(); 
        // std::cout << "WSSection() subgraphs: " << num_subgraphs << std::endl;
        
        header.type = SectionType::WS;

        header.length = sizeof(num_subgraphs);
        for (auto elf : elf_blobs) {
            header.length += sizeof(elf->header) + elf->header.length;
        } 
        // std::cout << "WSSection() header.length: " << header.length << std::endl;
    }

    explicit WSSection(SectionHeader header) : ISection(header) { };

    void serialize(std::ostream& stream) override {
        header.serialize(stream);
        stream.write(reinterpret_cast<const char*>(&num_subgraphs), sizeof(num_subgraphs));

        for (auto elf : elf_blobs) {
            elf->serialize(stream);
        }
    }

    void read_value(std::istream& stream) override {
        stream.read(reinterpret_cast<char*>(&num_subgraphs), sizeof(num_subgraphs));

        elf_blobs.resize(num_subgraphs);

        for (int i = 0; i < num_subgraphs; i++) {
            stream.read(reinterpret_cast<char*>(&header), sizeof(header));

            auto section = std::make_shared<ELFSection>(header);
            section->read_value(stream);
            elf_blobs[i] = section;
        }
    }

    void read_value(const uint8_t* data) override {
        int64_t curr = 0;
        num_subgraphs = *(const int64_t*)(&data[curr]);
        curr += sizeof(num_subgraphs);

        elf_blobs.resize(num_subgraphs);

        for (int i = 0; i < num_subgraphs; i++) {
            SectionHeader *header = (SectionHeader*)&data[curr];

            auto section = std::make_shared<ELFSection>(*header);
            curr += sizeof(SectionHeader);
            section->read_value(&data[curr]);
            curr += header->length;
            elf_blobs[i] = section;
        }
    }
};

// Self-registration
namespace sections::ws
{
    const bool registered = []()
    {
        SectionFactory::instance().registerSection(
            SectionType::WS,
            [](SectionHeader& header)
            {
                // std::cout << "returning a shared ptr of WSSection" << std::endl;
                return std::make_shared<WSSection>(header);
            }
        );
        return true;
    }();
}

#endif // _WS_HPP_