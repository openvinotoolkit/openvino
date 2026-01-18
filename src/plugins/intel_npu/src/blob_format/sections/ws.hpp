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
        
        header.type = SectionType::WS;

        header.length = sizeof(num_subgraphs);
        for (auto elf : elf_blobs) {
            header.length += SectionHeader::size_on_disk() + elf->header.length;
        } 
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
            SectionHeader inner_header;
            inner_header.read(stream);

            auto section = std::make_shared<ELFSection>(inner_header);
            section->read_value(stream);
            elf_blobs[i] = section;
        }
    }

    void read_value(const uint8_t* data) override {
        int64_t curr = 0;
        memcpy(reinterpret_cast<char*>(&num_subgraphs), &data[curr], sizeof(num_subgraphs));
        curr += sizeof(num_subgraphs);

        elf_blobs.resize(num_subgraphs);

        for (int i = 0; i < num_subgraphs; i++) {
            SectionHeader inner_header;
            curr += inner_header.read(&data[curr]);

            auto section = std::make_shared<ELFSection>(inner_header);
            section->read_value(&data[curr]);
            curr += inner_header.length;
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