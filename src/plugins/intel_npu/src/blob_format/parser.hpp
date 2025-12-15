#ifndef _BLOB_FORMAT_HPP_
 #define _BLOB_FORMAT_HPP_

#include "isection.hpp"
#include "factory.hpp"

void read_sections_from_data(const uint8_t* ptr, uint64_t size, std::vector<std::shared_ptr<ISection>>& sections){
    uint64_t curr = 0;
    while(curr + sizeof(SectionHeader) <= size){
        SectionHeader *header = (SectionHeader*)&ptr[curr];
        std::cout << "Found header.type: " << header->type << " header.length: " << header->length << std::endl;
        std::shared_ptr<ISection> section = SectionFactory::instance().create(*header);
        curr += sizeof(SectionHeader);
        section->read_value(&ptr[curr]);
        curr += header->length;
        sections.push_back(section);
    }
}

void read_sections(std::istream& stream, std::vector<std::shared_ptr<ISection>>& sections) {
    SectionHeader header;

    while (stream.read(reinterpret_cast<char*>(&header), sizeof(header))) {
        std::cout << "Found header.type: " << header.type << " header.length: " << header.length << std::endl;
        std::shared_ptr<ISection> section = SectionFactory::instance().create(header);
        section->read_value(stream);
        sections.push_back(section);
    }
}

void serialize_sections(std::ostream& stream, std::vector<std::shared_ptr<ISection>>& sections) {
    for (auto section : sections) {
        section->serialize(stream);
    }
}

#endif // _BLOB_FORMAT_HPP_