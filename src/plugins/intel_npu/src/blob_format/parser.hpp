#ifndef _BLOB_FORMAT_HPP_
 #define _BLOB_FORMAT_HPP_

#include "header.hpp"
#include "factory.hpp"
#include "isection.hpp"

void read_sections_from_data(const uint8_t* ptr, uint64_t size, std::vector<std::shared_ptr<ISection>>& sections)
{
    std::cout << "read_sections_from_data " << ptr << " size " << size << std::endl; 
    uint64_t curr = 0;
    while(curr < size){
        std::cout << "curr: " << curr << " size " << size << std::endl; 
        SectionHeader header;
        auto bytes = header.read(&ptr[curr]);
        std::cout << "Found header.type: " << header.type << " header.length: " << header.length << std::endl;
        std::shared_ptr<ISection> section = SectionFactory::instance().create(header);
        curr += bytes;
        section->read_value(&ptr[curr]);
        curr += header.length;
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
    
void read_blob(std::istream& stream, std::vector<std::shared_ptr<ISection>>& sections) {
    Header header;
    header.read(stream);
    read_sections(stream, sections);
}

void read_blob_from_data(const uint8_t* data, uint64_t size, std::vector<std::shared_ptr<ISection>>& sections)
{
    std::cout << "read_blob_from_data " << (uint64_t)data << " size " << size << std::endl; 

    Header header;
    uint64_t bytes = header.read(data);
    read_sections_from_data(&data[bytes], size - bytes, sections);
}

void serialize_sections(std::ostream& stream, std::vector<std::shared_ptr<ISection>>& sections) {
    for (auto section : sections) {
        section->serialize(stream);
    }
}

#endif // _BLOB_FORMAT_HPP_