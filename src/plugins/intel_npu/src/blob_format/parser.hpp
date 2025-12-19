#ifndef _BLOB_FORMAT_HPP_
 #define _BLOB_FORMAT_HPP_

#include "header.hpp"
#include "factory.hpp"
#include "isection.hpp"

// I think this can be a one-liner
Header read_header(std::istream& stream) {
    Header header;
    header.read(stream);
    return header;
}

std::shared_ptr<ISection> read_expression(std::istream& stream) {
    SectionHeader header;

    while (header.read(stream)) {
        if (header.type == CRE) {
            std::shared_ptr<ISection> expression_section = SectionFactory::instance().create(header);

            expression_section->read_value(stream);

            return expression_section;
        }
        stream.seekg(header.length);
    }
    // probably throw an exception instead?
    return nullptr;
}
// OVNPU | MAJOR | MINOR | TABLE OFFSET | ~~~~ SECTIONS ~~~~ | SECTION TABLE
bool read_sections_from_data(const uint8_t* ptr, uint64_t size, std::vector<std::shared_ptr<ISection>>& sections)
{
    uint64_t offset = 0;
    while(offset < size){
        // Ensure that the header fits into remaining bytes
        if(size - offset < SectionHeader::size_on_disk())
            return false;
    
        SectionHeader header;
        offset += header.read(ptr + offset);
        // std::cout << "Found header.type: " << header.type << " header.length: " << header.length << std::endl;
        // Ensure that the payload fits
        if(size - offset < header.length)
            return false;

        // Create section
        std::shared_ptr<ISection> section = SectionFactory::instance().create(header);
        section->read_value(ptr + offset);
        offset += header.length;
        sections.push_back(section);
    }

    return offset == size;
}

void read_sections(std::istream& stream, std::vector<std::shared_ptr<ISection>>& sections) {
    SectionHeader header;

    while (header.read(stream)) {
        // std::cout << "Found header.type: " << header.type << " header.length: " << header.length << std::endl;
        std::shared_ptr<ISection> section = SectionFactory::instance().create(header);
        section->read_value(stream);
        sections.push_back(section);
    }
}

// at any point, we might actually read favorable noise regarding header.length
// so before reading the payload of header.length size we should check if it could be out of bounds
void read_blob(std::istream& stream, std::vector<std::shared_ptr<ISection>>& sections) {
    // check if header is valid?
    // would it be helpful in invalid case to check if the first bytes are ELF magic? (old/raw blob format)
    Header header = read_header(stream);
    int sections_offset = stream.tellg();

    // optimization: keep length + offset of expression section to skip it while reading the rest of the sections
    auto expression_section = read_expression(stream);
    if (!expression_section) {
        throw std::runtime_error("Expression section not found");
    }

    stream.seekg(sections_offset, std::ios::beg);
    read_sections(stream, sections);
}

bool read_blob_from_data(const uint8_t* data, uint64_t size, std::vector<std::shared_ptr<ISection>>& sections)
{
    Header header;
    uint64_t header_size = header.read(data);
    return read_sections_from_data(data + header_size, size - header_size, sections);
}

void serialize_sections(std::ostream& stream, std::vector<std::shared_ptr<ISection>>& sections) {
    for (auto section : sections) {
        section->serialize(stream);
    }
}

#endif // _BLOB_FORMAT_HPP_