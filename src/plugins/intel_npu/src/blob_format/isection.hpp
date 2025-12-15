#ifndef _ISECTION_HPP_
#define _ISECTION_HPP_

enum SectionType : uint16_t {
    // Reserved Ops
    AND             = 0xF001,
    OR              = 0xF002,
    OPEN            = 0xF003,
    CLOSE           = 0xF004,
    // Regular types
    CRE             = 0x0001,
    ELF             = 0x0002,
    WS              = 0x0003,
    BS              = 0x0004,
    IO_LAYOUTS      = 0x0005,
};

struct SectionHeader{
    SectionType type;
    uint64_t length; // Length in bytes

    // Compute the size of this structure in its serialized form
    // sizeof(SectionHeader) might include internal padding
    static constexpr std::size_t disk_alignment = alignof(uint64_t);
    static constexpr std::size_t header_size = sizeof(SectionType) + sizeof(uint64_t);
    static constexpr std::size_t padding_size = (disk_alignment - (header_size % disk_alignment)) % disk_alignment;

    void serialize(std::ostream& stream){
        stream.write(reinterpret_cast<const char*>(&type), sizeof(type));
        stream.write(reinterpret_cast<const char*>(&length), sizeof(length));

        static constexpr uint8_t zeros[padding_size] = {};
        stream.write(reinterpret_cast<const char*>(zeros), padding_size);
    }

    // returns true if read succeeded, false otherwise
    bool read(std::istream& stream) {
        if(!stream.read(reinterpret_cast<char*>(&type), sizeof(type))) return false;
        if(!stream.read(reinterpret_cast<char*>(&length), sizeof(length))) return false;
        
        stream.ignore(padding_size);
        return stream.good();
    }

    static constexpr uint64_t size_on_disk(){
        return header_size + padding_size;
    }

    uint64_t read(const uint8_t* data){
        uint64_t bytes = 0;
        memcpy(reinterpret_cast<char*>(&type), &data[bytes], sizeof(type));
        bytes += sizeof(type);
        memcpy(reinterpret_cast<char*>(&length), &data[bytes], sizeof(length));
        bytes += sizeof(length);
        bytes += padding_size;
        
        return bytes;
    }
};

struct ISection {
    SectionHeader header;

    ISection() {};
    ISection(SectionHeader& header) : header(header) {};

    virtual void serialize(std::ostream& stream) = 0;

    virtual void read_value(std::istream& stream) = 0;

    // No data ownership
    virtual void read_value(const uint8_t* data) {};
};

#endif // _ISECTION_HPP_