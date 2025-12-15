#ifndef _ISECTION_HPP_
#define _ISECTION_HPP_

enum SectionType : uint64_t { 
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
    int64_t length; // Length in bytes

    void serialize(std::ostream& stream){
        stream.write(reinterpret_cast<const char*>(&type), sizeof(type));
        stream.write(reinterpret_cast<const char*>(&length), sizeof(length));
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