#ifndef _HEADER_HPP_
 #define _HEADER_HPP_

#include <iostream>

#define RESERVED_BYTES (4)
struct Header
{
    uint8_t magic[8] = {'O', 'V', 'N', 'P', 'U'};
    uint32_t version = 0x30000; // 3.0;
    uint8_t reserved[RESERVED_BYTES];

    void serialize(std::ostream& stream) {
        stream.write(reinterpret_cast<const char*>(magic), sizeof(magic));
        stream.write(reinterpret_cast<const char*>(&version), sizeof(version));
        stream.write(reinterpret_cast<const char*>(&reserved), sizeof(reserved));
    }

    void read(std::istream& stream) {
        stream.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        stream.read(reinterpret_cast<char*>(&version), sizeof(version));
        stream.read(reinterpret_cast<char*>(&reserved), sizeof(reserved));
    }

    // We need this function only because we assume that padding might be added 
    // between the fields of the structure. While there is no padding in the 
    // serialized form. If there is a way to guarantee that does not happen,
    // we can replace this function with a cast
    uint64_t read(const uint8_t* data){
        uint64_t bytes = 0;
        memcpy(reinterpret_cast<char*>(&magic), &data[bytes], sizeof(magic));
        bytes += sizeof(magic);
        memcpy(reinterpret_cast<char*>(&version), &data[bytes], sizeof(version));
        bytes += sizeof(version);
        memcpy(reinterpret_cast<char*>(&reserved), &data[bytes], sizeof(reserved));
        bytes += sizeof(reserved);
        
        return bytes;
    }
};

#endif // _HEADER_HPP_
