#ifndef _HEADER_HPP_
 #define _HEADER_HPP_

#include <iostream>

struct Header
{
    uint8_t magic[8] = {'O', 'V', 'N', 'P', 'U'};
    uint32_t version = 0x30000; // 3.0;
    uint8_t reserved[4];

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
};

#endif // _HEADER_HPP_