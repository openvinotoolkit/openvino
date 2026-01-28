#ifndef _UNKNOWN_HPP_
 #define _UNKNOWN_HPP_

#include <iostream>

#include "isection.hpp"

struct UnknownSection : ISection {
    explicit UnknownSection(SectionHeader header) : ISection(header) { };

    void serialize(std::ostream& stream) override {
        // OPENVINO_THROW("Not supported");
    }

    void read_value(std::istream& stream) override {
        stream.seekg(header.length, std::ios::cur);
    }

    void read_value(const uint8_t* data) override {
        // do nothing
    }
};

#endif // _UNKNOWN_HPP_