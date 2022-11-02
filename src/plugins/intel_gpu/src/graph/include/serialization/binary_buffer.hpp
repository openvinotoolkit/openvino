// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <sstream>
#include <stdexcept>
#include <type_traits>
#include "buffer.hpp"
#include "helpers.hpp"
#include "bind.hpp"

namespace cldnn {
class BinaryOutputBuffer : public OutputBuffer<BinaryOutputBuffer> {
public:
    BinaryOutputBuffer(std::ostream& stream) : OutputBuffer<BinaryOutputBuffer>(this), stream(stream) {}

    void write(void const * data, std::streamsize size) {
        auto const written_size = stream.rdbuf()->sputn(reinterpret_cast<const char*>(data), size);
        if (written_size != size) {
            throw std::runtime_error("Failed to write " + std::to_string(size) + " bytes to stream! Wrote " + std::to_string(written_size));
        }
    }

private:
    std::ostream& stream;
};

class BinaryInputBuffer : public InputBuffer<BinaryInputBuffer> {
public:
    BinaryInputBuffer(std::istream& stream, engine& engine) : InputBuffer(this, engine), stream(stream) {}

    void read(void* const data, std::streamsize size) {
        auto const read_size = stream.rdbuf()->sgetn(reinterpret_cast<char*>(data), size);
        if (read_size != size) {
            throw std::runtime_error("Failed to read " + std::to_string(size) + " bytes from stream! Read " + std::to_string(read_size));
        }
    }

private:
    std::istream& stream;
};

template <typename T>
class Serializer<BinaryOutputBuffer, T, typename std::enable_if<std::is_arithmetic<T>::value>::type> {
public:
    static void save(BinaryOutputBuffer& buffer, const T& object) {
        buffer.write(std::addressof(object), sizeof(object));
    }
};

template <typename T>
class Serializer<BinaryInputBuffer, T, typename std::enable_if<std::is_arithmetic<T>::value>::type> {
public:
    static void load(BinaryInputBuffer& buffer, T& object) {
        buffer.read(std::addressof(object), sizeof(object));
    }
};

template <typename T>
class Serializer<BinaryOutputBuffer, Data<T>> {
public:
    static void save(BinaryOutputBuffer& buffer, const Data<T>& bin_data) {
        buffer.write(bin_data.data, static_cast<std::streamsize>(bin_data.number_of_bytes));
    }
};

template <typename T>
class Serializer<BinaryInputBuffer, Data<T>> {
public:
    static void load(BinaryInputBuffer& buffer, Data<T>& bin_data) {
        buffer.read(bin_data.data, static_cast<std::streamsize>(bin_data.number_of_bytes));
    }
};

}  // namespace cldnn

#define BIND_BINARY_BUFFER_WITH_TYPE(cls_name, obj_type) \
            namespace cldnn {                            \
            const object_type cls_name::type = obj_type; \
            BIND_TO_BUFFER(BinaryOutputBuffer, cls_name) \
            BIND_TO_BUFFER(BinaryInputBuffer, cls_name)  \
            }
