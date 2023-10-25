// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include "buffer.hpp"
#include "helpers.hpp"
#include "bind.hpp"

namespace cldnn {
struct memory;

class BinaryOutputBuffer : public OutputBuffer<BinaryOutputBuffer> {
public:
    BinaryOutputBuffer(std::ostream& stream)
    : OutputBuffer<BinaryOutputBuffer>(this), stream(stream), _impl_params(nullptr) {}

    void write(void const * data, std::streamsize size) {
        auto const written_size = stream.rdbuf()->sputn(reinterpret_cast<const char*>(data), size);
        OPENVINO_ASSERT(written_size == size,
            "[GPU] Failed to write " + std::to_string(size) + " bytes to stream! Wrote " + std::to_string(written_size));
    }

    void setKernelImplParams(void* impl_params) { _impl_params = impl_params; }
    void* getKernelImplParams() const { return _impl_params; }

private:
    std::ostream& stream;
    void* _impl_params;
};

class BinaryInputBuffer : public InputBuffer<BinaryInputBuffer> {
public:
    BinaryInputBuffer(std::istream& stream, engine& engine)
    : InputBuffer(this, engine), _stream(stream), _impl_params(nullptr) {}

    void read(void* const data, std::streamsize size) {
        auto const read_size = _stream.rdbuf()->sgetn(reinterpret_cast<char*>(data), size);
        OPENVINO_ASSERT(read_size == size,
            "[GPU] Failed to read " + std::to_string(size) + " bytes from stream! Read " + std::to_string(read_size));
    }

    void setKernelImplParams(void* impl_params) { _impl_params = impl_params; }
    void* getKernelImplParams() const { return _impl_params; }
    void addConstData(const uint32_t net_id, const std::string& prim_id, const std::shared_ptr<memory> mem_ptr) {
        while (_const_data_map.size() <= net_id) {
            _const_data_map.emplace_back(std::unordered_map<std::string, std::shared_ptr<memory>>());
        }
        OPENVINO_ASSERT(_const_data_map[net_id].find(prim_id) == _const_data_map[net_id].end(), "[GPU] duplicated primitive id " + prim_id);
        _const_data_map[net_id][prim_id] = mem_ptr;
    }
    std::shared_ptr<memory> getConstData(const uint32_t net_id, const std::string& prim_id) {
        OPENVINO_ASSERT(_const_data_map[net_id].find(prim_id) != _const_data_map[net_id].end(), "[GPU] Not found primitive id " + prim_id);
        return _const_data_map[net_id][prim_id];
    }

    std::streampos tellg() { return _stream.tellg(); }
    void seekg(std::streampos pos) { _stream.seekg(pos); }

private:
    std::istream& _stream;
    void* _impl_params;
    std::vector<std::unordered_map<std::string, std::shared_ptr<memory>>> _const_data_map;
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

#define ASSIGN_TYPE_NAME(cls_name) \
            namespace cldnn {                            \
            }

#define BIND_BINARY_BUFFER_WITH_TYPE(cls_name) \
            namespace cldnn {                            \
            BIND_TO_BUFFER(BinaryOutputBuffer, cls_name) \
            BIND_TO_BUFFER(BinaryInputBuffer, cls_name)  \
            }
