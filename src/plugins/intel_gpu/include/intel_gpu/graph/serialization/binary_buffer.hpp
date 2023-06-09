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
class stream;

class BinaryOutputBuffer : public OutputBuffer<BinaryOutputBuffer> {
public:
    BinaryOutputBuffer(std::ostream& ostream, engine& engine)
    : OutputBuffer<BinaryOutputBuffer>(this), _ostream(ostream), _impl_params(nullptr), _engine(engine) {}

    void write(void const * data, std::streamsize size) {
        auto const written_size = _ostream.rdbuf()->sputn(reinterpret_cast<const char*>(data), size);
        OPENVINO_ASSERT(written_size == size,
            "[GPU] Failed to write " + std::to_string(size) + " bytes to stream! Wrote " + std::to_string(written_size));
    }

    void setKernelImplParams(void* impl_params) { _impl_params = impl_params; }
    void* getKernelImplParams() const { return _impl_params; }

    engine& get_engine() { return _engine; }

    void set_stream(std::shared_ptr<stream> exec_stream) { _exec_stream = exec_stream; }
    stream& get_stream() const { return *_exec_stream; }

private:
    std::ostream& _ostream;
    void* _impl_params;
    engine& _engine;
    std::shared_ptr<stream> _exec_stream;
};

class BinaryInputBuffer : public InputBuffer<BinaryInputBuffer> {
public:
    BinaryInputBuffer(std::istream& istream, engine& engine)
    : InputBuffer(this, engine), _istream(istream), _impl_params(nullptr), _num_networks(0), _stream_id(0), _engine(engine) {}

    void read(void* const data, std::streamsize size) {
        auto const read_size = _istream.rdbuf()->sgetn(reinterpret_cast<char*>(data), size);
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

    std::streampos tellg() { return _istream.tellg(); }
    void seekg(std::streampos pos) { _istream.seekg(pos); }

    void new_network_added() { _num_networks += 1; }
    int get_num_networks() const { return _num_networks; }

    void set_stream_id(uint16_t stream_id) { _stream_id = stream_id; }
    uint16_t get_stream_id() const { return _stream_id; }

    engine& get_engine() { return _engine; }

    void set_stream(std::shared_ptr<stream> exec_stream) { _exec_stream = exec_stream; }
    stream& get_stream() const { return *_exec_stream; }

private:
    std::istream& _istream;
    void* _impl_params;
    std::vector<std::unordered_map<std::string, std::shared_ptr<memory>>> _const_data_map;
    int _num_networks;
    uint16_t _stream_id;
    engine& _engine;
    std::shared_ptr<stream> _exec_stream;
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
            const std::string cls_name::type_for_serialization = #cls_name; \
            }

#define BIND_BINARY_BUFFER_WITH_TYPE(cls_name) \
            namespace cldnn {                            \
            const std::string cls_name::type_for_serialization = #cls_name; \
            BIND_TO_BUFFER(BinaryOutputBuffer, cls_name) \
            BIND_TO_BUFFER(BinaryInputBuffer, cls_name)  \
            }
