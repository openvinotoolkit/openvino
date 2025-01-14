// Copyright (C) 2018-2025 Intel Corporation
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
    : OutputBuffer<BinaryOutputBuffer>(this), stream(stream), _impl_params(nullptr), _strm(nullptr) {}

    virtual ~BinaryOutputBuffer() = default;

    virtual void write(void const* data, std::streamsize size) {
        auto const written_size = stream.rdbuf()->sputn(reinterpret_cast<const char*>(data), size);
        OPENVINO_ASSERT(written_size == size,
                        "[GPU] Failed to write " + std::to_string(size) + " bytes to stream! Wrote " +
                            std::to_string(written_size));
    }

    virtual void flush() {}

    void setKernelImplParams(void* impl_params) { _impl_params = impl_params; }
    void* getKernelImplParams() const { return _impl_params; }
    void set_stream(void* strm) { _strm = strm; }
    void* get_stream() const { return _strm; }

private:
    std::ostream& stream;
    void* _impl_params;
    void* _strm;
};

class BinaryInputBuffer : public InputBuffer<BinaryInputBuffer> {
public:
    BinaryInputBuffer(std::istream& stream, engine& engine)
    : InputBuffer<BinaryInputBuffer>(this, engine), _stream(stream), _impl_params(nullptr) {}

    virtual ~BinaryInputBuffer() = default;

    virtual void read(void* const data, std::streamsize size) {
        auto const read_size = _stream.rdbuf()->sgetn(reinterpret_cast<char*>(data), size);
        OPENVINO_ASSERT(read_size == size,
            "[GPU] Failed to read " + std::to_string(size) + " bytes from stream! Read " + std::to_string(read_size));
    }

    void setKernelImplParams(void* impl_params) { _impl_params = impl_params; }
    void* getKernelImplParams() const { return _impl_params; }

private:
    std::istream& _stream;
    void* _impl_params;
};

class EncryptedBinaryOutputBuffer : public BinaryOutputBuffer {
public:
    EncryptedBinaryOutputBuffer(std::ostream& stream, std::function<std::string(const std::string&)> encrypt)
        : BinaryOutputBuffer(stream),
          encrypt(encrypt) {
        OPENVINO_ASSERT(encrypt);
    }

    ~EncryptedBinaryOutputBuffer() override = default;

    void write(void const* data, std::streamsize size) override {
        plaintext_str.append(reinterpret_cast<const char*>(data), size);
    }

    void flush() override {
        auto encrypted_str = encrypt(plaintext_str);
        size_t bytes = encrypted_str.size();
        BinaryOutputBuffer::write(make_data(&bytes, sizeof(bytes)).data, sizeof(bytes));
        BinaryOutputBuffer::write(make_data(encrypted_str.c_str(), encrypted_str.size()).data, encrypted_str.size());
    }

private:
    std::string
        plaintext_str;  // Not using stringstream here because passing to encrypt() would produce an additional copy.
    std::function<std::string(const std::string&)> encrypt;
};

class EncryptedBinaryInputBuffer : public BinaryInputBuffer {
public:
    EncryptedBinaryInputBuffer(std::istream& stream,
                               engine& engine,
                               std::function<std::string(const std::string&)> decrypt)
        : BinaryInputBuffer(stream, engine),
          decrypt(decrypt) {
        OPENVINO_ASSERT(decrypt);

        size_t bytes;
        BinaryInputBuffer::read(make_data(&bytes, sizeof(bytes)).data, sizeof(bytes));

        // Not reading directly to plaintext_stream because decrypt(plaintext_stream.str()) would create an additional
        // copy.
        std::string str(bytes, 0);
        BinaryInputBuffer::read(
            make_data(const_cast<void*>(reinterpret_cast<const void*>(str.c_str())), str.size()).data,
            str.size());
        plaintext_stream.str(decrypt(str));
    }

    ~EncryptedBinaryInputBuffer() override = default;

    void read(void* const data, std::streamsize size) override {
        auto const read_size = plaintext_stream.rdbuf()->sgetn(reinterpret_cast<char*>(data), size);
        OPENVINO_ASSERT(
            read_size == size,
            "[GPU] Failed to read " + std::to_string(size) + " bytes from stream! Read " + std::to_string(read_size));
    }

private:
    std::stringstream plaintext_stream;
    std::function<std::string(const std::string&)> decrypt;
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
