// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <utility>
#include <type_traits>
#include "openvino/core/except.hpp"
#include "serializer.hpp"

namespace cldnn {

class engine;

template <typename BufferType>
class Buffer {
public:
    Buffer(BufferType* const buffer) : buffer(buffer) {}

    template <typename ... Types>
    inline BufferType& operator()(Types&& ... args) {
        process(std::forward<Types>(args)...);
        return *buffer;
    }

protected:
    inline BufferType& getBuffer() {
        return *buffer;
    }

    BufferType* const buffer;

private:
    template <typename T, typename ... OtherTypes>
    inline void process(T&& first, OtherTypes&& ... remains) {
        process(std::forward<T>(first));
        process(std::forward<OtherTypes>(remains)...);
    }

    template <typename T>
    inline void process(T&& object){
        buffer->process(std::forward<T>(object));
    }
};

template <typename BufferType>
class OutputBuffer : public Buffer<BufferType> {
    friend class Buffer<BufferType>;
public:
    OutputBuffer(BufferType* const buffer) : Buffer<BufferType>(buffer) {}

    template <typename T>
    inline BufferType& operator<<(T&& arg) {
        process(std::forward<T>(arg));
        return Buffer<BufferType>::getBuffer();
    }
private:
    template <typename T>
    inline void process(T&& object) {
        Serializer<BufferType, typename std::remove_const<typename std::remove_reference<T>::type>::type>::save(*Buffer<BufferType>::buffer, object);
    }
};

template <typename BufferType>
class InputBuffer : public Buffer<BufferType> {
    friend class Buffer<BufferType>;
public:
    InputBuffer(BufferType* const buffer, engine& engine) : Buffer<BufferType>(buffer), _engine(engine) {}

    template <typename T>
    inline BufferType& operator>>(T&& arg) {
        process(std::forward<T>(arg));
        return Buffer<BufferType>::getBuffer();
    }

    engine& get_engine() { return _engine; }
private:
    template <typename T>
    inline void process(T&& object) {
        Serializer<BufferType, typename std::remove_reference<T>::type>::load(*Buffer<BufferType>::buffer, object);
    }

    engine& _engine;
};
}  // namespace cldnn
