#pragma once
#include <utility>
#include <type_traits>
#include "intel_gpu/runtime/engine.hpp"
#include "serializer.hpp"

namespace cldnn {

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
private:
    template <typename T>
    inline /*typename std::enable_if<!std::is_same<std::unique_ptr<B>, typename std::remove_reference<T>::type>::value>::type*/ void process(T&& object) {
        Serializer<BufferType, typename std::remove_reference<T>::type>::load(*Buffer<BufferType>::buffer, object);
    }

    // template <typename T>
    // inline /*typename std::enable_if<std::is_same<std::unique_ptr<B>, typename std::remove_reference<T>::type>::value>::type*/ void process(std::unique_ptr<T>& object) {
    //     Serializer<BufferType, std::unique_ptr<T>>::load(*Buffer<BufferType>::buffer, object, _engine);
    // }

    engine& _engine;
};
}