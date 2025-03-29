// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <random>

#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"

namespace {
template <typename T>
void copy_data(const ov::Tensor& tv, const std::vector<T>& data) {
    size_t data_size = data.size() * sizeof(T);
    if (data_size > 0) {
        OPENVINO_ASSERT(tv.get_byte_size() >= data_size);

        memcpy(tv.data(), data.data(), data_size);
    }
}

template <>
inline void copy_data<bool>(const ov::Tensor& tv, const std::vector<bool>& data) {
    std::vector<char> data_char(data.begin(), data.end());
    copy_data(tv, data_char);
}

template <typename T>
void init_int_tv(const ov::Tensor& tv, std::default_random_engine& engine, T min, T max) {
    size_t size = tv.get_size();
    std::uniform_int_distribution<T> dist(min, max);
    std::vector<T> vec(size);
    for (T& element : vec) {
        element = dist(engine);
    }
    size_t data_size = vec.size() * sizeof(T);
    OPENVINO_ASSERT(tv.get_byte_size() >= data_size);
    memcpy(tv.data(), vec.data(), data_size);
}

template <>
inline void init_int_tv<char>(const ov::Tensor& tv, std::default_random_engine& engine, char min, char max) {
    size_t size = tv.get_size();
    std::uniform_int_distribution<int16_t> dist(static_cast<short>(min), static_cast<short>(max));
    std::vector<char> vec(size);
    for (char& element : vec) {
        element = static_cast<char>(dist(engine));
    }
    size_t data_size = vec.size() * sizeof(char);
    OPENVINO_ASSERT(tv.get_byte_size() >= data_size);
    memcpy(tv.data(), vec.data(), data_size);
}

template <>
inline void init_int_tv<int8_t>(const ov::Tensor& tv, std::default_random_engine& engine, int8_t min, int8_t max) {
    size_t size = tv.get_size();
    std::uniform_int_distribution<int16_t> dist(static_cast<short>(min), static_cast<short>(max));
    std::vector<int8_t> vec(size);
    for (int8_t& element : vec) {
        element = static_cast<int8_t>(dist(engine));
    }
    size_t data_size = vec.size() * sizeof(int8_t);
    OPENVINO_ASSERT(tv.get_byte_size() >= data_size);
    memcpy(tv.data(), vec.data(), data_size);
}

template <>
inline void init_int_tv<uint8_t>(const ov::Tensor& tv, std::default_random_engine& engine, uint8_t min, uint8_t max) {
    size_t size = tv.get_size();
    std::uniform_int_distribution<int16_t> dist(static_cast<short>(min), static_cast<short>(max));
    std::vector<uint8_t> vec(size);
    for (uint8_t& element : vec) {
        element = static_cast<uint8_t>(dist(engine));
    }
    size_t data_size = vec.size() * sizeof(uint8_t);
    OPENVINO_ASSERT(tv.get_byte_size() >= data_size);
    memcpy(tv.data(), vec.data(), data_size);
}

template <typename T>
void init_real_tv(const ov::Tensor& tv, std::default_random_engine& engine, T min, T max) {
    size_t size = tv.get_size();
    std::uniform_real_distribution<T> dist(min, max);
    std::vector<T> vec(size);
    for (T& element : vec) {
        element = dist(engine);
    }
    size_t data_size = vec.size() * sizeof(T);
    OPENVINO_ASSERT(tv.get_byte_size() >= data_size);
    memcpy(tv.data(), vec.data(), data_size);
}

inline void random_init(const ov::Tensor& tv, std::default_random_engine& engine) {
    ov::element::Type et = tv.get_element_type();
    if (et == ov::element::boolean) {
        init_int_tv<char>(tv, engine, 0, 1);
    } else if (et == ov::element::f32) {
        init_real_tv<float>(tv, engine, std::numeric_limits<float>::min(), 1.0f);
    } else if (et == ov::element::f64) {
        init_real_tv<double>(tv, engine, std::numeric_limits<double>::min(), 1.0);
    } else if (et == ov::element::i8) {
        init_int_tv<int8_t>(tv, engine, -1, 1);
    } else if (et == ov::element::i16) {
        init_int_tv<int16_t>(tv, engine, -1, 1);
    } else if (et == ov::element::i32) {
        init_int_tv<int32_t>(tv, engine, 0, 1);
    } else if (et == ov::element::i64) {
        init_int_tv<int64_t>(tv, engine, 0, 1);
    } else if (et == ov::element::u8) {
        init_int_tv<uint8_t>(tv, engine, 0, 1);
    } else if (et == ov::element::u16) {
        init_int_tv<uint16_t>(tv, engine, 0, 1);
    } else if (et == ov::element::u32) {
        init_int_tv<uint32_t>(tv, engine, 0, 1);
    } else if (et == ov::element::u64) {
        init_int_tv<uint64_t>(tv, engine, 0, 1);
    } else {
        OPENVINO_THROW("unsupported type");
    }
}
}  // namespace

template <ov::element::Type_t ET>
ov::Tensor make_tensor(const ov::Shape& shape,
                       const std::vector<typename ov::element_type_traits<ET>::value_type>& data) {
    OPENVINO_ASSERT(shape_size(shape) == data.size(), "Incorrect number of initialization elements");
    auto tensor = ov::Tensor(ET, shape);
    copy_data(tensor, data);
    return tensor;
}

template <ov::element::Type_t ET>
ov::Tensor make_tensor(const ov::Shape& shape) {
    auto tensor = ov::Tensor(ET, shape);
    static std::default_random_engine engine(2112);
    random_init(tensor, engine);
    return tensor;
}
