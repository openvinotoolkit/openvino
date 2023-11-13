// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/util.hpp"

#include <algorithm>
#include <deque>
#include <forward_list>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <unordered_set>

#include "openvino/util/common_util.hpp"
#include "openvino/util/log.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START

namespace ngraph {
void dump(std::ostream& out, const void* _data, size_t _size) {
    auto flags = out.flags();
    const uint8_t* data = reinterpret_cast<const uint8_t*>(_data);
    size_t len = _size;
    size_t index = 0;
    while (index < len) {
        out << std::hex << std::setw(8) << std::setfill('0') << index;
        for (int i = 0; i < 8; i++) {
            if (index + i < len) {
                out << " " << std::hex << std::setw(2) << std::setfill('0') << static_cast<uint32_t>(data[i]);
            } else {
                out << "   ";
            }
        }
        out << "  ";
        for (int i = 8; i < 16; i++) {
            if (index + i < len) {
                out << " " << std::hex << std::setw(2) << std::setfill('0') << static_cast<uint32_t>(data[i]);
            } else {
                out << "   ";
            }
        }
        out << "  ";
        for (int i = 0; i < 16; i++) {
            char ch = (index + i < len ? data[i] : ' ');
            out << ((ch < 32) ? '.' : ch);
        }
        out << "\n";
        data += 16;
        index += 16;
    }
    out.flags(flags);
}

std::string to_lower(const std::string& s) {
    return ov::util::to_lower(s);
}

std::string to_upper(const std::string& s) {
    return ov::util::to_upper(s);
}

std::string trim(const std::string& s) {
    return ov::util::trim(s);
}

std::vector<std::string> split(const std::string& src, char delimiter, bool do_trim) {
    return ov::util::split(src, delimiter, do_trim);
}

size_t hash_combine(const std::vector<size_t>& list) {
    return ov::util::hash_combine(list);
}

void* ngraph_malloc(size_t size) {
    auto ptr = malloc(size);
    if (size != 0 && !ptr) {
        OPENVINO_ERR << "malloc failed to allocate memory of size " << size;
        throw std::bad_alloc();
    }
    return ptr;
}

void ngraph_free(void* ptr) {
    if (ptr) {
        free(ptr);
    }
}

size_t round_up(size_t size, size_t alignment) {
    if (alignment == 0) {
        return size;
    }

    size_t remainder = size % alignment;
    if (remainder == 0) {
        return size;
    }

    return size + alignment - remainder;
}

size_t stopwatch::get_call_count() const {
    return m_total_count;
}

size_t stopwatch::get_seconds() const {
    return std::chrono::duration_cast<std::chrono::seconds>(get_timer_value()).count();
}

size_t stopwatch::get_milliseconds() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(get_timer_value()).count();
}

size_t stopwatch::get_microseconds() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(get_timer_value()).count();
}

size_t stopwatch::get_nanoseconds() const {
    return get_timer_value().count();
}

std::chrono::nanoseconds stopwatch::get_timer_value() const {
    if (m_active) {
        return (m_clock.now() - m_start_time);
    } else {
        return m_last_time;
    }
}

size_t stopwatch::get_total_seconds() const {
    return std::chrono::duration_cast<std::chrono::seconds>(m_total_time).count();
}

size_t stopwatch::get_total_milliseconds() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(m_total_time).count();
}

size_t stopwatch::get_total_microseconds() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(m_total_time).count();
}

size_t stopwatch::get_total_nanoseconds() const {
    return m_total_time.count();
}

template <>
float parse_string<float>(const std::string& s) {
    const char* tmp = s.c_str();
    char* end;
    float result = strtof(tmp, &end);
    if (*end != 0) {
        throw std::runtime_error("Could not parse literal '" + s + "'");
    }
    return result;
}

template <>
double parse_string<double>(const std::string& s) {
    const char* tmp = s.c_str();
    char* end;
    double result = strtod(tmp, &end);
    if (*end != 0) {
        throw std::runtime_error("Could not parse literal '" + s + "'");
    }
    return result;
}

template <>
int8_t parse_string<int8_t>(const std::string& s) {
    char* err;
    int8_t result = static_cast<int8_t>(strtol(s.c_str(), &err, 10));

    // Check that (1) parsing succeeded and (2) the entire string was used.
    if (*err != 0) {
        throw std::runtime_error("Could not parse literal '" + s + "'");
    }

    return result;
}

template <>
uint8_t parse_string<uint8_t>(const std::string& s) {
    char* err;
    int8_t result = static_cast<int8_t>(strtol(s.c_str(), &err, 10));

    // Check that (1) parsing succeeded and (2) the entire string was used.
    if (*err != 0) {
        throw std::runtime_error("Could not parse literal '" + s + "'");
    }

    return result;
}

AxisVector get_default_order(const Shape& shape) {
    return get_default_order(shape.size());
}

AxisVector get_default_order(const PartialShape& shape) {
    return get_default_order(shape.rank());
}

AxisVector get_default_order(size_t rank) {
    AxisVector default_order(rank);
    std::iota(begin(default_order), end(default_order), 0);
    return default_order;
}

AxisVector get_default_order(const Rank& rank) {
    OPENVINO_ASSERT(rank.is_static(), "Can not calculate default order for dynamic rank");

    AxisVector default_order(rank.get_length());
    std::iota(begin(default_order), end(default_order), 0);
    return default_order;
}

void parse_version_string(std::string version, size_t& major, size_t& minor, size_t& patch, std::string& extra) {
    // Since regex is broken in gcc 4.8 I will just manually parse the version string
    // Version strings look like `0.25.0-rc.0+7c32240` or `v0.25.0-rc.0+7c32240`
    size_t start;
    size_t end;
    extra = "";
    start = (version[0] == 'v' ? 1 : 0);
    end = version.find_first_of('.', start);
    std::string major_str = version.substr(start, end - start);
    start = end + 1;

    end = version.find_first_of('.', start);
    std::string minor_str = version.substr(start, end - start);
    start = end + 1;

    end = version.find_first_of("-+", start);
    std::string patch_str = version.substr(start, end - start);
    start = end;

    if (start != std::string::npos) {
        extra = version.substr(start);
    }

    size_t err;
    bool error = false;
    try {
        major = stoi(major_str, &err);
        if (err != major_str.size()) {
            error = true;
        }
        minor = stoi(minor_str, &err);
        if (err != minor_str.size()) {
            error = true;
        }
        patch = stoi(patch_str, &err);
        if (err != patch_str.size()) {
            error = true;
        }
    } catch (...) {
        error = true;
    }
    if (error) {
        OPENVINO_THROW("Error parsing version string '", version, "'");
    }
}
}  // namespace ngraph

std::vector<float> read_float_vector(std::shared_ptr<ngraph::runtime::Tensor> tv) {
    std::vector<float> float_vec;
    ov::element::Type element_type = tv->get_element_type();

    if (element_type == ov::element::boolean) {
        std::vector<char> vec = read_vector<char>(tv);
        // Changed from vector ctor to explicit for loop to add static_cast
        // This silences MSVC warnings
        for (char value : vec) {
            float_vec.push_back(static_cast<float>(value));
        }
    } else if (element_type == ov::element::bf16) {
        std::vector<ov::bfloat16> vec = read_vector<ov::bfloat16>(tv);
        float_vec = ov::bfloat16::to_float_vector(vec);
    } else if (element_type == ov::element::f16) {
        std::vector<ov::float16> vec = read_vector<ov::float16>(tv);
        for (ov::float16 value : vec) {
            float_vec.push_back(static_cast<float>(value));
        }
    } else if (element_type == ov::element::f32) {
        std::vector<float> vec = read_vector<float>(tv);
        for (float value : vec) {
            float_vec.push_back(static_cast<float>(value));
        }
    } else if (element_type == ov::element::f64) {
        std::vector<double> vec = read_vector<double>(tv);
        for (double value : vec) {
            float_vec.push_back(static_cast<float>(value));
        }
    } else if (element_type == ov::element::i8) {
        std::vector<int8_t> vec = read_vector<int8_t>(tv);
        for (int8_t value : vec) {
            float_vec.push_back(static_cast<float>(value));
        }
    } else if (element_type == ov::element::i16) {
        std::vector<int16_t> vec = read_vector<int16_t>(tv);
        for (int16_t value : vec) {
            float_vec.push_back(static_cast<float>(value));
        }
    } else if (element_type == ov::element::i32) {
        std::vector<int32_t> vec = read_vector<int32_t>(tv);
        for (int32_t value : vec) {
            float_vec.push_back(static_cast<float>(value));
        }
    } else if (element_type == ov::element::i64) {
        std::vector<int64_t> vec = read_vector<int64_t>(tv);
        for (int64_t value : vec) {
            float_vec.push_back(static_cast<float>(value));
        }
    } else if (element_type == ov::element::u8) {
        std::vector<uint8_t> vec = read_vector<uint8_t>(tv);
        for (uint8_t value : vec) {
            float_vec.push_back(static_cast<float>(value));
        }
    } else if (element_type == ov::element::u16) {
        std::vector<uint16_t> vec = read_vector<uint16_t>(tv);
        for (uint16_t value : vec) {
            float_vec.push_back(static_cast<float>(value));
        }
    } else if (element_type == ov::element::u32) {
        std::vector<uint32_t> vec = read_vector<uint32_t>(tv);
        for (uint32_t value : vec) {
            float_vec.push_back(static_cast<float>(value));
        }
    } else if (element_type == ov::element::u64) {
        std::vector<uint64_t> vec = read_vector<uint64_t>(tv);
        for (uint64_t value : vec) {
            float_vec.push_back(static_cast<float>(value));
        }
    } else {
        OPENVINO_THROW("Unsupported OpenVINO element type.");
    }

    return float_vec;
}

std::vector<int64_t> read_index_vector(std::shared_ptr<ngraph::runtime::Tensor> tv) {
    std::vector<int64_t> index_vec;
    ov::element::Type element_type = tv->get_element_type();

    if (element_type == ov::element::boolean) {
        std::vector<char> vec = read_vector<char>(tv);
        // Changed from vector ctor to explicit for loop to add static_cast
        // This silences MSVC warnings
        for (char value : vec) {
            index_vec.push_back(static_cast<int64_t>(value));
        }
    } else if (element_type == ov::element::bf16) {
        std::vector<ov::bfloat16> vec = read_vector<ov::bfloat16>(tv);
        std::vector<float> float_vec = ov::bfloat16::to_float_vector(vec);
        for (float value : float_vec) {
            index_vec.push_back(static_cast<int64_t>(value));
        }
    } else if (element_type == ov::element::f16) {
        std::vector<ov::float16> vec = read_vector<ov::float16>(tv);
        for (ov::float16 value : vec) {
            index_vec.push_back(static_cast<int64_t>(static_cast<float>(value)));
        }
    } else if (element_type == ov::element::f32) {
        std::vector<float> vec = read_vector<float>(tv);
        for (float value : vec) {
            index_vec.push_back(static_cast<int64_t>(value));
        }
    } else if (element_type == ov::element::f64) {
        std::vector<double> vec = read_vector<double>(tv);
        for (double value : vec) {
            index_vec.push_back(static_cast<int64_t>(value));
        }
    } else if (element_type == ov::element::i8) {
        std::vector<int8_t> vec = read_vector<int8_t>(tv);
        for (int8_t value : vec) {
            index_vec.push_back(static_cast<int64_t>(value));
        }
    } else if (element_type == ov::element::i16) {
        std::vector<int16_t> vec = read_vector<int16_t>(tv);
        for (int16_t value : vec) {
            index_vec.push_back(static_cast<int64_t>(value));
        }
    } else if (element_type == ov::element::i32) {
        std::vector<int32_t> vec = read_vector<int32_t>(tv);
        for (int32_t value : vec) {
            index_vec.push_back(static_cast<int64_t>(value));
        }
    } else if (element_type == ov::element::i64) {
        index_vec = read_vector<int64_t>(tv);
    } else if (element_type == ov::element::u8) {
        std::vector<uint8_t> vec = read_vector<uint8_t>(tv);
        for (uint8_t value : vec) {
            index_vec.push_back(static_cast<int64_t>(value));
        }
    } else if (element_type == ov::element::u16) {
        std::vector<uint16_t> vec = read_vector<uint16_t>(tv);
        for (uint16_t value : vec) {
            index_vec.push_back(static_cast<int64_t>(value));
        }
    } else if (element_type == ov::element::u32) {
        std::vector<uint32_t> vec = read_vector<uint32_t>(tv);
        for (uint32_t value : vec) {
            index_vec.push_back(static_cast<int64_t>(value));
        }
    } else if (element_type == ov::element::u64) {
        std::vector<uint64_t> vec = read_vector<uint64_t>(tv);
        for (uint64_t value : vec) {
            index_vec.push_back(static_cast<int64_t>(value));
        }
    } else {
        OPENVINO_THROW("Unsupported OpenVINO element type.");
    }

    return index_vec;
}
