// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(NGRAPH_LEGACY_HEADER_INCLUDED)
#    define NGRAPH_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>  // llvm 8.1 gets confused about `malloc` otherwise
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "ngraph/axis_vector.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/shape.hpp"
#include "openvino/core/enum_mask.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
class Node;
}
namespace ngraph {
using ov::EnumMask;
using ov::Node;
class stopwatch;
class Tensor;

OPENVINO_SUPPRESS_DEPRECATED_START
template <typename T>
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
std::string join(const T& v, const std::string& sep = ", ") {
    std::ostringstream ss;
    size_t count = 0;
    for (const auto& x : v) {
        if (count++ > 0) {
            ss << sep;
        }
        ss << x;
    }
    return ss.str();
}

template <typename T>
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
std::string vector_to_string(const T& v) {
    std::ostringstream os;
    os << "[ " << ngraph::join(v) << " ]";
    return os.str();
}

OPENVINO_API
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
size_t hash_combine(const std::vector<size_t>& list);
OPENVINO_API
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
void dump(std::ostream& out, const void*, size_t);
OPENVINO_API
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
std::string to_lower(const std::string& s);
OPENVINO_API
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
std::string to_upper(const std::string& s);
OPENVINO_API
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
std::string trim(const std::string& s);
OPENVINO_API
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
std::vector<std::string> split(const std::string& s, char delimiter, bool trim = false);

template <typename T>
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
std::string locale_string(T x) {
    std::stringstream ss;
    ss.imbue(std::locale(""));
    ss << x;
    return ss.str();
}

class OPENVINO_API OPENVINO_DEPRECATED("It is obsolete structure and will be removed soon") stopwatch {
public:
    void start() {
        if (m_active == false) {
            m_total_count++;
            m_active = true;
            m_start_time = m_clock.now();
        }
    }

    void stop() {
        if (m_active == true) {
            auto end_time = m_clock.now();
            m_last_time = end_time - m_start_time;
            m_total_time += m_last_time;
            m_active = false;
        }
    }

    size_t get_call_count() const;
    size_t get_seconds() const;
    size_t get_milliseconds() const;
    size_t get_microseconds() const;
    std::chrono::nanoseconds get_timer_value() const;
    size_t get_nanoseconds() const;

    size_t get_total_seconds() const;
    size_t get_total_milliseconds() const;
    size_t get_total_microseconds() const;
    size_t get_total_nanoseconds() const;

private:
    std::chrono::high_resolution_clock m_clock;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start_time;
    bool m_active = false;
    std::chrono::nanoseconds m_total_time = std::chrono::high_resolution_clock::duration::zero();
    std::chrono::nanoseconds m_last_time = std::chrono::high_resolution_clock::duration::zero();
    size_t m_total_count = 0;
};

/// Parses a string containing a literal of the underlying type.
template <typename T>
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
T parse_string(const std::string& s) {
    T result;
    std::stringstream ss;

    ss << s;
    ss >> result;

    // Check that (1) parsing succeeded and (2) the entire string was used.
    if (ss.fail() || ss.rdbuf()->in_avail() != 0) {
        OPENVINO_THROW("Could not parse literal '" + s + "'");
    }

    return result;
}

/// template specializations for float and double to handle INFINITY, -INFINITY
/// and NaN values.
template <>
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
OPENVINO_API float parse_string<float>(const std::string& s);
template <>
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
OPENVINO_API double parse_string<double>(const std::string& s);

/// template specializations for int8_t and uint8_t to handle the fact that default
/// implementation ends up treating values as characters so that the number "0" turns into
/// the parsed value 48, which is it's ASCII value
template <>
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
OPENVINO_API int8_t parse_string<int8_t>(const std::string& s);
template <>
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
OPENVINO_API uint8_t parse_string<uint8_t>(const std::string& s);

/// Parses a list of strings containing literals of the underlying type.
template <typename T>
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
std::vector<T> parse_string(const std::vector<std::string>& ss) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    std::vector<T> result(ss.size());
    std::transform(ss.begin(), ss.end(), result.begin(), [](const std::string& s) {
        return parse_string<T>(s);
    });
    return result;
    OPENVINO_SUPPRESS_DEPRECATED_END
}

template <typename T>
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
T ceil_div(const T& x, const T& y) {
    return (x == 0 ? 0 : (1 + (x - 1) / y));
}

template <typename T>
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
T subtract_or_zero(T x, T y) {
    return y > x ? 0 : x - y;
}

OPENVINO_API
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
void* ngraph_malloc(size_t size);
OPENVINO_API
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
void ngraph_free(void*);

OPENVINO_API
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
size_t round_up(size_t size, size_t alignment);

OPENVINO_API
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
AxisVector get_default_order(size_t rank);

OPENVINO_API
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
AxisVector get_default_order(const ov::Rank& rank);

OPENVINO_API
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
AxisVector get_default_order(const ov::Shape& shape);

OPENVINO_API
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
AxisVector get_default_order(const ov::PartialShape& shape);

/// \brief Function to query parsed version information of the version of ngraph which
/// contains this function. Version information strictly follows Semantic Versioning
/// http://semver.org
/// \param version The major part of the version
/// \param major Returns the major part of the version
/// \param minor Returns the minor part of the version
/// \param patch Returns the patch part of the version
/// \param extra Returns the extra part of the version. This includes everything following
/// the patch version number.
///
/// \note Throws a runtime_error if there is an error during parsing
OPENVINO_API
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
void parse_version_string(std::string version, size_t& major, size_t& minor, size_t& patch, std::string& extra);

template <typename T>
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
T double_to_int(double x, double float_to_int_converter(double)) {
    if (!std::is_integral<T>()) {
        OPENVINO_THROW("Function double_to_int template parameter must be an integral type.");
    }

    x = float_to_int_converter(x);

    double min_t = static_cast<double>(std::numeric_limits<T>::min());
    if (x < min_t) {
        return std::numeric_limits<T>::min();
    }

    double max_t = static_cast<double>(std::numeric_limits<T>::max());
    if (x > max_t) {
        return std::numeric_limits<T>::max();
    }

    return static_cast<T>(x);
}
}  // end namespace ngraph

template <typename T>
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
std::vector<T> read_vector(std::shared_ptr<ov::Tensor> tv) {
    if (ov::element::from<T>() != tv->get_element_type()) {
        OPENVINO_THROW("read_vector type must match Tensor type");
    }
    size_t element_count = ngraph::shape_size(tv->get_shape());
    size_t size = element_count * sizeof(T);
    std::vector<T> rc(element_count);
    std::memcpy(rc.data(), tv->data(), size);
    return rc;
}

template <class T, ov::element::Type_t ET>
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
std::vector<T> array_2_vector(typename ov::element_type_traits<ET>::value_type* data, size_t size) {
    std::vector<T> result(size);
    for (size_t i = 0; i < size; i++) {
        result[i] = static_cast<T>(data[i]);
    }
    return result;
}

OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
std::vector<float> OPENVINO_API read_float_vector(std::shared_ptr<ov::Tensor> tv);

OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
std::vector<int64_t> OPENVINO_API read_index_vector(std::shared_ptr<ov::Tensor> tv);

OPENVINO_API
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
std::ostream& operator<<(std::ostream& os, const ov::NodeVector& nv);
OPENVINO_SUPPRESS_DEPRECATED_END
