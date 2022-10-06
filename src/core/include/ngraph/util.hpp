// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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
#include "ngraph/node.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/type/element_type_traits.hpp"
#include "openvino/core/enum_mask.hpp"

namespace ov {
class Node;
}
namespace ngraph {
using ov::EnumMask;
using ov::Node;
class stopwatch;

namespace runtime {
class Tensor;
}  // namespace runtime

template <typename T>
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
std::string vector_to_string(const T& v) {
    std::ostringstream os;
    os << "[ " << ngraph::join(v) << " ]";
    return os.str();
}

NGRAPH_API
NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
size_t hash_combine(const std::vector<size_t>& list);
NGRAPH_API
NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
void dump(std::ostream& out, const void*, size_t);
NGRAPH_API
std::string to_lower(const std::string& s);
NGRAPH_API
std::string to_upper(const std::string& s);
NGRAPH_API
std::string trim(const std::string& s);
NGRAPH_API
std::vector<std::string> split(const std::string& s, char delimiter, bool trim = false);

template <typename T>
NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
std::string locale_string(T x) {
    std::stringstream ss;
    ss.imbue(std::locale(""));
    ss << x;
    return ss.str();
}

class NGRAPH_API NGRAPH_DEPRECATED("It is obsolete structure and will be removed soon") stopwatch {
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
NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
T parse_string(const std::string& s) {
    T result;
    std::stringstream ss;

    ss << s;
    ss >> result;

    // Check that (1) parsing succeeded and (2) the entire string was used.
    if (ss.fail() || ss.rdbuf()->in_avail() != 0) {
        throw std::runtime_error("Could not parse literal '" + s + "'");
    }

    return result;
}

/// template specializations for float and double to handle INFINITY, -INFINITY
/// and NaN values.
template <>
NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
NGRAPH_API float parse_string<float>(const std::string& s);
template <>
NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
NGRAPH_API double parse_string<double>(const std::string& s);

/// template specializations for int8_t and uint8_t to handle the fact that default
/// implementation ends up treating values as characters so that the number "0" turns into
/// the parsed value 48, which is it's ASCII value
template <>
NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
NGRAPH_API int8_t parse_string<int8_t>(const std::string& s);
template <>
NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
NGRAPH_API uint8_t parse_string<uint8_t>(const std::string& s);

/// Parses a list of strings containing literals of the underlying type.
template <typename T>
NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
std::vector<T> parse_string(const std::vector<std::string>& ss) {
    NGRAPH_SUPPRESS_DEPRECATED_START
    std::vector<T> result(ss.size());
    std::transform(ss.begin(), ss.end(), result.begin(), [](const std::string& s) {
        return parse_string<T>(s);
    });
    return result;
    NGRAPH_SUPPRESS_DEPRECATED_END
}

template <typename T>
T ceil_div(const T& x, const T& y) {
    return (x == 0 ? 0 : (1 + (x - 1) / y));
}

template <typename T>
NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
T subtract_or_zero(T x, T y) {
    return y > x ? 0 : x - y;
}

NGRAPH_API
NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
void* ngraph_malloc(size_t size);
NGRAPH_API
NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
void ngraph_free(void*);

NGRAPH_API
NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
size_t round_up(size_t size, size_t alignment);

NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
bool is_valid_permutation(ngraph::AxisVector permutation, ngraph::Rank rank = Rank::dynamic());
template <typename T>
NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
T apply_permutation(T input, ngraph::AxisVector order);

extern template NGRAPH_API NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
    AxisVector apply_permutation<AxisVector>(AxisVector input, AxisVector order);

extern template NGRAPH_API NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
    Coordinate apply_permutation<Coordinate>(Coordinate input, AxisVector order);

extern template NGRAPH_API NGRAPH_DEPRECATED("This method is deprecated and will be removed soon") Strides
    apply_permutation<Strides>(Strides input, AxisVector order);

extern template NGRAPH_API NGRAPH_DEPRECATED("This method is deprecated and will be removed soon") Shape
    apply_permutation<Shape>(Shape input, AxisVector order);

template <>
NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
NGRAPH_API PartialShape apply_permutation(PartialShape input, AxisVector order);

NGRAPH_API
AxisVector get_default_order(size_t rank);

NGRAPH_API
AxisVector get_default_order(const Rank& rank);

NGRAPH_API
AxisVector get_default_order(const Shape& shape);

NGRAPH_API
AxisVector get_default_order(const PartialShape& shape);

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
NGRAPH_API
NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
void parse_version_string(std::string version, size_t& major, size_t& minor, size_t& patch, std::string& extra);

template <typename T>
T double_to_int(double x, double float_to_int_converter(double)) {
    if (!std::is_integral<T>()) {
        throw std::runtime_error("Function double_to_int template parameter must be an integral type.");
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
std::vector<T> read_vector(std::shared_ptr<ngraph::runtime::Tensor> tv) {
    if (ngraph::element::from<T>() != tv->get_element_type()) {
        throw std::invalid_argument("read_vector type must match Tensor type");
    }
    size_t element_count = ngraph::shape_size(tv->get_shape());
    size_t size = element_count * sizeof(T);
    std::vector<T> rc(element_count);
    tv->read(rc.data(), size);
    return rc;
}

template <class T, ngraph::element::Type_t ET>
std::vector<T> array_2_vector(typename ngraph::element_type_traits<ET>::value_type* data, size_t size) {
    std::vector<T> result(size);
    for (size_t i = 0; i < size; i++) {
        result[i] = static_cast<T>(data[i]);
    }
    return result;
}
template <typename T>
std::vector<T> host_tensor_2_vector(ngraph::HostTensorPtr tensor) {
    NGRAPH_CHECK(tensor != nullptr, "Invalid Tensor received, can't read the data from a null pointer.");

    switch (tensor->get_element_type()) {
    case ngraph::element::Type_t::boolean: {
        auto p = tensor->get_data_ptr<ngraph::element::Type_t::boolean>();
        return array_2_vector<T, ngraph::element::Type_t::boolean>(p, tensor->get_element_count());
    }
    case ngraph::element::Type_t::bf16: {
        auto p = tensor->get_data_ptr<ngraph::element::Type_t::bf16>();
        return array_2_vector<T, ngraph::element::Type_t::bf16>(p, tensor->get_element_count());
    }
    case ngraph::element::Type_t::f16: {
        auto p = tensor->get_data_ptr<ngraph::element::Type_t::f16>();
        return array_2_vector<T, ngraph::element::Type_t::f16>(p, tensor->get_element_count());
    }
    case ngraph::element::Type_t::f32: {
        auto p = tensor->get_data_ptr<ngraph::element::Type_t::f32>();
        return array_2_vector<T, ngraph::element::Type_t::f32>(p, tensor->get_element_count());
    }
    case ngraph::element::Type_t::f64: {
        auto p = tensor->get_data_ptr<ngraph::element::Type_t::f64>();
        return array_2_vector<T, ngraph::element::Type_t::f64>(p, tensor->get_element_count());
    }
    case ngraph::element::Type_t::i8: {
        auto p = tensor->get_data_ptr<ngraph::element::Type_t::i8>();
        return array_2_vector<T, ngraph::element::Type_t::i8>(p, tensor->get_element_count());
    }
    case ngraph::element::Type_t::i16: {
        auto p = tensor->get_data_ptr<ngraph::element::Type_t::i16>();
        return array_2_vector<T, ngraph::element::Type_t::i16>(p, tensor->get_element_count());
    }
    case ngraph::element::Type_t::i32: {
        auto p = tensor->get_data_ptr<ngraph::element::Type_t::i32>();
        return array_2_vector<T, ngraph::element::Type_t::i32>(p, tensor->get_element_count());
    }
    case ngraph::element::Type_t::i64: {
        auto p = tensor->get_data_ptr<ngraph::element::Type_t::i64>();
        return array_2_vector<T, ngraph::element::Type_t::i64>(p, tensor->get_element_count());
    }
    case ngraph::element::Type_t::u1:
        NGRAPH_CHECK(false, "u1 element type is unsupported");
        break;
    case ngraph::element::Type_t::u8: {
        auto p = tensor->get_data_ptr<ngraph::element::Type_t::u8>();
        return array_2_vector<T, ngraph::element::Type_t::u8>(p, tensor->get_element_count());
    }
    case ngraph::element::Type_t::u16: {
        auto p = tensor->get_data_ptr<ngraph::element::Type_t::u16>();
        return array_2_vector<T, ngraph::element::Type_t::u16>(p, tensor->get_element_count());
    }
    case ngraph::element::Type_t::u32: {
        auto p = tensor->get_data_ptr<ngraph::element::Type_t::u32>();
        return array_2_vector<T, ngraph::element::Type_t::u32>(p, tensor->get_element_count());
    }
    case ngraph::element::Type_t::u64: {
        auto p = tensor->get_data_ptr<ngraph::element::Type_t::u64>();
        return array_2_vector<T, ngraph::element::Type_t::u64>(p, tensor->get_element_count());
    }
    default:
        NGRAPH_UNREACHABLE("unsupported element type");
    }
}

std::vector<float> NGRAPH_API read_float_vector(std::shared_ptr<ngraph::runtime::Tensor> tv);

std::vector<int64_t> NGRAPH_API read_index_vector(std::shared_ptr<ngraph::runtime::Tensor> tv);

NGRAPH_API
std::ostream& operator<<(std::ostream& os, const ngraph::NodeVector& nv);
