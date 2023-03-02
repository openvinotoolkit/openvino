// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "tensor.hpp"
#include "half.hpp"

#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <limits>
#include <string>
#include <functional>
#include <set>

#include <openvino/core/partial_shape.hpp>
#include <openvino/core/type/element_type.hpp>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{

/// @addtogroup cpp_memory Memory description and management
/// @{

constexpr size_t float_type_mask = 0x80;
constexpr size_t uint_type_mask = 0x40;
constexpr size_t bin_type_mask = 0x20;

/// @brief Possible data types could be stored in memory.
enum class data_types : size_t {
    bin = sizeof(int32_t) | bin_type_mask,
    u8 = sizeof(uint8_t) | uint_type_mask,
    i8 = sizeof(int8_t),
    f16 = sizeof(int16_t) | float_type_mask,
    f32 = sizeof(float) | float_type_mask,
    i32 = sizeof(int32_t),
    i64 = sizeof(int64_t)
};


/// Converts C++ type to @ref data_types .
template <typename T>
struct type_to_data_type;
#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <>
struct type_to_data_type<int8_t> { static constexpr data_types value = data_types::i8; };
template <>
struct type_to_data_type<uint8_t> { static constexpr data_types value = data_types::u8; };
template <>
struct type_to_data_type<int32_t> { static constexpr data_types value = data_types::i32; };
template <>
struct type_to_data_type<int64_t> { static constexpr data_types value = data_types::i64; };
template <>
struct type_to_data_type<half_t> { static constexpr data_types value = data_types::f16; };
template <>
struct type_to_data_type<float> { static constexpr data_types value = data_types::f32; };
#endif

/// Converts @ref data_types to C++ type.
template <data_types Data_Type>
struct data_type_to_type;
#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <>
struct data_type_to_type<data_types::bin> { typedef uint32_t type; };
template <>
struct data_type_to_type<data_types::u8> { typedef uint8_t type; };
template <>
struct data_type_to_type<data_types::i8> { typedef int8_t type; };
template <>
struct data_type_to_type<data_types::i32> { typedef int32_t type; };
template <>
struct data_type_to_type<data_types::i64> { typedef int64_t type; };
template <>
struct data_type_to_type<data_types::f16> { typedef half_t type; };
template <>
struct data_type_to_type<data_types::f32> { typedef float type; };
#endif

/// Helper class to identify key properties for data_types.
struct data_type_traits {
    static size_t size_of(data_types data_type) {
        return (static_cast<uint32_t>(data_type) & ~(float_type_mask | uint_type_mask | bin_type_mask));
    }

    static bool is_floating_point(data_types data_type) {
        return (static_cast<uint32_t>(data_type) & float_type_mask) != 0;
    }

    static bool is_i8_u8(data_types data_type) {
        return data_type == data_types::i8 || data_type == data_types::u8;
    }

    static size_t align_of(data_types data_type) {
        switch (data_type) {
            case data_types::bin:
                return alignof(data_type_to_type<data_types::bin>::type);
            case data_types::i8:
                return alignof(data_type_to_type<data_types::i8>::type);
            case data_types::u8:
                return alignof(data_type_to_type<data_types::u8>::type);
            case data_types::i32:
                return alignof(data_type_to_type<data_types::i32>::type);
            case data_types::i64:
                return alignof(data_type_to_type<data_types::i64>::type);
            case data_types::f16:
                return alignof(data_type_to_type<data_types::f16>::type);
            case data_types::f32:
                return alignof(data_type_to_type<data_types::f32>::type);
            default:
                return size_t(1);
        }
    }

    static std::string name(data_types data_type) {
        switch (data_type) {
            case data_types::bin:
                return "bin";
            case data_types::i8:
                return "i8";
            case data_types::u8:
                return "u8";
            case data_types::i32:
                return "i32";
            case data_types::i64:
                return "i64";
            case data_types::f16:
                return "f16";
            case data_types::f32:
                return "f32";
            default:
                assert(0);
                return "unknown (" + std::to_string(typename std::underlying_type<data_types>::type(data_type)) + ")";
        }
    }

    static data_types max_type(data_types dt1, data_types dt2) {
        if (dt1 == data_types::bin)
            return dt2;

        if (dt2 == data_types::bin)
            return dt1;

        if (size_of(dt1) < size_of(dt2))
            return dt2;

        if (size_of(dt1) > size_of(dt2))
            return dt1;

        if (is_floating_point(dt2))
            return dt2;

        return dt1;
    }

    static bool is_quantized(data_types dt) {
        return dt == data_types::u8 || dt == data_types::i8;
    }

    template <typename T>
    static T max(data_types data_type) {
        switch (data_type) {
            case data_types::i8:
                return static_cast<T>(std::numeric_limits<int8_t>::max());
            case data_types::u8:
                return static_cast<T>(std::numeric_limits<uint8_t>::max());
            case data_types::i32:
                return static_cast<T>(std::numeric_limits<int32_t>::max());
            case data_types::i64:
                return static_cast<T>(std::numeric_limits<int64_t>::max());
            case data_types::f16:
                return static_cast<T>(65504);
            case data_types::f32:
                return static_cast<T>(std::numeric_limits<float>::max());
            default:
                assert(0);
                return static_cast<T>(0);
        }
    }
    template <typename T>
    static T min(data_types data_type) {
        switch (data_type) {
            case data_types::i8:
                return static_cast<T>(std::numeric_limits<int8_t>::lowest());
            case data_types::u8:
                return static_cast<T>(std::numeric_limits<uint8_t>::lowest());
            case data_types::i32:
                return static_cast<T>(std::numeric_limits<int32_t>::lowest());
            case data_types::i64:
                return static_cast<T>(std::numeric_limits<int64_t>::lowest());
            case data_types::f16:
                return static_cast<T>(-65504);
            case data_types::f32:
                return static_cast<T>(std::numeric_limits<float>::lowest());
            default:
                assert(0);
                return static_cast<T>(0);
        }
    }
};

inline ::std::ostream& operator<<(::std::ostream& os, const data_types& dt) {
    return os << data_type_traits::name(dt);
}

/// Helper function to check if C++ type matches @p data_type.
template <typename T>
bool data_type_match(data_types data_type) {
    return data_type == type_to_data_type<T>::value;
}

inline data_types element_type_to_data_type(ov::element::Type t) {
    switch (t) {
    case ov::element::Type_t::i16:
    case ov::element::Type_t::u16:
    case ov::element::Type_t::f32:
    case ov::element::Type_t::f64:
        return cldnn::data_types::f32;
    case ov::element::Type_t::f16:
        return cldnn::data_types::f16;
    case ov::element::Type_t::u8:
        return cldnn::data_types::u8;
    case ov::element::Type_t::i8:
        return cldnn::data_types::i8;
    case ov::element::Type_t::i32:
    case ov::element::Type_t::u32:
    case ov::element::Type_t::u64:
        return cldnn::data_types::i32;
    case ov::element::Type_t::i64:
        return cldnn::data_types::i64;
    case ov::element::Type_t::boolean:
        return cldnn::data_types::i8;
    case ov::element::Type_t::u1:
        return cldnn::data_types::bin;
    default:
        throw std::runtime_error("Can't convert " + t.get_type_name() + " element type");
    }
}

inline ov::element::Type data_type_to_element_type(data_types t) {
    switch (t) {
    case cldnn::data_types::f32:
        return ov::element::Type_t::f32;
    case cldnn::data_types::f16:
        return ov::element::Type_t::f16;
    case cldnn::data_types::u8:
        return ov::element::Type_t::u8;
    case cldnn::data_types::i8:
        return ov::element::Type_t::i8;
    case cldnn::data_types::i32:
        return ov::element::Type_t::i32;
    case cldnn::data_types::i64:
        return ov::element::Type_t::i64;
    case cldnn::data_types::bin:
        return ov::element::Type_t::u1;
    default:
        throw std::runtime_error("Can't convert " + data_type_traits::name(t) + " precision");
    }
}

/// Helper function to get both data_types and format::type in a single, unique value. Useable in 'case' statement.
constexpr auto fuse(data_types dt, cldnn::format::type fmt) -> decltype(static_cast<std::underlying_type<data_types>::type>(dt) |
                                                                        static_cast<std::underlying_type<format::type>::type>(fmt)) {
    using dt_type = std::underlying_type<data_types>::type;
    using fmt_type = std::underlying_type<cldnn::format::type>::type;
    using fmt_narrow_type = int16_t;

    return static_cast<fmt_type>(fmt) <= std::numeric_limits<fmt_narrow_type>::max() &&
                   static_cast<dt_type>(dt) <= (std::numeric_limits<dt_type>::max() >> (sizeof(fmt_narrow_type) * 8))
               ? (static_cast<dt_type>(dt) << (sizeof(fmt_narrow_type) * 8)) |
                     (static_cast<fmt_type>(fmt) >= 0 ? static_cast<fmt_narrow_type>(fmt) : static_cast<fmt_narrow_type>(-1))
               : throw std::invalid_argument("data_type and/or format values are too big to be fused into single value");
}

/// @brief Represents data padding information.
struct padding {
    /// @brief Filling value for padding area.
    float filling_value() const { return _filling_value; }

    /// @brief Gets lower padding sizes. For spatials, it means size of left (X) and top (Y) padding.
    /// @return Tensor with padding for top/left/lower bounds of data.
    tensor lower_size() const { return _lower_size; }

    /// @brief Gets upper padding sizes. For spatials, it means size of right (X) and bottom (Y) padding.
    /// @return Tensor with padding for bottom/right/upper bounds of data.
    tensor upper_size() const { return _upper_size; }

    /// @brief
    /// @param lower_sizes Top-left padding sizes. See @ref tensor::tensor(const std::vector<value_type>&, value_type) for details.
    /// @param upper_sizes Bottom-right padding sizes. See @ref tensor::tensor(const std::vector<value_type>&, value_type) for details.
    /// @param filling_value Filling value for padding area.
    padding(const std::vector<tensor::value_type>& lower_sizes, const std::vector<tensor::value_type>& upper_sizes, float filling_value = 0.0f)
        : _lower_size(to_abs(lower_sizes), 0), _upper_size(to_abs(upper_sizes), 0), _filling_value(filling_value) {}

    /// @brief Constrcuts symmetric padding.
    /// @param sizes Top-left and bottom-right padding sizes. See @ref tensor::tensor(const std::vector<value_type>&, value_type) for details.
    /// @param filling_value Filling value for padding area.
    explicit padding(const std::vector<tensor::value_type>& sizes, float filling_value = 0.0f)
        : padding(sizes, sizes, filling_value) {}

    /// @brief Constructs "zero-sized" padding.
    padding() : padding({0, 0, 0, 0}, 0) {}

    /// @brief Returns true if padding size is not zero.
    explicit operator bool() const {
        return std::any_of(_lower_size.raw.begin(), _lower_size.raw.end(), [](const tensor::value_type& el) { return el != 0; }) ||
               std::any_of(_upper_size.raw.begin(), _upper_size.raw.end(), [](const tensor::value_type& el) { return el != 0; });
    }

    friend bool operator==(const padding& lhs, const padding& rhs) {
        return lhs._lower_size == rhs._lower_size && lhs._upper_size == rhs._upper_size && lhs._filling_value == rhs._filling_value;
    }

    friend bool operator!=(const padding& lhs, const padding& rhs) {
        return !(lhs == rhs);
    }

    friend bool operator<(const padding& lhs, const padding& rhs) {
        if (lhs._filling_value != rhs._filling_value)
            return (lhs._filling_value < rhs._filling_value);
        if (lhs._lower_size != rhs._lower_size)
            return (lhs._lower_size < rhs._lower_size);
        return (lhs._upper_size < rhs._upper_size);
    }

    static padding max(padding const& lhs, padding const& rhs, float filling_value = 0.0f) {
        auto lower = tensor::max(lhs.lower_size(), rhs.lower_size());
        auto upper = tensor::max(lhs.upper_size(), rhs.upper_size());
        return padding{lower.sizes(), upper.sizes(), filling_value};
    }

    size_t hash() const {
        size_t seed = 0;
        seed = cldnn::hash_combine(seed, _filling_value);
        seed = cldnn::hash_combine(seed, _lower_size.hash());
        seed = cldnn::hash_combine(seed, _upper_size.hash());
        return seed;
    }

private:
    tensor _lower_size;  ///< Lower padding sizes. For spatials, it means size of left (X) and top (Y) padding.
    tensor _upper_size;  ///< Upper padding sizes. For spatials, it means size of right (X) and bottom (Y) padding.
    // TODO: Add support for non-zero filling value (if necessary) or remove variable (if not necessary).
    float _filling_value;  ///< Filling value for an element of padding. If data type of elements is different than float it is converted
                           ///< to it using round-towards-nearest-even (for floating-point data types) or round-towards-zero (for integral
                           ///< data types).

    static std::vector<tensor::value_type> to_abs(const std::vector<tensor::value_type>& sizes) {
        std::vector<tensor::value_type> result;
        result.reserve(sizes.size());
        std::transform(sizes.cbegin(), sizes.cend(), std::back_inserter(result), [](const tensor::value_type& el) { return abs(el); });
        return result;  // NRVO
    }
};

/// @brief Describes memory layout.
/// @details Contains information about data stored in @ref memory.
struct layout {
    /// Constructs layout based on @p data_type and @p size information described by @ref tensor
    layout(data_types data_type, cldnn::format fmt, tensor size, padding apadding = padding())
        : data_type(data_type)
        , format(fmt)
        , data_padding(apadding) {
            auto sizes = fmt == format::any ? size.sizes() : size.sizes(format::get_default_format(fmt.dimension(),
                                                                                                   format::is_weights_format(fmt),
                                                                                                   format::is_grouped(fmt)));
            ov::Shape shape(sizes.begin(), sizes.end());
            this->size = ov::PartialShape(shape);
        }

    layout(ov::PartialShape size, data_types data_type, cldnn::format fmt, padding apadding = padding())
        : data_type(data_type)
        , format(fmt)
        , data_padding(apadding)
        , size(size) { }

    layout(const layout& other) = default;

    layout()
        : data_type(cldnn::data_types::bin)
        , format(cldnn::format::any)
        , data_padding(padding())
        , size(ov::PartialShape()) { }

    layout& operator=(const layout& other) {
        if (this == &other)
            return *this;
        data_type = other.data_type;
        format = other.format;
        size = other.size;
        data_padding = other.data_padding;
        return *this;
    }

    friend bool operator==(const layout& lhs, const layout& rhs) {
        auto get_pshape = [&](const layout& l){
            if (l.format != cldnn::format::any && l.size.size() < l.format.dimension()) {
                auto dims = l.get_dims();
                return ov::PartialShape(ov::Shape(dims.begin(), dims.end()));
            }
            return l.size;
        };

        if (lhs.get_partial_shape().rank() != rhs.get_partial_shape().rank())
            return false;

        auto check_pshape = (lhs.is_dynamic() || rhs.is_dynamic()) ? (lhs.size == rhs.size) : (get_pshape(lhs) == get_pshape(rhs));
        return lhs.data_type == rhs.data_type && lhs.format == rhs.format && check_pshape && lhs.data_padding == rhs.data_padding;
    }

    friend bool operator!=(const layout& lhs, const layout& rhs) {
        return !(lhs == rhs);
    }

    friend bool operator<(const layout& lhs, const layout& rhs) {
        if (lhs.data_type != rhs.data_type)
            return (lhs.data_type < rhs.data_type);
        if (lhs.format != rhs.format)
            return (lhs.format < rhs.format);
        if (lhs.count() < rhs.count())
            return (lhs.count() < rhs.count());
        return (lhs.data_padding < rhs.data_padding);
    }

    /// Number of elements to be stored in this memory layout
    size_t count() const;

    /// Layout size with padding included
    tensor get_buffer_size() const;

    tensor get_pitches() const;

    // @brief Calculates position within buffer of the data element pointed by the provided tensor.
    // element == { 0,0,0,0 } means first no-padding (i.e. data) element
    size_t get_linear_offset(tensor element = tensor(0)) const;

    /// @brief Get aligned linear size calculated as multiplication of all elements.
    size_t get_linear_size() const;

    /// Modify padding in layout
    layout with_padding(padding const& padd) const;

    /// Data type stored in @ref memory (see. @ref data_types)
    data_types data_type;

    /// Format stored in @ref memory (see. @ref format)
    cldnn::format format;

    /// Explicit padding of the @ref memory
    padding data_padding;

    /// Number of bytes needed to store this layout
    size_t bytes_count() const { return data_type_traits::size_of(data_type) * get_linear_size(); }

    size_t get_rank() const;

    size_t get_spatial_rank() const;

    tensor::value_type get_dim(size_t idx) const;

    tensor::value_type batch() const;

    tensor::value_type feature() const;

    tensor::value_type spatial(size_t spatial_idx) const;

    tensor::value_type group() const;

    tensor::value_type ofm() const;

    tensor::value_type ifm() const;

    std::vector<tensor::value_type> get_dims() const;

    std::vector<tensor::value_type> get_padded_dims() const;

    std::vector<tensor::value_type> get_ordered_dims() const;

    std::vector<size_t> get_dims_order() const;

    layout convert_to_weights_layout(bool is_grouped) const;

    std::string to_string() const;
    std::string to_short_string() const;

    bool is_dynamic() const;

    bool has_upper_bound() const {
        for (auto i : size) {
            if (i.get_max_length() == -1)
                return false;
        }
        return true;
    }

    bool is_static() const;

    ov::PartialShape get_partial_shape() const;

    ov::Shape get_shape() const;

    tensor get_tensor() const;

    template<typename T>
    T get() const;

    void set_tensor(const tensor& size);

    void set_partial_shape(const ov::PartialShape& size);

    // Returns true if other layout can be reinterpreted without need of reordering
    bool compatible(const layout& other) const;

    // Returns true if other layout is identical to this.
    // Note: layouts can only be considered identical if data size described by both layouts match (so no data are genereted
    // nor dropped). If layouts describe two buffers with different size, consider them not to be identical even if
    // smaller buffer can be considered to hold subsequence of larger buffer,  this behavior is required to force buffer allocation
    // for smaller buffer which, currently, should always be performed
    bool identical(const layout& other) const;

    ov::PartialShape transform(cldnn::format new_fmt) const;

    size_t hash() const {
        size_t seed = 0;
        seed = hash_combine(seed, data_padding.hash());
        seed = hash_combine(seed, format.value);
        seed = hash_combine(seed, data_type);

        auto pshape = get_partial_shape();
        for (size_t idx = 0; idx < pshape.size(); idx++) {
            seed = hash_combine(seed, pshape[idx].get_length());
        }
        return seed;
    }

private:
    /// The size of the @ref memory (excluding padding)
    ov::PartialShape size;
};

inline ::std::ostream& operator<<(::std::ostream& os, const layout& p) {
    return os << p.to_string();
}

/// @}
/// @}
}  // namespace cldnn
