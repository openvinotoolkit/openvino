// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "tensor.hpp"

#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <limits>
#include <string>
#include <functional>

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/element_type_traits.hpp"

#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/graph/serialization/vector_serializer.hpp"

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{

/// @addtogroup cpp_memory Memory description and management
/// @{

/// @brief Possible data types could be stored in memory.
using data_types = ov::element::Type_t;

/// Helper class to identify key properties for data_types.
struct data_type_traits {
    static size_t size_of(data_types data_type) {
        auto et = ov::element::Type(data_type);
        OPENVINO_ASSERT(et.bitwidth() >= 8, "[GPU] Unexpected data_type_traits::size_of call for type with bitwidth < 8 (", et.get_type_name(), ")");
        return et.size();
    }

    static bool is_floating_point(data_types data_type) {
        return ov::element::Type(data_type).is_real();
    }

    static bool is_i8_u8(data_types data_type) {
        auto et = ov::element::Type(data_type);
        return et.is_quantized() && et.bitwidth() == 8;
    }

    static ov::element::Type max_type(ov::element::Type t1, ov::element::Type t2) {
        if (t1 == ov::element::u1)
            return t2;

        if (t2 == ov::element::u1)
            return t1;

        if (t1.bitwidth() < t2.bitwidth())
            return t2;

        if (t1.bitwidth() > t2.bitwidth())
            return t1;

        if (t2.is_real())
            return t2;

        return t1;
    }

    static bool is_quantized(ov::element::Type t) {
        return t.is_quantized();
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
                return static_cast<T>(std::numeric_limits<ov::float16>::max());
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
                return static_cast<T>(std::numeric_limits<ov::float16>::lowest());
            case data_types::f32:
                return static_cast<T>(std::numeric_limits<float>::lowest());
            default:
                assert(0);
                return static_cast<T>(0);
        }
    }
};

inline ::std::ostream& operator<<(::std::ostream& os, const data_types& dt) {
    return os << ov::element::Type(dt);
}

inline data_types element_type_to_data_type(ov::element::Type t) {
    switch (t) {
    case ov::element::Type_t::i16:
    case ov::element::Type_t::u16:
    case ov::element::Type_t::f64:
        return cldnn::data_types::f32;
    case ov::element::Type_t::u32:
    case ov::element::Type_t::u64:
        return cldnn::data_types::i32;
    case ov::element::Type_t::boolean:
        return cldnn::data_types::u8;
    default: return t;
    }
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

    void set_dynamic_pad(const tensor& dynamic_pad_dims) {
        _dynamic_pad_dims = dynamic_pad_dims;
    }

    tensor get_dynamic_pad_dims() const {
        return _dynamic_pad_dims;
    }
    /// @brief
    /// @param lower_sizes Top-left padding sizes. See @ref tensor::tensor(const std::vector<value_type>&, value_type) for details.
    /// @param upper_sizes Bottom-right padding sizes. See @ref tensor::tensor(const std::vector<value_type>&, value_type) for details.
    /// @param filling_value Filling value for padding area.
    padding(const std::vector<tensor::value_type>& lower_sizes,
            const std::vector<tensor::value_type>& upper_sizes,
            float filling_value = 0.0f,
            const tensor& dynamic_pad_dims = tensor(0))
        : _lower_size(to_abs(lower_sizes), 0),
          _upper_size(to_abs(upper_sizes), 0),
          _filling_value(filling_value),
          _dynamic_pad_dims(dynamic_pad_dims) {}

    /// @brief Constrcuts symmetric padding.
    /// @param sizes Top-left and bottom-right padding sizes. See @ref tensor::tensor(const std::vector<value_type>&,
    /// value_type) for details.
    /// @param filling_value Filling value for padding area.
    explicit padding(const std::vector<tensor::value_type>& sizes, float filling_value = 0.0f, const tensor& dynamic_pad_dims = tensor(0))
        : padding(sizes, sizes, filling_value, dynamic_pad_dims) {}

    /// @brief Constructs "zero-sized" padding.
    padding() : padding({0, 0, 0, 0}, 0, tensor(0)) {}

    /// @brief Returns true if padding size is not zero.
    explicit operator bool() const {
        return std::any_of(_lower_size.raw.begin(), _lower_size.raw.end(), [](const tensor::value_type& el) { return el != 0; }) ||
               std::any_of(_upper_size.raw.begin(), _upper_size.raw.end(), [](const tensor::value_type& el) { return el != 0; });
    }

    friend bool operator==(const padding& lhs, const padding& rhs) {
        return lhs._lower_size == rhs._lower_size && lhs._upper_size == rhs._upper_size &&
               lhs._filling_value == rhs._filling_value && lhs._dynamic_pad_dims == rhs._dynamic_pad_dims;
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
        auto dynamic_pad_dims = tensor::max(lhs.get_dynamic_pad_dims(), rhs.get_dynamic_pad_dims());
        return padding{lower.sizes(), upper.sizes(), filling_value, dynamic_pad_dims};
    }

    size_t hash() const {
        size_t seed = 0;
        seed = cldnn::hash_combine(seed, _filling_value);
        seed = cldnn::hash_combine(seed, _lower_size.hash());
        seed = cldnn::hash_combine(seed, _upper_size.hash());
        seed = cldnn::hash_combine(seed, _dynamic_pad_dims.hash());
        return seed;
    }

    void save(BinaryOutputBuffer& ob) const {
        ob << _lower_size.sizes();
        ob << _upper_size.sizes();
        ob << _filling_value;
        ob << _dynamic_pad_dims.sizes();
    }

    void load(BinaryInputBuffer& ib) {
        std::vector<tensor::value_type> sizes;
        ib >> sizes;
        _lower_size = tensor(sizes);
        ib >> sizes;
        _upper_size = tensor(sizes);
        ib >> _filling_value;
        ib >> sizes;
        _dynamic_pad_dims = tensor(sizes);
    }

private:
    tensor _lower_size;  ///< Lower padding sizes. For spatials, it means size of left (X) and top (Y) padding.
    tensor _upper_size;  ///< Upper padding sizes. For spatials, it means size of right (X) and bottom (Y) padding.
    // TODO: Add support for non-zero filling value (if necessary) or remove variable (if not necessary).
    float _filling_value;  ///< Filling value for an element of padding. If data type of elements is different than float it is converted
                           ///< to it using round-towards-nearest-even (for floating-point data types) or round-towards-zero (for integral
                           ///< data types).
    tensor _dynamic_pad_dims = tensor(0);

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
    struct Hasher {
        size_t operator()(const layout &l) const {
            return l.hash();
        }
    };

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
        : data_type(cldnn::data_types::undefined)
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
    ov::element::Type_t data_type;

    /// Format stored in @ref memory (see. @ref format)
    cldnn::format format;

    /// Explicit padding of the @ref memory
    padding data_padding;

    /// Number of bytes needed to store this layout
    size_t bytes_count() const { return (ov::element::Type(data_type).bitwidth() * get_linear_size() + 7) >> 3; }

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
        for (const auto& dim : size) {
            if (dim.get_max_length() == -1)
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

    static size_t max_rank() { return 8; }
    static ov::PartialShape transform(const ov::PartialShape& pshape, cldnn::format old_fmt, cldnn::format new_fmt);

    size_t hash() const {
        size_t seed = 0;
        seed = hash_combine(seed, data_padding.hash());
        seed = hash_combine(seed, format.value);
        seed = hash_combine(seed, data_type);

        auto pshape = get_partial_shape();
        for (size_t idx = 0; idx < pshape.size(); idx++) {
            auto v = pshape[idx].is_dynamic() ? -1 : pshape[idx].get_length();
            seed = hash_combine(seed, v);
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
