// Copyright (C) 2018-2024 Intel Corporation
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
#include <array>
#include <bitset>

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

constexpr size_t SHAPE_RANK_MAX = 9;

/// @brief Represents data padding information.
struct padding {
    using DynamicDimsMask = std::bitset<SHAPE_RANK_MAX>;
    static constexpr DynamicDimsMask EMPTY_MASK{0x0};

    std::array<int32_t, SHAPE_RANK_MAX> _lower_size = {0};  ///< Lower padding sizes. For spatials, it means size of left (X) and top (Y) padding.
    std::array<int32_t, SHAPE_RANK_MAX> _upper_size = {0};  ///< Upper padding sizes. For spatials, it means size of right (X) and bottom (Y) padding.
    DynamicDimsMask _dynamic_dims_mask = EMPTY_MASK;         ///< A mask saying which dimension has dynamic pad

    /// @brief
    /// @param lower_sizes Top-left padding sizes, in the same size and order as shape.
    /// @param upper_sizes Bottom-right padding sizes, in the same size and order as shape.
    padding(const std::vector<int32_t>& lower_sizes,
            const std::vector<int32_t>& upper_sizes,
            const DynamicDimsMask& dynamic_pad_dims = EMPTY_MASK) {
            // paddings
            OPENVINO_ASSERT(lower_sizes.size() <= SHAPE_RANK_MAX);
            OPENVINO_ASSERT(upper_sizes.size() <= SHAPE_RANK_MAX);
            std::copy_n(lower_sizes.begin(), lower_sizes.size(), _lower_size.begin());
            std::copy_n(upper_sizes.begin(), upper_sizes.size(), _upper_size.begin());
            _dynamic_dims_mask = dynamic_pad_dims;
          }

    /// @brief Constrcuts symmetric padding.
    /// @param sizes Top-left and bottom-right padding sizes, in the same size and order as shape.
    explicit padding(const std::vector<int32_t>& sizes,
                     const DynamicDimsMask& dynamic_pad_dims = EMPTY_MASK)
        : padding(sizes, sizes, dynamic_pad_dims) {}

    /// @brief Constructs "zero-sized" padding.
    padding() : padding({}, EMPTY_MASK) {}

    /// @brief Returns true if padding size is not zero.
    explicit operator bool() const {
        return std::any_of(_lower_size.begin(), _lower_size.end(), [](int32_t i){ return i > 0; }) ||
               std::any_of(_upper_size.begin(), _upper_size.end(), [](int32_t i){ return i > 0; });
    }

    bool is_dynamic() const {
        return _dynamic_dims_mask.any();
    }

    friend bool operator==(const padding& lhs, const padding& rhs) {
        return lhs._dynamic_dims_mask == rhs._dynamic_dims_mask &&
               lhs._lower_size == rhs._lower_size &&
               lhs._upper_size == rhs._upper_size;
    }

    friend bool operator!=(const padding& lhs, const padding& rhs) {
        return !(lhs == rhs);
    }

    friend bool operator<(const padding& lhs, const padding& rhs) {
        // Compare only actual padding size not _dynamic_dims_mask
        if (lhs._lower_size < rhs._lower_size) return true;
        else if (lhs._lower_size > rhs._lower_size) return false;
        if (lhs._upper_size < rhs._upper_size) return true;
        return false;
    }

    static padding max(padding const& lhs, padding const& rhs, float filling_value = 0.0f) {
        auto ret = lhs;
        for (size_t i = 0; i < SHAPE_RANK_MAX; ++i) {
            ret._lower_size[i] = std::max(ret._lower_size[i], rhs._lower_size[i]);
            ret._upper_size[i] = std::max(ret._upper_size[i], rhs._upper_size[i]);
        }
        ret._dynamic_dims_mask = ret._dynamic_dims_mask | rhs._dynamic_dims_mask;
        return ret;
    }

    size_t hash() const {
        size_t seed = 0;
        seed = hash_range(seed, std::begin(_lower_size), std::end(_lower_size));
        seed = hash_range(seed, std::begin(_upper_size), std::end(_upper_size));
        seed = cldnn::hash_combine(seed, _dynamic_dims_mask);
        return seed;
    }

    void save(BinaryOutputBuffer& ob) const {
        std::vector<int32_t> sizes;
        sizes.assign(_lower_size.begin(), _lower_size.end());
        ob << sizes;
        sizes.assign(_upper_size.begin(), _upper_size.end());
        ob << sizes;
        OPENVINO_ASSERT(sizes.size() == _dynamic_dims_mask.size(), "invalid size.");
        for (size_t i = 0; i < _dynamic_dims_mask.size(); i++)
            sizes[i] = static_cast<int32_t>(_dynamic_dims_mask[i]);
        ob << sizes;
    }

    void load(BinaryInputBuffer& ib) {
        std::vector<int32_t> sizes;
        ib >> sizes;
        std::copy_n(sizes.begin(), sizes.size(), _lower_size.begin());
        ib >> sizes;
        std::copy_n(sizes.begin(), sizes.size(), _upper_size.begin());
        ib >> sizes;
        OPENVINO_ASSERT(sizes.size() == _dynamic_dims_mask.size(), "invalid size.");
        for (size_t i = 0; i < _dynamic_dims_mask.size(); i++)
            _dynamic_dims_mask[i] = static_cast<bool>(sizes[i]);
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
        , size(size) {}

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
        data_padding = other.data_padding;
        size = other.size;
        return *this;
    }

    layout clone_with_other_shape(const ov::PartialShape& new_shape) const {
        return layout(new_shape, this->data_type, this->format, this->data_padding);
    }

    layout clone_with_other_shape(const ov::Shape& new_shape) const {
        return clone_with_other_shape(ov::PartialShape(new_shape));
    }


    friend bool operator==(const layout& lhs, const layout& rhs) {
        return lhs.data_type == rhs.data_type && lhs.format == rhs.format && lhs.size == rhs.size && lhs.data_padding == rhs.data_padding;
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

    /// Number of elements to be stored in this layout
    size_t count() const;

    std::vector<tensor::value_type> get_pitches() const;

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

    const ov::PartialShape& get_partial_shape() const;

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
    static ov::PartialShape transform(const ov::PartialShape& pshape, const cldnn::format& old_fmt, const cldnn::format& new_fmt);

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

        if (format == format::custom) {
            for (auto& bs : format.traits().block_sizes) {
                seed = hash_combine(seed, bs.first);
                seed = hash_combine(seed, bs.second);
            }
        }
        return seed;
    }

    /// @brief Returns a vector of tensors values, ordered regarding to @p format from the default format.
    /// @param _sizes an array that supports operator[] and stores data in the same order as shape.
    /// e.g. it could be std::vector, std::array, or std::bitset, etc.
    template <class TArray>
    inline static std::vector<int32_t> format_sizes(const TArray _sizes, const cldnn::format &fmt,
                                                    const int32_t default_val = 1) {
        const auto& output_order = fmt.order();
        std::vector<int32_t> sizes(output_order.size(), default_val);

        auto default_fmt = format::get_default_format(sizes.size(), format::is_weights_format(fmt), format::is_grouped(fmt));
        const auto& default_order = default_fmt.order();

        for (size_t i = 0; i < sizes.size(); ++i) {
            auto c = output_order[i];
            auto pos = default_order.find(c);
            OPENVINO_ASSERT(pos != std::string::npos, "[GPU] Unknown coord type: ", c);

            sizes[i] = static_cast<int32_t>(_sizes[pos]);
        }

        return sizes;
    }

private:
    /// The size of the @ref memory (excluding padding)
    ov::PartialShape size;
};

inline ::std::ostream& operator<<(::std::ostream& os, const layout& p) {
    return os << p.to_string();
}

inline ::std::ostream& operator<<(::std::ostream& os, const std::vector<layout>& layouts) {
    std::stringstream ss;

    ss << "[";
    for (size_t i = 0; i < layouts.size(); i++) {
        ss << layouts[i].to_short_string();

        if (i + 1 != layouts.size())
            ss << ", ";
    }
    ss << "]";

    return os << ss.str();
}

using optional_data_type = optional_value<data_types>;
using optional_layout = optional_value<layout>;

/// @}
/// @}
}  // namespace cldnn
