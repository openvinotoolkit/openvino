// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cfenv>
#include <cstring>
#include <iterator>

#include "openvino/core/except.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/reference/utils/coordinate_index.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"

namespace ov {
namespace reference {
template <typename T>
size_t normalize_index(const T idx, const size_t dim_value) {
    if (idx < 0) {
        return static_cast<size_t>(idx + dim_value);
    } else {
        return static_cast<size_t>(idx);
    }
}

namespace {
void scatter_elem_update_no_reduction(const size_t data_elem_size,
                                      const int64_t* indices,
                                      const char* updates,
                                      const int64_t axis,
                                      char* out_buf,
                                      const Shape& data_shape,
                                      const Shape& indices_shape,
                                      const ov::op::v12::ScatterElementsUpdate::Reduction reduction_type,
                                      const bool use_init_val) {
    // 3D example
    // output[indices[i][j][k]][j][k] = updates[i][j][k] if axis = 0,
    // output[i][indices[i][j][k]][k] = updates[i][j][k] if axis = 1,
    // output[i][j][indices[i][j][k]] = updates[i][j][k] if axis = 2

    CoordinateTransformBasic indices_transform{indices_shape};
    CoordinateTransformBasic data_transform{data_shape};
    const auto indices_strides = row_major_strides(indices_shape);
    const auto data_strides = row_major_strides(data_shape);

    for (const Coordinate& indices_cord : indices_transform) {
        const size_t indices_idx =
            std::inner_product(indices_cord.begin(), indices_cord.end(), indices_strides.begin(), uint64_t(0));
        Coordinate out_cord(indices_cord);
        out_cord.at(axis) = normalize_index(indices[indices_idx], data_shape[axis]);
        const size_t out_idx = ov::coordinate_offset(out_cord, data_strides);
        std::memcpy(out_buf + out_idx * data_elem_size, updates + indices_idx * data_elem_size, data_elem_size);
    }
}
}  // namespace

template <typename T>
T reduction_neutral_value(const ov::op::v12::ScatterElementsUpdate::Reduction reduction_type) {
    switch (reduction_type) {
    case ov::op::v12::ScatterElementsUpdate::Reduction::MAX:
        return std::numeric_limits<T>::lowest();
    case ov::op::v12::ScatterElementsUpdate::Reduction::MIN:
        return std::numeric_limits<T>::max();
    case ov::op::v12::ScatterElementsUpdate::Reduction::PROD:
        return T{1};
    case ov::op::v12::ScatterElementsUpdate::Reduction::SUM:
    case ov::op::v12::ScatterElementsUpdate::Reduction::MEAN:
        return T{0};
    default:
        OPENVINO_THROW("Neutral value not available for this type of reduction");
    }
}

template <typename T>
std::function<T(const T, const T)> reduction_functor_for(
    const ov::op::v12::ScatterElementsUpdate::Reduction reduction_type) {
    switch (reduction_type) {
    case ov::op::v12::ScatterElementsUpdate::Reduction::MAX:
        return [](const T a, const T b) {
            return a > b ? a : b;
        };
    case ov::op::v12::ScatterElementsUpdate::Reduction::MIN:
        return [](const T a, const T b) {
            return a < b ? a : b;
        };
    case ov::op::v12::ScatterElementsUpdate::Reduction::PROD:
        return std::multiplies<T>{};
    case ov::op::v12::ScatterElementsUpdate::Reduction::SUM:
    case ov::op::v12::ScatterElementsUpdate::Reduction::MEAN:
        return std::plus<T>{};
    default:
        OPENVINO_THROW("No functor available for this type of reduction");
    }
}

template <>
inline std::function<char(const char, const char)> reduction_functor_for<char>(
    const ov::op::v12::ScatterElementsUpdate::Reduction reduction_type) {
    switch (reduction_type) {
    case ov::op::v12::ScatterElementsUpdate::Reduction::MAX:
        return [](const char a, const char b) {
            return a > b ? a : b;
        };
    case ov::op::v12::ScatterElementsUpdate::Reduction::MIN:
        return [](const char a, const char b) {
            return a < b ? a : b;
        };
    case ov::op::v12::ScatterElementsUpdate::Reduction::PROD:
        return [](const char a, const char b) {
            return static_cast<bool>(a) && static_cast<bool>(b);
        };
    case ov::op::v12::ScatterElementsUpdate::Reduction::SUM:
        return [](const char a, const char b) {
            return static_cast<bool>(a) || static_cast<bool>(b);
        };
    default:
        OPENVINO_THROW("No functor available for this type of reduction");
    }
}

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value || std::is_class<T>::value, T>::type arithmetic_mean(
    const T accumulator,
    const int32_t N) {
    return accumulator / N;
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type arithmetic_mean(const T accumulator, const int32_t N) {
    const T value = static_cast<T>(std::nearbyint(static_cast<double>(accumulator) / N));
    return value;
}

template <typename T>
struct RoundingDirectionGuard {
    RoundingDirectionGuard() {
        if (std::is_integral<T>::value) {
            m_original_mode = std::fegetround();
            std::fesetround(FE_DOWNWARD);
        }
    }

    ~RoundingDirectionGuard() {
        if (std::is_integral<T>::value) {
            std::fesetround(m_original_mode);
        }
    }

private:
    decltype(std::fegetround()) m_original_mode;
};

template <typename DataType>
void scatter_elem_update_with_reduction(const int64_t* indices,
                                        const DataType* updates,
                                        const int64_t axis,
                                        DataType* out_buf,
                                        const Shape& data_shape,
                                        const Shape& indices_shape,
                                        const ov::op::v12::ScatterElementsUpdate::Reduction reduction_type,
                                        const bool use_init_val) {
    CoordinateTransformBasic indices_transform{indices_shape};
    CoordinateTransformBasic data_transform{data_shape};
    const auto indices_strides = row_major_strides(indices_shape);
    const auto data_strides = row_major_strides(data_shape);

    struct Offsets {
        size_t idx_offset;
        size_t out_offset;
    };

    std::vector<Offsets> idx_to_output_element;
    idx_to_output_element.reserve(shape_size(indices_shape));
    for (const Coordinate& indices_cord : indices_transform) {
        const size_t indices_offset =
            std::inner_product(indices_cord.begin(), indices_cord.end(), indices_strides.begin(), uint64_t(0));
        Coordinate out_cord(indices_cord);
        out_cord.at(axis) = normalize_index(indices[indices_offset], data_shape[axis]);
        const size_t out_offset =
            std::inner_product(out_cord.begin(), out_cord.end(), data_strides.begin(), uint64_t(0));

        idx_to_output_element.push_back({indices_offset, out_offset});
    }

    // When this is false we need to substitute the copied values at target locations with values that will not affect
    // the particular reduction algorithms. Effectively what happens here is setting the initial value
    // for the reduction accumulators.
    if (!use_init_val) {
        const auto value = reduction_neutral_value<DataType>(reduction_type);
        for (const auto& offsets : idx_to_output_element) {
            out_buf[offsets.out_offset] = value;
        }
    }

    // keeps the count of numbers included in the initial sums accumulated in the output tensor (reduction: MEAN)
    // the values in this map will later be used to divide the sums and calculate the final means
    // the key is the output tensor's element index and the value is the count
    std::unordered_map<size_t, int32_t> mean_reduction_counters;

    const auto reduce = reduction_functor_for<DataType>(reduction_type);
    for (const auto& offsets : idx_to_output_element) {
        out_buf[offsets.out_offset] = reduce(out_buf[offsets.out_offset], updates[offsets.idx_offset]);
        if (reduction_type == ov::op::v12::ScatterElementsUpdate::Reduction::MEAN) {
            mean_reduction_counters[offsets.out_offset] += 1;
        }
    }

    if (reduction_type == ov::op::v12::ScatterElementsUpdate::Reduction::MEAN) {
        // this object will change the rounding mode only for integer types which is required to match torch
        // upon destruction the previously used rounding mode will be restored
        RoundingDirectionGuard<DataType> rounding_guard;
        for (const auto& counter : mean_reduction_counters) {
            // include the initial value in the arithmetic mean divisor (if needed)
            const auto N = counter.second + static_cast<int32_t>(use_init_val);
            out_buf[counter.first] = arithmetic_mean<DataType>(out_buf[counter.first], N);
        }
    }
}

template <typename InType, typename OutType>
const OutType* convert_indices(const InType* indices, const size_t indices_count, std::vector<OutType>& buffer) {
    if (std::is_same<typename std::decay<InType>::type, OutType>::value)
        return reinterpret_cast<const OutType*>(indices);

    buffer.resize(indices_count);
    for (auto i = indices_count; i-- > 0;)
        buffer[i] = indices[i];
    return buffer.data();
}

template <typename DataType, typename IndicesType>
void scatter_elem_update(const DataType* input_data,
                         const IndicesType* indices,
                         const DataType* updates,
                         const int64_t axis,
                         DataType* out_buf,
                         const Shape& data_shape,
                         const Shape& indices_shape,
                         const ov::op::v12::ScatterElementsUpdate::Reduction reduction_type =
                             ov::op::v12::ScatterElementsUpdate::Reduction::NONE,
                         const bool use_init_val = true) {
    std::memcpy(out_buf, input_data, sizeof(DataType) * shape_size(data_shape));

    std::vector<int64_t> buffer;
    const auto indices_i64 = convert_indices(indices, shape_size(indices_shape), buffer);

    if (reduction_type != ov::op::v12::ScatterElementsUpdate::Reduction::NONE) {
        scatter_elem_update_with_reduction(indices_i64,
                                           updates,
                                           axis,
                                           out_buf,
                                           data_shape,
                                           indices_shape,
                                           reduction_type,
                                           use_init_val);
    } else {
        scatter_elem_update_no_reduction(sizeof(DataType),
                                         indices_i64,
                                         reinterpret_cast<const char*>(updates),
                                         axis,
                                         reinterpret_cast<char*>(out_buf),
                                         data_shape,
                                         indices_shape,
                                         reduction_type,
                                         use_init_val);
    }
}
}  // namespace reference
}  // namespace ov
