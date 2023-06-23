// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cfenv>
#include <cstring>
#include <iterator>

#include "ngraph/check.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape.hpp"
#include "openvino/op/scatter_elements_update.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
using Reduction = ov::op::v12::ScatterElementsUpdate::Reduction;

template <typename DataType, typename IndicesType>
void scatter_elem_update_with_reduction(const DataType* input_data,
                                        const IndicesType* indices,
                                        const DataType* updates,
                                        const int64_t& axis,
                                        DataType* out_buf,
                                        const Shape& data_shape,
                                        const Shape& indices_shape,
                                        const ov::op::v12::ScatterElementsUpdate::Reduction reduction_type,
                                        const bool use_init_val);

template <typename DataType, typename IndicesType>
void scatter_elem_update(const DataType* input_data,
                         const IndicesType* indices,
                         const DataType* updates,
                         const int64_t& axis,
                         DataType* out_buf,
                         const Shape& data_shape,
                         const Shape& indices_shape,
                         const Reduction reduction_type = Reduction::NONE,
                         const bool use_init_val = true) {
    // Copy inputs to out
    std::memcpy(out_buf, input_data, sizeof(DataType) * shape_size(data_shape));

    if (reduction_type != Reduction::NONE) {
        scatter_elem_update_with_reduction(input_data,
                                           indices,
                                           updates,
                                           axis,
                                           out_buf,
                                           data_shape,
                                           indices_shape,
                                           reduction_type,
                                           use_init_val);
        return;
    }

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
        out_cord.at(axis) = indices[indices_idx];
        const auto out_idx = std::inner_product(out_cord.begin(), out_cord.end(), data_strides.begin(), uint64_t(0));
        out_buf[out_idx] = updates[indices_idx];
    }
}

template <typename T>
T reduction_neutral_value(const Reduction reduction_type) {
    switch (reduction_type) {
    case Reduction::MAX:
        return std::numeric_limits<T>::min();
    case Reduction::MIN:
        return std::numeric_limits<T>::max();
    case Reduction::PROD:
        return T{1};
    case Reduction::SUM:
    case Reduction::MEAN:
        return T{0};
    default:
        OPENVINO_ASSERT(false, "Neutral value not available for this type of reduction");
        return 0;
    }
}

//todo: ^specialize for bool? min and max wont work like this

template <typename T>
std::function<T(const T, const T)> reduction_functor_for(const Reduction reduction_type) {
    switch (reduction_type) {
    // case Reduction::MAX:
    //     return std::numeric_limits<T>::min();
    // case Reduction::MIN:
    //     return std::numeric_limits<T>::max();
    case Reduction::PROD:
        return std::multiplies<T>{};
    case Reduction::SUM:
    case Reduction::MEAN:
        return std::plus<T>{};
    default:
        OPENVINO_ASSERT(false, "No functor available for this type of reduction");
        return 0;
    }
}

template<typename T>
T arithmetic_mean_int(const T accumulator, const int32_t N) {
    const auto old_mode = std::fegetround();
    std::fesetround(FE_DOWNWARD);
    const T value = static_cast<T>(std::nearbyint(static_cast<double>(accumulator) / N));
    std::fesetround(old_mode);
    return value;
}

template<typename T>
T arithmetic_mean(const T accumulator, const int32_t N) {
    const auto et = element::from<T>();
    if (et.is_integral_number()) {
        return arithmetic_mean_int(accumulator, N);
    } else {
        return accumulator / N;
    }
}

template <typename DataType, typename IndicesType>
void scatter_elem_update_with_reduction(const DataType* input_data,
                                        const IndicesType* indices,
                                        const DataType* updates,
                                        const int64_t& axis,
                                        DataType* out_buf,
                                        const Shape& data_shape,
                                        const Shape& indices_shape,
                                        const Reduction reduction_type,
                                        const bool use_init_val) {
    CoordinateTransformBasic indices_transform{indices_shape};
    CoordinateTransformBasic data_transform{data_shape};
    const auto indices_strides = row_major_strides(indices_shape);
    const auto data_strides = row_major_strides(data_shape);

    std::map<size_t, size_t> idx_to_output;  // todo: change to vector if no multiple lookups?
    for (const Coordinate& indices_cord : indices_transform) {
        const size_t indices_idx =
            std::inner_product(indices_cord.begin(), indices_cord.end(), indices_strides.begin(), uint64_t(0));
        Coordinate out_cord(indices_cord);
        out_cord.at(axis) = indices[indices_idx];
        const auto out_idx = std::inner_product(out_cord.begin(), out_cord.end(), data_strides.begin(), uint64_t(0));

        idx_to_output[indices_idx] = out_idx;
    }

    // When this is false we need to substitute the copied values at target locations with values that will not affect
    // the particular reduction algorithms. Effectively what happens here is setting the initial value
    // for the reduction accumulators.
    if (!use_init_val) {
        const auto value = reduction_neutral_value<DataType>(reduction_type);
        for (const auto& offsets : idx_to_output) {
            out_buf[offsets.second] = value;
        }
    }

    // keeps the count of numbers included in the initial sums accumulated in the output tensor (reduction: MEAN)
    // the values in this map will later be used to divide the sums and calculate the final means
    // the key is the output tensor's element index and the value is the count
    std::unordered_map<size_t, int32_t> mean_reduction_counters;

    const auto reduce = reduction_functor_for<DataType>(reduction_type);
    for (const auto& offsets : idx_to_output) {
        out_buf[offsets.second] = reduce(out_buf[offsets.second], updates[offsets.first]);
        if (reduction_type == Reduction::MEAN) {
            mean_reduction_counters[offsets.second] += 1;
        }
    }

    if (reduction_type == Reduction::MEAN) {
        for (const auto& counter : mean_reduction_counters) {
            // include the initial value in the arithmetic mean divisor (if needed)
            const auto N = counter.second + static_cast<int32_t>(use_init_val);
            out_buf[counter.first] = arithmetic_mean<DataType>(out_buf[counter.first], N);
        }
    }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
