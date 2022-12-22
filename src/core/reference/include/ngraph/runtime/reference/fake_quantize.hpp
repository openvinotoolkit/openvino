// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

#include "ngraph/check.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/shape.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
namespace fake_quantize_details {
inline std::vector<size_t> calc_broadcast_index_offset(const std::vector<size_t>& memory_offsets,
                                                       const std::vector<size_t>& broadcast_shape) {
    std::vector<size_t> broadcast_offsets(broadcast_shape.size(), 0);
    for (int i = static_cast<int>(broadcast_shape.size()) - 2; i >= 0; --i) {
        if (broadcast_shape[i] == 1) {
            broadcast_offsets[i] = memory_offsets[i];
        }
    }
    const auto not_one = [](size_t i) {
        return i != 1;
    };
    if (std::any_of(broadcast_shape.begin(), broadcast_shape.end(), not_one) && broadcast_shape.back() == 1) {
        broadcast_offsets[broadcast_offsets.size() - 1] = 1;
    }
    if (broadcast_shape.back() == 1) {
        for (int i = static_cast<int>(broadcast_shape.size()) - 1; i >= 0; --i) {
            if (broadcast_shape[i] != 1) {
                broadcast_offsets[i] = memory_offsets[i] - 1;
                break;
            }
        }
    }
    return broadcast_offsets;
}

inline size_t calc_full_broadcast_offset(const std::vector<size_t>& current_dims, const std::vector<size_t>& offsets) {
    return std::inner_product(begin(current_dims), end(current_dims), begin(offsets), uint64_t(0));
}

inline Shape align_shape_sizes(const Shape& shape, const Shape& target_shape, const op::AutoBroadcastSpec& broadcast) {
    Shape s;
    switch (broadcast.m_type) {
    case op::AutoBroadcastType::NONE: {
        s = shape;
        break;
    }
    case op::AutoBroadcastType::NUMPY: {
        s = Shape(target_shape.size(), 1);
        std::copy(begin(shape), end(shape), prev(end(s), shape.size()));
        break;
    }
    case op::AutoBroadcastType::PDPD: {
        const size_t axis =
            broadcast.m_axis == -1 ? target_shape.size() - shape.size() : static_cast<size_t>(broadcast.m_axis);

        s = Shape(target_shape.size(), 1);
        const auto axis_to_copy = target_shape.size() - axis;
        const auto b = begin(shape);
        const auto e = b + axis_to_copy;  // from e to end(shape) should be only ones
        std::copy(b, e, next(begin(s), axis));
        break;
    }
    }
    return s;
}

inline void increment_current_dim(std::vector<size_t>& current_dims, const std::vector<size_t>& shape) {
    size_t incremented_dim_number = current_dims.size();
    while (incremented_dim_number-- > 0) {
        current_dims[incremented_dim_number] += 1;
        if (current_dims[incremented_dim_number] < shape[incremented_dim_number]) {
            break;
        }
        current_dims[incremented_dim_number] = 0;
    }
}

template <typename T>
class QuantizationBound {
public:
    enum class Bound {
        trivial,
        aligned,
        broadcast,
    };
    QuantizationBound(const T* const bound_data,
                      const Shape& bound_shape,
                      const Shape& arg_shape,
                      const op::AutoBroadcastSpec& broadcast_spec)
        : bounds(bound_data) {
        if (shape_size(bound_shape) == 1) {
            bound = Bound::trivial;
        } else if (bound_shape == arg_shape) {
            bound = Bound::aligned;
        } else {
            bound = Bound::broadcast;
            const auto arg_memory_offsets = row_major_strides(arg_shape);
            const auto unsqueezed_bound_shape = align_shape_sizes(bound_shape, arg_shape, broadcast_spec);
            row_strides = calc_broadcast_index_offset(arg_memory_offsets, unsqueezed_bound_shape);
        }
    }
    T get_value(const std::vector<size_t>& current_dim, size_t idx) const {
        T val{};
        switch (bound) {
        case Bound::trivial:
            val = *bounds;
            break;
        case Bound::aligned:
            val = bounds[idx];
            break;
        case Bound::broadcast: {
            const size_t index_offset = calc_full_broadcast_offset(current_dim, row_strides);
            NGRAPH_CHECK(0 <= index_offset && index_offset <= idx, "Incorrect index offset value!");
            val = bounds[idx - index_offset];
            break;
        }
        }
        return val;
    }

private:
    Bound bound;
    std::vector<size_t> row_strides;
    const T* const bounds;
};

template <typename T>
inline T quantize(const T& arg,
                  const T& in_low,
                  const T& in_high,
                  const T& out_low,
                  const T& out_high,
                  const size_t& levels) {
    if (arg <= std::min(in_low, in_high)) {
        return out_low;
    } else if (arg > std::max(in_low, in_high)) {
        return out_high;
    }
    return static_cast<T>(std::nearbyint((arg - in_low) / (in_high - in_low) * (levels - 1)) / (levels - 1) *
                              (out_high - out_low) +
                          out_low);
}

}  // namespace fake_quantize_details

template <typename T>
void fake_quantize(const T* const arg,
                   const T* const in_low,
                   const T* const in_high,
                   const T* const out_low,
                   const T* const out_high,
                   T* const out,
                   const Shape& arg_shape,
                   const Shape& in_low_shape,
                   const Shape& in_high_shape,
                   const Shape& out_low_shape,
                   const Shape& out_high_shape,
                   size_t levels,
                   const op::AutoBroadcastSpec& broadcast) {
    using namespace fake_quantize_details;

    if (shape_size(in_low_shape) == 1 && shape_size(in_high_shape) == 1 && shape_size(out_low_shape) == 1 &&
        shape_size(out_high_shape) == 1) {
        const size_t arg_size = shape_size(arg_shape);
        const auto q = [=](const T& a) {
            return quantize(a, *in_low, *in_high, *out_low, *out_high, levels);
        };
        for (size_t i = 0; i < arg_size; ++i) {
            out[i] = q(arg[i]);
        }
    } else {
        NGRAPH_CHECK(in_low_shape.size() <= arg_shape.size() && in_high_shape.size() <= arg_shape.size() &&
                         out_low_shape.size() <= arg_shape.size() && out_high_shape.size() <= arg_shape.size(),
                     "Tensors with input\\output ranges should have rank less or "
                     "equal to data tensor rank equal to ",
                     arg_shape.size());

        const QuantizationBound<T> in_low_bound(in_low, in_low_shape, arg_shape, broadcast);
        const QuantizationBound<T> in_high_bound(in_high, in_high_shape, arg_shape, broadcast);
        const QuantizationBound<T> out_low_bound(out_low, out_low_shape, arg_shape, broadcast);
        const QuantizationBound<T> out_high_bound(out_high, out_high_shape, arg_shape, broadcast);

        std::vector<size_t> current_dim(arg_shape.size(), 0);
        const auto arg_shape_size = shape_size(arg_shape);
        for (size_t index = 0; index < arg_shape_size; ++index) {
            const T in_low_val = in_low_bound.get_value(current_dim, index);
            const T in_high_val = in_high_bound.get_value(current_dim, index);
            const T out_low_val = out_low_bound.get_value(current_dim, index);
            const T out_high_val = out_high_bound.get_value(current_dim, index);

            out[index] = quantize(arg[index], in_low_val, in_high_val, out_low_val, out_high_val, levels);
            increment_current_dim(current_dim, arg_shape);
        }
    }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
