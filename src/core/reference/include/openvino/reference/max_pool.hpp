// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <numeric>

#include "openvino/reference/utils/coordinate_transform.hpp"

namespace ov {
namespace reference {
namespace {
void validate_max_pool_kernel_params(const size_t dims,
                                     const Shape& kernel,
                                     const Strides& kernel_strides,
                                     const Strides& kernel_dilations,
                                     const Shape& pads_begin,
                                     const Shape& pads_end) {
    OPENVINO_ASSERT(kernel.size() == dims && kernel_strides.size() == dims && kernel_dilations.size() == dims &&
                        pads_begin.size() == dims && pads_end.size() == dims,
                    "One of the MaxPool params does not match the ",
                    dims,
                    "D implementation.\nkernel=",
                    kernel,
                    "\nkernel_strides=",
                    kernel_strides,
                    "\nkernel_dilations=",
                    kernel_dilations,
                    "\npads_begin=",
                    pads_begin,
                    "\npads_end=",
                    pads_end);
}

/// \brief A helper struct representing spatial coordinates of a tensor element. It can use signed numbers as the
///        underlying type; this way it is possible to represent elements which belong to the padding area
///        (by using negative values).
///
/// \note  This struct can be used to represent a location of a pooling kernel in space (non-flattened version)
///        but at the same time it can represent pixel offsets in the filter itself (dilated or non-dilated)
template <typename T>
struct Coord : public std::vector<T> {
    Coord() = default;

    Coord(std::initializer_list<T>&& values) : std::vector<T>{std::move(values)} {}
};

inline bool elem_in_padding_area(const Coord<size_t>& kernel_position,
                                 const Coord<size_t>& kernel_offset,
                                 const Shape& data_shape) {
    for (size_t dim = 0; dim + 2 < data_shape.size(); ++dim) {
        if (static_cast<int64_t>(kernel_position[dim]) + static_cast<int64_t>(kernel_offset[dim]) < 0LL ||
            kernel_position[dim] + kernel_offset[dim] >= data_shape[dim + 2]) {
            return true;
        }
    }

    return false;
}

inline Coord<size_t> calculate_kernel_position(const Coord<size_t>& out_elem_coord,
                                               const Strides& kernel_strides,
                                               const Shape& pads_begin) {
    Coord<size_t> top_left_corner;
    top_left_corner.reserve(out_elem_coord.size());
    for (size_t i = 0u; i < out_elem_coord.size(); ++i) {
        top_left_corner.emplace_back(out_elem_coord[i] * kernel_strides[i] - pads_begin[i]);
    }
    return top_left_corner;
}

namespace kernel {
template <typename Values_t, typename Indices_t>
void max_pool_1d(const Values_t* data,
                 Values_t* values,
                 Indices_t* indices,
                 const size_t data_elems,
                 const size_t out_elems,
                 const size_t kernel_size,
                 const size_t kernel_stride,
                 const size_t kernel_dilation,
                 const size_t pads_begin,
                 const size_t pads_end,
                 const size_t indices_offset) {
    int kernel_position = 0 - static_cast<int>(pads_begin);
    // select max elem and its index for each "placeholder" in the out buffer (pointed to by out_idx)
    for (size_t out_idx = 0; out_idx < out_elems; ++out_idx) {
        Values_t max_elem = std::numeric_limits<Values_t>::lowest();
        Indices_t max_elem_idx = Indices_t{0};
        for (size_t kernel_elem = 0; kernel_elem < kernel_size; ++kernel_elem) {
            const size_t kernel_elem_offset = kernel_elem * kernel_dilation;
            // don't process the padding elements
            if (kernel_position + kernel_elem_offset >= 0 && kernel_position + kernel_elem_offset < data_elems &&
                data[kernel_position + kernel_elem_offset] > max_elem) {
                max_elem = data[kernel_position + kernel_elem_offset];
                max_elem_idx = static_cast<Indices_t>(kernel_position + kernel_elem_offset);
            }
        }
        values[out_idx] = max_elem;
        indices[out_idx] = static_cast<Indices_t>(max_elem_idx + indices_offset);
        kernel_position += static_cast<int>(kernel_stride);
    }
}

template <typename Values_t, typename Indices_t>
void max_pool_2d(const Values_t* data,
                 Values_t* values,
                 Indices_t* indices,
                 const Shape& data_shape,
                 const Shape& out_shape,
                 const Shape& kernel,
                 const Strides& kernel_strides,
                 const Strides& kernel_dilations,
                 const Shape& pads_begin,
                 const Shape& pads_end,
                 const size_t indices_offset) {
    validate_max_pool_kernel_params(2, kernel, kernel_strides, kernel_dilations, pads_begin, pads_end);

    // helper constants(axes) denoting dimensions in the input data shape and kernel shape
    constexpr size_t data_H = 2, data_W = 3;
    constexpr size_t kernel_H = 0, kernel_W = 1;

    // select max elem and its index for each "placeholder" in the out buffer (pointed to by out_idx)
    size_t out_idx = 0u;
    for (size_t out_row = 0u; out_row < out_shape[data_H]; ++out_row) {
        for (size_t out_col = 0u; out_col < out_shape[data_W]; ++out_col) {
            Values_t max_elem = std::numeric_limits<Values_t>::lowest();
            Indices_t max_elem_idx = Indices_t{0};

            const auto kernel_position = calculate_kernel_position({out_row, out_col}, kernel_strides, pads_begin);
            // find the max element in the area covered by a current position of the kernel
            for (size_t kernel_row = 0; kernel_row < kernel[kernel_H]; ++kernel_row) {
                for (size_t kernel_col = 0; kernel_col < kernel[kernel_W]; ++kernel_col) {
                    // offset from the top-left corner of the kernel for a given row and col
                    const Coord<size_t> kernel_offset{kernel_row * kernel_dilations[kernel_H],
                                                      kernel_col * kernel_dilations[kernel_W]};

                    // ignore the elements in the padding area
                    if (!elem_in_padding_area(kernel_position, kernel_offset, data_shape)) {
                        // index of the flattened tensor element under the current row & column of the kernel
                        const size_t data_elem_index =
                            data_shape[data_W] * (kernel_offset[kernel_H] + kernel_position[kernel_H]) +
                            kernel_offset[kernel_W] + kernel_position[kernel_W];

                        if (data[data_elem_index] > max_elem) {
                            max_elem = data[data_elem_index];
                            max_elem_idx = static_cast<Indices_t>(data_elem_index);
                        }
                    }
                }
            }

            values[out_idx] = max_elem;
            indices[out_idx] = static_cast<Indices_t>(max_elem_idx + indices_offset);
            ++out_idx;
        }
    }
}

template <typename Values_t, typename Indices_t>
void max_pool_3d(const Values_t* data,
                 Values_t* values,
                 Indices_t* indices,
                 const Shape& data_shape,
                 const Shape& out_shape,
                 const Shape& kernel,
                 const Strides& kernel_strides,
                 const Strides& kernel_dilations,
                 const Shape& pads_begin,
                 const Shape& pads_end,
                 const size_t indices_offset) {
    validate_max_pool_kernel_params(3, kernel, kernel_strides, kernel_dilations, pads_begin, pads_end);

    // helper constants(axes) denoting dimensions in the input data shape and kernel shape
    constexpr size_t data_D = 2, data_H = 3, data_W = 4;
    constexpr size_t kernel_D = 0, kernel_H = 1, kernel_W = 2;

    // select max elem and its index for each "placeholder" in the out buffer (pointed to by out_idx)
    size_t out_idx = 0u;
    for (size_t out_channel = 0u; out_channel < out_shape[data_D]; ++out_channel) {
        for (size_t out_row = 0u; out_row < out_shape[data_H]; ++out_row) {
            for (size_t out_col = 0u; out_col < out_shape[data_W]; ++out_col) {
                Values_t max_elem = std::numeric_limits<Values_t>::lowest();
                Indices_t max_elem_idx = Indices_t{0};

                const auto kernel_position =
                    calculate_kernel_position({out_channel, out_row, out_col}, kernel_strides, pads_begin);

                for (size_t kernel_channel = 0; kernel_channel < kernel[kernel_D]; ++kernel_channel) {
                    for (size_t kernel_row = 0; kernel_row < kernel[kernel_H]; ++kernel_row) {
                        for (size_t kernel_col = 0; kernel_col < kernel[kernel_W]; ++kernel_col) {
                            // offset from the top-left corner of the kernel for a given row and col
                            const Coord<size_t> kernel_offset{kernel_channel * kernel_dilations[kernel_D],
                                                              kernel_row * kernel_dilations[kernel_H],
                                                              kernel_col * kernel_dilations[kernel_W]};

                            // ignore the elements in the padding area
                            if (!elem_in_padding_area(kernel_position, kernel_offset, data_shape)) {
                                // index of the flattened tensor element under the current row & column of the kernel
                                const size_t data_elem_index =
                                    data_shape[data_H] * data_shape[data_W] *
                                        (kernel_offset[kernel_D] + kernel_position[kernel_D]) +
                                    data_shape[data_W] * (kernel_offset[kernel_H] + kernel_position[kernel_H]) +
                                    kernel_offset[kernel_W] + kernel_position[kernel_W];

                                if (data[data_elem_index] > max_elem) {
                                    max_elem = data[data_elem_index];
                                    max_elem_idx = static_cast<Indices_t>(data_elem_index);
                                }
                            }
                        }
                    }
                }
                values[out_idx] = max_elem;
                indices[out_idx] = static_cast<Indices_t>(max_elem_idx + indices_offset);
                ++out_idx;
            }
        }
    }
}
}  // namespace kernel
}  // namespace

template <typename Values_t, typename Indices_t>
void max_pool(const Values_t* data,
              Values_t* values,
              Indices_t* indices,
              const Shape& data_shape,
              const Shape& out_shape,
              const Shape& kernel,
              const Strides& strides,
              const Strides& dilations,
              const Shape& pads_begin,
              const Shape& pads_end,
              const int64_t axis = 0) {
    const auto data_batch_elems = shape_size(std::begin(data_shape) + 1, std::end(data_shape));
    const auto data_channel_elems = shape_size(std::begin(data_shape) + 2, std::end(data_shape));

    const auto out_batch_elems = shape_size(std::begin(out_shape) + 1, std::end(out_shape));
    const auto out_channel_elems = shape_size(std::begin(out_shape) + 2, std::end(out_shape));

    for (size_t b = 0; b < data_shape[0]; ++b) {
        const Indices_t batch_indices_offset = static_cast<Indices_t>(b * data_batch_elems);

        for (size_t c = 0; c < data_shape[1]; ++c) {
            // calculate the buffer offsets for a given channel "c" then execute an appropriate
            // kernel for each processed channel
            const Values_t* data_channel_first_elem = data + b * data_batch_elems + c * data_channel_elems;
            Values_t* out_channel_first_elem = values + b * out_batch_elems + c * out_channel_elems;
            Indices_t* indices_channel_first_elem = indices + b * out_batch_elems + c * out_channel_elems;
            const Indices_t channel_indices_offset = static_cast<Indices_t>(c * data_channel_elems);
            // total offset of the flattened tensor indices for currently processed batch and channel
            const Indices_t indices_offset = batch_indices_offset + channel_indices_offset;

            if (data_shape.size() == 3) {
                kernel::max_pool_1d<Values_t, Indices_t>(data_channel_first_elem,
                                                         out_channel_first_elem,
                                                         indices_channel_first_elem,
                                                         data_shape[2],
                                                         out_shape[2],
                                                         kernel[0],
                                                         strides[0],
                                                         dilations[0],
                                                         pads_begin[0],
                                                         pads_end[0],
                                                         indices_offset);
            } else if (data_shape.size() == 4) {
                kernel::max_pool_2d<Values_t, Indices_t>(data_channel_first_elem,
                                                         out_channel_first_elem,
                                                         indices_channel_first_elem,
                                                         data_shape,
                                                         out_shape,
                                                         kernel,
                                                         strides,
                                                         dilations,
                                                         pads_begin,
                                                         pads_end,
                                                         indices_offset);
            } else if (data_shape.size() == 5) {
                kernel::max_pool_3d<Values_t, Indices_t>(data_channel_first_elem,
                                                         out_channel_first_elem,
                                                         indices_channel_first_elem,
                                                         data_shape,
                                                         out_shape,
                                                         kernel,
                                                         strides,
                                                         dilations,
                                                         pads_begin,
                                                         pads_end,
                                                         indices_offset);
            } else {
                OPENVINO_THROW("Unsupported input shape ",
                               data_shape,
                               " passed to the MaxPool reference implementation. Supported shapes: 3D, 4D and 5D.");
            }
        }
    }

    // adjust the calculated indices to the requested range (specified by the axis attribute) if needed
    if (axis != 0) {
        const Indices_t max_index =
            static_cast<Indices_t>(shape_size(std::begin(data_shape) + axis, std::end(data_shape)));

        const auto indices_number = shape_size(out_shape);
        for (size_t i = 0; i < indices_number; ++i) {
            indices[i] %= max_index;
        }
    }
}

template <typename Value_t>
void max_pool(const Value_t* data,
              Value_t* values,
              const Shape& data_shape,
              const Shape& out_shape,
              const Shape& kernel,
              const Strides& strides,
              const Shape& pads_begin,
              const Shape& pads_end) {
    std::vector<int32_t> indices(shape_size(out_shape));
    const Strides dilations(kernel.size(), 1);
    max_pool<Value_t, int32_t>(data,
                               values,
                               indices.data(),
                               data_shape,
                               out_shape,
                               kernel,
                               strides,
                               dilations,
                               pads_begin,
                               pads_end);
}
}  // namespace reference
}  // namespace ov
