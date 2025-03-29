// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/pad.hpp"

#include <cassert>

#include "openvino/core/except.hpp"
#include "openvino/reference/utils/coordinate_index.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"

namespace ov {
namespace impl {
namespace {
template <typename T>
T clamp(T v, T lo, T hi) {
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}
struct PadBase {
    PadBase(const char* const data,
            const char* const pad_value,
            char* const out,
            const size_t elem_size,
            const Shape& data_shape,
            const Shape& out_shape,
            const CoordinateDiff& padding_begin,
            const CoordinateDiff& padding_end,
            const op::PadMode pad_mode)
        : data(data),
          pad_value(pad_value),
          out(out),
          elem_size(elem_size),
          data_shape(data_shape),
          out_shape(out_shape),
          padding_begin(padding_begin),
          padding_end(padding_end),
          pad_mode(pad_mode),
          coord(data_shape) {}

    virtual ~PadBase() = default;

    void run() const {
        check_inputs();

        CoordinateTransformBasic out_coordinate(out_shape);
        char* out_data = out;
        for (const auto& out_coord : out_coordinate) {
            const auto in_coord = transform_to_input_data_coord(out_coord);

            if (in_coord) {
                const auto in_index = coordinate_index(*in_coord, data_shape);
                const auto in_data = data + in_index * elem_size;
                std::copy(in_data, in_data + elem_size, out_data);
            } else {
                std::copy(pad_value, pad_value + elem_size, out_data);
            }
            out_data += elem_size;
        }
    }
    virtual const Coordinate* transform_to_input_data_coord(const Coordinate& out_coord) const = 0;

    virtual void check_inputs() const {}

    ///
    /// DATA
    ///
    const char* const data;
    const char* const pad_value;
    char* const out;
    const size_t elem_size;
    const Shape& data_shape;
    const Shape& out_shape;
    const CoordinateDiff& padding_begin;
    const CoordinateDiff& padding_end;
    const op::PadMode pad_mode;

    mutable Coordinate coord;
};

struct ConstPad : PadBase {
    using PadBase::PadBase;

    const Coordinate* transform_to_input_data_coord(const Coordinate& out_coord) const override {
        assert(out_coord.size() == coord.size());

        for (size_t i = 0; i != coord.size(); ++i) {
            const auto sc = static_cast<std::ptrdiff_t>(out_coord[i]);

            const auto cc = sc - padding_begin.at(i);
            if (0 <= cc && cc < static_cast<std::ptrdiff_t>(data_shape[i])) {
                coord[i] = cc;
            } else {
                return nullptr;
            }
        }
        return std::addressof(coord);
    }
};

struct EdgePad : PadBase {
    using PadBase::PadBase;

    const Coordinate* transform_to_input_data_coord(const Coordinate& out_coord) const override {
        assert(out_coord.size() == coord.size());

        for (size_t i = 0; i != coord.size(); ++i) {
            const auto sc = static_cast<std::ptrdiff_t>(out_coord[i]);

            const auto cc = sc - padding_begin.at(i);
            coord[i] = clamp<std::ptrdiff_t>(cc, 0, data_shape[i] - 1);
        }
        return std::addressof(coord);
    }
};

struct SymmetricAndReflectPad : PadBase {
    SymmetricAndReflectPad(const char* const data,
                           const char* const pad_value,
                           char* const out,
                           const size_t elem_size,
                           const Shape& data_shape,
                           const Shape& out_shape,
                           const CoordinateDiff& padding_begin,
                           const CoordinateDiff& padding_end,
                           const op::PadMode pad_mode)
        : PadBase(data, pad_value, out, elem_size, data_shape, out_shape, padding_begin, padding_end, pad_mode),
          axis_correction(pad_mode == op::PadMode::SYMMETRIC ? 1 : 0) {}

    const Coordinate* transform_to_input_data_coord(const Coordinate& out_coord) const override {
        assert(out_coord.size() == coord.size());

        for (size_t i = 0; i != coord.size(); ++i) {
            const auto shape_dim = static_cast<std::ptrdiff_t>(data_shape[i]);
            const auto sc = static_cast<std::ptrdiff_t>(out_coord[i]);

            const auto cc = sc - padding_begin.at(i);
            const auto rollfront_cc = cc >= 0 ? cc : -cc - axis_correction;
            const auto rollback_cc = shape_dim - (rollfront_cc + 2 - shape_dim) + axis_correction;
            coord[i] = rollfront_cc < shape_dim ? rollfront_cc : rollback_cc;
            assert(0 <= coord[i] && coord[i] < data_shape[i]);
        }
        return std::addressof(coord);
    }

    void check_inputs() const override {
        for (size_t i = 0; i != padding_begin.size(); ++i) {
            const auto axis_size = static_cast<std::ptrdiff_t>(data_shape[i]);
            OPENVINO_ASSERT(padding_begin.at(i) - axis_correction < axis_size,
                            "padding below should be less than data shape");
            OPENVINO_ASSERT(padding_end.at(i) - axis_correction < axis_size, "padding  should be less than data shape");
        }
    }

    int axis_correction{};
};

void pad(const char* data,
         const char* pad_value,
         char* out,
         const size_t elem_size,
         const Shape& data_shape,
         const Shape& out_shape,
         const CoordinateDiff& padding_below,
         const CoordinateDiff& padding_above,
         const op::PadMode pad_mode) {
    switch (pad_mode) {
    case op::PadMode::CONSTANT: {
        impl::ConstPad
            pad{data, pad_value, out, elem_size, data_shape, out_shape, padding_below, padding_above, pad_mode};
        pad.run();
    } break;
    case op::PadMode::EDGE: {
        impl::EdgePad
            pad{data, pad_value, out, elem_size, data_shape, out_shape, padding_below, padding_above, pad_mode};
        pad.run();
    } break;
    case op::PadMode::REFLECT:
    case op::PadMode::SYMMETRIC: {
        impl::SymmetricAndReflectPad
            pad{data, pad_value, out, elem_size, data_shape, out_shape, padding_below, padding_above, pad_mode};
        pad.run();
    } break;
    default:
        break;
    }
}
}  // namespace
}  // namespace impl

namespace reference {
void pad(const char* data,
         const char* pad_value,
         char* out,
         const size_t elem_size,
         const Shape& data_shape,
         const Shape& out_shape,
         const CoordinateDiff& padding_below,
         const CoordinateDiff& padding_above,
         const op::PadMode pad_mode) {
    impl::pad(data, pad_value, out, elem_size, data_shape, out_shape, padding_below, padding_above, pad_mode);
}
}  // namespace reference
}  // namespace ov
