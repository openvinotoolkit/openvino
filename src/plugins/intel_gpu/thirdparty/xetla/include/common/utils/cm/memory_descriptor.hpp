/*******************************************************************************
* Copyright (c) 2022-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/// @file
/// C++ API

#pragma once

#ifdef _WIN32
#include "../../../common/utils/cm/common.hpp"
#include "../../../common/utils/cm/raw_send_load_store.hpp"
#else
#include "common/utils/cm/common.hpp"
#include "common/utils/cm/raw_send_load_store.hpp"
#endif

namespace gpu::xetla {

template <int dim = 2>
struct mem_coord_t {};
template <>
struct mem_coord_t<2> {
    int x;
    int y;
    inline mem_coord_t(int x_, int y_) : x(x_), y(y_) {}
    inline mem_coord_t() = default;
    inline mem_coord_t(const mem_coord_t<2> &coord) {
        this->x = coord.x;
        this->y = coord.y;
    }
    inline mem_coord_t<2> &operator=(const mem_coord_t<2> &coord) {
        this->x = coord.x;
        this->y = coord.y;
        return *this;
    }
    inline void init(int x_, int y_) {
        this->x = x_;
        this->y = y_;
    }
};

template <>
struct mem_coord_t<4> {
    int x;
    int y;
    int z;
    int w;
};

template <int dim = 2>
struct mem_shape_t {};
template <>
struct mem_shape_t<2> {
    uint32_t x;
    uint32_t y;
    uint32_t stride_x;
    inline mem_shape_t() = default;
    inline mem_shape_t(
            uint32_t shape_x_, uint32_t shape_y_, uint32_t row_stride_)
        : x(shape_x_), y(shape_y_), stride_x(row_stride_) {}
    inline mem_shape_t(const mem_shape_t<2> &shape) {
        this->x = shape.x;
        this->y = shape.y;
        this->stride_x = shape.stride_x;
    }
    inline mem_shape_t<2> &operator=(const mem_shape_t<2> &shape) {
        this->x = shape.x;
        this->y = shape.y;
        this->stride_x = shape.stride_x;
        return *this;
    }
    inline void init(
            uint32_t shape_x_, uint32_t shape_y_, uint32_t row_stride_) {
        this->x = shape_x_;
        this->y = shape_y_;
        this->stride_x = row_stride_;
    }
};

template <>
struct mem_shape_t<4> {
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t z = 0;
    uint32_t w = 0;
    uint32_t stride_x = 0;
    uint32_t stride_y = 0;
    uint32_t stride_z = 0;

    inline mem_shape_t() = default;
    inline mem_shape_t(uint32_t size0_, uint32_t size1_, uint32_t size2_,
            uint32_t size3_, uint32_t stride0_, uint32_t stride1_,
            uint32_t stride2_)
        : x(size0_)
        , y(size1_)
        , z(size2_)
        , w(size3_)
        , stride_x(stride0_)
        , stride_y(stride1_)
        , stride_z(stride2_) {}
    inline mem_shape_t(
            uint32_t size0_, uint32_t size1_, uint32_t size2_, uint32_t size3_)
        : x(size0_)
        , y(size1_)
        , z(size2_)
        , w(size3_)
        , stride_x(size0_)
        , stride_y(size1_)
        , stride_z(size2_) {}
    // these methods are here because without them cm produces this error
    // "warning:  undefined value is referenced after decomposition"
    inline mem_shape_t(const mem_shape_t<4> &shape)
        : x(shape.x)
        , y(shape.y)
        , z(shape.z)
        , w(shape.w)
        , stride_x(shape.stride_x)
        , stride_y(shape.stride_y)
        , stride_z(shape.stride_z) {}
    inline mem_shape_t<4> &operator=(const mem_shape_t<4> &shape) {
        this->x = shape.x;
        this->y = shape.y;
        this->z = shape.z;
        this->w = shape.w;
        this->stride_x = shape.stride_x;
        this->stride_y = shape.stride_y;
        this->stride_z = shape.stride_z;
        return *this;
    }
};

template <typename dtype_, mem_space space_>
struct mem_base_t {};
template <typename dtype_>
struct mem_base_t<dtype_, mem_space::global> {
    using dtype = dtype_;
    dtype *base;
    inline mem_base_t() = default;
    inline mem_base_t(dtype *base_) : base(base_) {}
    inline mem_base_t(const mem_base_t<dtype, mem_space::global> &mem_base)
        : base(mem_base.base) {}
    inline mem_base_t<dtype, mem_space::global> &operator=(
            const mem_base_t<dtype, mem_space::global> &mem_base) {
        this->base = mem_base.base;
        return *this;
    }
    inline void init(dtype *base_) { base = base_; }
    inline void update(int offset) { base = base + offset; }
};
template <typename dtype_>
struct mem_base_t<dtype_, mem_space::local> {
    using dtype = dtype_;
    uint32_t base;
    inline mem_base_t() = default;
    inline mem_base_t(uint32_t base_) : base(base_) {}
    inline mem_base_t(const mem_base_t<dtype, mem_space::local> &mem_base)
        : base(mem_base.base) {}
    inline mem_base_t<dtype, mem_space::local> &operator=(
            const mem_base_t<dtype, mem_space::local> &mem_base) {
        this->base = mem_base.base;
        return *this;
    }
    inline void init(uint32_t base_) { base = base_; }
    inline void update(int offset) { base = base + offset * sizeof(dtype); }
};

template <typename dtype_, mem_layout layout_, mem_space space_,
        uint32_t alignment_ = 8, int dim_ = 2>
struct mem_desc_t {};

template <typename dtype_, mem_layout layout_, mem_space space_,
        uint32_t alignment_>
struct mem_desc_t<dtype_, layout_, space_, alignment_, 2> {
    using dtype = dtype_;
    static constexpr mem_layout layout = layout_;
    static constexpr mem_space space = space_;
    static constexpr int dim = 2;
    static constexpr uint32_t alignment = alignment_;
    static constexpr uint32_t alignment_in_bytes = alignment_ * sizeof(dtype);

    static constexpr bool is_col_major = layout == mem_layout::col_major;
    static constexpr bool is_local = space == mem_space::local;
    using shape_t = mem_shape_t<dim>;
    using coord_t = mem_coord_t<dim>;
    using base_t = mem_base_t<dtype, space>;

    using this_type_t = mem_desc_t<dtype, layout_, space_, alignment, 2>;

    inline mem_desc_t() = default;
    inline mem_desc_t(base_t base_, shape_t shape_, coord_t coord_)
        : base(base_), shape(shape_), coord(coord_) {}
    // Be aware of the risks: Rule of three (copy constructor, copy assignment, destructor)
    // Please check if you need to add self-define destructor
    // inline ~mem_desc_t(){}
    inline mem_desc_t(const this_type_t &mem_desc)
        : base(mem_desc.base), shape(mem_desc.shape), coord(mem_desc.coord) {}

    inline this_type_t &operator=(const this_type_t &mem_desc) {
        this->base = mem_desc.base;
        this->shape = mem_desc.shape;
        this->coord = mem_desc.coord;
        return *this;
    }
    inline void init(base_t base_, shape_t shape_, coord_t coord_) {
        base = base_;
        shape = shape_;
        coord = coord_;
    }
    inline void update_coord(int offset_x, int offset_y) {
        coord.x += offset_x;
        coord.y += offset_y;
    }

    inline void set_coord(int offset_x, int offset_y) {
        coord.x = offset_x;
        coord.y = offset_y;
    }

    inline void update_coord_x(int offset_x) { coord.x += offset_x; }
    inline void update_coord_y(int offset_y) { coord.y += offset_y; }
    inline xetla_tdescriptor get_tdesc() {
        uint32_t width = is_col_major ? shape.y : shape.x;
        uint32_t height = is_col_major ? shape.x : shape.y;
        uint32_t pitch = shape.stride_x;
        int coord_x = is_col_major ? coord.y : coord.x;
        int coord_y = is_col_major ? coord.x : coord.y;
        return xetla_get_tdesc<dtype>(
                base.base, width, height, pitch, coord_x, coord_y);
    }

    shape_t shape;
    coord_t coord;
    base_t base;
};

template <typename dtype_, mem_layout layout_, mem_space space_,
        uint32_t alignment_>
struct mem_desc_t<dtype_, layout_, space_, alignment_, 4> {
    using dtype = dtype_;
    static constexpr mem_layout layout = layout_;
    static constexpr mem_space space = space_;
    static constexpr uint32_t alignment = alignment_;
    static constexpr uint32_t alignment_in_bytes = alignment * sizeof(dtype);
    static constexpr int dim = 4;

    static constexpr bool is_local = (space == mem_space::local);
    using shape_t = mem_shape_t<dim>;
    using coord_t = mem_coord_t<dim>;
    using base_t = mem_base_t<dtype, space>;

    using this_type_t = mem_desc_t<dtype, layout_, space_, alignment, 4>;

    // without this constructor some compiler versions can incorrectly
    // initialize  the struct, probably due to some bug
    inline mem_desc_t(base_t base_, shape_t shape_, coord_t coord_)
        : base(base_), shape(shape_), coord(coord_) {};

    inline void update_coord(
            int offset_x, int offset_y, int offset_z, int offset_w) {
        coord.x += offset_x;
        coord.y += offset_y;
        coord.z += offset_z;
        coord.w += offset_w;
    }
    inline void update_coord_x(int offset_x) { coord.x += offset_x; }
    inline void update_coord_y(int offset_y) { coord.y += offset_y; }
    inline void update_coord_z(int offset_z) { coord.z += offset_z; }
    inline void update_coord_w(int offset_w) { coord.w += offset_w; }
    inline xetla_tdescriptor get_tdesc() {
        uint32_t width = shape.x; // in NHWC this is C
        uint32_t height = shape.y; // in NHWC this is W
        uint32_t pitch = shape.stride_x;
        int coord_x = coord.x; // in NHWC this is offset in channels
        int coord_y = coord.y; // in NHWC this is offset in width

        dtype *base_ptr = base.base;
        return xetla_get_tdesc<dtype>(
                base_ptr, width, height, pitch, coord_x, coord_y);
    }
    inline uint32_t get_shape_x(void) { return shape.x; }
    inline uint32_t get_shape_y(void) { return shape.y; }
    inline uint32_t get_shape_z(void) { return shape.z; }
    inline uint32_t get_shape_w(void) { return shape.w; }

    inline uint32_t get_stride_x(void) { return shape.stride_x; }
    inline uint32_t get_stride_y(void) { return shape.stride_y; }
    inline uint32_t get_stride_z(void) { return shape.stride_z; }

    inline int32_t get_coord_x(void) { return coord.x; }
    inline int32_t get_coord_y(void) { return coord.y; }
    inline int32_t get_coord_z(void) { return coord.z; }
    inline int32_t get_coord_w(void) { return coord.w; }

    template <int32_t N>
    inline xetla_vector<int32_t, N> get_base_offset_from_z(
            xetla_vector<int32_t, N> offset) {
        xetla_vector<int32_t, N> ret
                = (offset + coord.z) * (shape.stride_x * shape.stride_y);
        return ret;
    }

    inline int32_t get_base_offset_from_z(int32_t offset) {
        int32_t ret = (offset + coord.z) * shape.stride_x * shape.stride_y;
        return ret;
    }

    template <int32_t N>
    inline xetla_mask<N> get_mask_from_z(xetla_vector<int32_t, N> offset) {
        xetla_mask<N> mask
                = (offset + coord.z >= 0) & (offset + coord.z < shape.z);
        return mask;
    }

    inline int get_mask_from_z(int32_t offset) {
        int mask = (offset + coord.z >= 0) & (offset + coord.z < shape.z);
        return mask;
    }

    inline int32_t get_base_offset_from_w(int32_t offset) {
        int32_t ret = (offset + coord.w) * shape.stride_x * shape.stride_y
                * shape.stride_z;
        return ret;
    }

    template <int32_t N>
    inline xetla_vector<int32_t, N> get_base_offset_from_w(
            xetla_vector<int32_t, N> offset) {
        xetla_vector<int32_t, N> ret = (offset + coord.w) * shape.stride_x
                * shape.stride_y * shape.stride_z;
        return ret;
    }

    template <int32_t N>
    inline xetla_mask<N> get_mask_from_w(xetla_vector<int32_t, N> offset) {
        xetla_mask<N> mask
                = (offset + coord.w >= 0) & (offset + coord.w < shape.w);
        return mask;
    }

    inline int get_mask_from_w(int32_t offset) {
        int mask = (offset + coord.w >= 0) & (offset + coord.w < shape.w);
        return mask;
    }

    base_t base;
    shape_t shape;
    coord_t coord;
};

} // namespace gpu::xetla
