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

#include "subgroup/tile/api.hpp"
#include "subgroup/tile/impl/op_function.hpp"

namespace gpu::xetla::subgroup {

/// @brief Is to describe the global memory surface for block-2d load/store
/// for each block in one tile, a payload message is prepared here.
/// in tile_load case, memory transpose, register transpose, memory transform
/// and dword transpose can be enable.
/// While in tile store case, we only support row major store, no register
/// operations can be applied.
/// @tparam dtype Is the data type
/// @tparam tile_desc_ Is the tile descriptor
/// @tparam mem_layout_ Is the memory layout
template <typename dtype_, typename tile_desc_, mem_layout mem_layout_,
        gpu_arch arch_tag_, uint32_t alignment_, uint32_t dim_>
struct mem_payload_t<
        mem_desc_t<dtype_, mem_layout_, mem_space::global, alignment_, dim_>,
        tile_desc_, msg_type::block_2d, arch_tag_,
        std::enable_if_t<(arch_tag_ == gpu_arch::Xe)>> {
    using tile_desc = tile_desc_;
    using mem_desc_t = mem_desc_t<dtype_, mem_layout_, mem_space::global,
            alignment_, dim_>;
    using dtype = dtype_;
    static constexpr msg_type message_type = msg_type::block_2d;
    static constexpr mem_space memory_space = mem_space::global;
    static constexpr mem_layout memory_layout = mem_layout_;
    static constexpr gpu_arch arch_tag = arch_tag_;

private:
    static constexpr uint32_t block_size_x = tile_desc::block_size_x;
    static constexpr uint32_t block_size_y = tile_desc::block_size_y;
    static constexpr uint32_t num_block_x = tile_desc::num_block_x;
    static constexpr uint32_t num_block_y = tile_desc::num_block_y;
    static constexpr uint32_t num_block = tile_desc::num_block;
    static constexpr uint32_t remained_size_y = tile_desc::remained_size_y;
    using this_payload_t = mem_payload_t<mem_desc_t, tile_desc,
            msg_type::block_2d, arch_tag>;

public:
    static constexpr bool mem_transpose
            = memory_layout == mem_layout::col_major;

    static constexpr reg_layout register_layout = tile_desc::register_layout;
    static constexpr bool reg_transpose
            = register_layout == reg_layout::transpose_tiled;
    static constexpr bool trans = mem_transpose ^ reg_transpose;

    static constexpr bool mem_transform = (sizeof(dtype) < 4) && !mem_transpose
            && (register_layout == reg_layout::vnni_tiled
                    || register_layout == reg_layout::vnni_tiled_col_major);
    static constexpr bool mem_dword_transpose = (sizeof(dtype) < 4) && trans;

    using mem_dtype = typename std::conditional<mem_dword_transpose, uint32_t,
            dtype>::type;
    static constexpr uint32_t scale_factor = sizeof(mem_dtype) / sizeof(dtype);

    xetla_vector<uint32_t, 16 * num_block> payloads;

    inline mem_payload_t(const this_payload_t &rhs) {
        this->payloads = rhs.payloads;
    }

    inline mem_payload_t(mem_desc_t &mem_desc) {
        xetla_tdescriptor base_tdesc = mem_desc.get_tdesc();
        int32_t offset
                = gpu::xetla::detail::xetla_get_tensor_offset_x(base_tdesc)
                / int32_t(scale_factor);
        gpu::xetla::detail::xetla_set_tensor_offset_x(
                base_tdesc.xetla_format<uint32_t>(), offset);
        prepare_tdesc(base_tdesc);
    }

    inline mem_payload_t(dtype *p, uint32_t surface_width,
            uint32_t surface_height, uint32_t surface_pitch,
            int32_t surface_offset_x = 0, int32_t surface_offset_y = 0) {
        xetla_tdescriptor base_tdesc;
        xetla_fill_tdesc(base_tdesc.xetla_format<uint32_t>(), p, surface_width,
                surface_height, surface_pitch,
                surface_offset_x / int32_t(scale_factor), surface_offset_y);
        prepare_tdesc(base_tdesc);
    }

    __XETLA_API void init(mem_desc_t &mem_desc) {
        xetla_tdescriptor base_tdesc = mem_desc.get_tdesc();
        int32_t offset
                = gpu::xetla::detail::xetla_get_tensor_offset_x(base_tdesc)
                / int32_t(scale_factor);
        gpu::xetla::detail::xetla_set_tensor_offset_x(
                base_tdesc.xetla_format<uint32_t>(), offset);
        prepare_tdesc(base_tdesc);
    }

    __XETLA_API void init(xetla_tdescriptor base_tdesc) {
        int32_t offset
                = gpu::xetla::detail::xetla_get_tensor_offset_x(base_tdesc)
                / int32_t(scale_factor);
        gpu::xetla::detail::xetla_set_tensor_offset_x(
                base_tdesc.xetla_format<uint32_t>(), offset);
        prepare_tdesc(base_tdesc);
    }

    __XETLA_API void init(dtype *p, uint32_t surface_width,
            uint32_t surface_height, uint32_t surface_pitch,
            int32_t surface_offset_x = 0, int32_t surface_offset_y = 0) {
        xetla_tdescriptor base_tdesc;
        xetla_fill_tdesc(base_tdesc.xetla_format<uint32_t>(), p, surface_width,
                surface_height, surface_pitch,
                surface_offset_x / int32_t(scale_factor), surface_offset_y);
        prepare_tdesc(base_tdesc);
    }

    inline mem_payload_t() = default;
    // Be aware of the risks: Rule of three (copy constructor, copy assignment, destructor)
    // Please check if you need to add self-define destructor
    // ~mem_payload_t(){}

    inline this_payload_t &operator=(const this_payload_t &rhs) {
        this->payloads = rhs.payloads;
        return *this;
    }

    template <tdesc_update_dir update_dir = tdesc_update_dir::x_dir>
    __XETLA_API void update_tdesc(int offset) {
        auto payloads_2d = payloads.xetla_format<uint32_t, num_block, 16>();
        if constexpr (update_dir == tdesc_update_dir::x_dir) {
#pragma unroll
            for (int i = 0; i < num_block; i++) {
                xetla_update_tdesc_offsetx(
                        payloads_2d.row(i), offset / int32_t(scale_factor));
            }
        } else {
#pragma unroll
            for (int i = 0; i < num_block; i++) {
                xetla_update_tdesc_offsety(payloads_2d.row(i), offset);
            }
        }
    }

    __XETLA_API void update_tdesc_base_address(int offset) {
        auto payloads_2d = payloads.xetla_format<uint32_t, num_block, 16>();
#pragma unroll
        for (int i = 0; i < num_block; i++) {
            xetla_update_tdesc_base_address(payloads_2d.row(i), offset);
        }
    }

    __XETLA_API void set_tdesc_width(uint32_t size) {
        auto payloads_2d = payloads.xetla_format<uint32_t, num_block, 16>();
#pragma unroll
        for (int i = 0; i < num_block; i++) {
            xetla_set_tdesc_width<dtype>(payloads_2d.row(i), size);
        }
    }

    __XETLA_API void set_tdesc_pitch(uint32_t size) {
        auto payloads_2d = payloads.xetla_format<uint32_t, num_block, 16>();
#pragma unroll
        for (int i = 0; i < num_block; i++) {
            xetla_set_tdesc_pitch<dtype>(payloads_2d.row(i), size);
        }
    }

    __XETLA_API void set_tdesc_height(uint32_t size) {
        auto payloads_2d = payloads.xetla_format<uint32_t, num_block, 16>();
#pragma unroll
        for (int i = 0; i < num_block; i++) {
            xetla_set_tdesc_height(payloads_2d.row(i), size);
        }
    }

    __XETLA_API void update_tdesc_base_address_masked(
            int offset, uint16_t mask = 1) {
        auto payloads_2d = payloads.xetla_format<uint32_t, num_block, 16>();
#pragma unroll
        for (int i = 0; i < num_block; i++) {
            xetla_update_tdesc_base_address(payloads_2d.row(i), offset);
        }

#pragma unroll
        for (int i = 0; i < num_block; i++) {
            xetla_tdesc_mask_op(payloads_2d.row(i), mask);
        }
    }
    __XETLA_API void set_tdesc_base_address_masked(
            uint64_t offset, uint16_t mask = 1) {
        auto payloads_2d = payloads.xetla_format<uint32_t, num_block, 16>();
#pragma unroll
        for (int i = 0; i < num_block; i++) {
            gpu::xetla::detail::xetla_set_tensor_base_address(
                    payloads_2d.row(i), offset);
        }

#pragma unroll
        for (int i = 0; i < num_block; i++) {
            xetla_tdesc_mask_op(payloads_2d.row(i), mask);
        }
    }

    __XETLA_API void set_tdesc_base_address(uint64_t addr) {
        auto payloads_2d = payloads.xetla_format<uint32_t, num_block, 16>();
#pragma unroll
        for (int i = 0; i < num_block; i++) {
            gpu::xetla::detail::xetla_set_tensor_base_address(
                    payloads_2d.row(i), addr);
        }
    }

    __XETLA_API void set_offset(int32_t offset_x, int32_t offset_y) {
        auto payloads_2d = payloads.xetla_format<uint32_t, num_block, 16>();
        constexpr uint32_t arr_len = 1;
        int32_t base_offset_y = offset_y;
#pragma unroll
        for (int i = 0; i < num_block_y; i++) {
            auto tdesc_row_2d = payloads_2d.xetla_select<num_block_x, 1, 16, 1>(
                    i * num_block_x, 0);

            int32_t base_offset_x = offset_x;
#pragma unroll
            for (int j = 0; j < num_block_x; j++) {
                // To mimic dw transpose for word/byte data type with transpose and pack
                constexpr uint8_t block_width = mem_transpose
                        ? (block_size_y / scale_factor)
                        : (block_size_x / scale_factor);
                constexpr uint8_t block_height
                        = mem_transpose ? block_size_x : block_size_y;
                constexpr uint32_t block_widthx_widthy_arrlen
                        = (block_width - 1) | ((block_height - 1) << 8)
                        | ((arr_len - 1) << 16);
                gpu::xetla::detail::xetla_set_block_widthx_widthy_arrlen(
                        tdesc_row_2d.row(j), block_widthx_widthy_arrlen);

                int32_t offset_width = mem_transpose
                        ? (base_offset_y / int32_t(scale_factor))
                        : (base_offset_x / int32_t(scale_factor));
                int32_t offset_height
                        = mem_transpose ? base_offset_x : base_offset_y;

                gpu::xetla::detail::xetla_set_tensor_offset_x(
                        tdesc_row_2d.row(j), offset_width);
                gpu::xetla::detail::xetla_set_tensor_offset_y(
                        tdesc_row_2d.row(j), offset_height);

                base_offset_x += block_size_x * arr_len;
            }
            base_offset_y += block_size_y;
        }
        // process the tail
        if constexpr (remained_size_y > 0) {
            auto tdesc_row_2d = payloads_2d.xetla_select<num_block_x, 1, 16, 1>(
                    num_block_y * num_block_x, 0);
            // this is exactly copy paste from above. so maybe worth createing some function
            int32_t base_offset_x = offset_x;
#pragma unroll
            for (int j = 0; j < num_block_x; j++) {
                // To mimic dw transpose for word/byte data type with transpose and pack
                constexpr uint8_t block_width = mem_transpose
                        ? (block_size_y / scale_factor)
                        : (block_size_x / scale_factor);
                constexpr uint8_t block_height
                        = mem_transpose ? block_size_x : block_size_y;
                constexpr uint32_t block_widthx_widthy_arrlen
                        = (block_width - 1) | ((block_height - 1) << 8)
                        | ((arr_len - 1) << 16);
                gpu::xetla::detail::xetla_set_block_widthx_widthy_arrlen(
                        tdesc_row_2d.row(j), block_widthx_widthy_arrlen);

                int32_t offset_width = mem_transpose
                        ? (base_offset_y / int32_t(scale_factor))
                        : (base_offset_x / int32_t(scale_factor));
                int32_t offset_height
                        = mem_transpose ? base_offset_x : base_offset_y;

                gpu::xetla::detail::xetla_set_tensor_offset_x(
                        tdesc_row_2d.row(j), offset_width);
                gpu::xetla::detail::xetla_set_tensor_offset_y(
                        tdesc_row_2d.row(j), offset_height);

                base_offset_x += block_size_x;
            }
        }
    }

private:
    __XETLA_API void prepare_tdesc(xetla_tdescriptor base_tdesc) {
        auto payloads_2d = payloads.xetla_format<uint32_t, num_block, 16>();
        uint32_t base_offset_y = 0;
#pragma unroll
        for (int i = 0; i < num_block_y; i++) {
            auto tdesc_row_2d = payloads_2d.xetla_select<num_block_x, 1, 16, 1>(
                    i * num_block_x, 0);
            prepare_tile_desc_core<num_block_x, block_size_x, block_size_y, 1,
                    mem_transpose>(tdesc_row_2d, base_tdesc, base_offset_y);
            base_offset_y += block_size_y;
        }
        // process the tail
        if constexpr (remained_size_y > 0) {
            auto tdesc_row_2d = payloads_2d.xetla_select<num_block_x, 1, 16, 1>(
                    num_block_y * num_block_x, 0);
            prepare_tile_desc_core<num_block_x, block_size_x, remained_size_y,
                    1, mem_transpose>(tdesc_row_2d, base_tdesc, base_offset_y);
        }
    }

    template <uint32_t num_tdesc, uint32_t size_x, uint32_t size_y,
            uint8_t arr_len, bool trans>
    __XETLA_API static void prepare_tile_desc_core(
            xetla_matrix_ref<uint32_t, num_tdesc, 16> __REF__ payloads_row_2d,
            xetla_tdescriptor base_tdesc, uint32_t base_offset_y) {
        uint32_t base_offset_x = 0;
#pragma unroll
        for (int j = 0; j < num_tdesc; j++) {
            payloads_row_2d.row(j) = base_tdesc;
            // To mimic dw transpose for word/byte data type with transpose and pack
            constexpr uint8_t block_width
                    = trans ? (size_y / scale_factor) : (size_x / scale_factor);
            constexpr uint8_t block_height = trans ? size_x : size_y;
            constexpr uint32_t block_widthx_widthy_arrlen = (block_width - 1)
                    | ((block_height - 1) << 8) | ((arr_len - 1) << 16);
            gpu::xetla::detail::xetla_set_block_widthx_widthy_arrlen(
                    payloads_row_2d.row(j), block_widthx_widthy_arrlen);

            // To mimic dw transpose for word/byte data type with transpose and pack
            uint32_t offset_width = trans
                    ? (base_offset_y / int32_t(scale_factor))
                    : (base_offset_x / int32_t(scale_factor));
            uint32_t offset_height = trans ? base_offset_x : base_offset_y;

            xetla_update_tdesc_offsetx(payloads_row_2d.row(j), offset_width);
            xetla_update_tdesc_offsety(payloads_row_2d.row(j), offset_height);
            base_offset_x += size_x * arr_len;
        }
    }
};

/// @brief Is to describe the global memory surface for block-1d load/store
/// For a block-1d payload message we need to set the base address and
/// offset of surface.
/// memory transpose and register operations can not be applied here.
/// 2d and 1d surface both can be accepted here
/// @tparam dtype Is the data type
/// @tparam tile_desc_ Is the tile descriptor
/// @tparam mem_layout_ Is the memory layout
template <typename dtype_, typename tile_desc_, gpu_arch arch_tag_,
        uint32_t alignment_>
struct mem_payload_t<mem_desc_t<dtype_, mem_layout::row_major,
                             mem_space::global, alignment_>,
        tile_desc_, msg_type::block_1d, arch_tag_,
        std::enable_if_t<(arch_tag_ == gpu_arch::Xe)>> {
    using mem_desc_t = mem_desc_t<dtype_, mem_layout::row_major,
            mem_space::global, alignment_>;
    using dtype = dtype_;
    using tile_desc = tile_desc_;
    static constexpr mem_space memory_space = mem_space::global;
    static constexpr mem_layout memory_layout = mem_layout::row_major;
    static constexpr msg_type message_type = msg_type::block_1d;
    static constexpr uint32_t alignment_in_bytes
            = mem_desc_t::alignment_in_bytes;
    static constexpr gpu_arch arch_tag = arch_tag_;
    static_assert((alignment_in_bytes % sizeof(uint32_t)) == 0,
            "alignment should at least DW aligned");

private:
    static constexpr uint32_t tile_size_x = tile_desc::tile_size_x;
    static constexpr uint32_t tile_size_y = tile_desc::tile_size_y;
    static_assert(tile_size_y == 1,
            "For tile_size_y > 1 case, please use 2d block message! ");
    using this_payload_t = mem_payload_t<mem_desc_t, tile_desc,
            msg_type::block_1d, arch_tag>;

public:
    static constexpr uint32_t bytes_per_row = tile_size_x * sizeof(dtype);
    using mem_dtype = typename std::conditional<
            (bytes_per_row % sizeof(uint64_t) == 0)
                    && (alignment_in_bytes % sizeof(uint64_t) == 0),
            uint64_t,
            typename std::conditional<(bytes_per_row % sizeof(uint32_t) == 0),
                    uint32_t, dtype>::type>::type;
    static constexpr uint32_t scale_factor = sizeof(mem_dtype) / sizeof(dtype);

    uint64_t base_offset;
    mem_dtype *base_ptr;
    uint32_t pitch_in_bytes;

    inline mem_payload_t(mem_desc_t &mem_tdesc) {
        pitch_in_bytes = mem_tdesc.shape.stride_x * sizeof(dtype);
        uint32_t offset_x = mem_tdesc.coord.x;
        uint32_t offset_y = mem_tdesc.coord.y;
        base_offset = offset_y * pitch_in_bytes + offset_x * sizeof(dtype);
        base_ptr = (mem_dtype *)mem_tdesc.base.base;
    }

    inline mem_payload_t(dtype *p, int surface_width, int surface_height,
            int surface_pitch, int surface_offset_x, int surface_offset_y) {
        pitch_in_bytes = surface_pitch * sizeof(dtype);
        uint32_t offset_x = surface_offset_x;
        uint32_t offset_y = surface_offset_y;
        base_offset = offset_y * pitch_in_bytes + offset_x * sizeof(dtype);
        base_ptr = (mem_dtype *)p;
    }

    __XETLA_API void init(mem_desc_t &mem_tdesc) {
        pitch_in_bytes = mem_tdesc.shape.stride_x * sizeof(dtype);
        uint32_t offset_x = mem_tdesc.coord.x;
        uint32_t offset_y = mem_tdesc.coord.y;
        base_offset = offset_y * pitch_in_bytes + offset_x * sizeof(dtype);
        base_ptr = (mem_dtype *)mem_tdesc.base.base;
    }

    __XETLA_API void init(dtype *p, int surface_width, int surface_height,
            int surface_pitch, int surface_offset_x, int surface_offset_y) {
        pitch_in_bytes = surface_pitch * sizeof(dtype);
        uint32_t offset_x = surface_offset_x;
        uint32_t offset_y = surface_offset_y;
        base_offset = offset_y * pitch_in_bytes + offset_x * sizeof(dtype);
        base_ptr = (mem_dtype *)p;
    }

    inline mem_payload_t(const this_payload_t &rhs) {
        this->base_offset = rhs.base_offset;
        this->base_ptr = rhs.base_ptr;
        this->pitch_in_bytes = rhs.pitch_in_bytes;
    }

    inline mem_payload_t() = default;
    inline this_payload_t &operator=(const this_payload_t &rhs) {
        this->base_offset = rhs.base_offset;
        this->base_ptr = rhs.base_ptr;
        this->pitch_in_bytes = rhs.pitch_in_bytes;
        return *this;
    }

    template <tdesc_update_dir update_dir = tdesc_update_dir::x_dir>
    __XETLA_API void update_tdesc(int offset) {
        if constexpr (update_dir == tdesc_update_dir::x_dir) {
            base_offset += int64_t(offset) * sizeof(dtype);
        } else {
            base_offset += int64_t(offset) * pitch_in_bytes;
        }
    }
};

/// @brief Is to describe the global memory surface for atomic store
/// For atomic store, we need to prepare necessary information for
/// each simd channel.
/// @tparam dtype Is the data type
/// @tparam tile_desc_ Is the tile descriptor
/// @tparam mem_layout_ Is the memory layout
template <typename dtype_, typename tile_desc_, mem_layout mem_layout_,
        gpu_arch arch_tag_, uint32_t alignment_, uint32_t dim_>
struct mem_payload_t<
        mem_desc_t<dtype_, mem_layout_, mem_space::global, alignment_, dim_>,
        tile_desc_, msg_type::atomic_add, arch_tag_,
        std::enable_if_t<(arch_tag_ == gpu_arch::Xe)>> {
    using mem_desc_t = mem_desc_t<dtype_, mem_layout_, mem_space::global,
            alignment_, dim_>;

    using dtype = dtype_;
    using tile_desc = tile_desc_;
    static constexpr mem_space memory_space = mem_space::global;
    static constexpr mem_layout memory_layout = mem_layout_;
    static constexpr msg_type message_type = msg_type::atomic_add;
    static constexpr uint32_t alignment_in_bytes
            = mem_desc_t::alignment_in_bytes;
    static constexpr gpu_arch arch_tag = arch_tag_;
    static_assert(
            sizeof(dtype) >= 2, "for atomic add, we only support W, DW or QW");

private:
    static constexpr uint32_t tile_size_x = tile_desc::tile_size_x;
    static constexpr uint32_t tile_size_y = tile_desc::tile_size_y;
    static constexpr uint32_t block_size_x = tile_desc::block_size_x;
    static constexpr uint32_t block_size_y = tile_desc::block_size_y;
    using this_payload_t = mem_payload_t<mem_desc_t, tile_desc,
            msg_type::atomic_add, arch_tag>;

public:
    static constexpr uint32_t tile_bytes
            = tile_size_x * tile_size_y * sizeof(dtype);
    static constexpr uint32_t block_bytes
            = block_size_x * block_size_y * sizeof(dtype);

    // for pvc, we can use simd16 or simd32
    static constexpr uint32_t min_store_bytes = 16 * sizeof(dtype);
    static constexpr uint32_t max_store_bytes = 32 * sizeof(dtype);
    static constexpr uint32_t num_channel
            = ((tile_bytes % max_store_bytes) == 0
                      && (block_bytes % max_store_bytes) == 0)
            ? 32
            : 16;

    static constexpr uint32_t num_channel_x = block_size_x;
    static constexpr uint32_t num_channel_y = num_channel / num_channel_x;
    static constexpr uint32_t store_elems = num_channel_y * block_size_x;

    //may need to set it to be configurable later
    using Toffset = uint32_t;

    xetla_vector<Toffset, num_channel> channel_offset;
    xetla_vector<uint32_t, num_channel> step_x;
    xetla_vector<uint32_t, num_channel> step_y;
    uint32_t pitch_in_bytes;
    uint32_t width_in_elems;
    uint32_t height_in_elems;
    uint32_t base_x;
    uint32_t base_y;
    uint64_t base_pointer;

    inline mem_payload_t(mem_desc_t &mem_tdesc) {
        pitch_in_bytes = mem_tdesc.shape.stride_x * sizeof(dtype);
        base_x = mem_tdesc.coord.x;
        base_y = mem_tdesc.coord.y;
        width_in_elems = mem_tdesc.shape.x;
        height_in_elems = mem_tdesc.shape.y;
        base_pointer = (uint64_t)mem_tdesc.base.base;
        base_pointer += base_y * pitch_in_bytes + base_x * sizeof(dtype);
        xetla_vector<uint32_t, num_channel> channel_index
                = xetla_vector_gen<uint32_t, num_channel>(0, 1);
        step_x = channel_index % num_channel_x;
        step_y = channel_index / num_channel_x;
        channel_offset = step_x * sizeof(dtype) + step_y * pitch_in_bytes;
    }

    inline mem_payload_t(dtype *p, int surface_width, int surface_height,
            int surface_pitch, int surface_offset_x, int surface_offset_y) {
        pitch_in_bytes = surface_pitch * sizeof(dtype);
        base_x = surface_offset_x;
        base_y = surface_offset_y;
        width_in_elems = surface_width;
        height_in_elems = surface_height;
        base_pointer = (uint64_t)p;
        base_pointer += base_y * pitch_in_bytes + base_x * sizeof(dtype);
        xetla_vector<uint32_t, num_channel> channel_index
                = xetla_vector_gen<uint32_t, num_channel>(0, 1);
        step_x = channel_index % num_channel_x;
        step_y = channel_index / num_channel_x;
        channel_offset = step_x * sizeof(dtype) + step_y * pitch_in_bytes;
    }

    __XETLA_API void init(dtype *p, int surface_width, int surface_height,
            int surface_pitch, int surface_offset_x, int surface_offset_y) {
        pitch_in_bytes = surface_pitch * sizeof(dtype);
        base_x = surface_offset_x;
        base_y = surface_offset_y;
        width_in_elems = surface_width;
        height_in_elems = surface_height;
        base_pointer = (uint64_t)p;
        base_pointer += base_y * pitch_in_bytes + base_x * sizeof(dtype);
        xetla_vector<uint32_t, num_channel> channel_index
                = xetla_vector_gen<uint32_t, num_channel>(0, 1);
        step_x = channel_index % num_channel_x;
        step_y = channel_index / num_channel_x;
        channel_offset = step_x * sizeof(dtype) + step_y * pitch_in_bytes;
    }

    __XETLA_API void init(mem_desc_t &mem_tdesc) {
        pitch_in_bytes = mem_tdesc.shape.stride_x * sizeof(dtype);
        base_x = mem_tdesc.coord.x;
        base_y = mem_tdesc.coord.y;
        width_in_elems = mem_tdesc.shape.x;
        height_in_elems = mem_tdesc.shape.y;
        base_pointer = (uint64_t)mem_tdesc.base.base;
        base_pointer += base_y * pitch_in_bytes + base_x * sizeof(dtype);
        xetla_vector<uint32_t, num_channel> channel_index
                = xetla_vector_gen<uint32_t, num_channel>(0, 1);
        step_x = channel_index % num_channel_x;
        step_y = channel_index / num_channel_x;
        channel_offset = step_x * sizeof(dtype) + step_y * pitch_in_bytes;
    }

    inline mem_payload_t(const this_payload_t &rhs) {
        this->pitch_in_bytes = rhs.pitch_in_bytes;
        this->width_in_elems = rhs.width_in_elems;
        this->height_in_elems = rhs.height_in_elems;
        this->base_x = rhs.base_x;
        this->base_y = rhs.base_y;
        this->base_pointer = rhs.base_pointer;
        this->channel_offset = rhs.channel_offset;
        this->step_x = rhs.step_x;
        this->step_y = rhs.step_y;
    }

    inline mem_payload_t() = default;
    inline this_payload_t &operator=(const this_payload_t &rhs) {
        this->pitch_in_bytes = rhs.pitch_in_bytes;
        this->width_in_elems = rhs.width_in_elems;
        this->height_in_elems = rhs.height_in_elems;
        this->base_x = rhs.base_x;
        this->base_y = rhs.base_y;
        this->base_pointer = rhs.base_pointer;
        this->channel_offset = rhs.channel_offset;
        this->step_x = rhs.step_x;
        this->step_y = rhs.step_y;
        return *this;
    }

    template <tdesc_update_dir update_dir = tdesc_update_dir::x_dir>
    __XETLA_API void update_tdesc(int offset) {
        if constexpr (update_dir == tdesc_update_dir::x_dir) {
            base_pointer += int64_t(offset) * sizeof(dtype);
            base_x += offset;
        } else {
            base_pointer += int64_t(offset) * pitch_in_bytes;
            base_y += offset;
        }
    }

    __XETLA_API void set_tdesc_base_address_masked(
            uint64_t offset, uint16_t mask = 1) {
        base_pointer = offset;
        base_pointer += base_y * pitch_in_bytes + base_x * sizeof(dtype);
        base_x += width_in_elems & (uint32_t)(mask - 1);
    }
};

/// @brief Is to describe the shared local memory surface for block-1d load/store
/// 1. data located in shared local memory 2. tile will be loaded / stored in 1d mode.
/// @tparam dtype Is the data type
/// @tparam tile_desc_ Is the tile descriptor
/// @tparam mem_layout_ Is the memory layout
template <typename dtype_, typename tile_desc_, gpu_arch arch_tag_,
        uint32_t alignment_>
struct mem_payload_t<
        mem_desc_t<dtype_, mem_layout::row_major, mem_space::local, alignment_>,
        tile_desc_, msg_type::block_1d, arch_tag_,
        std::enable_if_t<(arch_tag_ == gpu_arch::Xe)>> {
    using mem_desc_t = mem_desc_t<dtype_, mem_layout::row_major,
            mem_space::local, alignment_>;
    using dtype = dtype_;
    using tile_desc = tile_desc_;
    static constexpr mem_space memory_space = mem_space::local;
    static constexpr mem_layout memory_layout = mem_layout::row_major;
    static constexpr msg_type message_type = msg_type::block_1d;
    static constexpr uint32_t alignment_in_bytes
            = mem_desc_t::alignment_in_bytes;
    static_assert((alignment_in_bytes % sizeof(uint32_t)) == 0,
            "alignment should at least DW aligned");
    static constexpr gpu_arch arch_tag = arch_tag_;

private:
    static constexpr uint32_t tile_size_x = tile_desc::tile_size_x;
    static constexpr uint32_t tile_size_y = tile_desc::tile_size_y;
    static constexpr uint32_t block_size_x = tile_desc::block_size_x;
    static constexpr uint32_t block_size_y = tile_desc::block_size_y;
    using this_payload_t = mem_payload_t<mem_desc_t, tile_desc,
            msg_type::block_1d, arch_tag>;

public:
    static constexpr uint32_t tile_bytes
            = tile_size_x * tile_size_y * sizeof(dtype);
    static constexpr uint32_t block_bytes
            = block_size_x * block_size_y * sizeof(dtype);
    static constexpr uint32_t bytes_per_row = block_size_x * sizeof(dtype);
    using mem_dtype = typename std::conditional<
            (bytes_per_row % sizeof(uint64_t) == 0)
                    && (alignment_in_bytes % sizeof(uint64_t) == 0),
            uint64_t,
            typename std::conditional<(bytes_per_row % sizeof(uint32_t) == 0),
                    uint32_t, dtype>::type>::type;
    static constexpr uint32_t scale_factor = sizeof(mem_dtype) / sizeof(dtype);

    uint32_t base_address;
    uint32_t address;
    uint32_t pitch_in_bytes;

    __XETLA_API void init(mem_desc_t &mem_tdesc) {
        base_address = mem_tdesc.base.base;
        pitch_in_bytes = mem_tdesc.shape.stride_x * sizeof(dtype);
        uint32_t offset_x = mem_tdesc.coord.x;
        uint32_t offset_y = mem_tdesc.coord.y;
        address = offset_y * pitch_in_bytes + offset_x * sizeof(dtype);
    }

    __XETLA_API void init(uint32_t base, uint32_t surface_width,
            uint32_t surface_height, uint32_t surface_pitch,
            int32_t surface_offset_x, int32_t surface_offset_y) {
        base_address = base;
        uint32_t offset_x = surface_offset_x;
        uint32_t offset_y = surface_offset_y;
        pitch_in_bytes = surface_pitch * sizeof(dtype);
        address = offset_y * pitch_in_bytes + offset_x * sizeof(dtype);
    }

    inline mem_payload_t(mem_desc_t &mem_tdesc) { init(mem_tdesc); }
    inline mem_payload_t(uint32_t base, uint32_t surface_width,
            uint32_t surface_height, uint32_t surface_pitch,
            int32_t surface_offset_x, int32_t surface_offset_y) {
        init(base, surface_width, surface_height, surface_pitch,
                surface_offset_x, surface_offset_y);
    }

    inline mem_payload_t(const this_payload_t &rhs) {
        this->base_address = rhs.base_address;
        this->address = rhs.address;
        this->pitch_in_bytes = rhs.pitch_in_bytes;
    }

    inline mem_payload_t() = default;
    inline this_payload_t &operator=(const this_payload_t &rhs) {
        this->base_address = rhs.base_address;
        this->address = rhs.address;
        this->pitch_in_bytes = rhs.pitch_in_bytes;
        return *this;
    }

    template <tdesc_update_dir update_dir = tdesc_update_dir::x_dir>
    __XETLA_API void update_tdesc(int32_t offset) {
        if constexpr (update_dir == tdesc_update_dir::x_dir) {
            address += offset * sizeof(dtype);
        } else {
            address += offset * pitch_in_bytes;
        }
    }

    __XETLA_API void set_base_address(uint32_t base) {
        this->base_address = base;
    }
};

/// @brief Is to describe the global memory surface for unaligned-2d load/store
/// for each block in one tile, a payload message is prepared here.
/// in tile_load case, memory transpose, register transpose, memory transform
/// and dword transpose can be enable.
/// While in tile store case, we only support row major store, no register
/// operations can be applied.
/// @tparam dtype Is the data type
/// @tparam tile_desc_ Is the tile descriptor
/// @tparam mem_layout_ Is the memory layout
template <typename dtype_, typename tile_desc_, mem_layout mem_layout_,
        uint32_t alignment_, gpu_arch arch_tag_>
struct mem_payload_t<
        mem_desc_t<dtype_, mem_layout_, mem_space::global, alignment_>,
        tile_desc_, msg_type::unaligned_2d, arch_tag_,
        std::enable_if_t<(arch_tag_ == gpu_arch::Xe)>> {
    using dtype = dtype_;
    using mem_desc_t
            = mem_desc_t<dtype_, mem_layout_, mem_space::global, alignment_>;
    using tile_desc = tile_desc_;
    static constexpr mem_space memory_space = mem_space::global;
    static constexpr mem_layout memory_layout = mem_layout_;
    static constexpr msg_type message_type = msg_type::unaligned_2d;
    static constexpr uint32_t alignment_in_bytes
            = mem_desc_t::alignment_in_bytes;
    static constexpr gpu_arch arch_tag = arch_tag_;

private:
    static constexpr uint32_t tile_size_x = tile_desc::tile_size_x;
    static constexpr uint32_t tile_size_y = tile_desc::tile_size_y;
    static constexpr uint32_t block_size_x = tile_desc::block_size_x;
    static constexpr uint32_t block_size_y = tile_desc::block_size_y;

    using this_payload_t = mem_payload_t<mem_desc_t, tile_desc,
            msg_type::unaligned_2d, arch_tag_>;

public:
    static constexpr bool mem_transpose
            = memory_layout == mem_layout::col_major;

    static constexpr reg_layout register_layout = tile_desc::register_layout;
    static constexpr bool reg_transpose
            = register_layout == reg_layout::transpose_tiled;
    static constexpr bool trans = mem_transpose ^ reg_transpose;

    static constexpr bool mem_transform = (sizeof(dtype) < 4)
            && (register_layout == reg_layout::vnni_tiled
                    || register_layout == reg_layout::vnni_tiled_col_major);

    static constexpr uint32_t tile_bytes
            = tile_size_x * tile_size_y * sizeof(dtype);
    static constexpr uint32_t block_bytes
            = block_size_x * block_size_y * sizeof(dtype);

    using mem_dtype = typename std::conditional<
            (alignment_in_bytes % sizeof(uint64_t) == 0), uint64_t,
            typename std::conditional<(alignment_in_bytes % sizeof(uint32_t)
                                              == 0),
                    uint32_t, dtype>::type>::type;
    static constexpr uint32_t scale_factor = sizeof(mem_dtype) / sizeof(dtype);

    // for pvc, we can use simd16 or simd32
    static constexpr uint32_t min_store_bytes = 16 * sizeof(dtype);
    static constexpr uint32_t max_store_bytes = 32 * sizeof(dtype);
    static constexpr uint32_t num_channel
            = ((tile_bytes % max_store_bytes) == 0
                      && (block_bytes % max_store_bytes) == 0)
            ? 32
            : 16;

    static constexpr uint32_t num_channel_x
            = block_size_x * sizeof(dtype) / sizeof(mem_dtype);
    static constexpr uint32_t num_channel_y = num_channel / num_channel_x;

    xetla_vector<uint32_t, num_channel> channel_offset;
    xetla_vector<uint32_t, num_channel> step_x;
    xetla_vector<uint32_t, num_channel> step_y;

    uint64_t base_offset;
    uint32_t base_x;
    uint32_t base_y;
    uint32_t width_in_elems;
    uint32_t height_in_elems;

    mem_dtype *base_ptr;
    uint32_t pitch_in_bytes;

    inline mem_payload_t(mem_desc_t &mem_tdesc) {
        pitch_in_bytes = mem_tdesc.shape.stride_x * sizeof(dtype);
        base_x = mem_tdesc.coord.x;
        base_y = mem_tdesc.coord.y;
        width_in_elems = mem_tdesc.shape.x;
        height_in_elems = mem_tdesc.shape.y;
        base_offset = trans ? base_x * pitch_in_bytes + base_y * sizeof(dtype)
                            : base_y * pitch_in_bytes + base_x * sizeof(dtype);
        base_ptr = (mem_dtype *)mem_tdesc.base.base;

        xetla_vector<uint32_t, num_channel> channel_index
                = xetla_vector_gen<uint32_t, num_channel>(0, 1);
        step_x = channel_index % num_channel_x;
        step_y = channel_index / num_channel_x;
        channel_offset = trans
                ? step_y * sizeof(mem_dtype) + step_x * pitch_in_bytes
                : step_x * sizeof(mem_dtype) + step_y * pitch_in_bytes;
    }

    inline mem_payload_t(dtype *p, int surface_width, int surface_height,
            int surface_pitch, int surface_offset_x, int surface_offset_y) {
        pitch_in_bytes = surface_pitch * sizeof(dtype);
        base_x = surface_offset_x;
        base_y = surface_offset_y;
        width_in_elems = surface_width;
        height_in_elems = surface_height;
        base_offset = trans ? base_x * pitch_in_bytes + base_y * sizeof(dtype)
                            : base_y * pitch_in_bytes + base_x * sizeof(dtype);
        base_ptr = (mem_dtype *)p;

        xetla_vector<uint32_t, num_channel> channel_index
                = xetla_vector_gen<uint32_t, num_channel>(0, 1);
        step_x = channel_index % num_channel_x;
        step_y = channel_index / num_channel_x;
        channel_offset = trans
                ? step_y * sizeof(mem_dtype) + step_x * pitch_in_bytes
                : step_x * sizeof(mem_dtype) + step_y * pitch_in_bytes;
    }

    __XETLA_API void init(mem_desc_t &mem_tdesc) {
        pitch_in_bytes = mem_tdesc.shape.stride_x * sizeof(dtype);
        base_x = mem_tdesc.coord.x;
        base_y = mem_tdesc.coord.y;
        width_in_elems = mem_tdesc.shape.x;
        height_in_elems = mem_tdesc.shape.y;
        base_offset = trans ? base_x * pitch_in_bytes + base_y * sizeof(dtype)
                            : base_y * pitch_in_bytes + base_x * sizeof(dtype);
        base_ptr = (mem_dtype *)mem_tdesc.base.base;

        xetla_vector<uint32_t, num_channel> channel_index
                = xetla_vector_gen<uint32_t, num_channel>(0, 1);
        step_x = channel_index % num_channel_x;
        step_y = channel_index / num_channel_x;
        channel_offset = trans
                ? step_y * sizeof(mem_dtype) + step_x * pitch_in_bytes
                : step_x * sizeof(mem_dtype) + step_y * pitch_in_bytes;
    }

    __XETLA_API void init(dtype *p, int surface_width, int surface_height,
            int surface_pitch, int surface_offset_x, int surface_offset_y) {
        pitch_in_bytes = surface_pitch * sizeof(dtype);
        base_x = surface_offset_x;
        base_y = surface_offset_y;
        width_in_elems = surface_width;
        height_in_elems = surface_height;
        base_offset = trans ? base_x * pitch_in_bytes + base_y * sizeof(dtype)
                            : base_y * pitch_in_bytes + base_x * sizeof(dtype);
        base_ptr = (mem_dtype *)p;

        xetla_vector<uint32_t, num_channel> channel_index
                = xetla_vector_gen<uint32_t, num_channel>(0, 1);
        step_x = channel_index % num_channel_x;
        step_y = channel_index / num_channel_x;
        channel_offset = trans
                ? step_y * sizeof(mem_dtype) + step_x * pitch_in_bytes
                : step_x * sizeof(mem_dtype) + step_y * pitch_in_bytes;
    }

    inline mem_payload_t(const this_payload_t &rhs) {
        this->base_offset = rhs.base_offset;
        this->base_ptr = rhs.base_ptr;
        this->pitch_in_bytes = rhs.pitch_in_bytes;
        this->base_x = rhs.base_x;
        this->base_y = rhs.base_y;
        this->width_in_elems = rhs.width_in_elems;
        this->height_in_elems = rhs.height_in_elems;

        this->step_x = rhs.step_x;
        this->step_y = rhs.step_y;

        this->channel_offset = rhs.channel_offset;
    }

    inline mem_payload_t() = default;
    inline this_payload_t &operator=(const this_payload_t &rhs) {
        this->base_offset = rhs.base_offset;
        this->base_ptr = rhs.base_ptr;
        this->pitch_in_bytes = rhs.pitch_in_bytes;
        this->base_x = rhs.base_x;
        this->base_y = rhs.base_y;
        this->width_in_elems = rhs.width_in_elems;
        this->height_in_elems = rhs.height_in_elems;

        this->step_x = rhs.step_x;
        this->step_y = rhs.step_y;
        this->channel_offset = rhs.channel_offset;

        return *this;
    }

    template <tdesc_update_dir update_dir = tdesc_update_dir::x_dir>
    __XETLA_API void update_tdesc(int offset) {
        if constexpr (update_dir == tdesc_update_dir::x_dir) {
            base_offset += int64_t(offset) * sizeof(dtype);
            trans ? base_y += offset : base_x += offset;
        } else {
            base_offset += int64_t(offset) * pitch_in_bytes;
            trans ? base_x += offset : base_y += offset;
        }
    }
};

/// @brief Is to describe the shared local memory surface for scatter load/store
/// 1. data located in shared local memory 2. tile will be loaded / stored in scatter mode
/// @tparam dtype Is the data type
/// @tparam tile_desc_ Is the tile descriptor
/// @tparam mem_layout_ Is the memory layout
template <typename dtype_, typename tile_desc_, gpu_arch arch_tag_,
        uint32_t alignment_>
struct mem_payload_t<
        mem_desc_t<dtype_, mem_layout::row_major, mem_space::local, alignment_>,
        tile_desc_, msg_type::scatter, arch_tag_,
        std::enable_if_t<(arch_tag_ == gpu_arch::Xe)>> {
    using mem_desc_t = mem_desc_t<dtype_, mem_layout::row_major,
            mem_space::local, alignment_>;
    using dtype = dtype_;
    using tile_desc = tile_desc_;
    static constexpr mem_space memory_space = mem_space::local;
    static constexpr mem_layout memory_layout = mem_layout::row_major;
    static constexpr msg_type message_type = msg_type::scatter;
    static constexpr uint32_t alignment_in_bytes
            = mem_desc_t::alignment_in_bytes;
    static constexpr gpu_arch arch_tag = arch_tag_;

private:
    static constexpr uint32_t tile_size_x = tile_desc::tile_size_x;
    static constexpr uint32_t tile_size_y = tile_desc::tile_size_y;
    static constexpr uint32_t block_size_x = tile_desc::block_size_x;
    static constexpr uint32_t block_size_y = tile_desc::block_size_y;
    using this_payload_t
            = mem_payload_t<mem_desc_t, tile_desc, msg_type::scatter, arch_tag>;

public:
    static constexpr reg_layout register_layout = tile_desc::register_layout;
    static constexpr bool mem_transform
            = (sizeof(dtype) < 4) && register_layout == reg_layout::vnni_tiled;

    static constexpr uint32_t tile_bytes
            = tile_size_x * tile_size_y * sizeof(dtype);
    static constexpr uint32_t block_bytes
            = block_size_x * block_size_y * sizeof(dtype);
    using mem_dtype = typename std::conditional<
            (block_bytes % (16 * sizeof(uint64_t)) == 0), uint64_t,
            typename std::conditional<(block_bytes % (16 * sizeof(uint32_t))
                                              == 0),
                    uint32_t, dtype>::type>::type;
    // we can use simd16 or simd32
    static constexpr uint32_t min_bytes = 16 * sizeof(mem_dtype);
    static constexpr uint32_t max_bytes = 32 * sizeof(mem_dtype);

    static constexpr uint32_t num_channel
            = ((tile_bytes % max_bytes) == 0 && (block_bytes % max_bytes) == 0)
            ? 32
            : 16;
    static constexpr uint32_t num_channel_x
            = block_size_x * sizeof(dtype) / sizeof(mem_dtype);
    static constexpr uint32_t num_channel_y = num_channel / num_channel_x;
    xetla_vector<uint32_t, num_channel> address;
    uint32_t pitch_in_bytes;
    uint32_t wg_width_in_bytes;
    uint32_t wg_height_in_elems;

    inline mem_payload_t(mem_desc_t &mem_tdesc) {
        xetla_tdescriptor base_tdesc = mem_tdesc.get_tdesc();
        pitch_in_bytes = base_tdesc[4];
        wg_width_in_bytes = base_tdesc[2];
        wg_height_in_elems = base_tdesc[3];
        uint32_t offset_x = base_tdesc[5];
        uint32_t offset_y = base_tdesc[6];
        uint32_t start_address = base_tdesc[0];
        start_address += offset_y * pitch_in_bytes + offset_x * sizeof(dtype);
        xetla_vector<uint32_t, num_channel> channel_index
                = xetla_vector_gen<uint32_t, num_channel>(0, 1);
        address = start_address
                + (channel_index % num_channel_x) * sizeof(mem_dtype)
                + (channel_index / num_channel_x) * pitch_in_bytes;
    }

    inline mem_payload_t(uint32_t base, int surface_width, int surface_height,
            int surface_pitch, int surface_offset_x, int surface_offset_y) {
        pitch_in_bytes = surface_pitch * sizeof(dtype);
        wg_width_in_bytes = surface_width * sizeof(dtype);
        wg_height_in_elems = surface_height;
        uint32_t offset_x = surface_offset_x;
        uint32_t offset_y = surface_offset_y;
        uint32_t start_address = base;
        start_address += offset_y * pitch_in_bytes + offset_x * sizeof(dtype);
        xetla_vector<uint32_t, num_channel> channel_index
                = xetla_vector_gen<uint32_t, num_channel>(0, 1);
        address = start_address
                + (channel_index % num_channel_x) * sizeof(mem_dtype)
                + (channel_index / num_channel_x) * pitch_in_bytes;
    }

    __XETLA_API void init(uint32_t base, int surface_width, int surface_height,
            int surface_pitch, int surface_offset_x, int surface_offset_y) {
        pitch_in_bytes = surface_pitch * sizeof(dtype);
        wg_width_in_bytes = surface_width * sizeof(dtype);
        wg_height_in_elems = surface_height;
        uint32_t offset_x = surface_offset_x;
        uint32_t offset_y = surface_offset_y;
        uint32_t start_address = base;
        start_address += offset_y * pitch_in_bytes + offset_x * sizeof(dtype);
        xetla_vector<uint32_t, num_channel> channel_index
                = xetla_vector_gen<uint32_t, num_channel>(0, 1);
        address = start_address
                + (channel_index % num_channel_x) * sizeof(mem_dtype)
                + (channel_index / num_channel_x) * pitch_in_bytes;
    }

    __XETLA_API void init(mem_desc_t &mem_tdesc) {
        xetla_tdescriptor base_tdesc = mem_tdesc.get_tdesc();
        pitch_in_bytes = base_tdesc[4];
        wg_width_in_bytes = base_tdesc[2];
        wg_height_in_elems = base_tdesc[3];
        uint32_t offset_x = base_tdesc[5];
        uint32_t offset_y = base_tdesc[6];
        uint32_t start_address = base_tdesc[0];
        start_address += offset_y * pitch_in_bytes + offset_x * sizeof(dtype);
        xetla_vector<uint32_t, num_channel> channel_index
                = xetla_vector_gen<uint32_t, num_channel>(0, 1);
        address = start_address
                + (channel_index % num_channel_x) * sizeof(mem_dtype)
                + (channel_index / num_channel_x) * pitch_in_bytes;
    }

    inline mem_payload_t(const this_payload_t &rhs) {
        this->address = rhs.address;
        this->pitch_in_bytes = rhs.pitch_in_bytes;
        this->wg_width_in_bytes = rhs.wg_width_in_bytes;
        this->wg_height_in_elems = rhs.wg_height_in_elems;
    }

    inline mem_payload_t() = default;
    inline this_payload_t &operator=(const this_payload_t &rhs) {
        this->address = rhs.address;
        this->pitch_in_bytes = rhs.pitch_in_bytes;
        this->wg_width_in_bytes = rhs.wg_width_in_bytes;
        this->wg_height_in_elems = rhs.wg_height_in_elems;
        return *this;
    }

    template <tdesc_update_dir update_dir = tdesc_update_dir::x_dir>
    __XETLA_API void update_tdesc(int offset) {
        if constexpr (update_dir == tdesc_update_dir::x_dir) {
            address += offset * sizeof(dtype);
        } else {
            address += offset * pitch_in_bytes;
        }
    }
};

/// @brief Is to describe the shared local memory surface for scattering store
/// 1. data located in shared local memory 2. tile will be stored in scatter mode
/// 3. data in register file is vnni packed and col majored.
/// @tparam dtype Is the data type
/// @tparam tile_desc_ Is the tile descriptor
/// @tparam mem_layout_ Is the memory layout
/// @note this is used for Atrans SLM path, so, we can add more limitation for best performance.
template <typename dtype_, uint32_t tile_size_x_, uint32_t tile_size_y_,
        uint32_t block_size_x_, uint32_t block_size_y_, gpu_arch arch_tag_,
        uint32_t alignment_>
struct mem_payload_t<
        mem_desc_t<dtype_, mem_layout::row_major, mem_space::local, alignment_>,
        tile_desc_t<tile_size_x_, tile_size_y_, block_size_x_, block_size_y_,
                reg_layout::vnni_tiled_col_major>,
        msg_type::scatter, arch_tag_,
        std::enable_if_t<(arch_tag_ == gpu_arch::Xe)>> {
    using mem_desc_t = mem_desc_t<dtype_, mem_layout::row_major,
            mem_space::local, alignment_>;
    using dtype = dtype_;
    using tile_desc = tile_desc_t<tile_size_x_, tile_size_y_, block_size_x_,
            block_size_y_, reg_layout::vnni_tiled_col_major>;
    static constexpr mem_space memory_space = mem_space::local;
    static constexpr mem_layout memory_layout = mem_layout::row_major;
    static constexpr msg_type message_type = msg_type::scatter;
    static constexpr uint32_t alignment_in_bytes
            = mem_desc_t::alignment_in_bytes;
    static constexpr gpu_arch arch_tag = arch_tag_;

private:
    static constexpr uint32_t tile_size_x = tile_desc::tile_size_x;
    static constexpr uint32_t tile_size_y = tile_desc::tile_size_y;
    static constexpr uint32_t block_size_x = tile_desc::block_size_x;
    static constexpr uint32_t block_size_y = tile_desc::block_size_y;
    using this_payload_t
            = mem_payload_t<mem_desc_t, tile_desc, msg_type::scatter, arch_tag>;

public:
    static constexpr uint32_t tile_bytes
            = tile_size_x * tile_size_y * sizeof(dtype);
    static constexpr uint32_t block_bytes
            = block_size_x * block_size_y * sizeof(dtype);
    using store_dtype = uint32_t;
    static constexpr uint32_t vnni_scale_factor
            = sizeof(store_dtype) / sizeof(dtype);
    static_assert(block_size_x % 16 == 0,
            "block size x at least need to be 16 channel aligned");
    static constexpr uint32_t num_channel = block_size_x;
    static constexpr uint32_t max_vector_size = (block_size_x == 16) ? 8 : 4;
    static constexpr uint32_t num_vector_size
            = detail::gcd<tile_size_y / vnni_scale_factor,
                    max_vector_size>::value;
    static constexpr uint32_t store_elems
            = num_channel * num_vector_size * vnni_scale_factor;

    uint32_t base_address;
    uint32_t pitch_in_bytes;
    xetla_vector<uint32_t, num_channel> channel_address;

    __XETLA_API void init(mem_desc_t &mem_tdesc) {
        base_address = mem_tdesc.base.base;
        pitch_in_bytes = mem_tdesc.shape.stride_x * sizeof(dtype);
        uint32_t offset_x = mem_tdesc.coord.x; //because this is row-major
        uint32_t offset_y = mem_tdesc.coord.y;
        uint32_t offset_address
                = offset_y * pitch_in_bytes + offset_x * sizeof(dtype);
        xetla_vector<uint32_t, num_channel> channel_index
                = xetla_vector_gen<uint32_t, num_channel>(0, 1);
        channel_address = offset_address + channel_index * pitch_in_bytes;
    }
    __XETLA_API void init(uint32_t base, uint32_t surface_width,
            uint32_t surface_height, uint32_t surface_pitch,
            int32_t surface_offset_x, int32_t surface_offset_y) {
        base_address = base;
        pitch_in_bytes = surface_pitch * sizeof(dtype);
        uint32_t offset_x = surface_offset_x;
        uint32_t offset_y = surface_offset_y;
        uint32_t offset_address
                = offset_y * pitch_in_bytes + offset_x * sizeof(dtype);
        xetla_vector<uint32_t, num_channel> channel_index
                = xetla_vector_gen<uint32_t, num_channel>(0, 1);
        channel_address = offset_address + channel_index * pitch_in_bytes;
    }

    inline mem_payload_t(uint32_t base, uint32_t surface_width,
            uint32_t surface_height, uint32_t surface_pitch,
            int32_t surface_offset_x, int32_t surface_offset_y) {
        init(base, surface_width, surface_height, surface_pitch,
                surface_offset_x, surface_offset_y);
    }
    // Be aware of the risks: Rule of three (copy constructor, copy assignment, destructor)
    // Please check if you need to add self-define destructor
    // ~mem_payload_t(){}
    inline mem_payload_t(mem_desc_t &mem_tdesc) { init(mem_tdesc); }

    inline mem_payload_t(const this_payload_t &rhs) {
        this->base_address = rhs.base_address;
        this->pitch_in_bytes = rhs.pitch_in_bytes;
        this->channel_address = rhs.channel_address;
    }

    inline mem_payload_t() = default;
    inline this_payload_t &operator=(const this_payload_t &rhs) {
        this->base_address = rhs.base_address;
        this->pitch_in_bytes = rhs.pitch_in_bytes;
        this->channel_address = rhs.channel_address;
        return *this;
    }

    template <tdesc_update_dir update_dir = tdesc_update_dir::x_dir>
    __XETLA_API void update_tdesc(int offset) {
        if constexpr (update_dir == tdesc_update_dir::x_dir) {
            channel_address += offset * sizeof(dtype);
        } else {
            channel_address += offset * pitch_in_bytes;
        }
    }

    __XETLA_API void set_base_address(uint32_t base) {
        this->base_address = base;
    }
};

/// @brief Is to describe the global memory surface to prefetch data to cache
/// data in global memory will be prefetched into 2d tile
/// @tparam tile_desc_ Is the tile descriptor
/// @tparam dtype Is the data type
/// @tparam mem_layout Is the data layout
/// @tparam num_coop_sg_ Is the thread nums to prefetch data
template <typename dtype_, uint32_t tile_size_x_, uint32_t tile_size_y_,
        uint32_t block_size_x_, uint32_t block_size_y_, mem_layout mem_layout_,
        uint32_t alignment_, uint32_t num_coop_sg_, reg_layout reg_layout_,
        gpu_arch arch_tag_>
struct prefetch_payload_t<
        mem_desc_t<dtype_, mem_layout_, mem_space::global, alignment_>,
        tile_desc_t<tile_size_x_, tile_size_y_, block_size_x_, block_size_y_,
                reg_layout_>,
        num_coop_sg_, arch_tag_,
        std::enable_if_t<(arch_tag_ == gpu_arch::Xe)>> {
    using dtype = dtype_;
    using mem_desc_t
            = mem_desc_t<dtype_, mem_layout_, mem_space::global, alignment_>;
    using tile_desc = tile_desc_t<tile_size_x_, tile_size_y_, block_size_x_,
            block_size_y_, reg_layout_>;
    static constexpr mem_space memory_space = mem_space::global;
    static constexpr mem_layout memory_layout = mem_layout_;
    static constexpr gpu_arch arch_tag = arch_tag_;

    static constexpr uint32_t tile_size_x = tile_desc::tile_size_x;
    static constexpr uint32_t tile_size_y = tile_desc::tile_size_y;

    // private:
    static constexpr bool is_col_major = mem_layout_ == mem_layout::col_major;
    static constexpr uint32_t mem_tile_size_w
            = is_col_major ? tile_size_y : tile_size_x;
    static constexpr uint32_t mem_tile_size_h
            = is_col_major ? tile_size_x : tile_size_y;
    using load_store_attr = typename arch_attr_t<
            arch_tag>::template load_store_attr<msg_type::block_2d>;
    static constexpr uint32_t special_prefetch_width
            = load_store_attr::special_prefetch_width_in_bytes / sizeof(dtype);
    static constexpr uint32_t normal_prefetch_width
            = load_store_attr::max_load_width_in_bytes / sizeof(dtype);
    static constexpr bool is_special_prefetch
            = (mem_tile_size_w % special_prefetch_width) == 0;

    static constexpr uint32_t block_size_w = is_special_prefetch
            ? special_prefetch_width
            : (normal_prefetch_width > mem_tile_size_w ? mem_tile_size_w
                                                       : normal_prefetch_width);
    static constexpr uint32_t block_size_h
            = load_store_attr::max_load_height_in_elem;
    //could have over-prefetch, but that's should be fine
    static constexpr uint32_t max_num_block_w
            = (mem_tile_size_w + block_size_w - 1) / block_size_w;
    static constexpr uint32_t num_coop_sg = num_coop_sg_;
    static constexpr uint32_t num_coop_sg_w
            = detail::gcd<num_coop_sg, max_num_block_w>::value;
    static constexpr uint32_t num_coop_sg_h = num_coop_sg / num_coop_sg_w;

    static constexpr uint32_t num_block_w = max_num_block_w / num_coop_sg_w;
    static constexpr uint32_t tile_size_w = block_size_w * num_block_w;
    static constexpr uint32_t tile_size_h
            = (mem_tile_size_h + num_coop_sg_h - 1) / num_coop_sg_h;
    static constexpr uint32_t num_block_h
            = (tile_size_h + block_size_h - 1) / block_size_h;
    using this_payload_t
            = prefetch_payload_t<mem_desc_t, tile_desc, num_coop_sg_, arch_tag>;

public:
    static constexpr uint32_t num_tdesc = num_block_w * num_block_h;
    xetla_vector<uint32_t, num_tdesc * 16> tdesc_prefetch;

    inline prefetch_payload_t(const this_payload_t &rhs) {
        this->tdesc_prefetch = rhs.tdesc_prefetch;
    }

    inline prefetch_payload_t() = default;

    inline this_payload_t &operator=(const this_payload_t &rhs) {
        this->tdesc_prefetch = rhs.tdesc_prefetch;
        return *this;
    }

    inline prefetch_payload_t(mem_desc_t &mem_desc, uint32_t coop_id = 0) {
        xetla_tdescriptor base_tdesc = mem_desc.get_tdesc();
        uint32_t coop_id_x = coop_id % num_coop_sg_w;
        uint32_t coop_id_y = coop_id / num_coop_sg_w;
        xetla_update_tdesc_offsetx(
                base_tdesc.xetla_format<uint32_t>(), coop_id_x * tile_size_w);
        xetla_update_tdesc_offsety(
                base_tdesc.xetla_format<uint32_t>(), coop_id_y * tile_size_h);
        prepare_tdesc(base_tdesc);
    }

    inline prefetch_payload_t(dtype *p, int surface_width, int surface_height,
            int surface_pitch, int surface_offset_x, int surface_offset_y,
            uint32_t coop_id = 0) {
        uint32_t coop_id_x = coop_id % num_coop_sg_w;
        uint32_t coop_id_y = coop_id / num_coop_sg_w;
        xetla_tdescriptor base_tdesc;
        xetla_fill_tdesc(base_tdesc.xetla_format<uint32_t>(), p, surface_width,
                surface_height, surface_pitch,
                surface_offset_x + coop_id_x * tile_size_w,
                surface_offset_y + coop_id_y * tile_size_h);
        prepare_tdesc(base_tdesc);
    }

    inline void init(xetla_tdescriptor base_tdesc, uint32_t coop_id = 0) {
        uint32_t coop_id_x = coop_id % num_coop_sg_w;
        uint32_t coop_id_y = coop_id / num_coop_sg_w;
        xetla_update_tdesc_offsetx(
                base_tdesc.xetla_format<uint32_t>(), coop_id_x * tile_size_w);
        xetla_update_tdesc_offsety(
                base_tdesc.xetla_format<uint32_t>(), coop_id_y * tile_size_h);
        prepare_tdesc(base_tdesc);
    }
    // Be aware of the risks: Rule of three (copy constructor, copy assignment, destructor)
    // Please check if you need to add self-define destructor
    // ~prefetch_payload_t(){}

    template <tdesc_update_dir update_dir = tdesc_update_dir::x_dir>
    __XETLA_API void update_tdesc(int offset) {
        auto tdesc_2d = tdesc_prefetch.xetla_format<uint32_t, num_tdesc, 16>();
        if constexpr (update_dir == tdesc_update_dir::x_dir) {
#pragma unroll
            for (int i = 0; i < num_tdesc; i++) {
                xetla_update_tdesc_offsetx(tdesc_2d.row(i), offset);
            }
        } else {
#pragma unroll
            for (int i = 0; i < num_tdesc; i++) {
                xetla_update_tdesc_offsety(tdesc_2d.row(i), offset);
            }
        }
    }
    __XETLA_API void set_tdesc_width(uint32_t size) {
        auto tdesc_2d = tdesc_prefetch.xetla_format<uint32_t, num_tdesc, 16>();
#pragma unroll
        for (int i = 0; i < num_tdesc; i++) {
            xetla_set_tdesc_width<dtype>(tdesc_2d.row(i), size);
        }
    }

    __XETLA_API void set_tdesc_pitch(uint32_t size) {
        auto tdesc_2d = tdesc_prefetch.xetla_format<uint32_t, num_tdesc, 16>();
#pragma unroll
        for (int i = 0; i < num_tdesc; i++) {
            xetla_set_tdesc_pitch<dtype>(tdesc_2d.row(i), size);
        }
    }

    __XETLA_API void set_tdesc_height(uint32_t size) {
        auto tdesc_2d = tdesc_prefetch.xetla_format<uint32_t, num_tdesc, 16>();
#pragma unroll
        for (int i = 0; i < num_tdesc; i++) {
            xetla_set_tdesc_height(tdesc_2d.row(i), size);
        }
    }

    __XETLA_API void update_tdesc_base_address(int offset) {
        auto tdesc_2d = tdesc_prefetch.xetla_format<uint32_t, num_tdesc, 16>();
#pragma unroll
        for (int i = 0; i < num_tdesc; i++) {
            xetla_update_tdesc_base_address(tdesc_2d.row(i), offset);
        }
    }

    __XETLA_API void set_tdesc_base_address(uint64_t addr) {
        auto tdesc_2d = tdesc_prefetch.xetla_format<uint32_t, num_tdesc, 16>();
#pragma unroll
        for (int i = 0; i < num_tdesc; i++) {
            gpu::xetla::detail::xetla_set_tensor_base_address(
                    tdesc_2d.row(i), addr);
        }
    }

    __XETLA_API void update_tdesc_base_address_masked(
            int offset, uint16_t mask = 1) {
        auto tdesc_2d = tdesc_prefetch.xetla_format<uint32_t, num_tdesc, 16>();
#pragma unroll
        for (int i = 0; i < num_tdesc; i++) {
            xetla_update_tdesc_base_address(tdesc_2d.row(i), offset);
        }

#pragma unroll
        for (int i = 0; i < num_tdesc; i++) {
            xetla_tdesc_mask_op(tdesc_2d.row(i), mask);
        }
    }

    __XETLA_API void set_offset(
            int32_t offset_x, int32_t offset_y, uint32_t coop_id = 0) {
        uint32_t coop_id_x = coop_id % num_coop_sg_w;
        uint32_t coop_id_y = coop_id / num_coop_sg_w;

        auto tdesc_2d = tdesc_prefetch.xetla_format<uint32_t, num_tdesc, 16>();
        int32_t base_offset_y = offset_y
                + (is_col_major ? coop_id_x * tile_size_w
                                : coop_id_y * tile_size_h);
#pragma unroll
        for (int i = 0; i < num_block_h; i++) {
            auto tdesc_row_2d = tdesc_2d.xetla_select<num_block_w, 1, 16, 1>(
                    i * num_block_w, 0);

            int32_t base_offset_x = offset_x
                    + (is_col_major ? coop_id_y * tile_size_h
                                    : coop_id_x * tile_size_w);
#pragma unroll
            for (int j = 0; j < num_block_w; j++) {
                int32_t offset_width
                        = is_col_major ? base_offset_y : base_offset_x;
                int32_t offset_height
                        = is_col_major ? base_offset_x : base_offset_y;
                gpu::xetla::detail::xetla_set_tensor_offset_x(
                        tdesc_row_2d.row(j), offset_width);
                gpu::xetla::detail::xetla_set_tensor_offset_y(
                        tdesc_row_2d.row(j), offset_height);

                base_offset_x += block_size_w;
            }
            base_offset_y += block_size_h;
        }
    }

private:
    __XETLA_API void prepare_tdesc(xetla_tdescriptor base_tdesc) {
        auto tdesc_2d = tdesc_prefetch.xetla_format<uint32_t, num_tdesc, 16>();
        uint32_t base_offset_y = 0;
#pragma unroll
        for (int i = 0; i < tile_size_h / block_size_h; i++) {
            auto tdesc_row_2d = tdesc_2d.xetla_select<num_block_w, 1, 16, 1>(
                    i * num_block_w, 0);
            prepare_tile_desc_core<num_block_w, block_size_w, block_size_h>(
                    tdesc_row_2d, base_tdesc, base_offset_y);
            base_offset_y += block_size_h;
        }
        if constexpr ((tile_size_h % block_size_h) != 0) {
            constexpr int i = tile_size_h / block_size_h;
            auto tdesc_row_2d = tdesc_2d.xetla_select<num_block_w, 1, 16, 1>(
                    i * num_block_w, 0);
            constexpr uint32_t remain_size_y = tile_size_h % block_size_h;
            prepare_tile_desc_core<num_block_w, block_size_w, remain_size_y>(
                    tdesc_row_2d, base_tdesc, base_offset_y);
        }
    }

    template <int32_t num_tdesc, uint32_t size_x, uint32_t size_y>
    __XETLA_API static void prepare_tile_desc_core(
            xetla_matrix_ref<uint32_t, num_tdesc, 16> __REF__ tdesc_2d,
            xetla_tdescriptor base_tdesc, uint32_t base_offset_y) {
        uint32_t base_offset_x = 0;
#pragma unroll
        for (int j = 0; j < num_tdesc; j++) {
            tdesc_2d.row(j) = base_tdesc;

            constexpr uint32_t block_widthx_widthy_arrlen
                    = (size_x - 1) | ((size_y - 1) << 8);
            gpu::xetla::detail::xetla_set_block_widthx_widthy_arrlen(
                    tdesc_2d.row(j), block_widthx_widthy_arrlen);

            xetla_update_tdesc_offsetx(tdesc_2d.row(j), base_offset_x);
            xetla_update_tdesc_offsety(tdesc_2d.row(j), base_offset_y);
            base_offset_x += size_x;
        }
    }
};

/// @brief Is to describe the memory memory to prefetch data to cache
/// data in global memory will be prefetched into 1d tile
/// @tparam tile_desc_ Is the tile descriptor
/// @tparam dtype Is the data type
/// @tparam mem_layout Is the data layout
/// @tparam num_coop_sg_ Is the thread nums to prefetch data
template <typename dtype_, uint32_t tile_size_x_, uint32_t block_size_x_,
        mem_layout mem_layout_, uint32_t alignment_, uint32_t num_coop_sg_,
        reg_layout reg_layout_, gpu_arch arch_tag_>
struct prefetch_payload_t<
        mem_desc_t<dtype_, mem_layout_, mem_space::global, alignment_>,
        tile_desc_t<tile_size_x_, 1, block_size_x_, 1, reg_layout_>,
        num_coop_sg_, arch_tag_,
        std::enable_if_t<(arch_tag_ == gpu_arch::Xe)>> {
    using dtype = dtype_;
    using mem_desc_t
            = mem_desc_t<dtype_, mem_layout_, mem_space::global, alignment_>;
    // CL aligned, so we can use uint64_t
    using prefetch_dtype = uint64_t;
    using tile_desc
            = tile_desc_t<tile_size_x_, 1, block_size_x_, 1, reg_layout_>;
    static constexpr mem_space memory_space = mem_space::global;
    static constexpr mem_layout memory_layout = mem_layout_;
    static constexpr gpu_arch arch_tag = arch_tag_;

private:
    // Fetches the entire CL.
    static constexpr uint32_t cacheline_elems = 64 / sizeof(dtype);
    static constexpr uint32_t mem_block_nums
            = (tile_desc::tile_size_x + cacheline_elems - 1) / cacheline_elems;
    static constexpr uint32_t num_coop_sg = num_coop_sg_;

    // For mem_tile_nums < num_coop_sg cases, mem_tile_size_x will be CL length
    // which might lead to illegal read.
    // there are num_coop_sg threads to prefetch mem_block_nums
    // each thread will prefetch mem_tile_size_x elements
    static constexpr uint32_t mem_tile_size_x = mem_block_nums > num_coop_sg
            ? (mem_block_nums + num_coop_sg - 1) / num_coop_sg *cacheline_elems
            : 0;
    using this_payload_t
            = prefetch_payload_t<mem_desc_t, tile_desc, num_coop_sg_, arch_tag>;

    // Fixed prefetch_dtype, close this assertion
    // static_assert(sizeof(prefetch_dtype) >= 4,
    //         "prefetch dtype size should at least DW aligned");

public:
    static constexpr uint32_t scale_factor
            = sizeof(prefetch_dtype) / sizeof(dtype);
    uint32_t base_offset;
    prefetch_dtype *base_ptr;
    uint32_t pitch_in_bytes;

    inline prefetch_payload_t(const this_payload_t &rhs) {
        this->base_offset = rhs.base_offset;
        this->base_ptr = rhs.base_ptr;
        this->pitch_in_bytes = rhs.pitch_in_bytes;
    }

    inline prefetch_payload_t() = default;

    inline this_payload_t &operator=(const this_payload_t &rhs) {
        this->base_offset = rhs.base_offset;
        this->base_ptr = rhs.base_ptr;
        this->pitch_in_bytes = rhs.pitch_in_bytes;
        return *this;
    }

    inline prefetch_payload_t(mem_desc_t &mem_desc, uint32_t coop_id = 0) {
        pitch_in_bytes = mem_desc.shape.stride_x * sizeof(dtype);
        uint32_t offset_x = mem_desc.coord.x;
        uint32_t offset_y = mem_desc.coord.y;
        base_offset = offset_y * pitch_in_bytes + offset_x * sizeof(dtype);
        uint64_t ptr_temp = (uint64_t)mem_desc.base.base;
        base_ptr = (prefetch_dtype *)ptr_temp
                + (coop_id % num_coop_sg) * mem_tile_size_x;
    }

    inline prefetch_payload_t(dtype *p, int surface_width, int surface_height,
            int surface_pitch, int surface_offset_x, int surface_offset_y,
            uint32_t coop_id = 0) {
        pitch_in_bytes = surface_pitch * sizeof(dtype);
        uint32_t offset_x = surface_offset_x;
        uint32_t offset_y = surface_offset_y;
        base_offset = offset_y * pitch_in_bytes + offset_x * sizeof(dtype);
        base_ptr = (prefetch_dtype *)p
                + (coop_id % num_coop_sg) * mem_tile_size_x;
    }

    template <tdesc_update_dir update_dir = tdesc_update_dir::x_dir>
    __XETLA_API void update_tdesc(int offset) {
        if constexpr (update_dir == tdesc_update_dir::x_dir) {
            base_offset += offset * sizeof(dtype);
        } else {
            base_offset += offset * pitch_in_bytes;
        }
    }
};

/// @brief Is to describe the memory infomation to prefetch data to cache
/// data located in shared local memory, nothing will do.
/// @tparam tile_desc_ Is the tile descriptor
/// @tparam dtype Is the data type
/// @tparam mem_layout Is the data layout
/// @tparam num_coop_sg_ Is the thread nums to prefetch data
template <typename dtype_, typename tile_desc_, mem_layout mem_layout_,
        uint32_t alignment_, uint32_t num_coop_sg_, gpu_arch arch_tag_>
struct prefetch_payload_t<
        mem_desc_t<dtype_, mem_layout_, mem_space::local, alignment_>,
        tile_desc_, num_coop_sg_, arch_tag_,
        std::enable_if_t<(arch_tag_ == gpu_arch::Xe)>> {
    using dtype = dtype_;
    using mem_desc_t
            = mem_desc_t<dtype_, mem_layout_, mem_space::local, alignment_>;
    using tile_desc = tile_desc_;
    static constexpr mem_space memory_space = mem_space::local;
    static constexpr mem_layout memory_layout = mem_layout_;
    static constexpr gpu_arch arch_tag = arch_tag_;

    inline prefetch_payload_t(mem_desc_t &mem_desc, uint32_t coop_id = 0) {}

    inline prefetch_payload_t(dtype *p, int surface_width, int surface_height,
            int surface_pitch, int surface_offset_x, int surface_offset_y,
            uint32_t coop_id = 0) {}

    template <tdesc_update_dir update_dir = tdesc_update_dir::x_dir>
    __XETLA_API void update_tdesc(int offset) {}
};

} // namespace gpu::xetla::subgroup
