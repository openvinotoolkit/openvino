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

#include "group/epilogue/api.hpp"
#include "group/epilogue/common.hpp"
#include "group/epilogue/epilogue_policy.hpp"

namespace gpu::xetla::group {

/// @addtogroup xetla_epilogue
/// @{

/// @brief Is the epilogue functor specialized for epilogue_policy_tile_op and Xe architecture.
template <typename tile_op_t_, typename tile_shape_, typename mem_desc_c_t_,
        gpu_arch arch_tag_>
class epilogue_t<epilogue_policy_tile_op<tile_op_t_, arch_tag_>, tile_shape_,
        mem_desc_c_t_,
        std::enable_if_t<(arch_tag_ == gpu_arch::Xe)
                && (mem_desc_c_t_::dim == 2)>> {
public:
    using epilogue_policy = epilogue_policy_tile_op<tile_op_t_, arch_tag_>;
    using tile_op_t = typename epilogue_policy::tile_op_t;
    using tile_shape = tile_shape_;
    using mem_desc_c_t = mem_desc_c_t_;
    static constexpr gpu_arch arch_tag = arch_tag_;
    static constexpr uint32_t barrier_count = 0;
    static constexpr uint32_t slm_size = mem_desc_c_t::is_local
            ? tile_shape::wg_tile_size_x * tile_shape::wg_tile_size_y
            : 0;

    /// @brief Epilogue arguments.
    struct arguments_t {
        /// @brief Is tile_op arguments, could be a single
        /// tile_op argument or chained_tile_op_args.
        typename tile_op_t::arguments_t tile_op_args;

        /// @brief Constructs a new arguments t object.
        inline arguments_t() = default;

        /// @brief Constructs a new arguments t object.
        /// @param tile_op_args_ Is tile_op arguments, could be a single
        /// tile_op argument or chained_tile_op_args.
        inline arguments_t(typename tile_op_t::arguments_t tile_op_args_)
            : tile_op_args(tile_op_args_) {}
        // Be aware of the risks: Rule of three (copy constructor, copy assignment, destructor)
        // Please check if you need to add self-define destructor
        // inline ~arguments_t(){}
        inline arguments_t(const arguments_t &args)
            : tile_op_args(args.tile_op_args) {}

        inline arguments_t &operator=(const arguments_t &args) {
            this->tile_op_args = args.tile_op_args;
            return *this;
        }

        /// @brief Explicit initialization function.
        /// @param tile_op_args_ Is tile_op arguments, could be a single
        /// tile_op argument or chained_tile_op_args.
        inline void init(typename tile_op_t::arguments_t tile_op_args_) {
            tile_op_args = tile_op_args_;
        }
    };

private:
    using work_group_t = typename tile_shape::work_group_t;
    static constexpr uint32_t sg_tile_m = tile_shape::sg_tile_size_y;
    static constexpr uint32_t sg_tile_n = tile_shape::sg_tile_size_x;
    static constexpr uint32_t wg_size_x = tile_shape::wg_size_x;
    static constexpr uint32_t wg_size_y = tile_shape::wg_size_y;
    using dtype_c = typename mem_desc_c_t::dtype;
    static constexpr mem_layout mem_layout_c = mem_desc_c_t::layout;
    static constexpr mem_space mem_space_c = mem_desc_c_t::space;

    /// @brief Updates tile base descriptor based on the tid.
    __XETLA_API static void update_sg_tile_tdesc(
            work_group_t &g, mem_desc_c_t &mem_desc_c) {
        int32_t sg_idx = g.get_id() % wg_size_x;
        int32_t sg_idy = g.get_id() / wg_size_x;
        int32_t tile_offset_n = sg_idx * sg_tile_n;
        int32_t tile_offset_m = sg_idy * sg_tile_m;
        mem_desc_c.update_coord(tile_offset_n, tile_offset_m);
    }

public:
    static constexpr msg_type msg_type_c
            = (mem_space_c == mem_space::global ? msg_type::block_2d
                                                : msg_type::scatter);

    /// @brief Default epilogue.
    /// 1) Call tile_op/chained_tile_op 2) Convert dtype_acc to dtype_c
    /// 3) Overwrite/reduce_sum to memory.
    /// @tparam matAcc_t Is the type of the input tile.
    /// @param g Is the workgroup of the current tile.
    /// @param matAcc Is the input tile.
    /// @param mem_desc_c Is the memory description of matC, including base, shape and coordinate.
    /// @param args Is the additional arguments for epilogue.
    /// @param slm_base Is the slm base address.
    /// @param nbarrier_base Is the named barrier base.
    template <typename matAcc_t>
    __XETLA_API KERNEL_FUNC void operator()(work_group_t &g, matAcc_t &matAcc,
            mem_desc_c_t mem_desc_c, arguments_t args = {},
            uint32_t slm_base = 0, uint32_t nbarrier_base = 0) {
        using mat_tile_desc = typename matAcc_t::tile_desc;
        using matC_t = subgroup::tile_t<dtype_c, mat_tile_desc>;
        using matC_payload_t = subgroup::mem_payload_t<mem_desc_c_t,
                mat_tile_desc, msg_type_c, arch_tag>;
        update_sg_tile_tdesc(g, mem_desc_c);
        tile_op_t tile_op;
        tile_op(matAcc, mem_desc_c.coord, args.tile_op_args, slm_base,
                nbarrier_base);
        matC_t matC;
        matC_payload_t matC_payload(mem_desc_c);
        subgroup::elemwise_cvt(matC, matAcc);
        subgroup::tile_store<cache_hint::streaming, cache_hint::write_back>(
                matC, matC_payload);
    }
};

/// @brief Is the epilogue functor specialized for epilogue_policy_tile_op and Xe architecture.
template <typename tile_op_t_, typename tile_shape_, typename mem_desc_c_t_,
        gpu_arch arch_tag_>
class epilogue_t<epilogue_policy_tile_op<tile_op_t_, arch_tag_>, tile_shape_,
        mem_desc_c_t_,
        std::enable_if_t<(arch_tag_ == gpu_arch::Xe)
                && (mem_desc_c_t_::dim == 4)>> {
public:
    using epilogue_policy = epilogue_policy_tile_op<tile_op_t_, arch_tag_>;
    using tile_op_t = typename epilogue_policy::tile_op_t;
    using tile_shape = tile_shape_;
    using mem_desc_c_t = mem_desc_c_t_;
    static constexpr gpu_arch arch_tag = arch_tag_;
    static constexpr uint32_t barrier_count = 0;
    static constexpr uint32_t slm_size = 0;

    /// @brief Epilogue arguments.
    struct arguments_t {
        /// @brief Is tile_op arguments, could be a single
        /// tile_op argument or chained_tile_op_args.
        typename tile_op_t::arguments_t tile_op_args;

        /// @brief Constructs a new arguments t object.
        inline arguments_t() = default;

        /// @brief Constructs a new arguments t object.
        /// @param tile_op_args_ Is tile_op arguments, could be a single
        /// tile_op argument or chained_tile_op_args.
        inline arguments_t(typename tile_op_t::arguments_t tile_op_args_)
            : tile_op_args(tile_op_args_) {}
        // Be aware of the risks: Rule of three (copy constructor, copy assignment, destructor)
        // Please check if you need to add self-define destructor
        // inline ~arguments_t(){}
        inline arguments_t(const arguments_t &args)
            : tile_op_args(args.tile_op_args) {}

        inline arguments_t &operator=(const arguments_t &args) {
            this->tile_op_args = args.tile_op_args;
            return *this;
        }

        /// @brief Explicit initialization function.
        /// @param tile_op_args_ Is tile_op arguments, could be a single
        /// tile_op argument or chained_tile_op_args.
        inline void init(typename tile_op_t::arguments_t tile_op_args_) {
            tile_op_args = tile_op_args_;
        }
    };

private:
    using work_group_t = typename tile_shape::work_group_t;
    static constexpr uint32_t wg_tile_n = tile_shape::wg_tile_size_n;
    static constexpr uint32_t wg_tile_p = tile_shape::wg_tile_size_p;
    static constexpr uint32_t wg_tile_q = tile_shape::wg_tile_size_q;
    static constexpr uint32_t wg_tile_k = tile_shape::wg_tile_size_k;

    static constexpr uint32_t sg_tile_n = tile_shape::sg_tile_size_n;
    static constexpr uint32_t sg_tile_p = tile_shape::sg_tile_size_p;
    static constexpr uint32_t sg_tile_q = tile_shape::sg_tile_size_q;
    static constexpr uint32_t sg_tile_k = tile_shape::sg_tile_size_k;

    static constexpr uint32_t wg_size_n = tile_shape::wg_size_n;
    static constexpr uint32_t wg_size_p = tile_shape::wg_size_p;
    static constexpr uint32_t wg_size_q = tile_shape::wg_size_q;
    static constexpr uint32_t wg_size_k = tile_shape::wg_size_k;
    using dtype_c = typename mem_desc_c_t::dtype;
    static constexpr mem_layout mem_layout_c = mem_desc_c_t::layout;
    static constexpr mem_space mem_space_c = mem_desc_c_t::space;
    static constexpr msg_type msg_type_c
            = (mem_space_c == mem_space::global ? msg_type::block_2d
                                                : msg_type::scatter);

    /// @brief Updates tile base descriptor based on the tid.
    __XETLA_API static void update_sg_tile_tdesc(
            work_group_t &g, mem_desc_c_t &mem_desc_c) {
        int32_t sg_idk = g.get_id() % wg_size_k;
        int32_t sg_idq = (g.get_id() / wg_size_k) % wg_size_q;
        int32_t sg_idp = (g.get_id() / (wg_size_k * wg_size_q)) % wg_size_p;
        int32_t sg_idn = (g.get_id() / (wg_size_k * wg_size_q * wg_size_p))
                % wg_size_n;

        int32_t tile_offset_n = sg_idn * sg_tile_n;
        int32_t tile_offset_p = sg_idp * sg_tile_p;
        int32_t tile_offset_q = sg_idq * sg_tile_q;
        int32_t tile_offset_k = sg_idk * sg_tile_k;
        mem_desc_c.update_coord(
                tile_offset_k, tile_offset_q, tile_offset_p, tile_offset_n);
    }

public:
    static constexpr bool is_2d_block_c = (msg_type_c == msg_type::block_2d);

    /// @brief Default epilogue.
    /// 1) Call tile_op/chained_tile_op 2) Convert dtype_acc to dtype_c
    /// 3) Overwrite/reduce_sum to memory.
    /// @tparam matAcc_t Is the type of the input tile.
    /// @param g Is the workgroup of the current tile.
    /// @param matAcc Is the input tile.
    /// @param mem_desc_c Is the memory description of matC, including base, shape and coordinate.
    /// @param args Is the additional arguments for epilogue.
    /// @param slm_base Is the slm base address.
    /// @param nbarrier_base Is the named barrier base.
    template <typename matAcc_t, int _sg_tile_n = sg_tile_n,
            int _sg_tile_p = sg_tile_p>
    __XETLA_API KERNEL_FUNC void operator()(work_group_t &g,
            matAcc_t matAcc[_sg_tile_n][_sg_tile_p], mem_desc_c_t mem_desc_c,
            arguments_t args = {}, uint32_t slm_base = 0,
            uint32_t nbarrier_base = 0) {
        using mat_tile_desc = typename matAcc_t::tile_desc;
        using matC_t = subgroup::tile_t<dtype_c, mat_tile_desc>;
        using matC_payload_t = subgroup::mem_payload_t<
                mem_desc_t<dtype_c, mem_layout_c, mem_space_c>, mat_tile_desc,
                msg_type_c, arch_tag>;

        update_sg_tile_tdesc(g, mem_desc_c);

#pragma unroll
        for (uint32_t n = 0; n < _sg_tile_n; n++) {
#pragma unroll
            for (uint32_t p = 0; p < _sg_tile_p; p++) {
                tile_op_t tile_op;

                typename mem_desc_c_t::shape_t shape = mem_desc_c.shape;
                typename mem_desc_c_t::coord_t coord = mem_desc_c.coord;
                int32_t coord_y = (coord.w + n) * shape.z * shape.y
                        + (coord.z + p) * shape.y + coord.y;
                mem_coord_t<2> mem_desc_c_coord(mem_desc_c.coord.x, coord_y);

                tile_op(matAcc[n][p], mem_desc_c_coord, args.tile_op_args,
                        slm_base, nbarrier_base);

                matC_t matC;
                matC_payload_t matC_payload;
                matC_payload.init(mem_desc_c.get_tdesc());

                subgroup::elemwise_cvt<matC_t, matAcc_t>(matC, matAcc[n][p]);

                int32_t offset_n = mem_desc_c.get_base_offset_from_w(n);
                int32_t offset_p = mem_desc_c.get_base_offset_from_z(p);
                int mask_n = mem_desc_c.get_mask_from_w(n);
                int mask_p = mem_desc_c.get_mask_from_z(p);
                int32_t base_offset = (offset_n + offset_p) * sizeof(dtype_c);

                matC_payload.update_tdesc_base_address_masked(
                        base_offset, mask_n & mask_p);

                subgroup::tile_store<cache_hint::streaming,
                        cache_hint::write_back>(matC, matC_payload);
            }
        }
    }
};

/// @} xetla_epilogue

} // namespace gpu::xetla::group
