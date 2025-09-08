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
        mem_desc_c_t_, std::enable_if_t<(arch_tag_ == gpu_arch::Xe)>> {
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

/// @} xetla_epilogue

} // namespace gpu::xetla::group
