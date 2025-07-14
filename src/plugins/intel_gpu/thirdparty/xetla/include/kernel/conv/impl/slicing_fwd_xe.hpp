/*******************************************************************************
* Copyright (c) 2022-2024 Intel Corporation
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

#include "kernel/conv/api.hpp"
#include "kernel/conv/common.hpp"
#include "kernel/conv/dispatch_policy.hpp"
#include "kernel/gemm/dispatch_policy.hpp"

namespace gpu::xetla::kernel {

/// @addtogroup xetla_conv
/// @{

/// @brief Default conv functor, specialized for Xe architecture.
///
/// @tparam num_local_slicing_ Is the c dim split ratio within a group.
/// @tparam num_local_slicing_ Is the type of memory used for local slicing.
/// mem_space::local is for SLM, mem_space::global for scratchpad memory.
/// @tparam brconv_t_ Is the brconv functor to compose a convoultion.
/// @tparam epilogue_t_ Is the epilogue functor to, compose a convoultion.
template <int num_global_slicing_, int num_local_slicing_,
        mem_space local_slicing_mem_space_, typename brconv_t_,
        typename epilogue_t_, typename group_swizzle_>
class conv_fwd_t<dispatch_policy_slicing<group_swizzle_, num_global_slicing_,
                         num_local_slicing_, local_slicing_mem_space_>,
        brconv_t_, epilogue_t_> {
    using brconv_t = brconv_t_;
    using epilogue_t = epilogue_t_;
    using brconv_args_t = typename brconv_t::arguments_t;
    using epilogue_args_t = typename epilogue_t::arguments_t;
    using tile_shape = typename brconv_t::tile_shape;
    using group_swizzle_t = group_swizzle_;

    static constexpr uint32_t wg_tile_n = brconv_t::wg_tile_n;
    static constexpr uint32_t wg_tile_p = brconv_t::wg_tile_p;
    static constexpr uint32_t wg_tile_q = brconv_t::wg_tile_q;
    static constexpr uint32_t wg_tile_k = brconv_t::wg_tile_k;

    static constexpr uint32_t sg_tile_n = brconv_t::sg_tile_n;
    static constexpr uint32_t sg_tile_p = brconv_t::sg_tile_p;
    static constexpr uint32_t sg_tile_q = brconv_t::sg_tile_q;
    static constexpr uint32_t sg_tile_k = brconv_t::sg_tile_k;

    static constexpr uint32_t wg_size_n = brconv_t::wg_size_n;
    static constexpr uint32_t wg_size_p = brconv_t::wg_size_p;
    static constexpr uint32_t wg_size_q = brconv_t::wg_size_q;
    static constexpr uint32_t wg_size_k = brconv_t::wg_size_k;

    static constexpr uint32_t accum_step = brconv_t::accum_step;
    using work_group_t = typename brconv_t::work_group_t;
    static constexpr uint32_t work_group_size = work_group_t::size;

    static constexpr gpu_arch arch_tag = group_swizzle_t::arch_tag;
    static_assert(arch_tag == brconv_t::arch_tag,
            "arch_tag of brconv and conv should be the same");
    static_assert(arch_tag == epilogue_t::arch_tag,
            "arch_tag of epilogue and conv should be the same");
    static_assert(std::is_same<typename brconv_t::tile_shape,
                          typename epilogue_t::tile_shape>::value,
            "tile_shape should be the same");

    using mem_desc_src_t = typename brconv_t::mem_desc_src_t;
    using mem_desc_weight_t = typename brconv_t::mem_desc_weight_t;
    using mem_desc_out_t = typename epilogue_t::mem_desc_c_t;
    using src_base_t = typename mem_desc_src_t::base_t;
    using weight_base_t = typename mem_desc_weight_t::base_t;
    using out_base_t = typename mem_desc_out_t::base_t;
    using dtype_src = typename mem_desc_src_t::dtype;
    using dtype_weight = typename mem_desc_weight_t::dtype;
    using dtype_out = typename mem_desc_out_t::dtype;
    using matAcc_t = typename brconv_t::matAcc_t;
    using dtype_acc = typename matAcc_t::dtype;
    using mem_desc_scratchpad_t
            = mem_desc_t<dtype_acc, mem_layout::row_major, mem_space::global>;
    using scratchpad_base_t = typename mem_desc_scratchpad_t::base_t;

    using mem_desc_acc_t
            = mem_desc_t<dtype_acc, mem_layout::nhwc, mem_space::global, 8, 4>;
    using mem_desc_cnt_t
            = mem_desc_t<uint32_t, mem_layout::nhwc, mem_space::global, 8, 4>;

    using acc_base_t = typename mem_desc_acc_t::base_t;
    using cnt_base_t = typename mem_desc_cnt_t::base_t;

    static constexpr uint32_t num_local_slicing = num_local_slicing_;
    static constexpr uint32_t num_global_slicing = num_global_slicing_;

    static_assert((num_local_slicing > 0), "min slicing ratio is 1");
    static_assert((num_local_slicing & (num_local_slicing - 1)) == 0,
            "num_local_slicing should be power of 2!");

    static constexpr mem_space local_slicing_mem_space
            = local_slicing_mem_space_;
    using local_slicing_t
            = group::cooperative_reduce_t<reduce_op::sum, tile_shape, matAcc_t,
                    num_local_slicing, arch_tag, local_slicing_mem_space>;

    using mat_slice_t = typename local_slicing_t::mat_slice_t;
    static constexpr uint32_t num_slice_p = local_slicing_t::num_slice_p;
    static constexpr uint32_t num_slice_n = local_slicing_t::num_slice_n;

    static constexpr uint32_t brconv_nbarr_count = brconv_t::barrier_count;
    static constexpr uint32_t brconv_slm_size = brconv_t::slm_size;

    static constexpr uint32_t epilogue_nbarr_count = epilogue_t::barrier_count;
    static constexpr uint32_t epilogue_slm_size = epilogue_t::slm_size;

    static constexpr uint32_t local_slicing_nbarr_count
            = local_slicing_t::barrier_count;
    static constexpr uint32_t local_slicing_mem_size
            = local_slicing_t::mem_size;

    static constexpr uint32_t local_slicing_slm_size
            = local_slicing_mem_space == mem_space::local
            ? local_slicing_mem_size
            : 0;

    // counter_size=8 is because with atomics being uint32_t each counter would require 32 bytes,
    // meaning 2 counters share the cache line. We can do 1 counter - 1 cache line,
    // but some experiments showed that the perf is the same and with two counters sharing
    // the cache line we require x2 less memory for counters.
    // More counters sharing the same cache line then would block each other for updates.
    static constexpr uint32_t counter_size = 8;
    static constexpr uint32_t alignment = 8 / sizeof(dtype_acc);

    using tile_shape_cnt
            = group::tile_shape_t<wg_size_n, wg_size_p * num_local_slicing,
                    wg_size_q, wg_size_k, 1, num_local_slicing, 1, 1>;

    using global_group_reduce_t = group::global_reduce_t<reduce_op::sum,
            tile_shape, tile_shape_cnt, mem_desc_acc_t, mem_desc_cnt_t,
            num_global_slicing, counter_size, arch_tag>;

public:
    /// @brief conv arguments.
    /// This is the interface for users to pass the application-related runtime variables.
    struct arguments_t {
        /// @brief batch size.
        uint32_t n;
        /// @brief input height.
        uint32_t h;
        /// @brief input width.
        uint32_t w;
        /// @brief output channels.
        uint32_t k;
        /// @brief input channels.
        uint32_t c;
        /// @brief Is the base address of src.
        src_base_t src_base;
        /// @brief Is the base address of weight.
        weight_base_t weight_base;
        /// @brief Is the base address of out.
        out_base_t out_base;
        /// @brief Is the base address of scratchpad buffer.
        scratchpad_base_t scratchpad_base;
        /// @brief Is the base address of accumulation buffer.
        acc_base_t acc_base;
        /// @brief Is the base address of counter buffer.
        cnt_base_t cnt_base;
        /// @brief Is the epilogue arguments.
        epilogue_args_t epilogue_args;

        /// @brief Set for device copyable
        static constexpr bool host_callable = true;

        /// @brief output height.
        uint32_t p;
        /// @brief output width.
        uint32_t q;

        /// @brief Constructs arguments with initialization list.
        /// @param n batch size.
        /// @param h input height.
        /// @param w input width.
        /// @param k output channels.
        /// @param c input channels.
        /// @param src_base Is the base address of src.
        /// @param weight_base Is the base address of weight.
        /// @param out_base Is the base address of out.
        /// @param epilogue_args Is the epilogue arguments.
        inline arguments_t(uint32_t n_, uint32_t h_, uint32_t w_, uint32_t k_,
                uint32_t c_, src_base_t src_base_, weight_base_t weight_base_,
                out_base_t out_base_, scratchpad_base_t scratchpad_base_ = {},
                acc_base_t acc_base_ = {}, cnt_base_t cnt_base_ = {},
                epilogue_args_t epilogue_args_ = {})
            : n(n_)
            , h(h_)
            , w(w_)
            , k(k_)
            , c(c_)
            , src_base(src_base_)
            , weight_base(weight_base_)
            , out_base(out_base_)
            , scratchpad_base(scratchpad_base_)
            , acc_base(acc_base_)
            , cnt_base(cnt_base_)
            , epilogue_args(epilogue_args_) {
            p = group::detail::conv_i2o_spatial_map<brconv_t::fh,
                    brconv_t::stride_h, brconv_t::pad_h>(h);
            q = group::detail::conv_i2o_spatial_map<brconv_t::fw,
                    brconv_t::stride_w, brconv_t::pad_w>(w);
        };
    };

    /// @brief Gets named_barrier id consumption count.
    /// Users query and get a named_barrier id consumption count in compile time.
    /// @return The count of named barriers required.
    __XETLA_API static constexpr uint32_t get_barrier_count() {
        constexpr uint32_t count = brconv_nbarr_count * num_local_slicing
                + local_slicing_nbarr_count
                + epilogue_nbarr_count * num_local_slicing;
        static_assert(
                count <= 32, "The named_barrier count should be less than 32!");
        return count;
    }

    /// @brief Gets local memory size consumption.
    /// Users query and get a local memory consumption size in compile time.
    /// @return The size of local memory required.
    __XETLA_API static constexpr uint32_t get_slm_size() {
        constexpr uint32_t size = brconv_slm_size * num_local_slicing
                + local_slicing_slm_size
                + epilogue_slm_size * num_local_slicing;
        static_assert(size <= (128 * 1024),
                "The local memory size should be less than 128KB!");
        return size;
    };

private:
    matAcc_t matAcc[sg_tile_n][sg_tile_p];
    mat_slice_t mat_slice[num_slice_n][num_slice_p];

public:
    /// @brief Main execution function for convolution.
    /// The processing order is 1) set group-level base and boundary -> 2) brconv -> 3) epilogue.
    /// @param ei Is the execution item, returns execution related information, such as workgroup id, subgroup id...
    /// @param args Is the conv arguments for application-related runtime variables.
    /// @param slm_base Is the slm base address.
    /// @param nbarrier_base Is the named barrier base.
    __XETLA_API KERNEL_FUNC void operator()(sycl::nd_item<3> &item,
            const arguments_t &args, uint32_t slm_base = 0,
            uint32_t nbarrier_base = 0) {
        work_group_t g;
        g.init(item.get_local_linear_id() % work_group_size);
        uint32_t wg_id = item.get_local_linear_id() / work_group_size;

        // set up workgroup level coordinates and boundaries
        uint32_t start_k = item.get_group(2) * wg_tile_k;

        uint32_t n_tiles = div_round_up(args.n, wg_tile_n);
        uint32_t p_tiles = div_round_up(args.p, wg_tile_p);
        uint32_t q_tiles = div_round_up(args.q, wg_tile_q);
        uint32_t k_tiles = div_round_up(args.k, wg_tile_k);

        uint32_t start_n = (item.get_group(1) % n_tiles) * wg_tile_n;
        uint32_t start_p
                = ((item.get_group(1) / n_tiles) % p_tiles) * wg_tile_p;
        uint32_t start_q = ((item.get_group(1) / (n_tiles * p_tiles)) % q_tiles)
                * wg_tile_q;

        uint32_t start_c = 0;
        uint32_t boundary_c = args.c;
        uint32_t wg_c = args.c;
        if constexpr (num_global_slicing > 1) {
            wg_c = div_round_up(wg_c, num_global_slicing);
            start_c = start_c + item.get_group(0) * wg_c;
            boundary_c = (start_c + wg_c) > args.c ? args.c : (start_c + wg_c);
        }
        if constexpr (num_local_slicing > 1) {
            wg_c = div_round_up(wg_c, num_local_slicing);
            start_c += wg_id * wg_c;
            boundary_c = (start_c + wg_c) > args.c ? args.c : (start_c + wg_c);
        }
        uint32_t boundary_n = args.n;

        // A mem config
        uint32_t src_shape_x = boundary_c;
        uint32_t src_shape_y = args.w;
        uint32_t src_shape_z = args.h;
        uint32_t src_shape_w = boundary_n;

        uint32_t src_stride_x = args.c;
        uint32_t src_stride_y = args.w;
        uint32_t src_stride_z = args.h;

        int src_start_x = start_c;
        int src_start_y = start_q * brconv_t::stride_w - brconv_t::pad_w;
        int src_start_z = start_p * brconv_t::stride_h - brconv_t::pad_h;
        int src_start_w = start_n;

        // B mem config
        uint32_t weight_shape_x = args.k;
        uint32_t weight_shape_y = boundary_c;
        uint32_t weight_shape_z = brconv_t::fw;
        uint32_t weight_shape_w = brconv_t::fh;

        uint32_t weight_stride_x = args.k;
        uint32_t weight_stride_y = args.c;
        uint32_t weight_stride_z = brconv_t::fw;

        int weight_start_x = start_k;
        int weight_start_y = start_c;
        int weight_start_z = 0;
        int weight_start_w = 0;

        // C mem config
        uint32_t out_shape_x = args.k;
        uint32_t out_shape_y = args.q;
        uint32_t out_shape_z = args.p;
        uint32_t out_shape_w = boundary_n;

        // set up arguments
        int out_start_x = start_k;
        int out_start_y = start_q;
        int out_start_z = start_p;
        int out_start_w = start_n;

        static_assert(!mem_desc_src_t::is_local,
                "mem_desc_src_t: mem_space::local is not supported yet.");
        static_assert(!mem_desc_weight_t::is_local,
                "mem_desc_weight_t: mem_space::local is not supported "
                "yet.");
        static_assert(!mem_desc_out_t::is_local,
                "mem_desc_out_t: mem_space::local is not supported "
                "yet.");

        mem_desc_src_t md_a {{args.src_base},
                {src_shape_x, src_shape_y, src_shape_z, src_shape_w,
                        src_stride_x, src_stride_y, src_stride_z},
                {src_start_x, src_start_y, src_start_z, src_start_w}};

        mem_desc_weight_t md_b {{args.weight_base},
                {weight_shape_x, weight_shape_y, weight_shape_z, weight_shape_w,
                        weight_stride_x, weight_stride_y, weight_stride_z},
                {weight_start_x, weight_start_y, weight_start_z,
                        weight_start_w}};

        mem_desc_out_t md_c {{args.out_base},
                {out_shape_x, out_shape_y, out_shape_z, out_shape_w},
                {out_start_x, out_start_y, out_start_z, out_start_w}};

        uint32_t brconv_slm_base = slm_base + wg_id * brconv_slm_size;
        uint32_t brconv_nbarr_base = nbarrier_base + wg_id * brconv_nbarr_count;

        uint32_t local_slicing_slm_base
                = slm_base + num_local_slicing * brconv_slm_size;
        uint32_t local_slicing_nbarr_base
                = nbarrier_base + num_local_slicing * brconv_nbarr_count;

        uint32_t epilogue_slm_base
                = local_slicing_slm_base + local_slicing_slm_size;
        uint32_t epilogue_nbarr_base
                = local_slicing_nbarr_base + local_slicing_nbarr_count;

        dtype_acc *scratchpad_base = args.scratchpad_base.base
                + item.get_group_linear_id() * local_slicing_mem_size;

        typename local_slicing_t::reduce_memspace_t reduce_mem;
        if constexpr (local_slicing_mem_space == mem_space::local)
            reduce_mem = slm_base;
        else if constexpr (local_slicing_mem_space == mem_space::global)
            reduce_mem = scratchpad_base;

        uint32_t inner_loop_count = div_round_up(wg_c, accum_step);

        brconv_args_t brconv_args {md_a, md_b, inner_loop_count};
        brconv_t brconv;
        epilogue_t epilogue;

#pragma unroll
        for (uint32_t n = 0; n < brconv.sg_tile_n; n++) {
#pragma unroll
            for (uint32_t p = 0; p < brconv.sg_tile_p; p++) {
                matAcc[n][p].init(0);
            }
        }

        local_slicing_t local_slicing(wg_id);

        brconv(g, matAcc, brconv_args, brconv_slm_base, brconv_nbarr_base);

        local_slicing(
                g, mat_slice, matAcc, reduce_mem, local_slicing_nbarr_base);

        int32_t acc_start_x = start_k;
        int32_t acc_start_y = start_q;
        int32_t acc_start_z = start_p;
        int32_t acc_start_w = start_n;

        uint32_t acc_shape_x = args.k;
        uint32_t acc_shape_y = args.q;
        uint32_t acc_shape_z = args.p;
        uint32_t acc_shape_w = boundary_n;

        mem_desc_acc_t mem_desc_acc {{args.acc_base},
                {acc_shape_x, acc_shape_y, acc_shape_z, acc_shape_w},
                {acc_start_x, acc_start_y, acc_start_z, acc_start_w}};

        mem_desc_acc.update_coord(local_slicing.coop_offset_k,
                local_slicing.coop_offset_q, local_slicing.coop_offset_p,
                local_slicing.coop_offset_n);

        int32_t cnt_start_w = (item.get_group(1) % n_tiles)
                * tile_shape_cnt::wg_tile_size_n;
        int32_t cnt_start_z = ((item.get_group(1) / n_tiles) % p_tiles)
                        * tile_shape_cnt::wg_tile_size_p
                + local_slicing.coop_id;
        int32_t cnt_start_y
                = ((item.get_group(1) / (n_tiles * p_tiles)) % q_tiles)
                * tile_shape_cnt::wg_tile_size_q;
        int32_t cnt_start_x
                = item.get_group(2) * tile_shape_cnt::wg_tile_size_k;

        uint32_t cnt_shape_w = n_tiles * tile_shape_cnt::wg_tile_size_n;
        uint32_t cnt_shape_z = p_tiles * tile_shape_cnt::wg_tile_size_p;
        uint32_t cnt_shape_y = q_tiles * tile_shape_cnt::wg_tile_size_q;
        uint32_t cnt_shape_x = k_tiles * tile_shape_cnt::wg_tile_size_k;

        mem_desc_cnt_t mem_desc_cnt {{args.cnt_base},
                {cnt_shape_x, cnt_shape_y, cnt_shape_z, cnt_shape_w},
                {cnt_start_x, cnt_start_y, cnt_start_z, cnt_start_w}};

        global_group_reduce_t global_group_reduce;
        global_group_reduce
                .template operator()<mat_slice_t, sg_tile_n, num_slice_p>(
                        g, mat_slice, mem_desc_acc, mem_desc_cnt);
        if (global_group_reduce.is_last_group()) {

            md_c.update_coord(local_slicing.coop_offset_k,
                    local_slicing.coop_offset_q, local_slicing.coop_offset_p,
                    local_slicing.coop_offset_n);

            epilogue.template operator()<mat_slice_t, num_slice_n, num_slice_p>(
                    g, mat_slice, md_c, args.epilogue_args, epilogue_slm_base,
                    epilogue_nbarr_base);
        }
    }
};

/// @} xetla_conv

} // namespace gpu::xetla::kernel
