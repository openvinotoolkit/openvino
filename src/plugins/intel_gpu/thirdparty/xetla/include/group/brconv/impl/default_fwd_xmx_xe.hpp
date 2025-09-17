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

#include "group/brconv/common.hpp"
#include "group/brconv/compute_policy.hpp"

namespace gpu::xetla::group {

/// @addtogroup xetla_gemm
/// @{

/// @brief Is the brconv functor for Xe architecture and matrix engine.
template <typename compute_attr_, typename perf_tuning_knob_,
        gpu_arch arch_tag_, typename tile_shape_, typename brconv_filter_attr_,
        typename mem_desc_src_t_, typename mem_desc_weight_t_>
class brconv_fwd_t<
        compute_policy_default_xmx<compute_attr_, perf_tuning_knob_, arch_tag_>,
        tile_shape_, brconv_filter_attr_, mem_desc_src_t_, mem_desc_weight_t_,
        std::enable_if_t<((arch_tag_ == gpu_arch::Xe)
                        )>> {
public:
    using mem_desc_src_t = mem_desc_src_t_;
    using mem_desc_weight_t = mem_desc_weight_t_;
    using tile_shape = tile_shape_;
    using filter_attr = brconv_filter_attr_;
    using compute_policy = compute_policy_default_xmx<compute_attr_,
            perf_tuning_knob_, arch_tag_>;

    static constexpr uint32_t accum_step = compute_policy::k_stride;
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

    static constexpr uint32_t fh = filter_attr::fh;
    static constexpr uint32_t fw = filter_attr::fw;
    static constexpr uint32_t pad_h = filter_attr::pad_h;
    static constexpr uint32_t pad_w = filter_attr::pad_w;
    static constexpr uint32_t stride_h = filter_attr::stride_h;
    static constexpr uint32_t stride_w = filter_attr::stride_w;
    static constexpr uint32_t dilation_h = filter_attr::dilation_h;
    static constexpr uint32_t dilation_w = filter_attr::dilation_w;

    using work_group_t = typename tile_shape::work_group_t;

    constexpr static gpu_arch arch_tag = compute_policy::arch_tag;

private:
    /******** set data type **********/
    using dtype_a = typename mem_desc_src_t::dtype;
    using dtype_b = typename mem_desc_weight_t::dtype;
    using dtype_mma_acc = typename compute_policy::dtype_mma_acc;
    using dtype_mma_a = typename compute_policy::dtype_mma_a;
    using dtype_mma_b = typename compute_policy::dtype_mma_b;

    /******** set memory attribute **********/
    static constexpr mem_layout mem_layout_a = mem_desc_src_t::layout;
    static constexpr mem_layout mem_layout_b = mem_desc_weight_t::layout;
    static constexpr mem_space mem_space_a = mem_desc_src_t::space;
    static constexpr mem_space mem_space_b = mem_desc_weight_t::space;

    static constexpr int stages = compute_policy::stages;
    // TODO: periodic sync is not implemented in brconv.
    // Implement it or remove sync_freq from here
    static constexpr int sync_freq = compute_policy::sync_freq;

    /******** set tile layout && worker scope **********/
    static constexpr uint32_t tile_size_x_a = accum_step;
    static constexpr uint32_t tile_size_y_a = sg_tile_q;
    static constexpr uint32_t tile_size_x_b = sg_tile_k;
    static constexpr uint32_t tile_size_y_b = accum_step;
    static constexpr uint32_t tile_size_x_c = sg_tile_k;
    static constexpr uint32_t tile_size_y_c = sg_tile_q;

    static constexpr uint32_t block_size_x_a
            = compute_policy::block_bytes_x_a / sizeof(dtype_mma_a);
    static constexpr uint32_t block_size_y_a
            = (compute_policy::block_size_y_a > tile_size_y_a)
            ? tile_size_y_a
            : compute_policy::block_size_y_a;
    static constexpr uint32_t block_size_x_b = compute_policy::block_size_x_b;
    static constexpr uint32_t block_size_y_b
            = compute_policy::block_bytes_y_b / sizeof(dtype_mma_b);

    /******** set tile  **********/
    // currently A is only row_major but we might need col_major
    // in fututre when implementing bwd by weights pass
    static constexpr bool is_col_major_a = false;
    static constexpr reg_layout reg_layout_a = reg_layout::tiled;
    using matA_tile_desc_t = subgroup::tile_desc_t<tile_size_x_a, tile_size_y_a,
            block_size_x_a, block_size_y_a, reg_layout_a>;
    using matA_t = subgroup::tile_t<dtype_a, matA_tile_desc_t>;
    using matA_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_a, mem_layout_a, mem_space_a>, matA_tile_desc_t,
            msg_type::block_2d, arch_tag>;

    using matA_acc_t = subgroup::tile_t<dtype_mma_a, matA_tile_desc_t>;

    using matA_prefetch_payload_t = subgroup::prefetch_payload_t<
            mem_desc_t<dtype_a, mem_layout_a, mem_space_a>,
            subgroup::tile_desc_t<tile_size_x_a,
                    (tile_size_y_a - 1) * stride_w + fw, 1, 1>,
            wg_size_k, arch_tag>;

    static constexpr reg_layout reg_layout_b
            = sizeof(dtype_b) < sizeof(uint32_t) ? reg_layout::vnni_tiled
                                                 : reg_layout::tiled;
    using matB_tile_desc_t = subgroup::tile_desc_t<tile_size_x_b, tile_size_y_b,
            block_size_x_b, block_size_y_b, reg_layout_b>;
    using matB_t = subgroup::tile_t<dtype_b, matB_tile_desc_t>;
    using matB_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_b, mem_layout_b, mem_space_b>, matB_tile_desc_t,
            subgroup::msg_type_v<matB_tile_desc_t, mem_space_b>, arch_tag>;

    using matB_acc_t = subgroup::tile_t<dtype_mma_b, matB_tile_desc_t>;

    using matB_prefetch_payload_t = subgroup::prefetch_payload_t<
            mem_desc_t<dtype_b, mem_layout_b, mem_space_b>,
            subgroup::tile_desc_t<tile_size_x_b, tile_size_y_b, 1, 1>,
            wg_size_n * wg_size_p * wg_size_q, arch_tag>;

public:
    using matAcc_tile_desc_t = subgroup::tile_desc_t<tile_size_x_c,
            tile_size_y_c, block_size_x_b, block_size_y_a, reg_layout::tiled>;
    using matAcc_t = subgroup::tile_t<dtype_mma_acc, matAcc_tile_desc_t>;

private:
    using tile_mma = subgroup::tile_mma_t<matAcc_t, matAcc_t, matB_acc_t,
            matA_acc_t, mma_engine::xmx, arch_tag>;

public:
    static constexpr uint32_t barrier_count = 0; //TODO: support it
    static constexpr uint32_t slm_size = 0; //TODO: support it

    static constexpr bool is_2d_block_a
            = matA_payload_t::message_type == msg_type::block_2d;
    static constexpr bool is_2d_block_b
            = matB_payload_t::message_type == msg_type::block_2d;

    /// @brief Arguments for brconv.
    /// User should prepare matA_base_desc, matB_base_desc, inner_loop_count...
    struct arguments_t {
        /// @brief Is the memory description of matA, including base, shape and coordinate.
        mem_desc_src_t matA_base_desc;
        /// @brief Is the memory description of matB, including base, shape and coordinate.
        mem_desc_weight_t matB_base_desc;
        /// @brief Is the total inner loop count required to compute the entire K-dim.
        uint32_t inner_loop_count;
    };

    static constexpr uint32_t num_tdesc_p = (sg_tile_p - 1) * stride_h + fh;

    // matA array is here and not in operator() since the bug with templates and arrays
    matA_t matA[sg_tile_n][sg_tile_p - 1 + fh];

    matA_prefetch_payload_t matA_prefetch_payload[sg_tile_n][num_tdesc_p];

    /// @brief Gets the subgroup-level tile output channel offset k.
    /// @param g Is the workgroup of the current tile.
    /// @return Subgroup-level tile output channel offset k.
    __XETLA_API static int get_matC_offset_k(work_group_t &g) {
        int32_t sg_idk = g.get_id() % wg_size_k;
        return sg_idk * sg_tile_k;
    }

    /// @brief Gets the subgroup-level tile height offset q.
    /// @param g Is the workgroup of the current tile.
    /// @return Subgroup-level tile height offset q.
    __XETLA_API static int get_matC_offset_q(work_group_t &g) {
        int32_t sg_idq = (g.get_id() / wg_size_k) % wg_size_q;
        return sg_idq * sg_tile_q;
    }

    /// @brief Gets the subgroup-level tile width offset p.
    /// @param g Is the workgroup of the current tile.
    /// @return Subgroup-level tile width offset p.
    __XETLA_API static int get_matC_offset_p(work_group_t &g) {
        int32_t sg_idp = (g.get_id() / (wg_size_k * wg_size_q)) % wg_size_p;
        return sg_idp * sg_tile_p;
    }

    /// @brief Gets the subgroup-level tile batch size offset n.
    /// @param g Is the workgroup of the current tile.
    /// @return Subgroup-level tile batch size offset n.
    __XETLA_API static int get_matC_offset_n(work_group_t &g) {
        int32_t sg_idn = (g.get_id() / (wg_size_k * wg_size_q * wg_size_p))
                % wg_size_n;
        return sg_idn * sg_tile_n;
    }

private:
    template <typename payload_t>
    __XETLA_API static void matB_payload_set_addr(payload_t &matB_payload,
            mem_desc_weight_t &matB_base_desc, int kh, int kw) {
        int32_t in_x_out_channels_b_bytes = matB_base_desc.get_stride_x()
                * matB_base_desc.get_stride_y() * sizeof(dtype_b);

        matB_payload.set_tdesc_base_address((uint64_t)matB_base_desc.base.base);
        matB_payload.update_tdesc_base_address(
                in_x_out_channels_b_bytes * (fw * kh + kw));
    }

    static __XETLA_API void matA_prefetch_payload_update(
            matA_prefetch_payload_t &matA_prefetch_payload) {
        matA_prefetch_payload.template update_tdesc<tdesc_update_dir::x_dir>(
                accum_step);
    }

    static __XETLA_API void matB_prefetch_payload_update(
            matB_prefetch_payload_t &matB_prefetch_payload,
            mem_desc_weight_t &matB_base_desc, int iter, int &curr_kh,
            uint32_t b_coop_id = 0) {
        int32_t in_x_out_channels_b_bytes = matB_base_desc.get_stride_x()
                * matB_base_desc.get_stride_y() * sizeof(dtype_b);

        // here we calculate what iteration of IC, kw and kh corresponds to the current iter
        uint32_t ic_iter = iter / (fh * fw);
        uint32_t kw_iter = (iter / fw) % fw;
        uint32_t kh_iter = iter % fh;

        curr_kh = calc_next_kh(curr_kh, kh_iter);

        // reset prefetch_payload to start
        matB_payload_set_addr(
                matB_prefetch_payload, matB_base_desc, curr_kh, kw_iter);

        matB_prefetch_payload.set_offset(matB_base_desc.get_coord_x(),
                matB_base_desc.get_coord_y() + accum_step * ic_iter, b_coop_id);
    }

    __XETLA_API static constexpr int calc_next_kh(int curr_kh, int kh_iter) {
        // calculates next kh offset taking into account matA reuse scheme for strides > 1
        int next_kh = 0;
        if (stride_h + curr_kh < fh) {
            next_kh = curr_kh + stride_h;
        } else {
            next_kh = (curr_kh % stride_h) + 1;
        }

        // zero next_kh if there is new kh loop
        next_kh *= div_round_up(kh_iter, fh);
        return next_kh;
    }

    matA_payload_t matA_payload[sg_tile_n][sg_tile_p + fh - 1];

    // round up to multiple of 8
    __XETLA_API static constexpr uint32_t roundup_8(uint32_t x) {
        return ((x + 7) >> 3) << 3;
    }

    xetla_mask<num_tdesc_p> mask;

    uint64_t base_offsets_vec[sg_tile_n][num_tdesc_p];

    int curr_offset_x;
    int curr_offset_y;

    __XETLA_API void matA_payload_set_kh(
            matA_payload_t matA_payload[sg_tile_n][sg_tile_p + fh - 1],
            uint32_t kh, int n, int p) {
        matA_payload[n][p].set_offset(curr_offset_x, curr_offset_y);
        matA_payload[n][p].set_tdesc_base_address_masked(
                base_offsets_vec[n][p * stride_h + kh],
                mask[p * stride_h + kh]);
    }

    __XETLA_API void set_base_offset(
            uint64_t base_offset, uint32_t n, uint32_t p) {
        base_offsets_vec[n][p] = base_offset;
    }

    __XETLA_API void init(
            matA_payload_t matA_payload[sg_tile_n][sg_tile_p + fh - 1],
            mem_desc_src_t &matA_base_desc) {
        uint32_t IC = matA_base_desc.get_stride_x();

#pragma unroll
        for (uint32_t n = 0; n < sg_tile_n; n++) {
#pragma unroll
            for (uint32_t p = 0; p < sg_tile_p + fh - 1; p++) {
                matA_payload[n][p].init(matA_base_desc.get_tdesc());
                matA_payload[n][p].set_tdesc_width(stride_w * IC);
                matA_payload[n][p].set_tdesc_pitch(stride_w * IC);
            }
        }
    }

    __XETLA_API void matA_payload_set_kw(
            matA_payload_t matA_payload[sg_tile_n][sg_tile_p + fh - 1],
            mem_desc_src_t &matA_base_desc, uint32_t kw) {
        matA_payload_set_offsets(matA_payload, matA_base_desc, kw);
        matA_payload_set_height(matA_payload, matA_base_desc);
    }

private:
    __XETLA_API void matA_payload_set_offsets(
            matA_payload_t matA_payload[sg_tile_n][sg_tile_p + fh - 1],
            mem_desc_src_t &matA_base_desc, uint32_t kw) {

        int ic_offset = matA_base_desc.get_coord_x();
        int w_offset = matA_base_desc.get_coord_y();
        uint32_t IC = matA_base_desc.get_stride_x();

        curr_offset_x = ic_offset + modulo(kw - pad_w, stride_w) * IC;
        curr_offset_y = div_round_down(w_offset + kw, stride_w);
#pragma unroll
        for (uint32_t n = 0; n < sg_tile_n; n++) {
#pragma unroll
            for (uint32_t p = 0; p < sg_tile_p + fh - 1; p++) {

                matA_payload[n][p].set_offset(curr_offset_x, curr_offset_y);
            }
        }
    }

    __XETLA_API void matA_payload_set_height(
            matA_payload_t matA_payload[sg_tile_n][sg_tile_p + fh - 1],
            mem_desc_src_t &matA_base_desc) {

        uint32_t W = matA_base_desc.get_stride_y();
        uint32_t IC = matA_base_desc.get_stride_x();

        uint32_t min_height = W / stride_w;
        uint32_t max_height = min_height + 1;
        uint32_t height_change_offset_x = (W % stride_w) * IC;
        uint32_t is_min_height = height_change_offset_x <= curr_offset_x;
        uint32_t height = is_min_height ? min_height : max_height;

#pragma unroll
        for (uint32_t n = 0; n < sg_tile_n; n++) {
#pragma unroll
            for (uint32_t p = 0; p < sg_tile_p + fh - 1; p++) {
                matA_payload[n][p].set_tdesc_height(height);
            }
        }
    }

public:
    /// @brief Main execution function for brconv.
    /// The basic process is load data -> matrix multiply.
    /// @param g Is the workgroup of the current tile.
    /// @param matAcc Is the array of the accumulation buffers.
    /// @param args Is the brconv::arguments_t.
    /// @param slm_base Is the slm base address.
    /// @param nbarrier_base Is the named barrier base.
    __XETLA_API KERNEL_FUNC void operator()(work_group_t &g,
            matAcc_t matAcc[sg_tile_n][sg_tile_p], arguments_t args,
            uint32_t slm_base = 0, uint32_t nbarrier_base = 0) {
        int32_t sg_idk = g.get_id() % wg_size_k;
        int32_t sg_idq = (g.get_id() / wg_size_k) % wg_size_q;
        int32_t sg_idp = (g.get_id() / (wg_size_k * wg_size_q)) % wg_size_p;
        int32_t sg_idn = (g.get_id() / (wg_size_k * wg_size_q * wg_size_p))
                % wg_size_n;

        update_mem_desc(sg_idk, sg_idq, sg_idp, sg_idn, args);

#pragma unroll
        for (uint32_t n = 0; n < sg_tile_n; n++) {
#pragma unroll
            for (uint32_t p = 0; p < num_tdesc_p; p++) {

                matA_prefetch_payload[n][p].init(
                        args.matA_base_desc.get_tdesc(), sg_idk);
            }
        }

        init(matA_payload, args.matA_base_desc);

        // prepare matrix B load descriptor
        matB_t matB;
        matB_payload_t matB_payload;
        matB_prefetch_payload_t matB_prefetch_payload;

        matB_payload.init(args.matB_base_desc.get_tdesc());
        uint32_t b_coop_id = g.get_id() / wg_size_k;
        matB_prefetch_payload.init(args.matB_base_desc.get_tdesc(), b_coop_id);
        int matB_prefetch_iter = 0;
        int matB_prefetch_iter_kh = 0;

        // prepare matrix A load descriptors and masks

        // num_tdesc_p8 here is because of the issues in compiler with arrays and masks
        constexpr uint32_t num_tdesc_p8 = roundup_8(num_tdesc_p);
        xetla_vector<int32_t, num_tdesc_p8> offset_p;
        offset_p = xetla_vector_gen<int32_t, num_tdesc_p8>(0, 1);

        // xetla_vector_gen produce some error from cmtl.h if vector length < 8
        // so some workaround here
        xetla_vector<int32_t, sg_tile_n> offset_n;
#pragma unroll
        for (uint32_t i = 0; i < sg_tile_n; i++) {
            offset_n[i] = i;
        }

        xetla_vector<int32_t, num_tdesc_p> base_offset_p;
        xetla_vector<int32_t, sg_tile_n> base_offset_n;

        base_offset_p
                = args.matA_base_desc
                          .template get_base_offset_from_z<num_tdesc_p>(
                                  offset_p.template select<num_tdesc_p, 1>())
                * sizeof(dtype_a);

        mask = args.matA_base_desc.get_mask_from_z(offset_p)
                       .template select<num_tdesc_p, 1>();

        base_offset_n
                = args.matA_base_desc
                          .template get_base_offset_from_w<sg_tile_n>(offset_n)
                * sizeof(dtype_a);
#pragma unroll
        for (uint32_t n = 0; n < sg_tile_n; n++) {
#pragma unroll
            for (uint32_t p = 0; p < num_tdesc_p; p++) {
                int32_t base_offset = base_offset_n[n] + base_offset_p[p];

                set_base_offset(
                        base_offset + (uint64_t)args.matA_base_desc.base.base,
                        n, p);
                matA_prefetch_payload[n][p].update_tdesc_base_address_masked(
                        base_offset, mask[p]);
            }
        }

        // prefetch A before the main loop
#pragma unroll
        for (int i = 0; i < div_round_up(stages, (fh * fw)); i++) {

#pragma unroll
            for (uint32_t n = 0; n < sg_tile_n; n++) {
#pragma unroll
                for (uint32_t p = 0; p < num_tdesc_p; p++) {
                    subgroup::tile_prefetch(matA_prefetch_payload[n][p]);
                    matA_prefetch_payload_update(matA_prefetch_payload[n][p]);
                }
            }
        }

#pragma unroll
        for (int i = 0; i < stages; i++) {
            subgroup::tile_prefetch(matB_prefetch_payload);
            matB_prefetch_iter++;
            matB_prefetch_payload_update(matB_prefetch_payload,
                    args.matB_base_desc, matB_prefetch_iter,
                    matB_prefetch_iter_kh, b_coop_id);
        }

        for (uint32_t i = 0; i < args.inner_loop_count; i++) {

#pragma unroll
            for (uint32_t kw = 0; kw < fw; kw++) {
                matA_payload_set_kw(matA_payload, args.matA_base_desc, kw);
#pragma unroll
                for (int start_kh = 0; start_kh < stride_h; start_kh++) {

#pragma unroll
                    for (uint32_t n = 0; n < sg_tile_n; n++) {
#pragma unroll
                        for (uint32_t p = 0; p < sg_tile_p - 1; p++) {
                            matA_payload_set_kh(matA_payload, start_kh, n, p);
                            subgroup::tile_load(matA[n][p], matA_payload[n][p]);
                        }
                    }

#pragma unroll
                    for (uint32_t kh = 0; start_kh + kh * stride_h < fh; kh++) {
                        // check if it is the last filter iteration for kw and kh
                        if ((kw == fw - 1) && (start_kh == stride_h - 1)
                                && (start_kh + (kh + 1) * stride_h >= fh)) {
                            // Prefetch A for the next inner loop count iteration
#pragma unroll
                            for (uint32_t n = 0; n < sg_tile_n; n++) {
#pragma unroll
                                for (uint32_t p = 0; p < num_tdesc_p; p++) {
                                    subgroup::tile_prefetch(
                                            matA_prefetch_payload[n][p]);
                                    matA_prefetch_payload_update(
                                            matA_prefetch_payload[n][p]);
                                }
                            }
                        }

                        subgroup::tile_prefetch(matB_prefetch_payload);
                        matB_prefetch_iter++;
                        matB_prefetch_payload_update(matB_prefetch_payload,
                                args.matB_base_desc, matB_prefetch_iter,
                                matB_prefetch_iter_kh, b_coop_id);
                        SW_BARRIER();
#pragma unroll
                        for (uint32_t n = 0; n < sg_tile_n; n++) {
                            matA_payload_set_kh(matA_payload, start_kh, n,
                                    sg_tile_p - 1 + kh);
                            subgroup::tile_load(matA[n][sg_tile_p - 1 + kh],
                                    matA_payload[n][sg_tile_p - 1 + kh]);
                        }

                        matB_payload_set_addr(matB_payload, args.matB_base_desc,
                                start_kh + kh * stride_h, kw);
                        subgroup::tile_load(matB, matB_payload);

                        SW_BARRIER();
#pragma unroll
                        for (uint32_t n = 0; n < sg_tile_n; n++) {
#pragma unroll
                            for (uint32_t p = 0; p < sg_tile_p; p++) {
                                tile_mma::mma(matAcc[n][p], matAcc[n][p], matB,
                                        matA[n][p + kh]);
                            }
                        }
                        SW_BARRIER();
                    }
                }
                SW_BARRIER();
            }

            // update A for the next IC
            args.matA_base_desc.update_coord_x(accum_step);
            matA_payload_set_kw(matA_payload, args.matA_base_desc, 0);

            // update B for the next IC
            matB_payload.template update_tdesc<tdesc_update_dir::y_dir>(
                    accum_step);
            SW_BARRIER();
        }
        SW_BARRIER();
    }

private:
    /// @brief Updates tile base descriptor based on the tid.
    /// @param sg_idk [in] id of subgroup tile in dimension k.
    /// @param sg_idq [in] id of subgroup tile in dimension q.
    /// @param sg_idp [in] id of subgroup tile in dimension p.
    /// @param sg_idn [in] id of subgroup tile in dimension n.
    /// @param args [in|out] Includes base descriptors.
    __XETLA_API static void update_mem_desc(int32_t sg_idk, int32_t sg_idq,
            int32_t sg_idp, int32_t sg_idn, arguments_t &args) {

        int32_t tile_offset_k = sg_idk * sg_tile_k;
        int32_t tile_offset_w = sg_idq * sg_tile_q * stride_w;
        int32_t tile_offset_h = sg_idp * sg_tile_p * stride_h;
        int32_t tile_offset_n = sg_idn * sg_tile_n;
        args.matA_base_desc.update_coord(
                0, tile_offset_w, tile_offset_h, tile_offset_n);
        args.matB_base_desc.update_coord(tile_offset_k, 0, 0, 0);
    }
};
} // namespace gpu::xetla::group
