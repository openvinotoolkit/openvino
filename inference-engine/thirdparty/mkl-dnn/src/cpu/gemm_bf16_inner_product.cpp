/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "mkldnn_thread.hpp"

#include "gemm_bf16_inner_product.hpp"
#include "bfloat16_utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::data_type;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::primitive_kind;
using namespace memory_tracking::names;
using namespace mkldnn::impl::cpu::bf16_cvt_utils;

template<data_type_t store_type>
gemm_bf16_ip_pp_kernel_t<store_type>::gemm_bf16_ip_pp_kernel_t(
        size_t row_length, bool do_bias, bool do_relu)
    : ker_(nullptr), row_length_(row_length)
    , do_bias_(do_bias), do_relu_(do_relu)
    , def_unroll_(4), max_unroll_(12)
    , vlen_(cpu_isa_traits<avx512_common>::vlen / sizeof(float))
    , is_cpx_(false), bf16_emu_(nullptr)
{
    is_ajustable_row_length_ = row_length_ == 0;
    if (do_bias_ && row_length_ == 0)
        // Post-ops kernel w/ bias can't have ajustable row length
        return;

    if (is_ajustable_row_length_)
        row_length_ = vlen_ * max_unroll_;

    if (store_type == data_type::f32 && !do_bias_ && !do_relu_)
        // Nothing to do for post-ops, dst must be used directly as accumulator
        // in gemm call
        return;

    if (!mayiuse(avx512_core))
        // bfloat16 not supported for older CPUs
        return;

    is_cpx_ = mayiuse(avx512_core_bf16);

    if (!is_cpx_)
        bf16_emu_ = new bf16_emulation_t(this,
                            bf16_emu_reserv_1, bf16_emu_reserv_2,
                            bf16_emu_reserv_3, bf16_emu_reserv_4,
                            bf16_emu_reserv_5, bf16_emu_reserv_6);

    generate();
}

template<data_type_t store_type>
void gemm_bf16_ip_pp_kernel_t<store_type>::generate()
{
    using namespace Xbyak;
    using namespace utils;

    auto vreg_dst = [&](int idx) { return Zmm(2 + idx * 2 + 0); };
    auto vreg_dst_ymm = [&](int idx) { return Ymm(2 + idx * 2 + 0); };
    auto vreg_bias = [&](int idx) { return Zmm(2 + idx * 2 + 1); };

    preamble();

#define PARAM_OFF(x) offsetof(ker_args, x)
    mov(reg_dst, ptr[reg_param + PARAM_OFF(dst)]);
    mov(reg_acc, ptr[reg_param + PARAM_OFF(acc)]);
    mov(reg_bias, ptr[reg_param + PARAM_OFF(bias)]);
    mov(reg_len, ptr[reg_param + PARAM_OFF(len)]);
    mov(reg_row_offset, ptr[reg_param + PARAM_OFF(row_offset)]);
    vbroadcastss(vreg_nslope, ptr[reg_param + PARAM_OFF(nslope)]);
#undef PARAM_OFF

    if (do_relu_)
        vxorps(vreg_zero, vreg_zero, vreg_zero);

    // Load accumulated value (float), apply bias (if any) and relu (if any);
    // then convert to destination type and store
    auto compute = [&](size_t offset, int idx, bool apply_mask) {
        if (!is_cpx_)
            bf16_emu_->init_vcvtneps2bf16();

        auto acc_addr = ptr[reg_acc + offset * sizeof(acc_data_t)];

        auto vreg_dst_ = vreg_dst(idx);
        if (apply_mask) {
            vxorps(vreg_dst_, vreg_dst_, vreg_dst_);
            vreg_dst_ = vreg_dst_ | kreg_rem_mask;
        }
        vmovups(vreg_dst_, acc_addr);

        if (do_bias_) {
            auto bias_addr = ptr[reg_bias + offset * sizeof(acc_data_t)];
            auto vreg_bias_ = vreg_bias(idx);
            if (apply_mask) {
                vxorps(vreg_bias_, vreg_bias_, vreg_bias_);
                vreg_bias_ = vreg_bias_ | kreg_rem_mask;
            }

            vmovups(vreg_bias_, bias_addr);
            vaddps(vreg_dst(idx), vreg_dst(idx), vreg_bias(idx));
        }

        if (do_relu_) {
            vcmpps(kreg_relu_cmp, vreg_dst(idx), vreg_zero, _cmp_lt_os);
            vmulps(vreg_dst(idx) | kreg_relu_cmp, vreg_dst(idx), vreg_nslope);
        }

        auto dst_addr = ptr[reg_dst + offset * sizeof(store_data_t)];
        if (store_type == data_type::bf16) {
            // TODO: implement store by zmm registers for bf16
            auto vreg_dst_ymm_ = vreg_dst_ymm(idx);
            if (is_cpx_)
                vcvtneps2bf16(vreg_dst_ymm_, vreg_dst(idx));
            else
                bf16_emu_->r_vcvtneps2bf16(vreg_dst_ymm_, vreg_dst(idx));

            if (apply_mask)
                vreg_dst_ymm_ = vreg_dst_ymm_ | kreg_rem_mask;

            vmovdqu16(dst_addr, vreg_dst_ymm_);
        } else if (store_type == data_type::f32)
            vmovups(dst_addr, vreg_dst_);
        else
            assert(!"unimplemented");
    };

    // Advance all pointers by an immediate
    auto advance_ptrs_imm = [&](size_t offset) {
        add(reg_dst, offset * sizeof(store_data_t));
        add(reg_acc, offset * sizeof(acc_data_t));
        if (do_bias_)
            add(reg_bias, offset * sizeof(acc_data_t));
    };

    // Advance all pointers by a value stored in a register
    auto advance_ptrs_reg = [&](Reg64 offset) {
        lea(reg_dst, ptr[reg_dst + offset * sizeof(store_data_t)]);
        lea(reg_acc, ptr[reg_acc + offset * sizeof(acc_data_t)]);
        if (do_bias_)
            lea(reg_bias, ptr[reg_bias + offset * sizeof(acc_data_t)]);
    };

    // Rewind pointers that point to data that is indexed by output channel
    // (bias or per-oc scaling factors)
    auto rewind_ptrs = [&]() {
        if (do_bias_)
            sub(reg_bias, row_length_ * sizeof(acc_data_t));
    };

    //  For fwd w/ bias:
    //
    //      <----------------- row_length_ = OC -------------------->
    //
    // ^    +....................+----------------------------------+
    // |    :   not accessed     |          Prologue loop           |
    // |    +--------------------+----------------------------------+
    //      |                                                       |
    // M    |                 Main loop (unrolled)                  |
    // B    |                                                       |
    //      +--------------------------------+----------------------+
    // |    |       Epilogue loop            |      not accessed    :
    // v    +--------------------------------+......................+

    if (!is_ajustable_row_length_)
    {
        Label prologue_end;
        cmp(reg_row_offset, 0);
        je(prologue_end, T_NEAR);

        // Prologue loop
        {
            mov(reg_tmp, row_length_);
            sub(reg_tmp, reg_row_offset);
            cmp(reg_tmp, reg_len);
            cmovg(reg_tmp, reg_len);
            sub(reg_len, reg_tmp);

            Label prologue_loop, prologue_loop_tail, prologue_loop_end;
            cmp(reg_tmp, vlen_);
            jl(prologue_loop_tail, T_NEAR);
            L(prologue_loop); {
                compute(0, 0, false);
                advance_ptrs_imm(vlen_);
                sub(reg_tmp, vlen_);
                cmp(reg_tmp, vlen_);
                jge(prologue_loop, T_NEAR);
            }

            L(prologue_loop_tail);
            mov(reg_rem_mask, 1);
            shl(reg_rem_mask, cl); // cl == reg_tmp because
                                   // reg_tmp < vlen_ here
            sub(reg_rem_mask, 1);
            jz(prologue_loop_end, T_NEAR);

            kmovq(kreg_rem_mask, reg_rem_mask);
            compute(0, 0, true);
            advance_ptrs_reg(reg_tmp);

            L(prologue_loop_end);
            rewind_ptrs();
        }
        L(prologue_end);
    }

    // Main loop
    Label main_loop_end;
    {
        cmp(reg_len, row_length_);
        jle(main_loop_end, T_NEAR);

        Label main_loop;
        L(main_loop); {
            size_t row_loop, row_tail;
            if (row_length_ <= max_unroll_ * vlen_) {
                // Fully unroll small loops
                row_loop = 0;
                row_tail = row_length_;
            } else {
                row_loop = vlen_ * def_unroll_;
                row_tail = row_length_ % row_loop;
            }

            assert(!!row_loop || !!row_tail);

            if (row_tail % vlen_) {
                int vlen_tail = row_tail % vlen_;
                unsigned tail_mask = (1 << vlen_tail) - 1;
                mov(reg_tmp, tail_mask);
                kmovq(kreg_rem_mask, reg_tmp);
            }

            if (row_loop) {
                mov(reg_tmp, rnd_dn(row_length_, row_loop));
                Label oc_loop;
                L(oc_loop); {
                    for (size_t offset = 0; offset < row_loop; offset += vlen_)
                        compute(offset, offset / vlen_, false);
                    advance_ptrs_imm(row_loop);
                    sub(reg_tmp, row_loop);
                    jnz(oc_loop);
                }
            }

            if (row_tail) {
                for (size_t offset = 0; offset < row_tail; offset += vlen_) {
                    bool use_mask = (offset + vlen_) > row_tail;
                    compute(offset, offset / vlen_, use_mask);
                }
                advance_ptrs_imm(row_tail);
            }

            rewind_ptrs();
            sub(reg_len, row_length_);
            cmp(reg_len, row_length_);
            jge(main_loop, T_NEAR);
        }
    }
    L(main_loop_end);

    // Epilogue loop
    Label epilogue_end;
    {
        cmp(reg_len, 0);
        je(epilogue_end, T_NEAR);

        Label epilogue_loop, epilogue_loop_tail;
        cmp(reg_len, vlen_);
        jl(epilogue_loop_tail, T_NEAR);
        L(epilogue_loop); {
            compute(0, 0, false);
            sub(reg_len, vlen_);
            advance_ptrs_imm(vlen_);
            cmp(reg_len, vlen_);
            jge(epilogue_loop, T_NEAR);
        }

        L(epilogue_loop_tail);
        mov(reg_tmp, reg_len); // reg_tmp is rcx, and we need cl for the shift
        mov(reg_rem_mask, 1);
        shl(reg_rem_mask, cl); // reg_tmp == rcx and reg_tail < vlen_ == 16
        sub(reg_rem_mask, 1);
        jz(epilogue_end, T_NEAR);
        kmovq(kreg_rem_mask, reg_rem_mask);
        compute(0, 0, true);
    }

    L(epilogue_end);

    postamble();

    ker_ = getCode<decltype(ker_)>();
}

template struct gemm_bf16_ip_pp_kernel_t<data_type::f32>;
template struct gemm_bf16_ip_pp_kernel_t<data_type::bf16>;

template<data_type_t store_type>
void gemm_bf16_ip_pp_kernel_t<store_type>::operator ()(
        store_data_t *dst, const acc_data_t *acc,
        const acc_data_t *bias, float nslope,
        size_t start, size_t end)
{
    if (end <= start)
        return;

    if (ker_) {
        // JIT
        ker_args args;
        size_t row_offset = start % row_length_;
        args.dst = dst + start;
        args.acc = acc + start;
        args.bias = bias ? bias + row_offset : nullptr;
        args.nslope = nslope;
        args.len = end - start;
        args.row_offset = row_offset;
        ker_(&args);
    }
};

template <data_type_t dst_data_type>
void gemm_bf16_inner_product_fwd_t<dst_data_type>::execute_forward() const {
    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const acc_data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory());

    const int M = pd()->OC();
    const int N = pd()->MB();
    const int K = pd()->IC_total_padded();

    bool wei_tr = !utils::one_of(pd()->weights_pd()->desc()->format,
             hwio, dhwio, io);

    acc_data_t *acc = pd()->dst_is_acc_
        ? (acc_data_t *)dst
        : scratchpad().template get<acc_data_t>(key_iprod_int_dat_in_acc_dt);

    float alpha = 1.0, beta = 0.0;
    mkldnn_gemm_bf16bf16f32(wei_tr ? "T" : "N", "N", &M, &N, &K,
            &alpha, weights, wei_tr ? &K : &M, src, &K, &beta, acc, &M);

    if (!pd()->dst_is_acc_ || pd()->do_relu_ || pd()->with_bias()) {
        const auto &post_ops = pd()->attr()->post_ops_;
        float nslope = pd()->do_relu_ ? post_ops.entry_[0].eltwise.alpha : 0.f;
        parallel(0, [&](int ithr, int nthr) {
            size_t start, end;
            balance211((size_t)M * N, nthr, ithr, start, end);
            (*pp_kernel_)(dst, acc, bias, nslope, start, end);
        });
    }
}

template <data_type_t diff_src_data_type>
void gemm_bf16_inner_product_bwd_data_t<diff_src_data_type>::
    execute_backward_data() const
{
    auto diff_dst =
        reinterpret_cast<const diff_dst_data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<diff_src_data_t*>(this->memory());

    const int M = pd()->IC_total_padded();
    const int N = pd()->MB();
    const int K = pd()->OC();

    bool wei_tr = utils::one_of(pd()->weights_pd()->desc()->format,
             hwio, dhwio, io);

    acc_data_t *acc = pd()->diff_src_is_acc_
        ? (acc_data_t *)diff_src
        : scratchpad().template get<acc_data_t>(key_iprod_int_dat_in_acc_dt);

    float alpha = 1.0, beta = 0.0;
    mkldnn_gemm_bf16bf16f32(wei_tr ? "T" : "N", "N", &M, &N, &K, &alpha,
            weights, wei_tr ? &K : &M, diff_dst, &K, &beta, acc, &M);

    if (!pd()->diff_src_is_acc_) {
        parallel(0, [&](int ithr, int nthr) {
            size_t start, end;
            balance211((size_t)M * N, nthr, ithr, start, end);
            if (end > start)
                cvt_float_to_bfloat16((mkldnn_bfloat16_t *)&diff_src[start],
                    (const float *)&acc[start],
                    end - start);
        });
    }
}

template <data_type_t diff_wei_data_type>
void gemm_bf16_inner_product_bwd_weights_t<diff_wei_data_type>::
    execute_backward_weights() const
{
    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto diff_dst =
        reinterpret_cast<const diff_dst_data_t *>(this->input_memory(1));
    auto diff_weights = reinterpret_cast<diff_wei_data_t *>(this->memory(0));
    auto diff_bias = reinterpret_cast<acc_data_t *>(this->memory(1));

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_pd());
    const memory_desc_wrapper diff_bias_d(pd()->diff_weights_pd(1));

    diff_dst += diff_dst_d.blocking_desc().offset_padding;

    const int MB = pd()->MB();
    const int OC = pd()->OC();
    const int IC = pd()->IC_total_padded();

    bool wei_tr = utils::one_of(pd()->diff_weights_pd()->desc()->format,
             hwio, dhwio, io);

    const int M = wei_tr ? OC : IC;
    const int N = wei_tr ? IC : OC;
    const int K = MB;

    acc_data_t *acc = pd()->diff_wei_is_acc_
        ? (acc_data_t *)diff_weights
        : scratchpad().template get<acc_data_t>(key_iprod_int_dat_in_acc_dt);

    float alpha = 1.0, beta = 0.0;
    mkldnn_gemm_bf16bf16f32("N", "T", &M, &N, &K, &alpha,
            wei_tr ? diff_dst : src, &M, wei_tr ? src : diff_dst, &N, &beta,
            acc, &M);

    if (!pd()->diff_wei_is_acc_) {
        parallel(0, [&](int ithr, int nthr) {
            size_t start, end;
            balance211((size_t)M * N, nthr, ithr, start, end);
            if (end > start)
                cvt_float_to_bfloat16((mkldnn_bfloat16_t *)&diff_weights[start],
                    (const float *)&acc[start],
                    end - start);
        });
    }

    if (pd()->with_bias()) {
        diff_bias += diff_bias_d.blocking_desc().offset_padding;
        constexpr int blksize = 16;
        const int OC_blocks = OC / blksize;
        const int rem_OC = OC % blksize;
        acc_data_t *ddst_ws = scratchpad().template get<acc_data_t>(
            key_iprod_dst_bf16_convert_wsp);
        parallel(0, [&](const int ithr, const int nthr) {
            int oc_st{0}, oc_e{0};
            balance211(OC_blocks, nthr, ithr, oc_st, oc_e);
            oc_st = oc_st * blksize;
            oc_e = oc_e * blksize;

            PRAGMA_OMP_SIMD()
            for (int oc = oc_st; oc < oc_e; ++oc)
                diff_bias[oc] = 0.0f;

            for (int mb = 0; mb < MB; ++mb) {
                if (oc_e > oc_st)
                    cvt_bfloat16_to_float((float *)&ddst_ws[oc_st],
                        (const mkldnn_bfloat16_t *)&diff_dst[mb * OC + oc_st],
                        oc_e - oc_st);

                PRAGMA_OMP_SIMD()
                for (int oc = oc_st; oc < oc_e; ++oc)
                    diff_bias[oc] += ddst_ws[oc];
            }

            if (rem_OC != 0 && ithr == nthr-1) {
                int oc_st = OC_blocks * blksize;
                for (int oc = OC_blocks * blksize; oc < OC; oc++)
                    diff_bias[oc] = 0.0f;
                for (int mb = 0; mb < MB; ++mb) {
                    cvt_bfloat16_to_float((float *)&ddst_ws[oc_st],
                        (const mkldnn_bfloat16_t *)&diff_dst[mb * OC + oc_st],
                        OC - oc_st);

                    for (int oc = oc_st; oc < OC; oc++)
                        diff_bias[oc] += ddst_ws[oc];
                }
            }
        });
    }

}

template struct gemm_bf16_inner_product_fwd_t<data_type::f32>;
template struct gemm_bf16_inner_product_fwd_t<data_type::bf16>;
template struct gemm_bf16_inner_product_bwd_data_t<data_type::f32>;
template struct gemm_bf16_inner_product_bwd_data_t<data_type::bf16>;
template struct gemm_bf16_inner_product_bwd_weights_t<data_type::f32>;
template struct gemm_bf16_inner_product_bwd_weights_t<data_type::bf16>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
