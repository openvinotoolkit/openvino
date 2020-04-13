/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "jit_avx512_common_lrn.hpp"
#include "jit_generator.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "jit_avx512_core_bf16cvt.hpp"


typedef float acc_data_t;

#define IRB_LOOP(statement)                     \
    for (int irb = 0; irb < loop_size; irb++) { \
        statement;                              \
    }

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;
using namespace data_type;

using namespace Xbyak;

struct nChw16c_across {
/*  version:
 *  -1: channels 0..15,
 *   1: channels C-16 .. C-1,
 *   0: other channels
 *   3: channels only for this kernel(without prev and next)
 */
    int H, W, version;
    nChw16c_across(int h, int w, int v) : H(h), W(w), version(v) {}
};

template <data_type_t d_type>
struct jit_avx512_common_lrn_fwd_t<d_type>::jit_avx512_common_lrn_kernel_f
    : public jit_generator {
    struct jit_args_fwd_t {
        const data_t *src;
        data_t *dst, *ws0, *ws1;
    };

    int xmm_size, zmm_size, buffer_block, buffer_nest_offset, src_prev_offset,
            vlen, reg_block;
    int HW, W;
    bool is_first;
    bool is_last;
    bool is_single;

    Reg64 src = rax;
    Reg64 dst = r8;
    Reg64 ws0 = rdx;
    Reg64 ws1 = rsi;
    Reg64 imm_addr64 = rbx;

    Zmm zalpha = zmm0;
    Xmm xalpha = xmm0;
    Zmm zk = zmm1;
    Xmm xk = xmm1;

    Reg64 param = abi_param1;
    Reg64 t = rsp;
    Reg64 hw = r9;
    Zmm bf16_emu_reserv_1 = Zmm(27);
    Zmm bf16_emu_reserv_2 = Zmm(28);
    Zmm bf16_emu_reserv_3 = Zmm(29);
    Reg64 bf16_emu_scratch = rax;
    Zmm bf16_emu_reserv_4 = Zmm(30);
    Zmm bf16_emu_reserv_5 = Zmm(31);

    const int xsrc_prev = 2;
    const int zsrc = 7;
    const int xsrc_next = 3;
    const int zc = 7;

    const int za = 2;
    const int zb = 3;
    const int zd = 5;
    const int ze = 6;
    const int zsum = 4;
    const int zdst = 2;
    const int zbase = 3;
    const int zsum2 = 5;

    prop_kind_t pk;
    int use_h_parallelism;

    float alpha, k;
    bf16_emulation_t *bf16_emu_;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_common_lrn_kernel_f)

    void (*ker)(jit_args_fwd_t *);
    void operator()(jit_args_fwd_t *arg) { ker(arg); }

    inline void compute_loop(int loop_size_param) {
        // loop_size - param for IRB_LOOP macro
        const int prf0_offt = 1 * reg_block;
        const int prf2_offt = 8 * reg_block;

        int loop_size = reg_block;

        auto xreg = [=](int irb, int i) { return Xmm(irb * 3 + i); };
        auto yreg = [=](int irb, int i) { return Ymm(irb * 7 + i); };
        auto zreg = [=](int irb, int i) { return Zmm(irb * 7 + i); };
        auto load_data = [=](Xmm reg, const Address p) {
            if (d_type == bf16) {
                vpmovzxwd(reg, p);
                vpslld(reg, reg, 0x10);
            } else
                vmovups(reg, p);
        };

        auto store_data = [=](const Address addr, Zmm zr, Ymm yr) {
            if (d_type == bf16) {
                if (mayiuse(avx512_core_bf16))
                    vcvtneps2bf16(yr, zr);
                else
                    bf16_emu_->r_vcvtneps2bf16(yr, zr);
                vmovdqu16(addr, yr);
            } else
                vmovups(addr, zr);
        };

        if (!is_first && !is_single) {
            IRB_LOOP(mic_prefetcht0(ptr[src + (irb + prf0_offt - HW)*vlen]));
            IRB_LOOP(mic_prefetcht2(ptr[src + (irb + prf2_offt - HW)*vlen]));
        }
        IRB_LOOP(mic_prefetcht0(
                EVEX_compress_addr(src, (irb + prf0_offt) * vlen)));
        IRB_LOOP(mic_prefetcht2(
                EVEX_compress_addr(src, (irb + prf2_offt) * vlen)));
        if (!is_last && !is_single) {
            IRB_LOOP(mic_prefetcht0(ptr[src + (irb + prf0_offt + HW)*vlen]));
            IRB_LOOP(mic_prefetcht2(ptr[src + (irb + prf2_offt + HW)*vlen]));
        }
        if (pk != prop_kind::forward_inference) {
            IRB_LOOP(mic_prefetcht0(
                    EVEX_compress_addr(ws0, (irb + prf0_offt) * vlen)));
            IRB_LOOP(mic_prefetcht2(
                    EVEX_compress_addr(ws0, (irb + prf2_offt) * vlen)));
        }
        IRB_LOOP(mic_prefetcht0(
                EVEX_compress_addr(dst, (irb + prf0_offt) * vlen)));
        IRB_LOOP(mic_prefetcht2(
                EVEX_compress_addr(dst, (irb + prf2_offt) * vlen)));
        if (pk != prop_kind::forward_inference) {
            IRB_LOOP(mic_prefetcht0(
                    EVEX_compress_addr(ws1, (irb + prf0_offt) * vlen)));
            IRB_LOOP(mic_prefetcht2(
                    EVEX_compress_addr(ws1, (irb + prf2_offt) * vlen)));
        }

        loop_size = loop_size_param;
        if (loop_size == 0)
            return;

        // --- loading source data to special buffer to form convenient data
        // layout for ACROSS lrn ---

        if (!is_first && !is_single) {
            IRB_LOOP(load_data(xreg(irb, xsrc_prev),
                    ptr[src + (irb - HW) * vlen + src_prev_offset]));
        }
        IRB_LOOP(load_data(
                zreg(irb, zsrc), EVEX_compress_addr(src, irb * vlen)));
        if (!is_last && !is_single) {
            IRB_LOOP(load_data(
                    xreg(irb, xsrc_next), ptr[src + (irb + HW) * vlen]));
        }

        if (!is_first && !is_single) {
            IRB_LOOP(
                    vmovups(ptr[t + irb * buffer_block], xreg(irb, xsrc_prev)));
        }
        IRB_LOOP(vmovups(EVEX_compress_addr(t, irb * buffer_block + xmm_size),
                zreg(irb, zsrc)));
        if (!is_last && !is_single) {
            IRB_LOOP(vmovups(ptr[t + irb * buffer_block + buffer_nest_offset],
                    xreg(irb, xsrc_next)));
        }

        // --- perform ACROSS lrn ---
        size_t acc_size = sizeof(acc_data_t);
        IRB_LOOP(vmovups(zreg(irb, za),
                EVEX_compress_addr(
                        t, irb * buffer_block + xmm_size - 2 * acc_size)));
        IRB_LOOP(vmovups(zreg(irb, zb),
                EVEX_compress_addr(
                        t, irb * buffer_block + xmm_size - acc_size)));
        IRB_LOOP(vmovups(zreg(irb, zd),
                EVEX_compress_addr(
                        t, irb * buffer_block + xmm_size + acc_size)));
        IRB_LOOP(vmovups(zreg(irb, ze),
                EVEX_compress_addr(
                        t, irb * buffer_block + xmm_size + 2 * acc_size)));

        assert(zc == zsrc);
        IRB_LOOP(vmulps(zreg(irb, zsum), zreg(irb, zc), zreg(irb, zc)));

        IRB_LOOP(vfmadd231ps(zreg(irb, zsum), zreg(irb, za), zreg(irb, za)));
        IRB_LOOP(vfmadd231ps(zreg(irb, zsum), zreg(irb, zb), zreg(irb, zb)));
        IRB_LOOP(vfmadd231ps(zreg(irb, zsum), zreg(irb, zd), zreg(irb, zd)));
        IRB_LOOP(vfmadd231ps(zreg(irb, zsum), zreg(irb, ze), zreg(irb, ze)));

        IRB_LOOP(vfmadd132ps(zreg(irb, zsum), zk, zalpha));

        IRB_LOOP(vmovaps(zreg(irb, zbase), zreg(irb, zsum)));

        IRB_LOOP(vmulps(zreg(irb, zsum2), zreg(irb, zsum), zreg(irb, zsum)));
        IRB_LOOP(vmulps(zreg(irb, zsum), zreg(irb, zsum), zreg(irb, zsum2)));

        IRB_LOOP(vsqrtps(zreg(irb, zsum), zreg(irb, zsum)));
        IRB_LOOP(vsqrtps(zreg(irb, zsum), zreg(irb, zsum)));

        const int ytmp = zsum2; // temporary ymm for f32->bf16 conversion
        if (pk != prop_kind::forward_inference) {
            // save intermediate results for lrn backward
            IRB_LOOP(store_data(EVEX_compress_addr(ws0, irb * vlen),
                    zreg(irb, zsum), yreg(irb, ytmp)));
        }
        IRB_LOOP(vdivps(zreg(irb, zdst), zreg(irb, zsrc), zreg(irb, zsum)));
        // storing to dst
        IRB_LOOP(store_data(EVEX_compress_addr(dst, irb * vlen),
                zreg(irb, zdst), yreg(irb, ytmp)));
        if (pk != prop_kind::forward_inference) {
            // calculate and save more intermediate results for lrn backward
            /* ws1 = zdst / zbase = zsrc / (zbase^1.75) */
            IRB_LOOP(
                    vdivps(zreg(irb, zsum), zreg(irb, zdst), zreg(irb, zbase)));
            IRB_LOOP(store_data(EVEX_compress_addr(ws1, irb * vlen),
                    zreg(irb, zsum), yreg(irb, ytmp)));
        }
    }

    jit_avx512_common_lrn_kernel_f(
        const struct nChw16c_across &J,
        prop_kind_t prop_kind,
        int use_h_parallel,
        float A,
        float K,
        void *code_ptr = nullptr,
        size_t code_size = 2 * Xbyak::DEFAULT_MAX_CODE_SIZE)
        : jit_generator(code_ptr, code_size)
        , pk(prop_kind)
        , use_h_parallelism(use_h_parallel)
        , alpha(A)
        , k(K)
        , bf16_emu_(nullptr)
    {
        vlen = d_type == bf16 ? 32 : 64;
        // some registers needed for conversion from bf16 to f32
        reg_block = (d_type == bf16 && !mayiuse(avx512_core_bf16)) ? 3 : 4;
        src_prev_offset = vlen - 4 * sizeof(data_t);

        xmm_size = 4 * sizeof(acc_data_t);
        zmm_size = 64;
        buffer_block = xmm_size + zmm_size + xmm_size;
        buffer_nest_offset = xmm_size + zmm_size;

        if (d_type == bf16 && !mayiuse(avx512_core_bf16)) {
            bf16_emu_ = new bf16_emulation_t(this, bf16_emu_reserv_1,
                    bf16_emu_reserv_2, bf16_emu_reserv_3, bf16_emu_scratch,
                    bf16_emu_reserv_4, bf16_emu_reserv_5);
            bf16_emu_->init_vcvtneps2bf16();
        }

        this->preamble();

#define GET_OFF(field) offsetof(jit_args_fwd_t, field)
        mov(src, ptr[param + GET_OFF(src)]);
        mov(dst, ptr[param + GET_OFF(dst)]);
        if (pk != prop_kind::forward_inference) {
            mov(ws0, ptr[param + GET_OFF(ws0)]);
            mov(ws1, ptr[param + GET_OFF(ws1)]);
        }
#undef GET_OFF
        is_first = J.version == -1 || J.version == -2;
        is_last  = J.version == +1 || J.version == -2;
        is_single = J.version == 3;

        W = J.W;
        HW = J.W*J.H;
        int LSB = use_h_parallelism ? W : HW;

        sub(t, reg_block * buffer_block);
        mov(imm_addr64, float2int(this->alpha));
        movq(xalpha, imm_addr64);
        vbroadcastss(zalpha, xalpha);

        mov(imm_addr64, float2int(this->k));
        movq(xk, imm_addr64);
        vbroadcastss(zk, xk);

        if (is_first || is_single) {
            vxorps(xmm2, xmm2, xmm2);
            for (int irb = 0; irb < reg_block; irb++) {
                vmovups(ptr[t + irb * buffer_block], xmm2);
            }
        }
        if (is_last || is_single) {
            vxorps(xmm2, xmm2, xmm2);
            for (int irb = 0; irb < reg_block; irb++) {
                vmovups(ptr[t + irb * buffer_block + buffer_nest_offset], xmm2);
            }
        }

        int LSREST = LSB % reg_block;
        int LS = LSB - LSREST;

        Label lrn_loop;

        if (LS > 0) {
            mov(hw, LS);

            L(lrn_loop);
            {
                compute_loop(reg_block);

                add(src, reg_block * vlen);
                add(dst, reg_block * vlen);
                if (pk != prop_kind::forward_inference) {
                    add(ws0, reg_block * vlen);
                    add(ws1, reg_block * vlen);
                }

                for (int irb = 0; irb < reg_block; irb++)
                    dec(hw);
                cmp(hw, 0);
                jne(lrn_loop, T_NEAR);
            }
        }

        compute_loop(LSREST);

        add(t, reg_block * buffer_block);
        this->postamble();

        ker = reinterpret_cast<decltype(ker)>(
                const_cast<uint8_t *>(this->getCode()));
    }
    ~jit_avx512_common_lrn_kernel_f() { delete bf16_emu_; }
};

template <data_type_t d_type>
status_t jit_avx512_common_lrn_fwd_t<d_type>::pd_t::init() {
    using namespace prop_kind;
    using namespace alg_kind;

    assert(engine()->kind() == engine_kind::cpu);

    const memory_desc_wrapper data_d(data_pd_.desc());
    bool ok = true
            && mayiuse(avx512_common)
            && one_of(desc()->prop_kind, forward_training, forward_inference)
            && !has_zero_dim_memory()
            && everyone_is(d_type, desc()->data_desc.data_type)
            && data_d.ndims() == 4
            && data_d.dims()[1] % vsize == 0
            && attr()->has_default_values();
    if (!ok)
        return unimplemented;

    if (desc()->prop_kind == forward_training) {
        memory_desc_t ws_d;
        dims_t ws_dims = { MB(), C(), H(), 2*W() };
        mkldnn_memory_desc_init(
                &ws_d, 4, ws_dims, d_type, memory_format::nChw16c);
        ws_pd_ = cpu_memory_t::pd_t(engine_, &ws_d);
    }

    bool args_ok_across = true
        && desc()->alg_kind == lrn_across_channels
        && desc()->local_size == 5
        && desc()->lrn_beta == 0.75
        && data_d.format() == nChw16c;

    return args_ok_across ? success : unimplemented;
}

template <data_type_t d_type>
jit_avx512_common_lrn_fwd_t<d_type>::jit_avx512_common_lrn_fwd_t(
        const pd_t *apd, const input_vector &inputs,
        const output_vector &outputs)
    : cpu_primitive_t(apd, inputs, outputs)
    , use_h_parallelism(0)
    , ker_(nullptr)
    , ker_first_(nullptr)
    , ker_last_(nullptr) {
    using namespace alg_kind;
    const int C = pd()->C();
    const int H = pd()->H();
    const int W = pd()->W();
    const int ls = pd()->desc()->local_size;
    const float alpha = pd()->desc()->lrn_alpha / ls;
    const float k = pd()->desc()->lrn_k;

    auto pk = pd()->desc()->prop_kind;

    use_h_parallelism = H > 28 ? 1 : 0;

    if (C / vsize == 1) {
        ker_ = new jit_avx512_common_lrn_kernel_f(
                nChw16c_across(H, W, 3), pk, use_h_parallelism, alpha, k);
    } else {
        ker_ = new jit_avx512_common_lrn_kernel_f(
                nChw16c_across(H, W, 0), pk, use_h_parallelism, alpha, k);
        ker_first_ = new jit_avx512_common_lrn_kernel_f(
                nChw16c_across(H, W, -1), pk, use_h_parallelism, alpha, k);
        ker_last_ = new jit_avx512_common_lrn_kernel_f(
                nChw16c_across(H, W, +1), pk, use_h_parallelism, alpha, k);
    }
}

template <data_type_t d_type>
jit_avx512_common_lrn_fwd_t<d_type>::~jit_avx512_common_lrn_fwd_t() {
    delete ker_;
    delete ker_first_;
    delete ker_last_;
}

template <data_type_t d_type>
void jit_avx512_common_lrn_fwd_t<d_type>::execute_forward() const {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t*>(this->memory(0));
    auto ws = reinterpret_cast<data_t*>(this->memory(1));

    const int N = pd()->MB();
    const int C = pd()->C();
    const int H = pd()->H();
    const int W = pd()->W();
    const int C16 = C / vsize;
    const size_t work_amount = use_h_parallelism ? N*C16*H : N*C16;

    parallel(0, work_amount, [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};

        balance211(work_amount, nthr, ithr, start, end);
        if (use_h_parallelism) {
            int n{0}, c16{0}, h{0};
            nd_iterator_init(start, n, N, c16, C16, h, H);
            for (size_t iwork = start; iwork < end; ++iwork) {
                auto offset
                        = n * C * H * W + c16 * H * W * vsize + h * W * vsize;
                auto ws_offset0 = n * C * H * 2 * W + c16 * H * 2 * W * vsize
                        + h * 2 * W * vsize;
                auto ws_offset1 = ws_offset0 + W*vsize;

                typename jit_avx512_common_lrn_kernel_f::jit_args_fwd_t args;
                args.src = &src[offset];
                args.dst = &dst[offset];
                args.ws0 = &ws[ws_offset0];
                args.ws1 = &ws[ws_offset1];

                if (C16 == 1)
                    (*ker_)(&args);
                else if (c16 == 0)
                    (*ker_first_)(&args);
                else if (c16 == C16 - 1)
                    (*ker_last_)(&args);
                else
                    (*ker_)(&args);
                nd_iterator_step(n, N, c16, C16, h, H);
            }
        } else {
            int n{0}, c16{0};
            nd_iterator_init(start, n, N, c16, C16);
            for (size_t iwork = start; iwork < end; ++iwork) {
                auto offset = n*C*H*W + c16*H*W*vsize;
                auto ws_offset0 = n*C*H*2*W + c16*H*2*W*vsize;
                auto ws_offset1 = ws_offset0 + H*W*vsize;

                typename jit_avx512_common_lrn_kernel_f::jit_args_fwd_t args;
                args.src = &src[offset];
                args.dst = &dst[offset];
                args.ws0 = &ws[ws_offset0];
                args.ws1 = &ws[ws_offset1];

                if (C16 == 1)
                    (*ker_)(&args);
                else if (c16 == 0)
                    (*ker_first_)(&args);
                else if (c16 == C16 - 1)
                    (*ker_last_)(&args);
                else
                    (*ker_)(&args);

                nd_iterator_step(n, N, c16, C16);
            }
        }
    });
}

template struct jit_avx512_common_lrn_fwd_t<data_type::f32>;
template struct jit_avx512_common_lrn_fwd_t<data_type::bf16>;

template <data_type_t d_type>
struct jit_avx512_common_lrn_bwd_t<d_type>::jit_avx512_common_lrn_kernel_f
    : public jit_generator {
    struct jit_args_bwd_t {
        const data_t *src, *diff_dst, *ws0, *ws1;
        data_t *diff_src;
    };

    int xmm_size, zmm_size, buffer_block, buffer_nest_offset, src_prev_offset,
            vlen, reg_block;
    int HW, W;
    bool is_first;
    bool is_last;
    bool is_single;

    Reg64 src = rax;
    Reg64 diffsrc = r8;
    Reg64 diffdst = r9;
    Reg64 workspace0 = rdx;
    Reg64 workspace1 = rsi;
    Reg64 imm_addr64 = rbx;

    Zmm znalphabeta = zmm0;
    Xmm xnalphabeta = xmm0;

    Reg64 param = abi_param1;
    Reg64 t = rsp;
    Reg64 hw = r10;
    Zmm bf16_emu_reserv_1 = Zmm(27);
    Zmm bf16_emu_reserv_2 = Zmm(28);
    Zmm bf16_emu_reserv_3 = Zmm(29);
    Reg64 bf16_emu_scratch = rax;
    Zmm bf16_emu_reserv_4 = Zmm(30);
    Zmm bf16_emu_reserv_5 = Zmm(31);

    const int xws1_prev = 1;
    const int xdiffdst_prev = 2;
    const int zws1 = 1;

    const int zsrc = 1;
    const int zdiffdst = 5;
    const int zdiffsrc = 6;

    const int xws1_next = 1;
    const int xdiffdst_next = 3;

    const int za = 1;
    const int zb = 2;
    const int zd = 3;
    const int ze = 4;
    const int zws0 = 2;

    float nalphabeta;

    int use_h_parallelism;
    bf16_emulation_t *bf16_emu_;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_common_lrn_kernel_f)

    void (*ker)(jit_args_bwd_t *);
    void operator()(jit_args_bwd_t *arg) { ker(arg); }

    inline void compute_loop(
            int loop_size_param, int prefetchL1, int prefetchL2) {
        // loop_size - param for IRB_LOOP macro
        int loop_size = loop_size_param;
        const int prf0_offt = 1 * reg_block;
        const int prf2_offt = 8 * reg_block;

        auto xreg = [=](int irb, int i) { return Xmm(irb * 6 + i); };

        auto zreg = [=](int irb, int i) { return Zmm(irb * 6 + i); };
        auto load_data = [=](Xmm reg, const Address p) {
            if (d_type == bf16) {
                vpmovzxwd(reg, p);
                vpslld(reg, reg, 0x10);
            } else
                vmovups(reg, p);
        };

        auto store_data = [=](bool nt, const Address addr, Zmm zr) {
            if (d_type == bf16) {
                Ymm yr = Ymm(zr.getIdx());
                if (mayiuse(avx512_core_bf16))
                    vcvtneps2bf16(yr, zr);
                else
                    bf16_emu_->r_vcvtneps2bf16(yr, zr);
                vmovdqu16(addr, yr);
            } else {
                if (nt)
                    uni_vmovntps(addr, zr);
                else
                    uni_vmovups(addr, zr);
            }
        };
        // ---- prefetching -------------------------------------------
        if (!is_first && !is_single) {
            if (prefetchL1)
                IRB_LOOP(mic_prefetcht0(
                        ptr[workspace1 + (irb + prf0_offt - 2 * HW) * vlen]));
            if (prefetchL1)
                IRB_LOOP(mic_prefetcht0(
                        ptr[diffdst + (irb + prf0_offt - HW) * vlen]));
        }

        if (prefetchL1)
            IRB_LOOP(mic_prefetcht0(ptr[src + (irb + prf0_offt) * vlen]));
        if (prefetchL2)
            IRB_LOOP(mic_prefetcht2(ptr[src + (irb + prf2_offt) * vlen]));

        if (prefetchL1)
            IRB_LOOP(
                    mic_prefetcht0(ptr[workspace1 + (irb + prf0_offt) * vlen]));

        if (prefetchL1)
            IRB_LOOP(mic_prefetcht0(ptr[diffdst + (irb + prf0_offt)*vlen]));

        if (!is_last && !is_single) {
            if (prefetchL1)
                IRB_LOOP(mic_prefetcht0(
                        ptr[workspace1 + (irb + prf0_offt + 2 * HW) * vlen]));
            if (prefetchL2)
                IRB_LOOP(mic_prefetcht2(
                        ptr[workspace1 + (irb + prf2_offt + 2 * HW) * vlen]));

            if (prefetchL1)
                IRB_LOOP(mic_prefetcht0(
                        ptr[diffdst + (irb + prf0_offt + HW) * vlen]));
            if (prefetchL2)
                IRB_LOOP(mic_prefetcht2(
                        ptr[diffdst + (irb + prf2_offt + HW) * vlen]));
        }
        if (prefetchL1)
            IRB_LOOP(
                    mic_prefetcht0(ptr[workspace0 + (irb + prf0_offt) * vlen]));
        if (prefetchL2)
            IRB_LOOP(
                    mic_prefetcht2(ptr[workspace0 + (irb + prf2_offt) * vlen]));
        // -----------------------------------------------------------

        if (loop_size_param == 0)
            return;

        if (!is_first && !is_single) {
            IRB_LOOP(load_data(xreg(irb, xws1_prev),
                    ptr[workspace1 + (irb - 2 * HW) * vlen + src_prev_offset]));
            IRB_LOOP(load_data(xreg(irb, xdiffdst_prev),
                    ptr[diffdst + (irb - HW) * vlen + src_prev_offset]));
            IRB_LOOP(vmulps(xreg(irb, xdiffdst_prev), xreg(irb, xdiffdst_prev),
                    xreg(irb, xws1_prev)));
        }

        IRB_LOOP(load_data(
                zreg(irb, zws1), EVEX_compress_addr(workspace1, irb * vlen)));
        IRB_LOOP(load_data(
                zreg(irb, zdiffdst), EVEX_compress_addr(diffdst, irb * vlen)));
        IRB_LOOP(vmulps(
                zreg(irb, zdiffsrc), zreg(irb, zdiffdst), zreg(irb, zws1)));

        if (!is_last && !is_single) {
            IRB_LOOP(load_data(xreg(irb, xws1_next),
                    ptr[workspace1 + (irb + 2 * HW) * vlen]));
            IRB_LOOP(load_data(xreg(irb, xdiffdst_next),
                    ptr[diffdst + (irb + HW) * vlen]));
            IRB_LOOP(vmulps(xreg(irb, xdiffdst_next), xreg(irb, xdiffdst_next),
                    xreg(irb, xws1_next)));
        }

        if (!is_first && !is_single) {
            IRB_LOOP(vmovups(
                    ptr[t + irb * buffer_block], xreg(irb, xdiffdst_prev)));
        }
        IRB_LOOP(vmovups(EVEX_compress_addr(t, irb * buffer_block + xmm_size),
                zreg(irb, zdiffsrc)));
        if (!is_last && !is_single) {
            IRB_LOOP(vmovups(ptr[t + irb * buffer_block + buffer_nest_offset],
                    xreg(irb, xdiffdst_next)));
        }
        size_t acc_size = sizeof(acc_data_t);
        IRB_LOOP(vmovups(zreg(irb, za),
                EVEX_compress_addr(
                        t, irb * buffer_block + xmm_size - 2 * acc_size)));
        IRB_LOOP(vmovups(zreg(irb, zb),
                EVEX_compress_addr(
                        t, irb * buffer_block + xmm_size - 1 * acc_size)));
        IRB_LOOP(vmovups(zreg(irb, zd),
                EVEX_compress_addr(
                        t, irb * buffer_block + xmm_size + 1 * acc_size)));
        IRB_LOOP(vmovups(zreg(irb, ze),
                EVEX_compress_addr(
                        t, irb * buffer_block + xmm_size + 2 * acc_size)));
        IRB_LOOP(vaddps(
                zreg(irb, zdiffsrc), zreg(irb, zdiffsrc), zreg(irb, za)));
        assert(zsrc == za);
        IRB_LOOP(load_data(
                zreg(irb, zsrc), EVEX_compress_addr(src, irb * vlen)));
        IRB_LOOP(vaddps(
                zreg(irb, zdiffsrc), zreg(irb, zdiffsrc), zreg(irb, zb)));
        IRB_LOOP(vaddps(
                zreg(irb, zdiffsrc), zreg(irb, zdiffsrc), zreg(irb, zd)));
        IRB_LOOP(vaddps(
                zreg(irb, zdiffsrc), zreg(irb, zdiffsrc), zreg(irb, ze)));
        IRB_LOOP(vmulps(zreg(irb, zsrc), zreg(irb, zsrc), znalphabeta));

        IRB_LOOP(load_data(
                zreg(irb, zws0), EVEX_compress_addr(workspace0, irb * vlen)));
        IRB_LOOP(vdivps(
                zreg(irb, zdiffdst), zreg(irb, zdiffdst), zreg(irb, zws0)));
        IRB_LOOP(vfmadd213ps(
                zreg(irb, zdiffsrc), zreg(irb, zsrc), zreg(irb, zdiffdst)));

        Label unaligned_store, end_store;
        test(diffsrc, vlen - 1);
        jnz(unaligned_store, T_NEAR);
        IRB_LOOP(store_data(true, EVEX_compress_addr(diffsrc, irb * vlen),
                zreg(irb, zdiffsrc)));
        jmp(end_store, T_NEAR);
        L(unaligned_store);
        {
            IRB_LOOP(store_data(false, EVEX_compress_addr(diffsrc, irb * vlen),
                    zreg(irb, zdiffsrc)));
        }
        L(end_store);
    }

    jit_avx512_common_lrn_kernel_f(
        const struct nChw16c_across &J,
        float A,
        float B,
        int use_h_parallel,
        void *code_ptr = nullptr,
        size_t code_size = 1 * Xbyak::DEFAULT_MAX_CODE_SIZE)
        : jit_generator(code_ptr, code_size)
        , nalphabeta(-2*A*B)
        , use_h_parallelism(use_h_parallel)
        , bf16_emu_(nullptr)
    {
        vlen = d_type == bf16 ? 32 : 64;
        reg_block = 3;
        src_prev_offset = vlen - 4 * sizeof(data_t);

        xmm_size = 4 * sizeof(acc_data_t);
        zmm_size = 64;
        buffer_block = xmm_size + zmm_size + xmm_size;
        buffer_nest_offset = xmm_size + zmm_size;

        if (d_type == bf16 && !mayiuse(avx512_core_bf16)) {
            bf16_emu_ = new bf16_emulation_t(this, bf16_emu_reserv_1,
                    bf16_emu_reserv_2, bf16_emu_reserv_3, bf16_emu_scratch,
                    bf16_emu_reserv_4, bf16_emu_reserv_5);
            bf16_emu_->init_vcvtneps2bf16();
        }

        this->preamble();

#define GET_OFF(field) offsetof(jit_args_bwd_t, field)
        mov(src, ptr[param + GET_OFF(src)]);
        mov(diffdst, ptr[param + GET_OFF(diff_dst)]);
        mov(workspace0, ptr[param + GET_OFF(ws0)]);
        mov(workspace1, ptr[param + GET_OFF(ws1)]);
        mov(diffsrc, ptr[param + GET_OFF(diff_src)]);
#undef GET_OFF

        W = J.W;
        HW = J.H*J.W;
        int LSB = this->use_h_parallelism ? W : HW;

        sub(t, reg_block * buffer_block);
        mov(imm_addr64, float2int(this->nalphabeta));
        movq(xnalphabeta, imm_addr64);
        vbroadcastss(znalphabeta, xnalphabeta);

        is_first = J.version == -1 || J.version == -2;
        is_last  = J.version == +1 || J.version == +2;
        is_single = J.version == 3;

        if (is_first || is_single) {
            vxorps(xmm1, xmm1, xmm1);
            for (int irb = 0; irb < reg_block; irb++) {
                vmovups(ptr[t + irb * buffer_block], xmm1);
            }
        }
        if (is_last || is_single) {
            vxorps(xmm1, xmm1, xmm1);
            for (int irb = 0; irb < reg_block; irb++) {
                vmovups(ptr[t + irb * buffer_block + buffer_nest_offset], xmm1);
            }
        }

        int LSREST = LSB % reg_block;
        int LS = LSB - LSREST;

        Label lrn_loop;

        if (LS > 0) {
            mov(hw, LS);

            L(lrn_loop);
            {
                compute_loop(reg_block, 1, 1);

                add(src, reg_block * vlen);
                add(diffsrc, reg_block * vlen);
                add(diffdst, reg_block * vlen);
                add(workspace0, reg_block * vlen);
                add(workspace1, reg_block * vlen);

                for (int irb = 0; irb < reg_block; irb++)
                    dec(hw);
                cmp(hw, 0);
                jne(lrn_loop, T_NEAR);
            }
        }

        compute_loop(LSREST, 1, this->use_h_parallelism ? 0 : 1);

        add(t, reg_block * buffer_block);
        this->postamble();

        ker = reinterpret_cast<decltype(ker)>(
                const_cast<uint8_t *>(this->getCode()));
    }
    ~jit_avx512_common_lrn_kernel_f() { delete bf16_emu_; }
};

template <data_type_t d_type>
status_t jit_avx512_common_lrn_bwd_t<d_type>::pd_t::init() {
    using namespace prop_kind;
    using namespace alg_kind;

    assert(engine()->kind() == engine_kind::cpu);

    const memory_desc_wrapper data_d(data_pd_.desc());
    bool ok = true
            && mayiuse(avx512_common)
            && utils::one_of(desc()->prop_kind, backward, backward_data)
            && utils::everyone_is(d_type, desc()->data_desc.data_type)
            && !has_zero_dim_memory()
            && data_d.ndims() == 4
            && data_d.dims()[1] % vsize == 0
            && attr()->has_default_values();
    if (!ok)
        return unimplemented;

    memory_desc_t ws_d;
    dims_t ws_dims = { MB(), C(), H(), 2*W() };
    mkldnn_memory_desc_init(&ws_d, 4, ws_dims, d_type, memory_format::nChw16c);
    ws_pd_ = cpu_memory_t::pd_t(engine_, &ws_d);

    auto fwd_ws_d_ = hint_fwd_pd_->workspace_pd()->desc();
    bool ws_ok = true
        && fwd_ws_d_->ndims == ws_pd_.desc()->ndims
        && fwd_ws_d_->format == ws_pd_.desc()->format
        && fwd_ws_d_->data_type == ws_pd_.desc()->data_type;
    if (!ws_ok) return unimplemented;

    bool args_ok_across = true
        && desc()->alg_kind == lrn_across_channels
        && desc()->local_size == 5
        && desc()->lrn_beta == 0.75
        && data_d.format() == nChw16c;

    return args_ok_across ? success : unimplemented;
}

template <data_type_t d_type>
jit_avx512_common_lrn_bwd_t<d_type>::jit_avx512_common_lrn_bwd_t(
        const pd_t *apd, const input_vector &inputs,
        const output_vector &outputs)
    : cpu_primitive_t(apd, inputs, outputs)
    , use_h_parallelism(0)
    , ker_(nullptr)
    , ker_first_(nullptr)
    , ker_last_(nullptr) {
    const int C = pd()->C();
    const int H = pd()->H();
    const int W = pd()->W();
    const int ls = pd()->desc()->local_size;
    const float alpha = pd()->desc()->lrn_alpha / ls;
    const float beta = pd()->desc()->lrn_beta;

    use_h_parallelism = H > 28 ? 1 : 0;

    if (C / vsize == 1) {
        ker_ = new jit_avx512_common_lrn_kernel_f(
                nChw16c_across(H, W, 3), alpha, beta, use_h_parallelism);
    } else {
        ker_ = new jit_avx512_common_lrn_kernel_f(
                nChw16c_across(H, W, 0), alpha, beta, use_h_parallelism);
        ker_first_ = new jit_avx512_common_lrn_kernel_f(
                nChw16c_across(H, W, -1), alpha, beta, use_h_parallelism);
        ker_last_ = new jit_avx512_common_lrn_kernel_f(
                nChw16c_across(H, W, +1), alpha, beta, use_h_parallelism);
    }
}

template <data_type_t d_type>
jit_avx512_common_lrn_bwd_t<d_type>::~jit_avx512_common_lrn_bwd_t() {
    delete ker_;
    delete ker_first_;
    delete ker_last_;
}

template <data_type_t d_type>
void jit_avx512_common_lrn_bwd_t<d_type>::execute_backward() const {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto ws = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto diff_src = reinterpret_cast<data_t *>(this->memory(0));

    const int N = pd()->MB();
    const int C = pd()->C();
    const int H = pd()->H();
    const int W = pd()->W();
    const int C16 = C / vsize;
    const size_t work_amount = use_h_parallelism ? N*C16*H : N*C16;

    parallel(0, work_amount, [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};

        balance211(work_amount, nthr, ithr, start, end);
        if (use_h_parallelism) {
            int n{0}, c16{0}, h{0};
            nd_iterator_init(start, n, N,  h, H, c16, C16);
            for (size_t iwork = start; iwork < end; ++iwork) {
                auto offset
                        = n * C * H * W + c16 * H * W * vsize + h * W * vsize;
                auto ws_offset0 = n * C * H * 2 * W + c16 * H * 2 * W * vsize
                        + h * 2 * W * vsize;
                auto ws_offset1 = ws_offset0 + W * vsize;

                typename jit_avx512_common_lrn_kernel_f::jit_args_bwd_t args;
                args.src = &src[offset];
                args.diff_dst = &diff_dst[offset];
                args.ws0 = &ws[ws_offset0];
                args.ws1 = &ws[ws_offset1];
                args.diff_src = &diff_src[offset];

                if (C16 == 1)
                    (*ker_)(&args);
                else if (c16 == 0)
                    (*ker_first_)(&args);
                else if (c16 == C16 - 1)
                    (*ker_last_)(&args);
                else
                    (*ker_)(&args);
                nd_iterator_step(n, N, h, H, c16, C16);
            }
        } else {
            int n{0}, c16{0};
            nd_iterator_init(start, n, N, c16, C16);
            for (size_t iwork = start; iwork < end; ++iwork) {
                auto offset = n*C*H*W + c16*H*W*vsize;
                auto ws_offset0 = n*C*H*2*W + c16*H*2*W*vsize;
                auto ws_offset1 = ws_offset0 + H*W*vsize;

                typename jit_avx512_common_lrn_kernel_f::jit_args_bwd_t args;
                args.src = &src[offset];
                args.diff_dst = &diff_dst[offset];
                args.ws0 = &ws[ws_offset0];
                args.ws1 = &ws[ws_offset1];
                args.diff_src = &diff_src[offset];

                if (C16 == 1)
                    (*ker_)(&args);
                else if (c16 == 0)
                    (*ker_first_)(&args);
                else if (c16 == C16 - 1)
                    (*ker_last_)(&args);
                else
                    (*ker_)(&args);

                nd_iterator_step(n, N, c16, C16);
            }
        }
    });
}

template struct jit_avx512_common_lrn_bwd_t<data_type::f32>;
template struct jit_avx512_common_lrn_bwd_t<data_type::bf16>;

}
}
}
