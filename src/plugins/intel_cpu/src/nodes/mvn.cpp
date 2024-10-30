// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn.h"

#include <algorithm>
#include <string>
#include <vector>
#include <memory>

#include "fake_quantize.h"
#include "eltwise.h"
#include "dnnl_extension_utils.h"
#include "utils/bfloat16.hpp"
#include "openvino/core/parallel.hpp"
#include "emitters/plugin/x64/jit_load_store_emitters.hpp"
#include "emitters/plugin/x64/jit_bf16_emitters.hpp"

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_uni_eltwise.hpp"
#include "cpu/x64/injectors/jit_uni_depthwise_injector.hpp"
#include "cpu/x64/injectors/jit_uni_quantization_injector.hpp"
#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"

#include <openvino/opsets/opset6.hpp>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "utils/cpu_utils.hpp"

using namespace dnnl;

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_mvn_call_args, field)

namespace ov {
namespace intel_cpu {
namespace node {
namespace {

struct MVNKey {
    MVNAttrs mvnAttrs;
    dnnl::primitive_attr attr;

    size_t hash() const;
    bool operator==(const MVNKey& rhs) const;
};

size_t MVNKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;
    seed = hash_combine(seed, mvnAttrs.initAcrossChannels_);
    seed = hash_combine(seed, mvnAttrs.execAcrossChannels_);
    seed = hash_combine(seed, mvnAttrs.normalizeVariance_);
    seed = hash_combine(seed, mvnAttrs.epsValue_);
    seed = hash_combine(seed, mvnAttrs.epsMode_);
    seed = hash_combine(seed, mvnAttrs.src_prc.hash());
    seed = hash_combine(seed, mvnAttrs.dst_prc.hash());
    seed = hash_combine(seed, mvnAttrs.layout);
    seed = hash_combine(seed, get_attr_hash(*attr.get()));
    return seed;
}

bool MVNKey::operator==(const MVNKey& rhs) const {
    bool retVal = true;
    retVal = retVal &&
             mvnAttrs.initAcrossChannels_ == rhs.mvnAttrs.initAcrossChannels_ &&
             mvnAttrs.execAcrossChannels_ == rhs.mvnAttrs.execAcrossChannels_ &&
             mvnAttrs.normalizeVariance_ == rhs.mvnAttrs.normalizeVariance_ &&
             mvnAttrs.epsValue_ == rhs.mvnAttrs.epsValue_ &&
             mvnAttrs.epsMode_ == rhs.mvnAttrs.epsMode_ &&
             mvnAttrs.src_prc == rhs.mvnAttrs.src_prc &&
             mvnAttrs.dst_prc == rhs.mvnAttrs.dst_prc &&
             mvnAttrs.layout == rhs.mvnAttrs.layout;
    retVal = retVal && *attr.get() == *rhs.attr.get();
    return retVal;
}
} // namespace

#if defined(OPENVINO_ARCH_X86_64)

// some utility functions
static inline bool isFloatCompatible(ov::element::Type prc) {
    return one_of(prc, ov::element::f32, ov::element::bf16, ov::element::f16);
}

// 8/4/2/1 tile
static const int kTileNum = 4;

// 4-7 for src, 8-11 for sum, 12-15 for mean. 4 vector for 8/4/2/1 tiles
static inline int get_tile_vr_id(const int& step) {
    int vec_reg_id = 3;
    if (step == 8) {
        vec_reg_id = 0;
    } else if (step == 4) {
        vec_reg_id = 1;
    } else if (step == 2) {
        vec_reg_id = 2;
    }
    return vec_reg_id;
}

// normalize_variance = false : src->mean
// normalize_variance = true : src+mean->variance:sqr(x-mean)
template <cpu_isa_t isa>
struct jit_uni_mvn_mean_variance_kernel_f32 : public jit_uni_mvn_mean_variance_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_mvn_mean_kernel_f32)

    explicit jit_uni_mvn_mean_variance_kernel_f32(jit_mvn_config_params jcp) : jit_uni_mvn_mean_variance_kernel(jcp), jit_generator(jit_name()) {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        ov::element::Type dst_prc = isFloatCompatible(jcp_.src_prc) ? ov::element::f32 : ov::element::i32;
        load_emitter[VECTOR] = std::unique_ptr<jit_load_emitter>(new jit_load_emitter(this, isa, jcp_.src_prc, dst_prc, vector_step));
        load_emitter[TAIL8] = std::unique_ptr<jit_load_emitter>(new jit_load_emitter(this, isa, jcp_.src_prc, dst_prc, 8));
        load_emitter[TAIL4] = std::unique_ptr<jit_load_emitter>(new jit_load_emitter(this, isa, jcp_.src_prc, dst_prc, 4));
        load_emitter[TAIL2] = std::unique_ptr<jit_load_emitter>(new jit_load_emitter(this, isa, jcp_.src_prc, dst_prc, 2));
        load_emitter[TAIL1] = std::unique_ptr<jit_load_emitter>(new jit_load_emitter(this, isa, jcp_.src_prc, dst_prc, 1));
        load_emitter[TAIL8_FILL] = std::unique_ptr<jit_load_emitter>(new jit_load_emitter(this, isa, jcp_.src_prc, dst_prc, 8, ov::element::f32, true, "zero"));
        load_emitter[TAIL4_FILL] = std::unique_ptr<jit_load_emitter>(new jit_load_emitter(this, isa, jcp_.src_prc, dst_prc, 4, ov::element::f32, true, "zero"));
        load_emitter[TAIL2_FILL] = std::unique_ptr<jit_load_emitter>(new jit_load_emitter(this, isa, jcp_.src_prc, dst_prc, 2, ov::element::f32, true, "zero"));
        load_emitter[TAIL1_FILL] = std::unique_ptr<jit_load_emitter>(new jit_load_emitter(this, isa, jcp_.src_prc, dst_prc, 1, ov::element::f32, true, "zero"));

        this->preamble();
        mov(reg_table, l_table);
        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        if (jcp_.normalize_variance) {
            mov(reg_mean, ptr[reg_params + GET_OFF(mean)]);
            mov(reg_variance, ptr[reg_params + GET_OFF(variance)]);
            uni_vpxor(vmm_variance, vmm_variance, vmm_variance);
        } else {
            mov(reg_sum, ptr[reg_params + GET_OFF(sum)]);
            uni_vpxor(vmm_sum, vmm_sum, vmm_sum);
        }
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);
        mov(reg_rt_shape, ptr[reg_params + GET_OFF(rt_shape_size)]);

        if (jcp_.normalize_variance) {
            if (jcp_.layout == MVNLayoutType::mvn_planar || jcp_.across_channels) {
                uni_vbroadcastss(vmm_mean, ptr[reg_mean]);
            } else {
                uni_vmovups(vmm_mean, ptr[reg_mean]);
            }
        }

        size_t data_step = (isa == cpu::x64::sse41 && jcp_.layout == MVNLayoutType::mvn_block) ? vector_step * 2 : vector_step;
        src_stride = data_step * jcp_.src_data_size;

        load_pool_gpr_idxs = {static_cast<size_t>(reg_load_store_mask.getIdx()), static_cast<size_t>(reg_load_table.getIdx())};

        if (jcp_.layout == MVNLayoutType::mvn_planar) {
            worker_vector_unroll();
            // for tails. [0-15] for avx512, [0-7] for avx2, [0-3] for sse
            auto tails_func = [&](int tile_size) {
                worker_block(tile_size, true);
                add(reg_src, tile_size * jcp_.src_data_size);
            };
            worker_tails(reg_rt_shape, tails_func);
            // hsum+store
            if (!jcp_.normalize_variance && !isFloatCompatible(jcp_.src_prc))
                uni_vcvtdq2ps(vmm_sum, vmm_sum);
            Vmm vmm_dst = jcp_.normalize_variance ? vmm_variance : vmm_sum;
            reduce_sum_store_vmm(vmm_dst.getIdx());
        } else if (jcp_.layout == MVNLayoutType::mvn_by_channel) {
            if (jcp_.across_channels)
                nspc_ac_ker();
            else
                nspc_pc_ker();
        } else {
            block_ker();
        }

        this->postamble();

        for (size_t i = 0; i < LOAD_EMITTERS_NUM; i++)
            load_emitter[i]->emit_data();

        prepare_table();
    }

private:
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;

    const int vlen = cpu_isa_traits<isa>::vlen;
    const int vector_step = vlen / sizeof(float);

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_mean = r9;
    Xbyak::Reg64 reg_variance = r10;
    Xbyak::Reg64 reg_work_amount = r11;
    Xbyak::Reg64 reg_sum = reg_mean;
    Xbyak::Reg64 reg_params = abi_param1;
    Xbyak::Reg64 reg_load_table = r13;
    Xbyak::Reg64 reg_load_store_mask = r14;
    Xbyak::Reg64 reg_aux = r15;

    Xbyak::Reg64 reg_rt_shape = rbx;
    Xbyak::Reg64 reg_table = rsi;
    Xbyak::Label l_table;

    Vmm vmm_val = Vmm(1);
    Vmm vmm_mean = Vmm(2);
    Vmm vmm_variance = Vmm(3);
    Vmm vmm_sum = vmm_mean;
    Xbyak::Xmm xmm_aux1 = Xbyak::Xmm(4);
    Xbyak::Xmm xmm_aux2 = Xbyak::Xmm(5);
    Xbyak::Xmm xmm_aux3 = Xbyak::Xmm(6);
    Vmm vmm_zero = Vmm(0);
    // 8-15 for unloop

    Xbyak::Opmask k_mask = Xbyak::Opmask(7);

    size_t src_stride = 0;

    enum { VECTOR, TAIL8, TAIL4, TAIL2, TAIL1, TAIL8_FILL, TAIL4_FILL, TAIL2_FILL, TAIL1_FILL, LOAD_EMITTERS_NUM };
    std::unique_ptr<jit_load_emitter> load_emitter[LOAD_EMITTERS_NUM];
    std::vector<size_t> load_pool_gpr_idxs;

    const int tile_size[kTileNum] = {8, 4, 2, 1};

    // nspc across channel
    inline void nspc_ac_ker() {
        Xbyak::Label loop_label;
        Xbyak::Label loop_end_label;
        L(loop_label);
        {
            cmp(reg_work_amount, vector_step);
            jl(loop_end_label, T_NEAR);

            worker_full_size();
            add(reg_src, vector_step * jcp_.src_data_size);

            sub(reg_work_amount, vector_step);
            jmp(loop_label, T_NEAR);
        }
        L(loop_end_label);

        auto tails_func = [&](int tile_size) {
            worker_block(tile_size, true);
            add(reg_src, tile_size * jcp_.src_data_size);
        };
        worker_tails(reg_work_amount, tails_func);

        if (!jcp_.normalize_variance && !isFloatCompatible(jcp_.src_prc))
            uni_vcvtdq2ps(vmm_sum, vmm_sum);
        Vmm vmm_dst = jcp_.normalize_variance ? vmm_variance : vmm_sum;
        reduce_sum_store_vmm(vmm_dst.getIdx());
    }

    // nspc per channel with unroll
    inline void nspc_pc_ker() {
        // 4 unroll vector
        // r12, rax, rdx, rbp, r15, rcx and rdi is available
        // r13 is available as no fill need for this layout
        // reg_rt_shape is C
        Xbyak::Reg64 reg_unroll_size = r12;
        Xbyak::Reg64 reg_unroll_num = rax;
        Xbyak::Reg64 reg_vector_num = rbp;
        Xbyak::Reg64 reg_tail_num = r13;
        // size_t unroll_size = 4;
        mov(reg_unroll_size, 4);
        // size_t vec_num = C / vector_step
        mov(rax, reg_rt_shape);
        mov(reg_vector_num, vector_step);
        xor_(rdx, rdx);
        div(reg_vector_num);    // reg_rt_shape / vector_step, rax is result, rdx is tails(remainder)
        mov(reg_vector_num, rax);
        mov(reg_tail_num, rdx);

        Xbyak::Reg64 reg_src_aux = rdx;
        Xbyak::Reg64 reg_work_amount_bk = r15;
        mov(reg_work_amount_bk, reg_work_amount);  // should before tail jmp

        Xbyak::Label tail_label;
        cmp(reg_vector_num, 0);
        je(tail_label, T_NEAR);

        // unroll_size = vec_num >= unroll_size ? unroll_size : vec_num;
        Xbyak::Label label_reset_unroll_size_end;
        cmp(reg_unroll_size, reg_vector_num);
        jle(label_reset_unroll_size_end, T_NEAR);
        mov(reg_unroll_size, reg_vector_num);
        L(label_reset_unroll_size_end);

        // last unroll_size
        Xbyak::Label label_reset_last_unroll_size;
        Xbyak::Label label_reset_last_unroll_size_end;
        Xbyak::Reg64 last_unroll_size = rcx;
        mov(rax, reg_vector_num);
        xor_(rdx, rdx);
        div(reg_unroll_size);  // rdx
        cmp(rdx, 0);
        je(label_reset_last_unroll_size, T_NEAR);
        mov(last_unroll_size, rdx);
        jmp(label_reset_last_unroll_size_end);
        L(label_reset_last_unroll_size);
        {
            mov(last_unroll_size, reg_unroll_size);
        }
        L(label_reset_last_unroll_size_end);

        // size_t unroll_number = div_up(vec_num, unroll_size); --> (vec_num + unroll_size - 1) / unroll_size;
        mov(rdi, reg_vector_num);
        add(rdi, reg_unroll_size);
        sub(rdi, 1);
        mov(rax, rdi);
        xor_(rdx, rdx);
        div(reg_unroll_size);  // result is in rax, that is reg_unroll_num, no mov need.

        // 4-15 for unroll. 4-7 for src, 8-11 for m/v sum, 12-15 for mean, 4 vector for 4 unroll
        int ur_base = 4;
        auto init = [&](int vmm_id) {
            uni_vpxor(Vmm(ur_base + 4 + vmm_id), Vmm(ur_base + 4 + vmm_id), Vmm(ur_base + 4 + vmm_id));
            if (jcp_.normalize_variance)
                uni_vmovups(Vmm(ur_base + 8 + vmm_id), ptr[reg_mean + vmm_id * vlen]);
        };
        auto load_src = [&](int vmm_id) {
            load_emitter[VECTOR]->emit_code({static_cast<size_t>(reg_src_aux.getIdx())}, {static_cast<size_t>(ur_base + vmm_id)}, {}, {load_pool_gpr_idxs});
            add(reg_src_aux, vector_step * jcp_.src_data_size);
        };
        auto mv = [&](int vmm_id) {
            if (jcp_.normalize_variance) {
                if (!isFloatCompatible(jcp_.src_prc)) {
                    uni_vcvtdq2ps(Vmm(ur_base + vmm_id), Vmm(ur_base + vmm_id));
                }
                uni_vsubps(Vmm(ur_base + vmm_id), Vmm(ur_base + vmm_id), Vmm(ur_base + 8 + vmm_id));
                uni_vfmadd231ps(Vmm(ur_base + 4 + vmm_id), Vmm(ur_base + vmm_id), Vmm(ur_base + vmm_id));
            } else {
                if (!isFloatCompatible(jcp_.src_prc))
                    uni_vpaddd(Vmm(ur_base + 4 + vmm_id), Vmm(ur_base + 4 + vmm_id), Vmm(ur_base + vmm_id));
                else
                    uni_vaddps(Vmm(ur_base + 4 + vmm_id), Vmm(ur_base + 4 + vmm_id), Vmm(ur_base + vmm_id));
            }
        };
        auto store = [&](int vmm_id) {
            if (jcp_.normalize_variance) {
                uni_vmovups(ptr[reg_variance + vmm_id * vector_step * sizeof(float)], Vmm(ur_base + 4 + vmm_id));
            } else {
                if (!isFloatCompatible(jcp_.src_prc))
                    uni_vcvtdq2ps(Vmm(ur_base + 4 + vmm_id), Vmm(ur_base + 4 + vmm_id));
                uni_vmovups(ptr[reg_sum + vmm_id * vector_step * sizeof(float)], Vmm(ur_base + 4 + vmm_id));
            }
        };

        auto vector_worker = [&](std::function<void(int)> func) {
            Xbyak::Label label_end;
            func(0);
            cmp(reg_unroll_size, 1);
            jle(label_end, T_NEAR);
            func(1);
            cmp(reg_unroll_size, 2);
            jle(label_end, T_NEAR);
            func(2);
            cmp(reg_unroll_size, 3);
            jle(label_end, T_NEAR);
            func(3);
            L(label_end);
        };

        Xbyak::Label label_unroll_num;
        Xbyak::Label label_unroll_num_end;
        L(label_unroll_num);
        {
            cmp(reg_unroll_num, 0);
            jle(label_unroll_num_end, T_NEAR);

            Xbyak::Label label_not_last;
            cmp(reg_unroll_num, 1);
            jne(label_not_last, T_NEAR);
            mov(reg_unroll_size, last_unroll_size);
            L(label_not_last);

            vector_worker(init);

            mov(reg_src_aux, reg_src);
            mov(reg_work_amount, reg_work_amount_bk);
            Xbyak::Label loop_label;
            Xbyak::Label loop_end_label;
            L(loop_label);
            {
                cmp(reg_work_amount, 0);
                jle(loop_end_label, T_NEAR);

                // load unroll
                vector_worker(load_src);

                // advance src and prefetch
                mov(rdi, reg_unroll_size);
                imul(rdi, rdi, vector_step * jcp_.src_data_size);
                sub(reg_src_aux, rdi);
                mov(rdi, reg_rt_shape);
                imul(rdi, rdi, jcp_.src_data_size);
                add(reg_src_aux, rdi);
                prefetcht0(ptr[reg_src_aux]);

                // mv compute
                vector_worker(mv);

                sub(reg_work_amount, 1);
                jmp(loop_label, T_NEAR);
            }
            L(loop_end_label);

            // store mv vector to memory
            vector_worker(store);

            // src advance
            mov(rdi, reg_unroll_size);
            imul(rdi, rdi, vector_step * jcp_.src_data_size);
            add(reg_src, rdi);
            // m/v advance
            mov(rdi, reg_unroll_size);
            imul(rdi, rdi, vlen);
            if (jcp_.normalize_variance) {
                add(reg_mean, rdi);
                add(reg_variance, rdi);
            } else {
                add(reg_sum, rdi);
            }
            sub(reg_unroll_num, 1);
            jmp(label_unroll_num, T_NEAR);
        }
        L(label_unroll_num_end);

        // tails
        L(tail_label);

        Xbyak::Label label_exit;
        cmp(reg_tail_num, 0);
        je(label_exit, T_NEAR);

        Xbyak::Reg64 reg_tails_num_active = reg_unroll_size;
        mov(reg_src_aux, reg_src);
        mov(reg_work_amount, reg_work_amount_bk);

        auto init_tails = [&](int step) {
            int vmm_id = get_tile_vr_id(step);
            uni_vpxor(Vmm(ur_base + 4 + vmm_id), Vmm(ur_base + 4 + vmm_id), Vmm(ur_base + 4 + vmm_id));
            if (jcp_.normalize_variance) {
                uni_vmovups(Vmm(ur_base + 8 + vmm_id), ptr[reg_mean]);
                add(reg_mean, step * sizeof(float));
            }
        };
        auto load_src_tails = [&](int step) {
            int vmm_id = get_tile_vr_id(step);
            int emitter_id = 4;
            if (step == 8) {
                emitter_id = 1;
            } else if (step == 4) {
                emitter_id = 2;
            } else if (step == 2) {
                emitter_id = 3;
            }
            load_emitter[emitter_id]->emit_code({static_cast<size_t>(reg_src_aux.getIdx())}, {static_cast<size_t>(ur_base + vmm_id)},
                                                {}, {load_pool_gpr_idxs});
            add(reg_src_aux, step * jcp_.src_data_size);
        };
        auto mv_tails = [&](int step) {
            int vmm_id = get_tile_vr_id(step);
            if (jcp_.normalize_variance) {
                if (!isFloatCompatible(jcp_.src_prc)) {
                    uni_vcvtdq2ps(Vmm(ur_base + vmm_id), Vmm(ur_base + vmm_id));
                }
                uni_vsubps(Vmm(ur_base + vmm_id), Vmm(ur_base + vmm_id), Vmm(ur_base + 8 + vmm_id));
                uni_vfmadd231ps(Vmm(ur_base + 4 + vmm_id), Vmm(ur_base + vmm_id), Vmm(ur_base + vmm_id));
            } else {
                if (!isFloatCompatible(jcp_.src_prc))
                    uni_vpaddd(Vmm(ur_base + 4 + vmm_id), Vmm(ur_base + 4 + vmm_id), Vmm(ur_base + vmm_id));
                else
                    uni_vaddps(Vmm(ur_base + 4 + vmm_id), Vmm(ur_base + 4 + vmm_id), Vmm(ur_base + vmm_id));
            }
        };
        auto store_tails = [&](size_t step) {
            int vmm_id = get_tile_vr_id(step);
            if (jcp_.normalize_variance) {
                uni_vmovups(ptr[reg_variance], Vmm(ur_base + 4 + vmm_id));
                add(reg_variance, step * sizeof(float));
            } else {
                if (!isFloatCompatible(jcp_.src_prc))
                    uni_vcvtdq2ps(Vmm(ur_base + 4 + vmm_id), Vmm(ur_base + 4 + vmm_id));
                uni_vmovups(ptr[reg_sum], Vmm(ur_base + 4 + vmm_id));
                add(reg_sum, step * sizeof(float));
            }
        };

        // init
        mov(reg_tails_num_active, reg_tail_num);
        worker_tails(reg_tails_num_active, init_tails);

        Xbyak::Label loop_tail_label;
        Xbyak::Label label_tails_end;

        L(loop_tail_label);
        {
            cmp(reg_work_amount, 0);
            jle(label_tails_end, T_NEAR);

            // load src
            mov(reg_tails_num_active, reg_tail_num);
            worker_tails(reg_tails_num_active, load_src_tails);

            // m/v compute
            mov(reg_tails_num_active, reg_tail_num);
            worker_tails(reg_tails_num_active, mv_tails);

            mov(rdi, reg_vector_num);
            imul(rdi, rdi, vector_step * jcp_.src_data_size);
            add(reg_src_aux, rdi);
            sub(reg_work_amount, 1);
            jmp(loop_tail_label, T_NEAR);
        }
        L(label_tails_end);

        // store tails
        mov(reg_tails_num_active, reg_tail_num);
        worker_tails(reg_tails_num_active, store_tails);

        L(label_exit);
    }

    inline void block_ker() {
        // safe to use abi reg now.
        Xbyak::Reg64 reg_src_bk = rcx;
        Xbyak::Reg64 reg_work_amount_bk = rdi;
        mov(reg_src_bk, reg_src);
        mov(reg_work_amount_bk, reg_work_amount);
        int repeats = (isa == cpu::x64::sse41) ? 2 : 1; // block size is also 8 on cpu::x64::sse41 with two step process

        auto reset_with_offset = [&](int offset) {
            add(reg_src_bk, offset * jcp_.src_data_size);
            mov(reg_src, reg_src_bk);
            mov(reg_work_amount, reg_work_amount_bk);
            if (jcp_.normalize_variance) {
                // mean and vaiance for variance kernel
                if (!jcp_.across_channels) {
                    // mean is bc when across_channel, no need shift
                    add(reg_mean, offset * sizeof(float));
                    uni_vmovups(vmm_mean, ptr[reg_mean]);
                }
                add(reg_variance, offset * sizeof(float));
                uni_vpxor(vmm_variance, vmm_variance, vmm_variance);
            } else {
                // sum for mean kernel
                add(reg_sum, offset * sizeof(float));
                uni_vpxor(vmm_sum, vmm_sum, vmm_sum);
            }
        };

        auto save_result = [&]() {
            // add input_base value and store for per_channel
            // store for across_channels
            if (jcp_.normalize_variance) {
                if (!jcp_.across_channels) {
                    uni_vmovups(vmm_val, ptr[reg_variance]);
                    uni_vaddps(vmm_variance, vmm_variance, vmm_val);
                }
                uni_vmovups(ptr[reg_variance], vmm_variance);
            } else {
                if (!isFloatCompatible(jcp_.src_prc))  // add with int for int-family data type, other compute go with float
                    uni_vcvtdq2ps(vmm_sum, vmm_sum);

                if (!jcp_.across_channels) {
                    uni_vmovups(vmm_val, ptr[reg_sum]);
                    uni_vaddps(vmm_sum, vmm_sum, vmm_val);
                }
                uni_vmovups(ptr[reg_sum], vmm_sum);
            }
        };

        auto worker_tails_unroll = [&]() {
            auto unroll_w = [&](int block_num) {
                Xbyak::Label loop_label;
                Xbyak::Label loop_end_label;
                L(loop_label);
                {
                    cmp(reg_work_amount, 0);
                    jle(loop_end_label, T_NEAR);

                    worker_block(block_num, true);

                    add(reg_src, src_stride);
                    sub(reg_work_amount, 1);

                    jmp(loop_label, T_NEAR);
                }
                L(loop_end_label);
            };
            auto tails_func = [&](int tile_size) {
                unroll_w(tile_size);
                save_result();
                reset_with_offset(tile_size);
            };
            worker_tails(reg_rt_shape, tails_func);
        };

        // cover vector and tails on avx512, avx2
        // cover on sse, 2 part vector, first part vector and second part tails, first part tails
        Xbyak::Label label_exit;
        for (int i = 0; i < repeats; i++) {
            if (i > 0) {
                reset_with_offset(4);
            }

            Xbyak::Label label_tails;
            Xbyak::Label label_end;
            cmp(reg_rt_shape, 0);
            jne(label_tails, T_NEAR);

            worker_vector_unroll();
            save_result();
            jmp(label_end, T_NEAR);

            L(label_tails);
            {
                Xbyak::Label label_sse_full_size;
                if (isa == cpu::x64::sse41) {
                    // on sse, first 4 could be done with vector manner
                    cmp(reg_rt_shape, 4);
                    jae(label_sse_full_size, T_NEAR);
                }

                worker_tails_unroll();
                // exit directly if tails is done, for all platforms
                jmp(label_exit, T_NEAR);

                L(label_sse_full_size);
                {
                    worker_vector_unroll();
                    save_result();
                    sub(reg_rt_shape, 4);
                    // exit directly if empty second half
                    cmp(reg_rt_shape, 0);
                    jbe(label_exit, T_NEAR);
                }
            }
            L(label_end);
        }
        L(label_exit);
    }

    inline void worker_vector_unroll() {
        // if mean(sum) for continous data, then fast pass for major part
        if (!jcp_.normalize_variance && jcp_.layout == MVNLayoutType::mvn_planar) {
            Vmm vmm_one = Vmm(15);
            // i8/u8 fast path
            if (mayiuse(avx512_core_vnni) && jcp_.src_data_size == 1) {
                uni_vmovups(vmm_one, ptr[reg_table]);
                Xbyak::Label loop_8bit_label;
                Xbyak::Label loop_8bit_end_label;
                L(loop_8bit_label);
                {
                    cmp(reg_work_amount, 4);
                    jl(loop_8bit_end_label, T_NEAR);

                    if (jcp_.src_prc == ov::element::i8) {
                        vpdpbusd(vmm_sum, vmm_one, ptr[reg_src]);
                    } else {
                        uni_vmovdqu(vmm_val, ptr[reg_src]);
                        vpdpbusd(vmm_sum, vmm_val, vmm_one);
                    }

                    add(reg_src, vlen);
                    sub(reg_work_amount, 4);

                    jmp(loop_8bit_label, T_NEAR);
                }
                L(loop_8bit_end_label);
            }
            // bf16 fast path
            if (mayiuse(avx512_core_bf16) && jcp_.src_prc == ov::element::bf16) {
                uni_vmovups(vmm_one, ptr[reg_table]);
                Xbyak::Label loop_bf16_label;
                Xbyak::Label loop_bf16_end_label;
                L(loop_bf16_label);
                {
                    cmp(reg_work_amount, 2);
                    jl(loop_bf16_end_label, T_NEAR);

                    vdpbf16ps(vmm_sum, vmm_one, ptr[reg_src]);

                    add(reg_src, vlen);
                    sub(reg_work_amount, 2);

                    jmp(loop_bf16_label, T_NEAR);
                }
                L(loop_bf16_end_label);
            }
        }

        Xbyak::Label loop_label;
        Xbyak::Label loop_end_label;
        L(loop_label);
        {
            cmp(reg_work_amount, 0);
            jle(loop_end_label, T_NEAR);

            worker_full_size();

            add(reg_src, src_stride);
            sub(reg_work_amount, 1);

            jmp(loop_label, T_NEAR);
        }
        L(loop_end_label);
    }

    inline void worker_full_size() {
        load_emitter[VECTOR]->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val.getIdx())},
                                       {}, {load_pool_gpr_idxs});

        if (jcp_.normalize_variance) {
            // all with float
            if (!isFloatCompatible(jcp_.src_prc))
                uni_vcvtdq2ps(vmm_val, vmm_val);

            uni_vsubps(vmm_val, vmm_val, vmm_mean);
            uni_vfmadd231ps(vmm_variance, vmm_val, vmm_val);
        } else {
            // for sum, int execute prc for int-family data type
            if (!isFloatCompatible(jcp_.src_prc))
                uni_vpaddd(vmm_sum, vmm_sum, vmm_val);
            else
                uni_vaddps(vmm_sum, vmm_sum, vmm_val);
        }
    }

    inline void worker_tails(Xbyak::Reg64& reg_tail_num, std::function<void(int)> func) {
        int tile_start_idx = (isa == cpu::x64::avx512_core) ? 0 : ((isa == cpu::x64::avx2) ? 1 : 2);
        Label tile_exit[kTileNum];
        for (int i = tile_start_idx; i < kTileNum; i++) {
            cmp(reg_tail_num, tile_size[i]);
            jl(tile_exit[i], T_NEAR);

            func(tile_size[i]);
            sub(reg_tail_num, tile_size[i]);

            L(tile_exit[i]);
        }
    }

    inline void worker_block(int block_num, bool is_zero_pad) {
        if (is_zero_pad) {
            switch (block_num) {
            case 8:
                load_emitter[TAIL8_FILL]->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val.getIdx())},
                                           {}, {load_pool_gpr_idxs});
                break;
            case 4:
                load_emitter[TAIL4_FILL]->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val.getIdx())},
                                           {}, {load_pool_gpr_idxs});
                break;
            case 2:
                load_emitter[TAIL2_FILL]->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val.getIdx())},
                                           {}, {load_pool_gpr_idxs});
                break;
            case 1:
                load_emitter[TAIL1_FILL]->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val.getIdx())},
                                           {}, {load_pool_gpr_idxs});
                break;
            default:
                assert(!"MVN layer tails is processed only with 8/4/2/1 blocks.");
                break;
            }
        } else {
            switch (block_num) {
            case 8:
                load_emitter[TAIL8]->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val.getIdx())},
                                           {}, {load_pool_gpr_idxs});
                break;
            case 4:
                load_emitter[TAIL4]->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val.getIdx())},
                                           {}, {load_pool_gpr_idxs});
                break;
            case 2:
                load_emitter[TAIL2]->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val.getIdx())},
                                           {}, {load_pool_gpr_idxs});
                break;
            case 1:
                load_emitter[TAIL1]->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val.getIdx())},
                                           {}, {load_pool_gpr_idxs});
                break;
            default:
                assert(!"MVN layer tails is processed only with 8/4/2/1 blocks.");
                break;
            }
        }
        if (jcp_.normalize_variance) {
            if (!isFloatCompatible(jcp_.src_prc))
                uni_vcvtdq2ps(vmm_val, vmm_val);
            uni_vsubps(vmm_val, vmm_val, vmm_mean);
            if (is_zero_pad) {
                uni_vpxor(vmm_zero, vmm_zero, vmm_zero);
                if (isa == cpu::x64::sse41) {
                    uint8 imm = 1;
                    imm = ~((imm << block_num) - imm);
                    blendps(vmm_val, vmm_zero, imm);
                } else if (isa == cpu::x64::avx2) {
                    uint8 imm = 1;
                    imm = ~((imm << block_num) - imm);
                    vblendps(vmm_val, vmm_val, vmm_zero, imm);
                } else if (isa == cpu::x64::avx512_core) {
                    uint64_t tail_mask = 1;
                    tail_mask = ~((tail_mask << block_num) - tail_mask);
                    mov(reg_aux, tail_mask);
                    kmovq(k_mask, reg_aux);
                    vblendmps(vmm_val | k_mask, vmm_val, vmm_zero);
                }
            }
            uni_vfmadd231ps(vmm_variance, vmm_val, vmm_val);
        } else {
            if (!isFloatCompatible(jcp_.src_prc))
                uni_vpaddd(vmm_sum, vmm_sum, vmm_val);
            else
                uni_vaddps(vmm_sum, vmm_sum, vmm_val);
        }
    }

    inline void reduce_sum_store_xmm(Xbyak::Xmm xmm_sum) {
        uni_vmovshdup(xmm_aux3, xmm_sum);            //  sum:1,2,3,4; aux3:2,2,4,4
        uni_vaddps(xmm_sum, xmm_sum, xmm_aux3);      //  sum:1+2,2+2,3+4,4+4
        uni_vmovhlps(xmm_aux3, xmm_aux3, xmm_sum);   //  aux3:3+4,4+4,4,4
        uni_vaddps(xmm_sum, xmm_sum,  xmm_aux3);     //  sum:1+2+3+4,...
        if (jcp_.normalize_variance) {
            uni_vmovss(ptr[reg_variance], xmm_sum);
        } else {
            uni_vmovss(ptr[reg_sum], xmm_sum);
        }
    }

    inline void reduce_sum_store_vmm(int vmm_idx) {
        if (isa == cpu::x64::sse41) {
            reduce_sum_store_xmm(Xmm(vmm_idx));
        } else if (isa == cpu::x64::avx2) {
            Xbyak::Ymm ymm_sum = Xbyak::Ymm(vmm_idx);
            vextractf128(xmm_aux1, ymm_sum, 0);
            vextractf128(xmm_aux2, ymm_sum, 1);
            uni_vaddps(xmm_aux1, xmm_aux1, xmm_aux2);
            reduce_sum_store_xmm(xmm_aux1);
        } else {
            Xbyak::Zmm zmm_sum = Xbyak::Zmm(vmm_idx);
            vextractf32x4(xmm_aux1, zmm_sum, 0);
            vextractf32x4(xmm_aux2, zmm_sum, 1);
            uni_vaddps(xmm_aux1, xmm_aux1, xmm_aux2);
            vextractf32x4(xmm_aux2, zmm_sum, 2);
            vextractf32x4(xmm_aux3, zmm_sum, 3);
            uni_vaddps(xmm_aux2, xmm_aux2, xmm_aux3);
            uni_vaddps(xmm_aux1, xmm_aux1, xmm_aux2);
            reduce_sum_store_xmm(xmm_aux1);
        }
    }

    void prepare_table() {
        const unsigned int cvals[] = {
            // 4 * 8 = 32 bit
            0x01010101,  // one byte
            0x3f803f80   // one bf16
        };

        align(64);
        L(l_table);

        if (mayiuse(avx512_core_vnni) && (jcp_.src_prc == ov::element::u8 || jcp_.src_prc == ov::element::i8)) {
            for (int d = 0; d < vector_step; ++d) {
                dd(cvals[0]);
            }
        }
        if (mayiuse(avx512_core_bf16) && jcp_.src_prc == ov::element::bf16) {
            for (int d = 0; d < vector_step; ++d) {
                dd(cvals[1]);
            }
        }
    }
};

// mean,variance->mvn
template <cpu_isa_t isa>
struct jit_uni_mvn_kernel_f32 : public jit_uni_mvn_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_mvn_kernel_f32)

    explicit jit_uni_mvn_kernel_f32(jit_mvn_config_params jcp, const dnnl_primitive_attr &attr) : jit_uni_mvn_kernel(jcp, attr), jit_generator(jit_name()) {
        const auto &p = attr_.post_ops_;
        bool opt_scaleshift_applicable = jcp_.layout == MVNLayoutType::mvn_by_channel && isa == cpu::x64::avx512_core && !jcp_.across_channels;
        if (opt_scaleshift_applicable) {
            for (int i = 0; i < p.len(); i++) {
                auto &post_op = p.entry_[i];
                if (post_op.is_depthwise()) {
                    if (0 == i && post_op.depthwise.alg == alg_kind::depthwise_scale_shift) {
                        optimized_scaleshift_num = 1;
                    } else if (1 == i && optimized_scaleshift_num == 1 && post_op.depthwise.alg == alg_kind::depthwise_scale_shift) {
                        optimized_scaleshift_num = 2;
                    }
                }
            }
        }
    }

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        const auto &p = attr_.post_ops_;
        for (int i = 0; i < p.len(); i++) {
            auto &post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors.push_back(std::make_shared<jit_uni_eltwise_injector_f32<isa>>(
                        this, post_op.eltwise.alg, post_op.eltwise.alpha, post_op.eltwise.beta, post_op.eltwise.scale));
            } else if (post_op.is_depthwise()) {
                depthwise_injectors.push_back(std::make_shared<jit_uni_depthwise_injector_f32<isa>>(
                        this, post_op));
            } else if (post_op.is_quantization()) {
                quantization_injectors.push_back(std::make_shared<jit_uni_quantization_injector_f32<isa>>(
                        this, post_op, vmm_d_weights, vmm_d_bias, reg_d_weights, reg_d_bias));
            }
        }

        load_emitter[VECTOR] = std::unique_ptr<jit_load_emitter>(new jit_load_emitter(this, isa, jcp_.src_prc, ov::element::f32, vector_step));
        load_emitter[TAIL8] = std::unique_ptr<jit_load_emitter>(new jit_load_emitter(this, isa, jcp_.src_prc, ov::element::f32, 8));
        load_emitter[TAIL4] = std::unique_ptr<jit_load_emitter>(new jit_load_emitter(this, isa, jcp_.src_prc, ov::element::f32, 4));
        load_emitter[TAIL2] = std::unique_ptr<jit_load_emitter>(new jit_load_emitter(this, isa, jcp_.src_prc, ov::element::f32, 2));
        load_emitter[TAIL1] = std::unique_ptr<jit_load_emitter>(new jit_load_emitter(this, isa, jcp_.src_prc, ov::element::f32, 1));
        store_emitter[VECTOR] = std::unique_ptr<jit_store_emitter>(new jit_store_emitter(this, isa, ov::element::f32, jcp_.dst_prc, vector_step));
        store_emitter[TAIL8] = std::unique_ptr<jit_store_emitter>(new jit_store_emitter(this, isa, ov::element::f32, jcp_.dst_prc, 8));
        store_emitter[TAIL4] = std::unique_ptr<jit_store_emitter>(new jit_store_emitter(this, isa, ov::element::f32, jcp_.dst_prc, 4));
        store_emitter[TAIL2] = std::unique_ptr<jit_store_emitter>(new jit_store_emitter(this, isa, ov::element::f32, jcp_.dst_prc, 2));
        store_emitter[TAIL1] = std::unique_ptr<jit_store_emitter>(new jit_store_emitter(this, isa, ov::element::f32, jcp_.dst_prc, 1));

        this->preamble();

        mov(reg_post_ops_data, ptr[reg_params + GET_OFF(post_op_data)]);
        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_mean, ptr[reg_params + GET_OFF(mean)]);
        if (jcp_.normalize_variance)
            mov(reg_variance_inv, ptr[reg_params + GET_OFF(variance)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);
        mov(reg_rt_shape, ptr[reg_params + GET_OFF(rt_shape_size)]);
        mov(reg_oc_off, ptr[reg_params + GET_OFF(oc_off)]);

        size_t data_step = (isa == cpu::x64::sse41 && jcp_.layout == MVNLayoutType::mvn_block) ? vector_step * 2 : vector_step;
        src_stride = data_step * jcp_.src_data_size;
        dst_stride = data_step * jcp_.dst_data_size;

        if (jcp_.layout == MVNLayoutType::mvn_planar || jcp_.across_channels) {
            uni_vbroadcastss(vmm_mean, ptr[reg_mean]);
            if (jcp_.normalize_variance)
                uni_vbroadcastss(vmm_variance_inv, ptr[reg_variance_inv]);
        } else {
            uni_vmovups(vmm_mean, ptr[reg_mean]);
            if (jcp_.normalize_variance)
                uni_vmovups(vmm_variance_inv, ptr[reg_variance_inv]);
        }

        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        load_pool_gpr_idxs = {static_cast<size_t>(reg_load_store_mask.getIdx()), static_cast<size_t>(reg_load_table.getIdx())};
        store_pool_gpr_idxs = {static_cast<size_t>(reg_load_store_mask.getIdx())};
        store_pool_vec_idxs = {static_cast<size_t>(vmm_zero.getIdx()), static_cast<size_t>(vmm_val.getIdx())};

        if (jcp_.layout == MVNLayoutType::mvn_planar) {
            worker_mvn_vector_unroll(reg_work_amount);
            // tails
            auto tails_func = [&](int tile_size) {
                worker_mvn_block(tile_size);
                add(reg_src, tile_size * jcp_.src_data_size);
                add(reg_dst, tile_size * jcp_.dst_data_size);
            };
            worker_mvn_tails(reg_rt_shape, tails_func);
        } else if (jcp_.layout == MVNLayoutType::mvn_by_channel) {
            if (jcp_.across_channels)
                norm_nspc_ac_ker();
            else
                norm_nspc_pc_ker();
        } else {
            norm_block_ker();
        }

        this->postamble();

        for (size_t i = 0; i < EMITTERS_NUM; i++)
            load_emitter[i]->emit_data();
        for (size_t i = 0; i < EMITTERS_NUM; i++)
            store_emitter[i]->emit_data();

        for (auto& inj : eltwise_injectors)
            inj->prepare_table();
    }

private:
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;

    const int vlen = cpu_isa_traits<isa>::vlen;
    const int vector_step = vlen / sizeof(float);

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_mean = r9;
    Xbyak::Reg64 reg_variance_inv = r10;
    Xbyak::Reg64 reg_dst = r11;
    Xbyak::Reg64 reg_work_amount = r12;
    Xbyak::Reg64 reg_params = abi_param1;

    Xbyak::Reg64 reg_oc_off = r13;
    Xbyak::Reg64 reg_d_weights = rbx;
    Xbyak::Reg64 reg_d_bias = r14;
    Xbyak::Reg64 reg_post_ops_data = rsi;

    Xbyak::Reg64 reg_rt_shape = r15;
    Xbyak::Reg64 reg_load_table = r15; // fill not needed, dummy
    Xbyak::Reg64 reg_load_store_mask = rbp;

    size_t src_stride = 0;
    size_t dst_stride = 0;

    Vmm vmm_val = Vmm(3);
    Vmm vmm_mean = Vmm(4);
    Vmm vmm_variance_inv = Vmm(5);
    Vmm vmm_zero = Vmm(2);

    Vmm vmm_d_weights = Vmm(0);
    Vmm vmm_d_bias = Vmm(1);

    enum { VECTOR, TAIL8, TAIL4, TAIL2, TAIL1, EMITTERS_NUM };
    std::unique_ptr<jit_load_emitter> load_emitter[EMITTERS_NUM];
    std::unique_ptr<jit_store_emitter> store_emitter[EMITTERS_NUM];
    std::vector<size_t> store_pool_gpr_idxs;
    std::vector<size_t> store_pool_vec_idxs;
    std::vector<size_t> load_pool_gpr_idxs;

    const int tile_size[kTileNum] = {8, 4, 2, 1};

    std::vector<std::shared_ptr<jit_uni_eltwise_injector_f32<isa>>> eltwise_injectors;
    std::vector<std::shared_ptr<jit_uni_depthwise_injector_f32<isa>>> depthwise_injectors;
    std::vector<std::shared_ptr<jit_uni_quantization_injector_f32<isa>>> quantization_injectors;

    inline void norm_block_ker() {
        Xbyak::Reg64 reg_src_bk = rax;
        Xbyak::Reg64 reg_dst_bk = rdx;
        Xbyak::Reg64 reg_work_amount_bk = rdi;
        mov(reg_src_bk, reg_src);
        mov(reg_dst_bk, reg_dst);
        mov(reg_work_amount_bk, reg_work_amount);

        auto reset_with_offset = [&](int offset) {
            add(reg_src_bk, offset * jcp_.src_data_size);
            add(reg_dst_bk, offset * jcp_.dst_data_size);
            add(reg_oc_off, offset * sizeof(float));  // for post ops
            mov(reg_src, reg_src_bk);
            mov(reg_dst, reg_dst_bk);
            mov(reg_work_amount, reg_work_amount_bk);
            if (!jcp_.across_channels) {
                add(reg_mean, offset * sizeof(float));
                uni_vmovups(vmm_mean, ptr[reg_mean]);
                if (jcp_.normalize_variance) {
                    add(reg_variance_inv, offset * sizeof(float));
                    uni_vmovups(vmm_variance_inv, ptr[reg_variance_inv]);
                }
            }
        };

        // unroll for block layout, w/o zero pading
        auto worker_tails_unroll = [&]() {
            auto unroll_w = [&](int block_num) {
                Xbyak::Label loop_label;
                Xbyak::Label loop_end_label;
                L(loop_label);
                {
                    cmp(reg_work_amount, 0);
                    jle(loop_end_label, T_NEAR);

                    worker_mvn_block(block_num);

                    add(reg_src, src_stride);
                    add(reg_dst, dst_stride);
                    sub(reg_work_amount, 1);

                    jmp(loop_label, T_NEAR);
                }
                L(loop_end_label);
            };
            auto tails_func = [&](int tile_size) {
                unroll_w(tile_size);
                reset_with_offset(tile_size);
            };
            worker_mvn_tails(reg_rt_shape, tails_func);
        };

        // cover vector and tails on avx512, avx2
        // cover on sse, 2 part vector, first part vector and second part tails, first part tails
        Xbyak::Label label_exit;
        int repeats = (isa == cpu::x64::sse41) ? 2 : 1;
        for (int i = 0; i < repeats; i++) {
            if (i > 0) {
                reset_with_offset(4);
            }

            Xbyak::Label label_tails;
            Xbyak::Label label_end;
            cmp(reg_rt_shape, 0);
            jne(label_tails, T_NEAR);

            worker_mvn_vector_unroll(reg_work_amount);
            jmp(label_end, T_NEAR);

            L(label_tails);
            {
                Xbyak::Label label_sse_full_size;
                if (isa == cpu::x64::sse41) {
                    // on sse, first 4 could be done with vector manner
                    cmp(reg_rt_shape, 4);
                    jae(label_sse_full_size, T_NEAR);
                }

                worker_tails_unroll();
                // exit directly if tails is done, for all platforms
                jmp(label_exit, T_NEAR);

                L(label_sse_full_size);
                {
                    worker_mvn_vector_unroll(reg_work_amount);
                    sub(reg_rt_shape, 4);
                    // exit directly if empty second half
                    cmp(reg_rt_shape, 0);
                    jbe(label_exit, T_NEAR);
                }
            }
            L(label_end);
        }
        L(label_exit);
    }

    // nspc norm per channel with unroll
    inline void norm_nspc_pc_ker() {
        // stack used as no more GPR.
        const int gpr_size = 8;
        sub(rsp, 7 * gpr_size);
        const Xbyak::Address addr_unroll_size = qword[rsp];
        const Xbyak::Address addr_unroll_num = qword[rsp + 8];
        const Xbyak::Address addr_vector_num = qword[rsp + 16];
        const Xbyak::Address addr_tail_num = qword[rsp + 24];
        const Xbyak::Address addr_last_unroll_size = qword[rsp + 32];
        const Xbyak::Address addr_work_amount_bk = qword[rsp + 40];
        const Xbyak::Address addr_oc_off_bk = qword[rsp + 48];

        // size_t vec_num = C / vector_step
        mov(rax, reg_rt_shape);
        mov(addr_vector_num, vector_step);
        xor_(rdx, rdx);
        div(addr_vector_num);    // reg_rt_shape / vector_step, rax is result, rdx is tails
        mov(addr_vector_num, rax);
        mov(addr_tail_num, rdx);

        // should before tail jmp
        Xbyak::Reg64 reg_src_aux = rcx;
        Xbyak::Reg64 reg_dst_aux = rdi;
        mov(addr_work_amount_bk, reg_work_amount);
        mov(addr_oc_off_bk, reg_oc_off);

        Xbyak::Label tail_label;
        cmp(addr_vector_num, 0);
        je(tail_label, T_NEAR);

        // unroll_size = vec_num >= unroll_size ? unroll_size : vec_num;
        mov(addr_unroll_size, 4);  // default is 4 for addr_unroll_size
        mov(rax, addr_unroll_size);
        Xbyak::Label label_reset_unroll_size_end;
        cmp(rax, addr_vector_num);
        jle(label_reset_unroll_size_end, T_NEAR);
        mov(rax, addr_vector_num);
        mov(addr_unroll_size, rax);
        L(label_reset_unroll_size_end);

        // last unroll_size: vector_num % unroll_size
        Xbyak::Label label_reset_last_unroll_size;
        Xbyak::Label label_reset_last_unroll_size_end;
        mov(rax, addr_vector_num);
        xor_(rdx, rdx);
        div(addr_unroll_size);  // rdx
        cmp(rdx, 0);
        je(label_reset_last_unroll_size, T_NEAR);
        mov(addr_last_unroll_size, rdx);
        jmp(label_reset_last_unroll_size_end);
        L(label_reset_last_unroll_size);
        {
            mov(rax, addr_unroll_size);
            mov(addr_last_unroll_size, rax);
        }
        L(label_reset_last_unroll_size_end);

        // unroll_number = div_up(vec_num, unroll_size) --> (vec_num + unroll_size - 1) / unroll_size;
        mov(rax, addr_vector_num);
        add(rax, addr_unroll_size);
        sub(rax, 1);
        xor_(rdx, rdx);
        div(addr_unroll_size);
        mov(addr_unroll_num, rax);

        int ur_base = 4;
        auto load_mv = [&](int vmm_id, int step) {
            uni_vmovups(Vmm(ur_base + 4 + vmm_id), ptr[reg_mean]);
            add(reg_mean, step * sizeof(float));
            if (jcp_.normalize_variance) {
                uni_vmovups(Vmm(ur_base + 8 + vmm_id), ptr[reg_variance_inv]);
                add(reg_variance_inv, step * sizeof(float));
            }
        };

        // optimized scaleshift fusion data init
        int ss_repeat_id = 0;
        auto load_weight_bias = [&](int vmm_id, int step) {
            uni_vmovups(Vmm(16 + ss_repeat_id * 4 + vmm_id), ptr[reg_d_weights]);
            add(reg_d_weights, step * sizeof(float));
            uni_vmovups(Vmm(24 + ss_repeat_id * 4 + vmm_id), ptr[reg_d_bias]);
            add(reg_d_bias, step * sizeof(float));
        };

        auto load_src = [&](int vmm_id, int step) {
            load_emitter[VECTOR]->emit_code({static_cast<size_t>(reg_src_aux.getIdx())},
                                           {static_cast<size_t>(ur_base + vmm_id)}, {}, {load_pool_gpr_idxs});
            add(reg_src_aux, step * jcp_.src_data_size);
        };

        auto norm = [&](int vmm_id, int step) {
            uni_vsubps(Vmm(ur_base + vmm_id), Vmm(ur_base + vmm_id), Vmm(ur_base + 4 + vmm_id));
            if (jcp_.normalize_variance) {
                uni_vmulps(Vmm(ur_base + vmm_id), Vmm(ur_base + vmm_id), Vmm(ur_base + 8 + vmm_id));
            }
        };

        // optimized scaleshift fusion
        auto optimized_ss = [&](int vmm_id, int step) {
            uni_vfmadd132ps(Vmm(ur_base + vmm_id), Vmm(24 + ss_repeat_id * 4 + vmm_id), Vmm(16 + ss_repeat_id * 4 + vmm_id));
        };

        auto post_ops = [&](int vmm_id, int step) {
            apply_post_ops(jcp_.dst_prc, ur_base + vmm_id, false);
            add(reg_oc_off, step * sizeof(float));
        };

        auto store_dst = [&](int vmm_id, int step) {
            store_emitter[VECTOR]->emit_code({static_cast<size_t>(ur_base + vmm_id)}, {static_cast<size_t>(reg_dst_aux.getIdx())},
                {store_pool_vec_idxs}, {store_pool_gpr_idxs});
            add(reg_dst_aux, step * jcp_.dst_data_size);
        };

        auto vector_worker = [&](std::function<void(int, int)> func) {
            Xbyak::Label label_end;
            func(0, vector_step);
            cmp(addr_unroll_size, 1);
            jle(label_end, T_NEAR);
            func(1, vector_step);
            cmp(addr_unroll_size, 2);
            jle(label_end, T_NEAR);
            func(2, vector_step);
            cmp(addr_unroll_size, 3);
            jle(label_end, T_NEAR);
            func(3, vector_step);
            L(label_end);
        };

        Xbyak::Label label_unroll_num;
        Xbyak::Label label_unroll_num_end;
        L(label_unroll_num);
        {
            cmp(addr_unroll_num, 0);
            jle(label_unroll_num_end, T_NEAR);

            Xbyak::Label label_not_last;
            cmp(addr_unroll_num, 1);
            jne(label_not_last, T_NEAR);
            mov(rax, addr_last_unroll_size);
            mov(addr_unroll_size, rax);
            L(label_not_last);

            mov(reg_src_aux, reg_src);
            mov(reg_dst_aux, reg_dst);
            mov(reg_work_amount, addr_work_amount_bk);

            // 4-15 for unroll. 4-7 for src, 8-11 for m, 12-15 for v
            // load m/v
            vector_worker(load_mv);

            // optimized scaleshift fusion arg init. 16-23 for weight, 24-31 for bias.
            // reg_post_ops_data[0]:----w0---- ----b0---- reg_post_ops_data[1]:----w1---- ----b1----
            mov(reg_oc_off, addr_oc_off_bk);
            size_t post_ops_data_offset = 0;
            ss_repeat_id = 0;
            for (int i = 0; i < optimized_scaleshift_num; i++) {
                mov(reg_d_weights, ptr[reg_post_ops_data + post_ops_data_offset]);
                add(reg_d_weights, reg_oc_off);
                // bias = weight + C
                mov(reg_d_bias, reg_d_weights);
                mov(rax, reg_rt_shape);
                imul(rax, rax, sizeof(float));
                add(reg_d_bias, rax);

                vector_worker(load_weight_bias);

                post_ops_data_offset += sizeof(float*);
                ss_repeat_id++;
            }

            Xbyak::Label loop_label;
            Xbyak::Label loop_end_label;
            L(loop_label);
            {
                cmp(reg_work_amount, 0);
                jle(loop_end_label, T_NEAR);

                // load
                vector_worker(load_src);

                // to next iteration(next work_amount)
                mov(rax, addr_unroll_size);
                imul(rax, rax, vector_step * jcp_.src_data_size);
                sub(reg_src_aux, rax);
                mov(rax, reg_rt_shape);
                imul(rax, rax, jcp_.src_data_size);
                add(reg_src_aux, rax);
                prefetcht0(ptr[reg_src_aux]);

                // norm
                vector_worker(norm);

                // optimized ss fusion
                ss_repeat_id = 0;
                for (int i = 0; i < optimized_scaleshift_num; i++) {
                    vector_worker(optimized_ss);
                    ss_repeat_id++;
                }

                // post-ops
                if (attr_.post_ops_.len() != 0) {
                    vector_worker(post_ops);
                }

                // store
                vector_worker(store_dst);

                // dst advance
                mov(rax, addr_unroll_size);
                imul(rax, rax, vector_step * jcp_.dst_data_size);
                sub(reg_dst_aux, rax);
                mov(rax, reg_rt_shape);
                imul(rax, rax, jcp_.dst_data_size);
                add(reg_dst_aux, rax);
                prefetcht0(ptr[reg_dst_aux]);

                // reg_oc_off reset
                mov(rax, addr_unroll_size);
                imul(rax, rax, vector_step * sizeof(float));
                sub(reg_oc_off, rax);

                sub(reg_work_amount, 1);
                jmp(loop_label, T_NEAR);
            }
            L(loop_end_label);

            // src/dst advance
            mov(rax, addr_unroll_size);
            imul(rdx, rax, vector_step * jcp_.src_data_size);
            add(reg_src, rdx);
            imul(rdx, rax, vector_step * jcp_.dst_data_size);
            add(reg_dst, rdx);
            imul(rdx, rax, vector_step * sizeof(float));
            add(addr_oc_off_bk, rdx);

            sub(addr_unroll_num, 1);
            jmp(label_unroll_num, T_NEAR);
        }
        L(label_unroll_num_end);

        // tails
        L(tail_label);

        Xbyak::Label label_exit;
        cmp(addr_tail_num, 0);
        je(label_exit, T_NEAR);

        mov(reg_src_aux, reg_src);
        mov(reg_dst_aux, reg_dst);
        mov(reg_work_amount, addr_work_amount_bk);
        Xbyak::Reg64 reg_tails_num_active = rdx;
        mov(reg_tails_num_active, addr_tail_num);

        auto get_tile_emitter_id = [&](const int& step) -> int {
            int emitter_id = 4;
            if (step == 8) {
                emitter_id = 1;
            } else if (step == 4) {
                emitter_id = 2;
            } else if (step == 2) {
                emitter_id = 3;
            }
            return emitter_id;
        };
        auto load_mv_tails = [&](int step) {
            load_mv(get_tile_vr_id(step), step);
        };
        auto load_weight_bias_tails = [&](int step) {
            load_weight_bias(get_tile_vr_id(step), step);
        };
        auto norm_tails = [&](int step) {
            norm(get_tile_vr_id(step), step);
        };
        auto optimized_ss_tails = [&](int step) {
            optimized_ss(get_tile_vr_id(step), step);
        };
        auto post_ops_tails = [&](int step) {
            post_ops(get_tile_vr_id(step), step);
        };
        auto load_src_tails = [&](int step) {
            int vmm_id = get_tile_vr_id(step);
            int emitter_id = get_tile_emitter_id(step);
            load_emitter[emitter_id]->emit_code({static_cast<size_t>(reg_src_aux.getIdx())},
                                                {static_cast<size_t>(ur_base + vmm_id)}, {}, {load_pool_gpr_idxs});
            add(reg_src_aux, step * jcp_.src_data_size);
        };
        auto store_tails = [&](int step) {
            int vmm_id = get_tile_vr_id(step);
            int emitter_id = get_tile_emitter_id(step);
            store_emitter[emitter_id]->emit_code({static_cast<size_t>(ur_base + vmm_id)}, {static_cast<size_t>(reg_dst_aux.getIdx())},
                                       {store_pool_vec_idxs}, {store_pool_gpr_idxs});
            add(reg_dst_aux, step * jcp_.dst_data_size);
        };

        // load m/v m:8-11, v:12-15
        worker_mvn_tails(reg_tails_num_active, load_mv_tails);

        // optimized scaleshift. 16-23 for weight, 24-31 for bias.
        // reg_post_ops_data[0]:----w0---- ----b0---- reg_post_ops_data[1]:----w1---- ----b1----
        mov(reg_oc_off, addr_oc_off_bk);
        size_t post_ops_data_offset = 0;
        ss_repeat_id = 0;
        for (int i = 0; i < optimized_scaleshift_num; i++) {
            mov(reg_tails_num_active, addr_tail_num);
            mov(reg_d_weights, ptr[reg_post_ops_data + post_ops_data_offset]);
            add(reg_d_weights, reg_oc_off);
            // bias = weight + C
            mov(reg_d_bias, reg_d_weights);
            mov(rax, reg_rt_shape);
            imul(rax, rax, sizeof(float));
            add(reg_d_bias, rax);

            worker_mvn_tails(reg_tails_num_active, load_weight_bias_tails);

            post_ops_data_offset += sizeof(float*);
            ss_repeat_id++;
        }

        Xbyak::Label loop_tails_label;
        Xbyak::Label loop_tails_end_label;
        L(loop_tails_label);
        {
            cmp(reg_work_amount, 0);
            jle(loop_tails_end_label, T_NEAR);
            mov(reg_tails_num_active, addr_tail_num);

            worker_mvn_tails(reg_tails_num_active, load_src_tails);

            // to next iteration(next work_amount)
            mov(rax, addr_vector_num);
            imul(rax, rax, vector_step * jcp_.src_data_size);
            add(reg_src_aux, rax);

            // norm
            mov(reg_tails_num_active, addr_tail_num);
            worker_mvn_tails(reg_tails_num_active, norm_tails);

            // optimized scaleShift
            ss_repeat_id = 0;
            for (int i = 0; i < optimized_scaleshift_num; i++) {
                mov(reg_tails_num_active, addr_tail_num);
                worker_mvn_tails(reg_tails_num_active, optimized_ss_tails);
                ss_repeat_id++;
            }

            // post-ops
            if (attr_.post_ops_.len() != 0) {
                mov(reg_tails_num_active, addr_tail_num);
                worker_mvn_tails(reg_tails_num_active, post_ops_tails);
            }

            // store
            mov(reg_tails_num_active, addr_tail_num);
            worker_mvn_tails(reg_tails_num_active, store_tails);

            // dst advance
            mov(rax, reg_rt_shape);
            sub(rax, addr_tail_num);
            imul(rax, rax, jcp_.dst_data_size);
            add(reg_dst_aux, rax);

            // reg_oc_off reset
            mov(rax, addr_tail_num);
            imul(rax, rax, sizeof(float));
            sub(reg_oc_off, rax);

            sub(reg_work_amount, 1);
            jmp(loop_tails_label, T_NEAR);
        }
        L(loop_tails_end_label);
        L(label_exit);
        add(rsp, 7 * gpr_size);
    }

    inline void norm_nspc_ac_ker() {
        Xbyak::Reg64 reg_rt_shape_bk = rdx;
        Xbyak::Reg64 reg_oc_off_bk = rax;
        mov(reg_rt_shape_bk, reg_rt_shape);
        if (attr_.post_ops_.len() != 0) {
            mov(reg_oc_off_bk, reg_oc_off);
        }

        Xbyak::Label loop_label;
        Xbyak::Label loop_end_label;
        L(loop_label);
        {
            cmp(reg_work_amount, 0);
            jle(loop_end_label, T_NEAR);

            mov(reg_rt_shape, reg_rt_shape_bk);
            if (attr_.post_ops_.len() != 0) {
                mov(reg_oc_off, reg_oc_off_bk);
            }

            worker_mvn_vector_unroll(reg_rt_shape);
            auto tails_func = [&](int tile_size) {
                worker_mvn_block(tile_size);
                add(reg_src, tile_size * jcp_.src_data_size);
                add(reg_dst, tile_size * jcp_.dst_data_size);
                if (attr_.post_ops_.len() != 0)
                    add(reg_oc_off, tile_size * sizeof(float));
            };
            worker_mvn_tails(reg_rt_shape, tails_func);

            sub(reg_work_amount, 1);
            jmp(loop_label, T_NEAR);
        }
        L(loop_end_label);
    }

    inline void worker_mvn_vector_unroll(Xbyak::Reg64& reg_work_amount) {
        Xbyak::Label mvn_loop_label;
        Xbyak::Label mvn_loop_end_label;

        int step_sub = jcp_.layout == MVNLayoutType::mvn_by_channel ? vector_step : 1;
        int step_left = jcp_.layout == MVNLayoutType::mvn_by_channel ? vector_step - 1 : 0;

        L(mvn_loop_label);
        {
            cmp(reg_work_amount, step_left);
            jle(mvn_loop_end_label, T_NEAR);

            worker_mvn_vector();

            add(reg_src, src_stride);
            add(reg_dst, dst_stride);
            if (jcp_.layout == MVNLayoutType::mvn_by_channel && attr_.post_ops_.len() != 0)
                add(reg_oc_off, vector_step * sizeof(float));

            sub(reg_work_amount, step_sub);

            jmp(mvn_loop_label, T_NEAR);
        }
        L(mvn_loop_end_label);
    }

    inline void worker_mvn_vector() {
        load_emitter[VECTOR]->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val.getIdx())},
            {}, {load_pool_gpr_idxs});

        uni_vsubps(vmm_val, vmm_val, vmm_mean);
        if (jcp_.normalize_variance)
            uni_vmulps(vmm_val, vmm_val, vmm_variance_inv);

        apply_post_ops(jcp_.dst_prc, vmm_val.getIdx(), jcp_.layout == MVNLayoutType::mvn_planar);

        store_emitter[VECTOR]->emit_code({static_cast<size_t>(vmm_val.getIdx())}, {static_cast<size_t>(reg_dst.getIdx())},
            {store_pool_vec_idxs}, {store_pool_gpr_idxs});
    }

    inline void worker_mvn_tails(Xbyak::Reg64& reg_tail_num, std::function<void(int)> func) {
        int tile_start_idx = (isa == cpu::x64::avx512_core) ? 0 : ((isa == cpu::x64::avx2) ? 1 : 2);
        Label tile_exit[kTileNum];
        for (int i = tile_start_idx; i < kTileNum; i++) {
            cmp(reg_tail_num, tile_size[i]);
            jl(tile_exit[i], T_NEAR);

            func(tile_size[i]);
            sub(reg_tail_num, tile_size[i]);

            L(tile_exit[i]);
        }
    }

    inline void worker_mvn_block(int block_num) {
        switch (block_num) {
        case 8:
            load_emitter[TAIL8]->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val.getIdx())},
                {}, {load_pool_gpr_idxs});
            break;
        case 4:
            load_emitter[TAIL4]->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val.getIdx())},
                {}, {load_pool_gpr_idxs});
            break;
        case 2:
            load_emitter[TAIL2]->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val.getIdx())},
                {}, {load_pool_gpr_idxs});
            break;
        case 1:
            load_emitter[TAIL1]->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val.getIdx())},
                {}, {load_pool_gpr_idxs});
            break;
        default:
            assert(!"MVN layer tails is processed only with 8/4/2/1 blocks.");
            break;
        }

        uni_vsubps(vmm_val, vmm_val, vmm_mean);
        if (jcp_.normalize_variance)
            uni_vmulps(vmm_val, vmm_val, vmm_variance_inv);

        apply_post_ops(jcp_.dst_prc, vmm_val.getIdx(), jcp_.layout == MVNLayoutType::mvn_planar);

        switch (block_num) {
        case 8:
            store_emitter[TAIL8]->emit_code({static_cast<size_t>(vmm_val.getIdx())}, {static_cast<size_t>(reg_dst.getIdx())},
                {store_pool_vec_idxs}, {store_pool_gpr_idxs});
            break;
        case 4:
            store_emitter[TAIL4]->emit_code({static_cast<size_t>(vmm_val.getIdx())}, {static_cast<size_t>(reg_dst.getIdx())},
                {store_pool_vec_idxs}, {store_pool_gpr_idxs});
            break;
        case 2:
            store_emitter[TAIL2]->emit_code({static_cast<size_t>(vmm_val.getIdx())}, {static_cast<size_t>(reg_dst.getIdx())},
                {store_pool_vec_idxs}, {store_pool_gpr_idxs});
            break;
        case 1:
            store_emitter[TAIL1]->emit_code({static_cast<size_t>(vmm_val.getIdx())}, {static_cast<size_t>(reg_dst.getIdx())},
                {store_pool_vec_idxs}, {store_pool_gpr_idxs});
            break;
        default:
            assert(!"MVN layer tails is processed only with 8/4/2/1 blocks.");
            break;
        }
    }

    void apply_post_ops(ov::element::Type dst_prc, size_t vmm_idx, bool is_broadcast) {
        const auto &p = attr_.post_ops_;
        int eltwise_inj_idx = 0;
        int depthwise_inj_idx = 0;
        int quantization_inj_idx = 0;
        int post_ops_data_offset = 0;
        for (int i = 0; i < p.len(); i++) {
            auto& post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors[eltwise_inj_idx]->compute_vector_range(vmm_idx, vmm_idx + 1);
                eltwise_inj_idx++;
            } else if (post_op.is_depthwise()) {
                if (post_op.depthwise.alg == alg_kind::depthwise_scale_shift && i < optimized_scaleshift_num) {
                    post_ops_data_offset += depthwise_injectors[depthwise_inj_idx]->memoryStep();
                    depthwise_inj_idx++;
                    continue;
                }
                mov(reg_d_weights, ptr[reg_post_ops_data + post_ops_data_offset]);
                add(reg_d_weights, reg_oc_off);

                depthwise_injectors[depthwise_inj_idx]->compute_vector_range(
                        vmm_idx, vmm_idx + 1, reg_d_weights, reg_d_weights, is_broadcast);

                post_ops_data_offset += depthwise_injectors[depthwise_inj_idx]->memoryStep();
                depthwise_inj_idx++;
            } else if (post_op.is_quantization()) {
                bool do_dequantization = post_op.quantization.alg == alg_kind::quantization_quantize_dequantize;
                bool do_rounding = do_dequantization || isFloatCompatible(dst_prc) || i != p.len() - 1;

                quantization_injectors[quantization_inj_idx]->init_crop_ptrs(reg_post_ops_data + post_ops_data_offset, reg_oc_off);
                quantization_injectors[quantization_inj_idx]->compute_crop(vmm_idx, vmm_idx + 1, 0, 0, is_broadcast);

                quantization_injectors[quantization_inj_idx]->init_input_scale_shift_ptrs(reg_post_ops_data + post_ops_data_offset, reg_oc_off);
                quantization_injectors[quantization_inj_idx]->compute_input_scale_shift(vmm_idx, vmm_idx + 1, 0, do_rounding, 0, is_broadcast);

                quantization_injectors[quantization_inj_idx]->init_output_scale_shift_ptrs(reg_post_ops_data + post_ops_data_offset, reg_oc_off);
                quantization_injectors[quantization_inj_idx]->compute_output_scale_shift(vmm_idx, vmm_idx + 1, 0, 0, is_broadcast);

                post_ops_data_offset += quantization_injectors[quantization_inj_idx]->memoryStep();
                quantization_inj_idx++;
            }
        }
    }
};

#endif // OPENVINO_ARCH_X86_64

//////////////////////////////////////////////////////////////////////////////////

bool MVN::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (op->get_output_partial_shape(0).rank().is_dynamic()) {
            errorMessage = "Unsupported dynamic input rank.";
            return false;
        }
        const auto& inDataRank = op->get_output_partial_shape(0).rank().get_length();
        if (inDataRank < 1 || inDataRank > 5) {
            errorMessage = "First input accepts ranks from 1 to 5. Actual: " + std::to_string(inDataRank);
            return false;
        }

        if (auto mvnOp = ov::as_type_ptr<const ov::op::v6::MVN>(op)) {
            auto axesOp = ov::as_type_ptr<ov::op::v0::Constant>(mvnOp->get_input_node_shared_ptr(1));
            if (!axesOp) {
                errorMessage = "Constant expected as the second input.";
                return false;
            }

            auto epsMode = mvnOp->get_eps_mode();
            if (epsMode != ov::op::MVNEpsMode::INSIDE_SQRT &&
                    epsMode != ov::op::MVNEpsMode::OUTSIDE_SQRT) {
                errorMessage = std::string("Just INSIDE_SQRT and OUTSIDE_SQRT epsilon mods are supported. Actual: ") +
                        std::to_string(static_cast<int>(epsMode));
                return false;
            }
            // Validates MVN node axes to check whether it can be executed on the current CPU implementation.
            // Supported cases:
            // 1D: axes: [0]
            // 2D: axes: [1]
            // 3D: axes: [1,2], [2]
            // 4D: axes: [1,2,3], [2,3]
            // 5D: axes: [1,2,3,4], [2,3,4]
            auto axesVal = axesOp->cast_vector<int>();
            for (int& axe : axesVal)
                axe = axe < 0 ? axe + inDataRank : axe;
            std::sort(axesVal.begin(), axesVal.end());
            if (inDataRank == 1) {
                if (axesVal.size() != 1 || axesVal[0] != 0) {
                    errorMessage = "Unsupported axes.";
                    return false;
                }
            } else {
                if (inDataRank > 5 || (static_cast<size_t>(inDataRank) != axesVal.size() + 1 &&
                                       static_cast<size_t>(inDataRank) != axesVal.size() + 2)) {
                    errorMessage = "Unsupported axes.";
                    return false;
                }
                int value = inDataRank - 1;
                for (int i = axesVal.size() - 1; i >= 0; i--, value--) {
                    if (axesVal[i] != value) {
                        errorMessage = "Unsupported axes.";
                        return false;
                    }
                }
            }
        } else if (auto mvnOp = ov::as_type_ptr<const ov::op::v0::MVN>(op)) {
        } else {
            errorMessage = "Node is not an instance of the MVN operation.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MVN::MVN(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
        : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    mvnAttrs.epsMode_ = INSIDE_SQRT;
    if (auto mvnOp = ov::as_type_ptr<ov::op::v6::MVN>(op)) {
        mvnAttrs.normalizeVariance_ = mvnOp->get_normalize_variance();
        mvnAttrs.epsValue_ = mvnOp->get_eps();
        if (mvnOp->get_eps_mode() == ov::op::MVNEpsMode::OUTSIDE_SQRT) {
            mvnAttrs.epsMode_ = OUTSIDE_SQRT;
        }

        mvnAttrs.initAcrossChannels_ = false;
        const auto& inDataShapeSize = getInputShapeAtPort(0).getRank();
        if (inDataShapeSize == mvnOp->input_value(1).get_shape()[0] + 1 || inDataShapeSize == 1)
            mvnAttrs.initAcrossChannels_ = true;
    } else if (auto mvnOp = ov::as_type_ptr<ov::op::v0::MVN>(op)) {
        mvnAttrs.normalizeVariance_ = mvnOp->get_normalize_variance();
        mvnAttrs.epsValue_ = mvnOp->get_eps();
        mvnAttrs.initAcrossChannels_ = mvnOp->get_across_channels();
    } else {
        OPENVINO_THROW_NOT_IMPLEMENTED("Node is not an instance of MVN from the operation set v0 or v6");
    }
    mvnAttrs.execAcrossChannels_ = mvnAttrs.initAcrossChannels_;
}

void MVN::getSupportedDescriptors() {}

static inline bool isUnaryEltwise(const NodePtr& node) {
    return one_of(node->getAlgorithm(), Algorithm::EltwiseRelu,
                                        Algorithm::EltwiseGeluErf,
                                        Algorithm::EltwiseGeluTanh,
                                        Algorithm::EltwiseElu,
                                        Algorithm::EltwiseSigmoid,
                                        Algorithm::EltwiseClamp,
                                        Algorithm::EltwiseTanh,
                                        Algorithm::EltwiseSwish,
                                        Algorithm::EltwiseHswish,
                                        Algorithm::EltwiseMish,
                                        Algorithm::EltwiseHsigmoid,
                                        Algorithm::EltwiseRoundHalfToEven,
                                        Algorithm::EltwiseRoundHalfAwayFromZero,
                                        Algorithm::EltwiseAbs,
                                        Algorithm::EltwiseSqrt,
                                        Algorithm::EltwiseSoftRelu);
}

void MVN::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    ov::element::Type inputPrecision = getOriginalInputPrecisionAtPort(0);
    ov::element::Type outputPrecision = getOriginalOutputPrecisionAtPort(0);
    if (!hasHardwareSupport(outputPrecision))
        outputPrecision = ov::element::f32;

    if (!fusedWith.empty()) {
        outputPrecision = fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0);
        onlyUnaryPostOps = true;
        for (auto &node : fusedWith) {
            if (isUnaryEltwise(node)) {
                continue;
            } else {
                onlyUnaryPostOps = false;
                break;
            }
        }
    }
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
    // ref with float planar and no fusion
    if (!mayiuse(cpu::x64::sse41)) {
        inputPrecision = outputPrecision = ov::element::f32;
    }
#endif
//Output precision has to be equal to input precision in ACL MVN
#if defined(OV_CPU_WITH_ACL)
    outputPrecision = inputPrecision;
#endif
    // TODO [DS]: inplace
    bool canBeInplace = !isDynamicNode() && (inputPrecision.size() == outputPrecision.size()) &&
                        (getParentEdgeAt(0)->getParent()->getChildEdges().size() == 1) &&
                        !getParentEdgeAt(0)->getParent()->isConstant();

    const size_t inputsNum = getParentEdges().size();
    NodeConfig config;
    config.inConfs.resize(inputsNum);
    config.outConfs.resize(1);
    config.inConfs[0].constant(false);
    config.outConfs[0].constant(false);
    config.inConfs[0].inPlace(-1);
    config.outConfs[0].inPlace(canBeInplace ? 0 : -1);
    if (inputsNum == 2) {
        config.inConfs[1].setMemDesc(std::make_shared<CpuBlockedMemoryDesc>(ov::element::i32, getInputShapeAtPort(1)));
        config.inConfs[1].constant(true);
    }

    auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    auto pushDesc = [&](LayoutType format, impl_desc_type impl_type, bool useAclExecutor = false) {
        config.inConfs[0].setMemDesc(creatorsMap.at(format)->createSharedDesc(inputPrecision, getInputShapeAtPort(0)));
        config.outConfs[0].setMemDesc(creatorsMap.at(format)->createSharedDesc(outputPrecision, getOutputShapeAtPort(0)));

        if (useAclExecutor) {
            std::vector<MemoryDescPtr> srcMemoryDescs;
            for (size_t i = 0; i < config.inConfs.size(); i++) {
                srcMemoryDescs.push_back(config.inConfs[i].getMemDesc());
            }
            std::vector<MemoryDescPtr> dstMemoryDescs;
            for (size_t i = 0; i < config.outConfs.size(); i++) {
                dstMemoryDescs.push_back(config.outConfs[i].getMemDesc());
            }

            auto factory = std::make_shared<MVNExecutorFactory>(mvnAttrs, srcMemoryDescs, dstMemoryDescs,
                                                                        std::make_shared<ExecutorContext>(context, getImplPriority()));
            if (!factory->isEmpty()) {
                supportedPrimitiveDescriptors.push_back({config, impl_type, factory});
            }
        } else {
            supportedPrimitiveDescriptors.push_back({config, impl_type});
        }
    };

#if defined(OV_CPU_WITH_ACL)
        pushDesc(LayoutType::nspc, undef, true);
        pushDesc(LayoutType::ncsp, undef, true);
        canUseAclExecutor = !supportedPrimitiveDescriptors.empty();
        if (canUseAclExecutor)
            return;
        else
            // Reference MVN implementation does not support fp16, so set fp32 explicitly
            inputPrecision = outputPrecision = ov::element::f32;
#endif // OV_CPU_WITH_ACL

    impl_desc_type impl_type;
    if (mayiuse(cpu::x64::avx512_core)) {
        impl_type = impl_desc_type::jit_avx512;
    } else if (mayiuse(cpu::x64::avx2)) {
        impl_type = impl_desc_type::jit_avx2;
    } else if (mayiuse(cpu::x64::sse41)) {
        impl_type = impl_desc_type::jit_sse42;
    } else {
        impl_type = impl_desc_type::ref;
    }

    if (mayiuse(cpu::x64::sse41)) {
        // nspc
        if (getInputShapeAtPort(0).getRank() == 4 || getInputShapeAtPort(0).getRank() == 5) {
            pushDesc(LayoutType::nspc, impl_type);
        }
        // blk
        if (impl_desc_type::jit_avx512 == impl_type) {
            if (getInputShapeAtPort(0).getRank() == 4 || getInputShapeAtPort(0).getRank() == 5) {
                pushDesc(LayoutType::nCsp16c, impl_type);
            }
        } else if (impl_desc_type::jit_avx2 ==  impl_type || impl_desc_type::jit_sse42 == impl_type) {
            if (getInputShapeAtPort(0).getRank() == 4 || getInputShapeAtPort(0).getRank() == 5) {
                pushDesc(LayoutType::nCsp8c, impl_type);
            }
        }
    }

    // planar
    if (canBeInplace)
        config.inConfs[0].inPlace(0);
    pushDesc(LayoutType::ncsp, impl_type);
}

MVN::MVNExecutorBase::MVNExecutorBase(const MVNAttrs& mvnAttrs)
    : mvnAttrs(mvnAttrs),
      src_data_size(mvnAttrs.src_prc.size()),
      dst_data_size(mvnAttrs.dst_prc.size()) {}

MVN::MVNJitExecutor::MVNJitExecutor(const MVNAttrs& mvnAttrs,
                                              const dnnl::primitive_attr& attr):
                                              MVNExecutorBase(mvnAttrs) {
    auto jcp = jit_mvn_config_params();
    jcp.src_prc = mvnAttrs.src_prc;
    jcp.dst_prc = mvnAttrs.dst_prc;
    jcp.src_data_size = src_data_size;
    jcp.dst_data_size = dst_data_size;
    jcp.layout = mvnAttrs.layout;
    jcp.normalize_variance = mvnAttrs.normalizeVariance_;
    jcp.across_channels = mvnAttrs.execAcrossChannels_;
#if defined(OPENVINO_ARCH_X86_64)
    if (mayiuse(cpu::x64::avx512_core)) {
        mvn_kernel.reset(new jit_uni_mvn_kernel_f32<cpu::x64::avx512_core>(jcp, *attr.get()));
        jcp.normalize_variance = false;
        mvn_mean_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<cpu::x64::avx512_core>(jcp));
        if (mvnAttrs.normalizeVariance_) {
            jcp.normalize_variance = true;
            mvn_variance_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<cpu::x64::avx512_core>(jcp));
        }
    } else if (mayiuse(cpu::x64::avx2)) {
        mvn_kernel.reset(new jit_uni_mvn_kernel_f32<cpu::x64::avx2>(jcp, *attr.get()));
        jcp.normalize_variance = false;
        mvn_mean_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<cpu::x64::avx2>(jcp));
        if (mvnAttrs.normalizeVariance_) {
            jcp.normalize_variance = true;
            mvn_variance_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<cpu::x64::avx2>(jcp));
        }
    } else if (mayiuse(cpu::x64::sse41)) {
        mvn_kernel.reset(new jit_uni_mvn_kernel_f32<cpu::x64::sse41>(jcp, *attr.get()));
        jcp.normalize_variance = false;
        mvn_mean_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<cpu::x64::sse41>(jcp));
        if (mvnAttrs.normalizeVariance_) {
            jcp.normalize_variance = true;
            mvn_variance_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<cpu::x64::sse41>(jcp));
        }
    } else {
        OPENVINO_THROW("Can't create jit MVN kernel");
    }
#endif // OPENVINO_ARCH_X86_64
    if (mvn_kernel)
        mvn_kernel->create_ker();
    if (mvn_mean_kernel)
        mvn_mean_kernel->create_ker();
    if (mvn_variance_kernel)
        mvn_variance_kernel->create_ker();
}

void MVN::MVNJitExecutor::exec(const uint8_t *src_data, uint8_t *dst_data, const void *post_ops_data_, const VectorDims& shape5d) {
    if (!mvn_mean_kernel || (mvnAttrs.normalizeVariance_ && !mvn_variance_kernel) || !mvn_kernel) {
        OPENVINO_THROW("MVN layer doesn't create kernel to execute on sse41 above platform.");
    }
    if (mvnAttrs.layout == MVNLayoutType::mvn_planar) {
        mvn_pln(src_data, dst_data, post_ops_data_, shape5d);
    } else if (mvnAttrs.layout == MVNLayoutType::mvn_by_channel) {
        mvn_nspc(src_data, dst_data, post_ops_data_, shape5d);
    } else {
        mvn_blk(src_data, dst_data, post_ops_data_, shape5d);
    }
}

MVN::MVNRefExecutor::MVNRefExecutor(const MVNAttrs& mvnAttrs):MVNExecutorBase(mvnAttrs) {}

void MVN::MVNRefExecutor::exec(const uint8_t *src_data, uint8_t *dst_data, const void *post_ops_data_, const VectorDims& shape5d) {
    mvn_ref(src_data, dst_data, shape5d);
}

void MVN::prepareParams() {
    auto dstMemPtr = getDstMemoryAtPort(0);
    auto srcMemPtr = getSrcMemoryAtPort(0);
    if (!dstMemPtr || !dstMemPtr->isDefined())
        OPENVINO_THROW("Destination memory is undefined.");
    if (!srcMemPtr || !srcMemPtr->isDefined())
        OPENVINO_THROW("Input memory is undefined.");
    if (getSelectedPrimitiveDescriptor() == nullptr)
        OPENVINO_THROW("Preferable primitive descriptor is not set.");

    const VectorDims in_dims = srcMemPtr->getStaticDims();
    transformTo5DCase(in_dims);

#if defined(OPENVINO_ARCH_X86_64)
    // New shape5D always need prepare via transformTo5DCase(), which is need in exec().
    // MVN itself and unary post ops is totally shape agnostic, execPtr can be reused directly w/o recompilation and setPostOps when shape is changed.
    // As key have not shape, if shape changes and new post ops attr is also the same, execPtr can still hit.
    // If new shape(channel changes) impact post ops attr, such as entry.quantization.offset, entry.depthwise.offset, entry.quantization.per_channel,
    // which is participate in compilation, even postOpsData is passed in runtime, still need recompilation.
    if (execPtr != nullptr && (fusedWith.empty() || onlyUnaryPostOps)) {
        return;
    }
#endif

    auto selectedPD = getSelectedPrimitiveDescriptor();
    mvnAttrs.src_prc = selectedPD->getConfig().inConfs[0].getMemDesc()->getPrecision();
    mvnAttrs.dst_prc = selectedPD->getConfig().outConfs[0].getMemDesc()->getPrecision();
    if (getParentEdgeAt(0)->getMemory().getDesc().hasLayoutType(LayoutType::ncsp)) {
        mvnAttrs.layout = MVNLayoutType::mvn_planar;
    } else if (getParentEdgeAt(0)->getMemory().getDesc().hasLayoutType(LayoutType::nspc)) {
        mvnAttrs.layout = MVNLayoutType::mvn_by_channel;
    } else {
        mvnAttrs.layout = MVNLayoutType::mvn_block;
    }

    if (canUseAclExecutor) {
        std::vector<MemoryDescPtr> srcMemoryDescs;
        for (size_t i = 0; i < getParentEdges().size(); i++) {
            srcMemoryDescs.push_back(getSrcMemoryAtPort(i)->getDescPtr());
        }
        std::vector<MemoryDescPtr> dstMemoryDescs;
        dstMemoryDescs.push_back(getDstMemoryAtPort(0)->getDescPtr());

        auto selectedPD = getSelectedPrimitiveDescriptor();
        aclExecPtr = selectedPD->getExecutorFactoryAs<MVNExecutorFactory>()->makeExecutor(mvnAttrs, srcMemoryDescs, dstMemoryDescs, {});
        selectedPD->setImplementationType(aclExecPtr->getImplType());

        return;
    }

    MVNKey key = {mvnAttrs, dnnl::primitive_attr()};
    setPostOps(key.attr, true);

    auto builder = [&](const MVNKey& key) -> std::shared_ptr<MVNExecutorBase> {
        std::shared_ptr<MVNExecutorBase> executor;
        if (mayiuse(cpu::x64::sse41)) {
            executor = std::make_shared<MVNJitExecutor>(key.mvnAttrs, key.attr);
        } else {
            executor = std::make_shared<MVNRefExecutor>(key.mvnAttrs);
        }
        return executor;
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, builder);
    execPtr = result.first;
}

void MVN::transformTo5DCase(const VectorDims& shape) {
    size_t rank = shape.size();
    // for 1 and 2 rank, if initAcrossChannels_ is true, adjust shape to fully vectorize under unified 5d procedure.
    // otherwise there are not enough data in spatial dimension to process in one kernel.
    switch (rank) {
        case 1 :  // C
            if (mvnAttrs.initAcrossChannels_) {
                shape5D = {1, 1, 1, 1, shape[0]};
                mvnAttrs.execAcrossChannels_ = false;
                break;
            } else {
                shape5D = {1, shape[0], 1, 1, 1};
                break;
            }
        case 2 :  // NC
            if (mvnAttrs.initAcrossChannels_) {
                shape5D = {1, shape[0], 1, shape[1], 1};
                mvnAttrs.execAcrossChannels_ = false;
                break;
            } else {
                shape5D = {shape[0], shape[1], 1, 1, 1};
                break;
            }
        case 3 : { shape5D = {shape[0], shape[1], 1, shape[2], 1}; break; }
        case 4 : { shape5D = {shape[0], shape[1], 1, shape[2], shape[3]}; break; }
        case 5 : { shape5D = {shape[0], shape[1], shape[2], shape[3], shape[4]}; break; }
        default: {
            OPENVINO_THROW("MVN layer with name '",
                           getName(),
                           "' doesn't support planar layout with rank: ",
                           shape.size());
        }
    }
}

void MVN::setPostOps(dnnl::primitive_attr &attr, bool initWeights) {
    dnnl::post_ops ops;
    postOpsDataPtrs.clear();
    for (auto &node : fusedWith) {
        auto* fakeQuantizeNode = dynamic_cast<FakeQuantize *>(node.get());
        if (fakeQuantizeNode) {
            fakeQuantizeNode->appendPostOps(ops, {}, postOpsDataPtrs);
            continue;
        }

        auto* eltwiseNode = dynamic_cast<Eltwise *>(node.get());
        if (eltwiseNode) {
            eltwiseNode->appendPostOps(ops, shape5D, postOpsDataPtrs);
            continue;
        }
        OPENVINO_THROW("Fusing of ",
                       NameFromType(node->getType()),
                       " operation to ",
                       NameFromType(this->getType()),
                       " node is not implemented");
    }
    attr.set_post_ops(ops);
}

void MVN::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

void MVN::execute(dnnl::stream strm) {
    auto dstMemPtr = getDstMemoryAtPort(0);
    auto srcMemPtr = getSrcMemoryAtPort(0);

    if (execPtr) {
        uint8_t *dst_data = dstMemPtr->getDataAs<uint8_t>();
        uint8_t *src_data = srcMemPtr->getDataAs<uint8_t>();
        execPtr->exec(src_data, dst_data, postOpsDataPtrs.data(), shape5D);
    } else if (aclExecPtr) {
        aclExecPtr->exec({srcMemPtr}, {dstMemPtr}, postOpsDataPtrs.data());
    } else {
        OPENVINO_THROW("Can't execute Interpolate node. Primitive didn't created");
    }
}

void MVN::MVNJitExecutor::mvn_pln(const uint8_t* src_data, uint8_t* dst_data, const void *post_ops_data_, const VectorDims& shape5d) {
    size_t blk_size = 1;  // blk size in vmm
    if (mayiuse(cpu::x64::avx512_core)) {
        blk_size = 16;
    } else if (mayiuse(cpu::x64::avx2)) {
        blk_size = 8;
    } else if (mayiuse(cpu::x64::sse41)) {
        blk_size = 4;
    }

    const size_t N = shape5d[0];
    const size_t C = shape5d[1];
    const size_t D = shape5d[2];
    const size_t H = shape5d[3];
    const size_t W = shape5d[4];

    size_t C1 = H * W;
    size_t C2 = C1 * D;
    size_t C3 = C2 * C;

    if (mvnAttrs.execAcrossChannels_) {
        parallel_for(N, [&](int b) {
            size_t cb = b * C3;
            // Calculate mean value for one instance in batch
            // Parallel sum for each channel
            float C3inv = 1.f / static_cast<float>(C3);
            float mean_temp = 0.0f;
            mean_temp = parallel_sum(C, mean_temp, [&](size_t c)->float {
                float mean_internal = 0.0f;
                size_t cc = cb + c * C2;
                auto arg = jit_mvn_call_args();
                arg.src = src_data + cc * src_data_size;
                arg.sum = static_cast<float*>(&mean_internal);
                arg.work_amount = static_cast<size_t>(C2 / blk_size); // for vector part
                arg.rt_shape_size = static_cast<size_t>(C2 % blk_size);
                arg.post_op_data = post_ops_data_;
                (*mvn_mean_kernel)(&arg);
                return mean_internal;
            });

            float mean = mean_temp * C3inv;

            // calculate variance value for one instance in batch
            // parallel sum for each channel
            if (mvnAttrs.normalizeVariance_) {
                float variance_temp = 0.0f;
                variance_temp = parallel_sum(C, variance_temp, [&](size_t c)->float {
                    float variance_internal = 0.0f;
                    size_t cc = cb + c * C2;
                    auto arg = jit_mvn_call_args();
                    arg.src = src_data + cc * src_data_size;
                    arg.mean = static_cast<float*>(&mean);
                    arg.variance = static_cast<float*>(&variance_internal);
                    arg.work_amount = static_cast<size_t>(C2 / blk_size);  // vector part
                    arg.rt_shape_size = static_cast<size_t>(C2 % blk_size);  // for tails
                    arg.post_op_data = post_ops_data_;
                    (*mvn_variance_kernel)(&arg);
                    return variance_internal;
                });

                float variance = 1.f;
                if (mvnAttrs.epsMode_ == INSIDE_SQRT)
                    variance /= sqrtf(variance_temp * C3inv + mvnAttrs.epsValue_);
                else if (mvnAttrs.epsMode_ == OUTSIDE_SQRT)
                    variance /= sqrtf(variance_temp * C3inv) + mvnAttrs.epsValue_;

                // mvn for one instance in batch
                parallel_for(C, [&](int c) {
                    size_t cc = cb + c * C2;
                    auto arg = jit_mvn_call_args();
                    arg.src = src_data + cc * src_data_size;
                    arg.dst = dst_data + cc * dst_data_size;
                    arg.mean = static_cast<float*>(&mean);
                    arg.variance = static_cast<float*>(&variance);
                    arg.work_amount = static_cast<size_t>(C2 / blk_size);  // work amount for vector part
                    arg.rt_shape_size = static_cast<size_t>(C2 % blk_size);  // for tails
                    arg.oc_off = sizeof(float) * c;
                    arg.post_op_data = post_ops_data_;
                    (*mvn_kernel)(&arg);
                });
            } else {
                // mvn for one instance in batch
                parallel_for(C, [&](int c) {
                    size_t cc = cb + c * C2;
                    auto arg = jit_mvn_call_args();
                    arg.src = src_data + cc * src_data_size;
                    arg.dst = dst_data + cc * dst_data_size;
                    arg.mean = static_cast<float*>(&mean);
                    arg.work_amount = static_cast<size_t>(C2 / blk_size);
                    arg.rt_shape_size = static_cast<size_t>(C2 % blk_size);  // for tails
                    arg.oc_off = sizeof(float) * c;
                    arg.post_op_data = post_ops_data_;
                    (*mvn_kernel)(&arg);
                });
            }
        });
    } else {
        parallel_for2d(N, C, [&](size_t b, size_t c) {
            size_t cb = b * C3;
            size_t cc = cb + c * C2;
            float C2inv = 1.f / static_cast<float>(C2);

            // mean for this channel
            float mean = 0.f;
            // the same arg for three kernels
            auto arg = jit_mvn_call_args();
            arg.src = src_data + cc * src_data_size;
            arg.dst = dst_data + cc * dst_data_size;
            arg.sum = static_cast<float*>(&mean);
            arg.work_amount = static_cast<size_t>(C2 / blk_size);
            arg.rt_shape_size = static_cast<size_t>(C2 % blk_size);
            arg.oc_off = static_cast<size_t>(c * sizeof(float));
            arg.post_op_data = post_ops_data_;
            (*mvn_mean_kernel)(&arg);

            mean *= C2inv;

            if (mvnAttrs.normalizeVariance_) {
                // variance for this channel
                float variance = 0.f;
                arg.mean = static_cast<float*>(&mean);
                arg.variance = static_cast<float*>(&variance);
                (*mvn_variance_kernel)(&arg);

                if (mvnAttrs.epsMode_ == INSIDE_SQRT)
                    variance = 1.f / sqrtf(variance * C2inv + mvnAttrs.epsValue_);
                else if (mvnAttrs.epsMode_ == OUTSIDE_SQRT)
                    variance = 1.f / (sqrtf(variance * C2inv) + mvnAttrs.epsValue_);

                // mvn for this channel
                (*mvn_kernel)(&arg);
            } else {
                // mvn for this channel
                arg.mean = static_cast<float*>(&mean);
                (*mvn_kernel)(&arg);
            }
        });
    }
}

void MVN::MVNRefExecutor::mvn_ref(const uint8_t* src_data, uint8_t* dst_data, const VectorDims& shape5d) {
    const float *src_data_ptr = reinterpret_cast<const float *>(src_data);
    float *dst_data_ptr = reinterpret_cast<float *>(dst_data);
    const size_t N = shape5d[0];
    const size_t C = shape5d[1];
    const size_t D = shape5d[2];
    const size_t H = shape5d[3];
    const size_t W = shape5d[4];

    size_t C1 = H * W;
    size_t C2 = C1 * D;
    size_t C3 = C2 * C;

    parallel_for(N, [&](int b) {
        size_t cb = b * C3;
        if (mvnAttrs.execAcrossChannels_) {
            // Parallel sum for each channel for mean
            float C3inv = 1.f / static_cast<float>(C3);
            float mean_temp = 0.0f;

            mean_temp = parallel_sum(C, mean_temp, [&](size_t c)->float {
                float mean_internal = 0.0f;
                size_t cc = cb + c * C2;
                for (size_t sp = 0lu; sp < C2; sp++) {
                    mean_internal += src_data_ptr[cc + sp];
                }
                return mean_internal;
            });

            float mean = mean_temp * C3inv;

            if (mvnAttrs.normalizeVariance_) {
                // parallel sum for each channel for variance
                float variance_temp = 0.0f;
                variance_temp = parallel_sum(C, variance_temp, [&](size_t c)->float {
                    float variance_internal = 0.0f;
                    size_t cc = cb + c * C2;
                    for (size_t sp = 0lu; sp < C2; sp++) {
                        variance_internal += (src_data_ptr[cc + sp] - mean) * (src_data_ptr[cc + sp] - mean);
                    }
                    return variance_internal;
                });

                float variance = 1.f;
                if (mvnAttrs.epsMode_ == INSIDE_SQRT)
                    variance = 1.f / sqrtf(variance_temp * C3inv + mvnAttrs.epsValue_);
                else if (mvnAttrs.epsMode_ == OUTSIDE_SQRT)
                    variance = 1.f / (sqrtf(variance_temp * C3inv) + mvnAttrs.epsValue_);

                parallel_for(C, [&](int c) {
                    size_t cc = cb + c * C2;
                    for (size_t sp = 0lu; sp < C2; sp++) {
                        dst_data_ptr[cc + sp] = (src_data_ptr[cc + sp] - mean) * variance;
                    }
                });
            } else {
                parallel_for(C, [&](int c) {
                    size_t cc = cb + c * C2;
                    for (size_t sp = 0lu; sp < C2; sp++) {
                        dst_data_ptr[cc + sp] = src_data_ptr[cc + sp] - mean;
                    }
                });
            }
        } else {  // per channel
            float C2inv = 1.f / static_cast<float>(C2);
            parallel_for(C, [&](size_t c) {
                // mean for this channel
                float mean = 0.f;
                size_t cc = cb + c * C2;
                for (size_t sp = 0lu; sp < C2; sp++) {
                    mean += src_data_ptr[cc + sp];
                }
                mean *= C2inv;

                if (mvnAttrs.normalizeVariance_) {
                    // variance for this channel
                    float variance = 0.f;
                    for (size_t sp = 0lu; sp < C2; sp++) {
                        variance += (src_data_ptr[cc + sp] - mean) * (src_data_ptr[cc + sp] - mean);
                    }

                    if (mvnAttrs.epsMode_ == INSIDE_SQRT)
                        variance = 1.f / sqrtf(variance * C2inv + mvnAttrs.epsValue_);
                    else if (mvnAttrs.epsMode_ == OUTSIDE_SQRT)
                        variance = 1.f / (sqrtf(variance * C2inv) + mvnAttrs.epsValue_);

                    // mvn for this channel
                    for (size_t sp = 0lu; sp < C2; sp++) {
                        dst_data_ptr[cc + sp] = (src_data_ptr[cc + sp] - mean) * variance;
                    }
                } else {
                    // mvn for this channel
                    for (size_t sp = 0lu; sp < C2; sp++) {
                        dst_data_ptr[cc + sp] = src_data_ptr[cc + sp] - mean;
                    }
                }
            });
        }
    });
}

void MVN::MVNJitExecutor::mvn_nspc(const uint8_t* src_data, uint8_t* dst_data, const void *post_ops_data_, const VectorDims& shape5d) {
    size_t blk_size = 1;  // channel blk for memory layout
    if (mayiuse(cpu::x64::avx512_core)) {
        blk_size = 16;
    } else if (mayiuse(cpu::x64::avx2)) {
        blk_size = 8;
    } else {
        blk_size = 4;
    }

    const size_t N = shape5d[0];
    const size_t C = shape5d[1];
    const size_t D = shape5d[2];
    const size_t H = shape5d[3];
    const size_t W = shape5d[4];

    const size_t threads_num = parallel_get_max_threads();
    size_t aux_buffer_size = mvnAttrs.execAcrossChannels_ ? 1 : rnd_up(C, blk_size) + blk_size;
    auto b_loop = [&](size_t b) {
        std::vector<float> mean_buffer(aux_buffer_size * threads_num, 0.f);
        std::vector<float> variance_buffer;
        if (mvnAttrs.normalizeVariance_) {
            variance_buffer.resize(aux_buffer_size * threads_num, 0.f);
        }
        size_t b_offset = b * C * D * H * W;

        // kernel_type: 0 for mean, 1 for variance, 2 for normalization
        auto worker = [&](const bool across_channel, const int kernel_type) {
            parallel_nt(threads_num, [&](const int ithr, const int nthr) {
                size_t start = 0, end = 0;
                splitter(D * H * W, nthr, ithr, start, end);

                auto arg = jit_mvn_call_args();
                arg.src = src_data + (b_offset + (start * C)) * src_data_size;
                if (0 == kernel_type) {
                    arg.sum = &mean_buffer[aux_buffer_size * ithr];
                } else if (1 == kernel_type) {
                    arg.mean = &mean_buffer[0];
                    arg.variance = &variance_buffer[aux_buffer_size * ithr];
                } else if (2 == kernel_type) {
                    arg.dst = dst_data + (b_offset + (start * C)) * dst_data_size;
                    arg.mean = &mean_buffer[0];
                    if (mvnAttrs.normalizeVariance_)
                        arg.variance = &variance_buffer[0];
                    arg.oc_off = 0;
                    arg.post_op_data = post_ops_data_;
                }
                if (across_channel) {
                    if (kernel_type == 2) {
                        arg.work_amount = end - start;
                        arg.rt_shape_size = C;
                    } else {
                        arg.work_amount = (end - start) * C;
                    }
                } else {
                    arg.work_amount = (end - start);
                    arg.rt_shape_size = C;
                }

                if (0 == kernel_type) {
                    (*mvn_mean_kernel)(&arg);
                } else if (1 == kernel_type) {
                    (*mvn_variance_kernel)(&arg);
                } else if (2 == kernel_type) {
                    (*mvn_kernel)(&arg);
                }
            });
        };

        if (mvnAttrs.execAcrossChannels_) {
            float size_inv = 1.f / static_cast<float>(C * D * H * W);
            worker(true, 0);
            for (size_t i = 1; i < threads_num; i++) {
                mean_buffer[0] += mean_buffer[i];
            }
            mean_buffer[0] *= size_inv;
            if (mvnAttrs.normalizeVariance_) {
                worker(true, 1);
                for (size_t i = 1; i < threads_num; i++) {
                    variance_buffer[0] += variance_buffer[i];
                }
                if (mvnAttrs.epsMode_ == INSIDE_SQRT)
                    variance_buffer[0] = 1.f / sqrtf(variance_buffer[0] * size_inv + mvnAttrs.epsValue_);
                else if (mvnAttrs.epsMode_ == OUTSIDE_SQRT)
                    variance_buffer[0] = 1.f / (sqrtf(variance_buffer[0] * size_inv) + mvnAttrs.epsValue_);
            }
            worker(true, 2);
        } else {  // for per_channel
            float size_inv = 1.f / static_cast<float>(D * H * W);
            worker(false, 0);
            for (size_t i = 1; i < threads_num; i++) {
                for (size_t c = 0; c < C; c++)
                    mean_buffer[c] += mean_buffer[c + aux_buffer_size * i];
            }
            for (size_t c = 0; c < C; c++)
                mean_buffer[c] *= size_inv;
            if (mvnAttrs.normalizeVariance_) {
                worker(false, 1);
                for (size_t i = 1; i < threads_num; i++) {
                    for (size_t c = 0; c < C; c++)
                        variance_buffer[c] += variance_buffer[c + aux_buffer_size * i];
                }
                for (size_t c = 0; c < C; c++) {
                    if (mvnAttrs.epsMode_ == INSIDE_SQRT)
                        variance_buffer[c] = 1.f / sqrtf(variance_buffer[c] * size_inv + mvnAttrs.epsValue_);
                    else if (mvnAttrs.epsMode_ == OUTSIDE_SQRT)
                        variance_buffer[c] = 1.f / (sqrtf(variance_buffer[c] * size_inv) + mvnAttrs.epsValue_);
                }
            }
            worker(false, 2);
        }
    };

    parallel_nt_static(threads_num, [&](const int ithr, const int nthr) {
        for_1d(ithr, nthr, N, b_loop);
    });
}

void MVN::MVNJitExecutor::mvn_blk(const uint8_t* src_data, uint8_t* dst_data, const void *post_ops_data_, const VectorDims& shape5d) {
    size_t blk_size = 1;  // channel blk for memory layout
    if (mayiuse(cpu::x64::avx512_core)) {
        blk_size = 16;
    } else {
        blk_size = 8;
    }

    const size_t N = shape5d[0];
    const size_t C = shape5d[1];
    const size_t D = shape5d[2];
    const size_t H = shape5d[3];
    const size_t W = shape5d[4];

    const size_t CB = div_up(C, blk_size);

    const size_t C0 = W * blk_size;
    const size_t C1 = C0 * H;
    const size_t C2 = C1 * D;
    const size_t C3 = C2 * CB;
    const size_t C5 = C * D * H * W;

    const size_t threads_num = parallel_get_max_threads();
    size_t aux_buffer_size = mvnAttrs.execAcrossChannels_ ? blk_size : rnd_up(C, blk_size);
    aux_buffer_size += blk_size;
    std::vector<float> mean_buffer(aux_buffer_size * threads_num);
    std::vector<float> variance_buffer(aux_buffer_size * threads_num);

    for (size_t b = 0lu; b < N; b++) {
        size_t b_offset = b * C3;
        if (mvnAttrs.execAcrossChannels_) {
            // mean for this instance in batch
            float C5inv = 1.f / static_cast<float>(C5);
            float mean_temp = 0.0f;
            mean_temp = parallel_sum3d(CB, D, H, mean_temp, [&](size_t cb, size_t d, size_t h)->float {
                size_t src_offset = b_offset + cb * C2 + d * C1 + h * C0;

                float mean_internal = 0.0f;
                /////////////////////////////////
                //          W           //  |
                //                      //  |
                //                      //  |
                //blk +  +  +  +  +  +  //  |  +
                //                      //  |
                //                      //  |
                //                      // \|/
                /////////////////////////////////
                auto thread_idx = static_cast<size_t>(parallel_get_thread_num());
                if (thread_idx >= threads_num) {
                    return mean_internal;
                }
                auto mean_buffer_ptr = &mean_buffer[aux_buffer_size * thread_idx];
                for (size_t i = 0; i < blk_size; i++)
                    mean_buffer_ptr[i] = 0.f;

                auto arg = jit_mvn_call_args();
                arg.src = src_data + src_offset * src_data_size;
                arg.sum = mean_buffer_ptr;
                arg.work_amount = static_cast<size_t>(W);
                // real tail number or tail is 0(for full vector block).
                arg.rt_shape_size = (C - cb * blk_size) < blk_size ? static_cast<size_t>(C % blk_size) : 0;
                arg.oc_off = static_cast<size_t>(cb * blk_size * sizeof(float));  // for tail process
                (*mvn_mean_kernel)(&arg); // for W * blk

                size_t min_cb = (std::min)(blk_size, C - cb * blk_size);
                for (size_t i = 0; i < min_cb; i++)
                    mean_internal += mean_buffer_ptr[i];
                return mean_internal;
            });
            float mean = mean_temp * C5inv;

            if (mvnAttrs.normalizeVariance_) {
                // variance: sum((x-mean)*(x-mean)) for one instance in batch
                float variance_temp = 0.0f;
                variance_temp = parallel_sum3d(CB, D, H, variance_temp, [&](size_t cb, size_t d, size_t h)->float {
                    size_t src_offset = b_offset + cb * C2 + d * C1 + h * C0;

                    float variance_internal = 0.0f;
                    auto variance_buffer_ptr = &variance_buffer[aux_buffer_size * static_cast<size_t>(parallel_get_thread_num())];
                    for (size_t i = 0; i < blk_size; i++)
                        variance_buffer_ptr[i] = 0.f;

                    auto arg = jit_mvn_call_args();
                    arg.src = src_data + src_offset * src_data_size;
                    arg.mean = static_cast<float*>(&mean);
                    arg.variance = variance_buffer_ptr;
                    arg.work_amount = static_cast<size_t>(W);
                    arg.rt_shape_size = (C - cb * blk_size) < blk_size ? static_cast<size_t>(C % blk_size) : 0;
                    arg.oc_off = cb * blk_size * sizeof(float);
                    arg.post_op_data = post_ops_data_;
                    (*mvn_variance_kernel)(&arg);

                    size_t min_cb = (std::min)(blk_size, C - cb * blk_size);
                    for (size_t i = 0; i < min_cb; i++)
                        variance_internal += variance_buffer_ptr[i];
                    return variance_internal;
                });

                float variance = 1.f;
                if (mvnAttrs.epsMode_ == INSIDE_SQRT)
                    variance /= sqrtf(variance_temp * C5inv + mvnAttrs.epsValue_);
                else if (mvnAttrs.epsMode_ == OUTSIDE_SQRT)
                    variance /= sqrtf(variance_temp * C5inv) + mvnAttrs.epsValue_;

                // mvn for one instance in batch
                parallel_for3d(CB, D, H, [&](size_t cb, size_t d, size_t h) {
                    size_t src_offset = b_offset + cb * C2 + d * C1 + h * C0;
                    auto arg = jit_mvn_call_args();
                    arg.src = src_data + src_offset * src_data_size;
                    arg.dst = dst_data + src_offset * dst_data_size;
                    arg.mean = static_cast<float*>(&mean);
                    arg.variance = static_cast<float*>(&variance);
                    arg.work_amount = static_cast<size_t>(W);
                    arg.rt_shape_size = (C - cb * blk_size) < blk_size ? static_cast<size_t>(C % blk_size) : 0;
                    arg.oc_off = cb * blk_size * sizeof(float);
                    arg.post_op_data = post_ops_data_;
                    (*mvn_kernel)(&arg);
                });
            } else {
                // mvn for one instance in batch
                parallel_for3d(CB, D, H, [&](size_t cb, size_t d, size_t h) {
                    size_t src_offset = b_offset + cb * C2 + d * C1 + h * C0;
                    auto arg = jit_mvn_call_args();
                    arg.src = src_data + src_offset * src_data_size;
                    arg.dst = dst_data + src_offset * dst_data_size;
                    arg.mean = static_cast<float*>(&mean);
                    arg.work_amount = static_cast<size_t>(W);
                    arg.rt_shape_size = (C - cb * blk_size) < blk_size ? static_cast<size_t>(C % blk_size) : 0;
                    arg.oc_off = cb * blk_size * sizeof(float);
                    arg.post_op_data = post_ops_data_;
                    (*mvn_kernel)(&arg);
                });
            }
        } else {  // for per_channel
            float size_inv = 1.f / static_cast<float>(D * H * W);
            for (size_t i = 0; i < mean_buffer.size(); i++)
                mean_buffer[i] = 0.f;

            // one thread for one C*W size(the same H) to get C size result for the same H, added to last group result
            // keep the compute order the same as planar
            auto dh_loop = [&](size_t thr_idx, size_t d, size_t h) {
                for (size_t cb = 0; cb < CB; cb++) {
                    size_t src_offset = b_offset + cb * C2 + d * C1 + h * C0;
                    auto mean_buffer_ptr = &mean_buffer[blk_size * cb + aux_buffer_size * thr_idx];

                    auto arg = jit_mvn_call_args();
                    arg.src = src_data + src_offset * src_data_size;
                    arg.sum = mean_buffer_ptr;
                    arg.work_amount = static_cast<size_t>(W);
                    arg.rt_shape_size = (C - cb * blk_size) < blk_size ? static_cast<size_t>(C % blk_size) : 0;
                    arg.oc_off = cb * blk_size * sizeof(float);
                    arg.post_op_data = post_ops_data_;
                    (*mvn_mean_kernel)(&arg);
                }
            };

            parallel_nt_static(threads_num, [&](const int ithr, const int nthr) {
                for_2d(ithr, nthr, D, H, dh_loop);
            });

            for (size_t i = 1; i < threads_num; i++) {
                for (size_t c = 0; c < C; c++)
                    mean_buffer[c] += mean_buffer[c + aux_buffer_size * i];
            }
            for (size_t c = 0; c < C; c++)
                mean_buffer[c] *= size_inv;

            if (mvnAttrs.normalizeVariance_) {
                for (size_t i = 0; i < variance_buffer.size(); i++)
                    variance_buffer[i] = 0.f;

                auto dh_loop = [&](size_t thr_idx, size_t d, size_t h) {
                    for (size_t cb = 0; cb < CB; cb++) {
                        size_t src_offset = b_offset + cb * C2 + d * C1 + h * C0;
                        auto mean_buffer_ptr = &mean_buffer[blk_size * cb];
                        auto variance_buffer_ptr = &variance_buffer[blk_size * cb + aux_buffer_size * thr_idx];

                        auto arg = jit_mvn_call_args();
                        arg.src = src_data + src_offset * src_data_size;
                        arg.mean = mean_buffer_ptr;
                        arg.variance = variance_buffer_ptr;
                        arg.work_amount = static_cast<size_t>(W);
                        arg.rt_shape_size = (C - cb * blk_size) < blk_size ? static_cast<size_t>(C % blk_size) : 0;
                        arg.oc_off = cb * blk_size * sizeof(float);
                        arg.post_op_data = post_ops_data_;
                        (*mvn_variance_kernel)(&arg);
                    }
                };

                parallel_nt_static(threads_num, [&](const int ithr, const int nthr) {
                    for_2d(ithr, nthr, D, H, dh_loop);
                });

                for (size_t i = 1; i < threads_num; i++) {
                    for (size_t c = 0; c < C; c++)
                        variance_buffer[c] += variance_buffer[c + aux_buffer_size * i];
                }
                for (size_t c = 0; c < C; c++) {
                    if (mvnAttrs.epsMode_ == INSIDE_SQRT)
                        variance_buffer[c] = 1.f / sqrtf(variance_buffer[c] * size_inv + mvnAttrs.epsValue_);
                    else if (mvnAttrs.epsMode_ == OUTSIDE_SQRT)
                        variance_buffer[c] = 1.f / (sqrtf(variance_buffer[c] * size_inv) + mvnAttrs.epsValue_);
                }

                parallel_for2d(D, H, [&](size_t d, size_t h) {
                    for (size_t cb = 0; cb < CB; cb++) {
                        size_t src_offset = b_offset + cb * C2 + d * C1 + h * C0;
                        auto mean_buffer_ptr = &mean_buffer[blk_size * cb];
                        auto variance_buffer_ptr = &variance_buffer[blk_size * cb];

                        auto arg = jit_mvn_call_args();
                        arg.src = src_data + src_offset * src_data_size;
                        arg.dst = dst_data + src_offset * dst_data_size;
                        arg.mean = mean_buffer_ptr;
                        arg.variance = variance_buffer_ptr;
                        arg.work_amount = static_cast<size_t>(W);
                        arg.rt_shape_size = (C - cb * blk_size) < blk_size ? static_cast<size_t>(C % blk_size) : 0;
                        arg.oc_off = cb * blk_size * sizeof(float);
                        arg.post_op_data = post_ops_data_;
                        (*mvn_kernel)(&arg);
                    }
                });
            } else {
                // normalizeVariance_ == false
                parallel_for2d(D, H, [&](size_t d, size_t h) {
                    for (size_t cb = 0; cb < CB; cb++) {
                        size_t src_offset = b_offset + cb * C2 + d * C1 + h * C0;
                        auto mean_buffer_ptr = &mean_buffer[blk_size * cb];

                        auto arg = jit_mvn_call_args();
                        arg.src = src_data + src_offset * src_data_size;
                        arg.dst = dst_data + src_offset * dst_data_size;
                        arg.mean = mean_buffer_ptr;
                        arg.work_amount = static_cast<size_t>(W);
                        arg.rt_shape_size = (C - cb * blk_size) < blk_size ? static_cast<size_t>(C % blk_size) : 0;
                        arg.oc_off = cb * blk_size * sizeof(float);
                        arg.post_op_data = post_ops_data_;
                        (*mvn_kernel)(&arg);
                    }
                });
            }
        }
    }
}

bool MVN::canFuse(const NodePtr& node) const {
    if (!mayiuse(cpu::x64::sse41)) {
        return false;
    }
    // limit post ops to unary when shape transformed on channel
    // 1D only fused with unary
    int inputRank = getInputShapeAtPort(0).getRank();
    bool unaryEltwise = isUnaryEltwise(node);
    if ((inputRank == 1 && !unaryEltwise) ||
        (inputRank == 2 && !unaryEltwise && mvnAttrs.initAcrossChannels_)) {
        return false;
    }

    return canFuseSimpleOperation(node);
}

bool MVN::created() const {
    return getType() == Type::MVN;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
