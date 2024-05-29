// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rope.h"

#include "common/bfloat16.hpp"
#include "common/cpu_memcpy.h"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_uni_eltwise.hpp"
#include "dnnl_extension_utils.h"
#include "emitters/plugin/x64/jit_load_store_emitters.hpp"
#include "openvino/core/type/element_type.hpp"
#include "shape_inference/shape_inference_internal_dyn.hpp"
#include "utils/plain_tensor.hpp"

#include <chrono>
#include <string>
#include <vector>

using namespace dnnl;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
using namespace Xbyak;

namespace ov {
namespace intel_cpu {
namespace node {

struct jit_rotary_compile_params {
    ov::element::Type src_prc;
    ov::element::Type dst_prc;
    size_t rotary_ndims;
    bool interleave;
    bool mix_cos_sin;
};
struct jit_rotary_call_args {
    void* src;
    float* cos;
    float* sin;
    void* dst;
};
struct jit_uni_rotary_kernel {
    void (*ker_)(const jit_rotary_call_args*);
    void operator()(const jit_rotary_call_args* call_args) {
        assert(ker_);
        ker_(call_args);
    }
    explicit jit_uni_rotary_kernel(const jit_rotary_compile_params& jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jit_uni_rotary_kernel() {}
    virtual void create_ker() = 0;
    jit_rotary_compile_params jcp_;
};

template <cpu_isa_t isa>
struct jit_rotary_kernel : public jit_uni_rotary_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rotary_kernel)
    static constexpr size_t vec_size = dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen / sizeof(float);
    explicit jit_rotary_kernel(const jit_rotary_compile_params& jcp) : jit_uni_rotary_kernel(jcp), jit_generator(jit_name()) {}
    virtual ~jit_rotary_kernel() {}
    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }
private:
    using Vmm = typename dnnl::impl::utils::conditional3<isa == cpu_isa_t::sse41, Xmm, isa == cpu_isa_t::avx2, Ymm, Zmm>::type;
    void generate() override {
        this->preamble();
#define GET_OFF(field) offsetof(jit_rotary_call_args, field)
        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_cos, ptr[reg_params + GET_OFF(cos)]);
        mov(reg_sin, ptr[reg_params + GET_OFF(sin)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        uni_vpxor(vmm_src0, vmm_src0, vmm_src0);
        uni_vpxor(vmm_src1, vmm_src1, vmm_src1);
        uni_vpxor(vmm_cos, vmm_cos, vmm_cos);
        uni_vpxor(vmm_sin, vmm_sin, vmm_sin);
        if (jcp_.interleave) {
            // dst: 0-2 4-6 8-10 12-14 16-18 20-22 24-26 28-30 ->
            // lower 64bit/128 lane
            //      0-2        4-6        8-10       12-14
            // higher 64bit/128 lane
            //           16-18      20-22      24-26       28-30
            static const uint64_t mask_zmm[] = {
                0, 4, 1, 5, 2, 6, 3, 7
            };
            if (isa == cpu_isa_t::avx512_core) {
                mov(reg_tmp, reinterpret_cast<uintptr_t>(mask_zmm));
                uni_vmovups(vmm_idx, ptr[reg_tmp]);
            }
            auto half_rotary_ndims = jcp_.rotary_ndims / 2;
            for (size_t i = 0; i < half_rotary_ndims / vec_size; i++) {
                rotary_interleave(vec_size);
            }
        } else {
            auto half_rotary_ndims = jcp_.rotary_ndims / 2;
            size_t steps = 0;
            for (size_t i = 0; i < half_rotary_ndims / vec_size; i++) {
                rotary_half(vec_size);
                steps += vec_size;
            }
            if (half_rotary_ndims % vec_size != 0) {
                rotary_half(half_rotary_ndims % vec_size);
                steps += half_rotary_ndims % vec_size;
            }
        }
        this->postamble();
        for (const auto& emitter : emitters) {
            if (emitter.second)
                emitter.second->emit_data();
        }
    }
    void rotary_half(size_t step) {
        // for (; i < half_rotary_dims; i++) {
        //     auto src0 = src[i];
        //     auto src1 = src[i + half_rotary_dims];
        //     dst[i] = cos[i] * src0 - sin[i] * src1;
        //     dst[i + half_rotary_dims] = cos[i + half_rotary_dims] * src1 + sin[i + half_rotary_dims] * src0;
        // }
        auto half_rotary_ndims = jcp_.rotary_ndims / 2;
        // src0: src[i]
        load(vmm_src0, reg_src, jcp_.src_prc, step, false);
        // src1: src[i + halfRotaryNdims]
        lea(reg_tmp, ptr[reg_src + half_rotary_ndims * jcp_.src_prc.size()]);
        load(vmm_src1, reg_tmp, jcp_.src_prc, step, false);
        // cos[i]
        load(vmm_cos, reg_cos, ov::element::f32, step, false);
        // sin[i]
        load(vmm_sin, reg_sin, ov::element::f32, step, false);
        // sin[i] * src1
        uni_vmulps(vmm_dst0, vmm_sin, vmm_src1);
        // cos[i] * src0 - sin[i] * src1
        vfmsub231ps(vmm_dst0, vmm_cos, vmm_src0);
        store(reg_dst, vmm_dst0, jcp_.dst_prc, step);

        // cos[i + halfRotaryNdims]
        lea(reg_tmp, ptr[reg_cos + half_rotary_ndims * sizeof(float)]);
        load(vmm_cos, reg_tmp, ov::element::f32, step, false);
        // sin[i + halfRotaryNdims]
        lea(reg_tmp, ptr[reg_sin + half_rotary_ndims * sizeof(float)]);
        load(vmm_sin, reg_tmp, ov::element::f32, step, false);
        // cos[i + half_rotary_dims] * src1
        uni_vmulps(vmm_dst0, vmm_cos, vmm_src1);
        // cos[i + half_rotary_dims] * src1 + sin[i + half_rotary_dims] * src0
        vfmadd231ps(vmm_dst0, vmm_sin, vmm_src0);
        lea(reg_tmp, ptr[reg_dst + half_rotary_ndims * jcp_.dst_prc.size()]);
        store(reg_tmp, vmm_dst0, jcp_.dst_prc, step);

        add(reg_src, jcp_.src_prc.size() * step);
        add(reg_dst, jcp_.dst_prc.size() * step);
        add(reg_cos, sizeof(float) * step);
        add(reg_sin, sizeof(float) * step);
    }
    void rotary_interleave(size_t step) {
        // for (size_t j = 0; i < rotary_dims; i += 2, j++) {
        //     dst[i] = cos[j] * x[i] - sin[j] * x[i + 1];
        //     dst[i + 1] = cos[j] * x[i + 1] + sin[j] * x[i];
        // }
        load(vmm_src0, reg_src, jcp_.src_prc, step, false);
        lea(reg_tmp, ptr[reg_src + step * jcp_.src_prc.size()]);
        load(vmm_src1, reg_tmp, jcp_.src_prc, step, false);
        auto deinterlace = [&] (Vmm& src0, Vmm& src1, Vmm& tmp0, Vmm& tmp1) {
            if (isa == cpu_isa_t::avx2) {
                // src0: 0 1  2  3  4  5  6  7
                // src1: 8 9 10 11 12 13 14 15
                // 0 1 2 3  8  9 10 11
                vperm2i128(tmp0, src0, src1, 0x20);
                // 4 5 6 7 12 13 14 15
                vperm2i128(tmp1, src0, src1, 0x31);
                // src0 x[i]:     0 2 4 6 8 10 12 14
                vshufps(src0, tmp0, tmp1, 0x88);
                // src1 x[i + 1]: 1 3 5 7 9 11 13 15
                vshufps(src1, tmp0, tmp1, 0xdd);
            } else {
                // src0: 0   1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
                // src1: 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
                // 0 1 2 3  8  9 10 11 16 17 18 19 24 25 26 27
                vshuff32x4(tmp0, src0, src1, 0x88);
                // 4 5 6 7 12 13 14 15 20 21 22 23 28 29 30 31
                vshuff32x4(tmp1, src0, src1, 0xdd);
                // src0 x[i]:     0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30
                vshufps(src0, tmp0, tmp1, 0x88);
                // src1 x[i + 1]: 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31
                vshufps(src1, tmp0, tmp1, 0xdd);
            }
        };
        deinterlace(vmm_src0, vmm_src1, vmm_dst0, vmm_dst1);
        // cos[j]
        load(vmm_cos, reg_cos, ov::element::f32, step, false);
        // sin[j]
        if (jcp_.mix_cos_sin) {
            lea(reg_tmp, ptr[reg_cos + step * sizeof(float)]);
            load(vmm_sin, reg_tmp, ov::element::f32, step, false);
            deinterlace(vmm_cos, vmm_sin, vmm_dst0, vmm_dst1);
        } else {
            load(vmm_sin, reg_sin, ov::element::f32, step, false);
        }
        // sin[j] * src1
        uni_vmulps(vmm_dst0, vmm_sin, vmm_src1);
        // cos[j] * src0 - sin[j] * src1
        vfmsub231ps(vmm_dst0, vmm_cos, vmm_src0);

        // cos[j] * src1
        uni_vmulps(vmm_dst1, vmm_cos, vmm_src1);
        // cos[j] * src1 + sin[j] * src0
        vfmadd231ps(vmm_dst1, vmm_sin, vmm_src0);
        if (isa == cpu_isa_t::avx2) {
            // dst0: 0 2 4 6 8 10 12 14
            // dst1: 1 3 5 7 9 11 13 15
            // 0 1 2 3  8  9 10 11
            vunpcklps(vmm_cos, vmm_dst0, vmm_dst1);
            // 4 5 6 7 12 13 14 15
            vunpckhps(vmm_sin, vmm_dst0, vmm_dst1);
            // 0 1  2  3  4  5  6  7
            vperm2i128(vmm_dst0, vmm_cos, vmm_sin, 0x20);
            // 8 9 10 11 12 13 14 15
            vperm2i128(vmm_dst1, vmm_cos, vmm_sin, 0x31);
        } else {
            // dst0: 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30
            // dst1: 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31
            // 0 2 16 18 4 6 20 22 8 10 24 26 12 14 28 30
            vpermq(vmm_cos, vmm_idx, vmm_dst0);
            // 1 3 17 19 5 7 21 23 9 11 25 27 13 15 29 31
            vpermq(vmm_sin, vmm_idx, vmm_dst1);
            // 0   1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
            vunpcklps(vmm_dst0, vmm_cos, vmm_sin);
            // 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
            vunpckhps(vmm_dst1, vmm_cos, vmm_sin);
        }
        store(reg_dst, vmm_dst0, jcp_.dst_prc, step);
        lea(reg_tmp, ptr[reg_dst + step * jcp_.dst_prc.size()]);
        store(reg_tmp, vmm_dst1, jcp_.dst_prc, step);
        add(reg_src, jcp_.src_prc.size() * step * 2);
        add(reg_dst, jcp_.dst_prc.size() * step * 2);
        if (jcp_.mix_cos_sin) {
            add(reg_cos, 2 * sizeof(float) * step);
        } else {
            add(reg_cos, sizeof(float) * step);
            add(reg_sin, sizeof(float) * step);
        }
    }
#undef GET_OFF
    inline void load(const Vmm& vmm_dst, const Xbyak::Reg64& reg_src, ov::element::Type src_prc, const int& elt_num, bool fill) {
        const auto seed = load_emitter_params(src_prc, ov::element::f32, elt_num, fill, "float_min").hash();
        if (!emitters[seed]) {
            emitters[seed].reset(new jit_load_emitter(this, isa, src_prc, ov::element::f32, elt_num, ov::element::f32, fill, "float_min"));
        }
        emitters[seed]->emit_code({static_cast<size_t>(reg_src.getIdx()), 0}, {static_cast<size_t>(vmm_dst.getIdx())},
                                  pool_aux_vmm_idxs, pool_aux_gpr_idxs);
    }
    inline void store(const Xbyak::Reg64& reg_dst, const Vmm& vmm_src, ov::element::Type dst_prc, const int& elt_num) {
        const auto seed = store_emitter_params(ov::element::f32, dst_prc, elt_num).hash();
        if (!emitters[seed]) {
            emitters[seed].reset(new jit_store_emitter(this, isa,ov::element::f32, dst_prc, elt_num));
        }
        emitters[seed]->emit_code({static_cast<size_t>(vmm_src.getIdx()), 0}, {static_cast<size_t>(reg_dst.getIdx())},
                                  pool_aux_vmm_idxs, pool_aux_gpr_idxs);
    }
    Vmm vmm_src0 = Vmm(0);
    Vmm vmm_src1 = Vmm(1);
    Vmm vmm_cos = Vmm(2);
    Vmm vmm_sin = Vmm(3);
    Vmm vmm_dst0 = Vmm(4);
    Vmm vmm_dst1 = Vmm(5);
    Vmm vmm_idx = Vmm(7);
    Reg64 reg_src = r8;
    Reg64 reg_cos = r10;
    Reg64 reg_sin = r11;
    Reg64 reg_dst = r12;
    Reg64 reg_tmp = rdx;
    Reg64 reg_params = abi_param1;
    Reg64 reg_not_params = abi_not_param1;
    std::unordered_map<size_t, std::unique_ptr<jit_emitter>> emitters;
    const std::vector<size_t> pool_aux_gpr_idxs = { static_cast<size_t>(reg_params.getIdx()), static_cast<size_t>(reg_not_params.getIdx()) };
    const std::vector<size_t> pool_aux_vmm_idxs = { 6 };
};

RoPE::RoPE(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW("CPU: " + errorMessage);
    }

    const auto node = std::dynamic_pointer_cast<const op::internal::RoPE>(op);
    m_config = node->get_config();
}

template <typename T>
struct RoPE::RoPEExecutorRotateHalf : public RoPE::Executor {
    std::unique_ptr<jit_uni_rotary_kernel> m_rotaryKernel;
    bool m_init = false;

    void execute(dnnl::stream strm,
                 const op::internal::RoPE::Config& config,
                 const std::vector<MemoryPtr>& inputs,
                 const std::vector<MemoryPtr>& outputs) override {
        ov::intel_cpu::PlainTensor t_src(inputs[0]);
        ov::intel_cpu::PlainTensor t_cos(inputs[1]);
        ov::intel_cpu::PlainTensor t_sin(inputs[2]);
        ov::intel_cpu::PlainTensor t_dst(outputs[0]);
        ov::intel_cpu::PlainTensor gather;
        auto rotary_dims = config.rotary_ndims;

        if (!m_init) {
            m_init = true;
            jit_rotary_compile_params jcp;
            jcp.src_prc = precision_of<T>::value;
            jcp.dst_prc = precision_of<T>::value;
            jcp.rotary_ndims = rotary_dims;
            jcp.interleave = false;
            if (mayiuse(cpu_isa_t::avx512_core)) {
                m_rotaryKernel.reset(new jit_rotary_kernel<cpu_isa_t::avx512_core>(jcp));
            } else if (mayiuse(cpu_isa_t::avx2)) {
                m_rotaryKernel.reset(new jit_rotary_kernel<cpu_isa_t::avx2>(jcp));
            }
            if (m_rotaryKernel)
                m_rotaryKernel->create_ker();
        }

        bool can_inplace = true;
        if (config.slice_stop - config.slice_start > 0) {
            t_src = t_src.slice(3, config.slice_start, config.slice_stop);
            can_inplace = false;
        }
        if (config.input_trans0213) {
            t_src = t_src.permute({0, 2, 1, 3});
            can_inplace = false;
        }
        if (config.gather_position_arg_id > 0) {
            gather.reset(inputs[config.gather_position_arg_id]);
        }

        if (t_cos.m_rank == 2) {
            t_cos = t_cos.reshape({1, 1, t_cos.size(0), t_cos.size(1)});
        }
        if (t_sin.m_rank == 2) {
            t_sin = t_sin.reshape({1, 1, t_sin.size(0), t_sin.size(1)});
        }

        auto batch_size = t_src.size(0);
        auto head_cnt = t_src.size(1);
        auto seq_len = t_src.size(2);
        auto feature_size = t_src.size(3);

        parallel_for3d(batch_size, head_cnt, seq_len, [&](size_t b, size_t h, size_t p) {
            auto cos_pos = p;
            if (gather) {
                if (gather.m_rank == 4)
                    cos_pos = gather.at<int32_t>({b, h, p, 0}, true);
                else
                    cos_pos = gather.at<int32_t>({b, p}, true);
            }
            auto* src = t_src.ptr<T>(b, h, p);
            auto* cos = &t_cos.at<float>({b, h, cos_pos, 0}, true);
            auto* sin = &t_sin.at<float>({b, h, cos_pos, 0}, true);
            auto* dst = t_dst.ptr<T>(b, h, p, 0);

            if (m_rotaryKernel) {
                jit_rotary_call_args call_args;
                call_args.src = src;
                call_args.cos = cos;
                call_args.sin = sin;
                call_args.dst = dst;
                (*m_rotaryKernel)(&call_args);
            } else {
                auto half_rotary_dims = rotary_dims / 2;
                size_t i = 0;
                for (; i < half_rotary_dims; i++) {
                    auto src0 = src[i];
                    auto src1 = src[i + half_rotary_dims];
                    dst[i] = cos[i] * src0 - sin[i] * src1;
                    dst[i + half_rotary_dims] = cos[i + half_rotary_dims] * src1 + sin[i + half_rotary_dims] * src0;
                }
            }
            if (!can_inplace) {
                memcpy(dst + rotary_dims, src + rotary_dims, (feature_size - rotary_dims) * sizeof(T));
            }
        });
    }
};

template <typename T>
struct RoPE::RoPEExecutorInterleaved : public RoPE::Executor {
    std::unique_ptr<jit_uni_rotary_kernel> m_rotaryKernel;
    bool m_init = false;

    void execute(dnnl::stream strm,
                 const op::internal::RoPE::Config& config,
                 const std::vector<MemoryPtr>& inputs,
                 const std::vector<MemoryPtr>& outputs) override {
        ov::intel_cpu::PlainTensor t_src(inputs[0]);
        ov::intel_cpu::PlainTensor t_sin_cos(inputs[1]);
        ov::intel_cpu::PlainTensor t_dst(outputs[0]);

        auto batch_size = t_src.size(0);
        auto seq_len = t_src.size(1);
        auto head_cnt = t_src.size(2);
        auto head_dims = t_src.size(3);

        auto rotary_dims = config.rotary_ndims;
        auto half_rotary_dims = rotary_dims / 2;
        if (!m_init) {
            m_init = true;
            jit_rotary_compile_params jcp;
            jcp.src_prc = precision_of<T>::value;
            jcp.dst_prc = precision_of<T>::value;
            jcp.rotary_ndims = rotary_dims;
            jcp.interleave = true;
            jcp.mix_cos_sin = false;
            if (mayiuse(cpu_isa_t::avx512_core)) {
                auto vec_size = jit_rotary_kernel<cpu_isa_t::avx512_core>::vec_size;
                // no tails will hit fast path
                if (rotary_dims % (2 * vec_size) == 0)
                    m_rotaryKernel.reset(new jit_rotary_kernel<cpu_isa_t::avx512_core>(jcp));
            } else if (mayiuse(cpu_isa_t::avx2)) {
                auto vec_size = jit_rotary_kernel<cpu_isa_t::avx2>::vec_size;
                if (rotary_dims % (2 * vec_size) == 0)
                    m_rotaryKernel.reset(new jit_rotary_kernel<cpu_isa_t::avx2>(jcp));
            }

            if (m_rotaryKernel)
                m_rotaryKernel->create_ker();
        }

        parallel_for3d(batch_size, seq_len, head_cnt, [&](size_t b, size_t p, size_t h) {
            auto* x = t_src.ptr<T>(b, p, h);
            float* sin = &t_sin_cos.at<float>({b, p, 0}, true);
            float* cos = &t_sin_cos.at<float>({b, p, half_rotary_dims}, true);
            auto* dst = t_dst.ptr<T>(b, h, p);

            if (m_rotaryKernel) {
                jit_rotary_call_args call_args;
                call_args.src = x;
                call_args.cos = cos;
                call_args.sin = sin;
                call_args.dst = dst;
                (*m_rotaryKernel)(&call_args);
            } else {
                size_t i = 0;
                for (size_t j = 0; i < rotary_dims; i += 2, j++) {
                    dst[i] = cos[j] * x[i] - sin[j] * x[i + 1];
                    dst[i + 1] = cos[j] * x[i + 1] + sin[j] * x[i];
                }
            }
            memcpy(dst + rotary_dims, x + rotary_dims, (head_dims - rotary_dims) * sizeof(T));
        });
    }
};

template <typename T>
struct RoPE::RoPEExecutorChatGLM : public RoPE::Executor {
    std::unique_ptr<jit_uni_rotary_kernel> m_rotaryKernel;
    bool m_init = false;

    void execute(dnnl::stream strm,
                 const op::internal::RoPE::Config& config,
                 const std::vector<MemoryPtr>& inputs,
                 const std::vector<MemoryPtr>& outputs) override {
        ov::intel_cpu::PlainTensor t_src(inputs[0]);
        ov::intel_cpu::PlainTensor t_cos_sin(inputs[1]);
        ov::intel_cpu::PlainTensor t_dst(outputs[0]);

        // [seq_len, batch_size, (hidden_states_q + hidden_states_k + hidden_states_v)]
        if (config.slice_stop - config.slice_start > 0) {
            t_src = t_src.slice(2, config.slice_start, config.slice_stop);
        }
        auto seq_len = t_src.size(0);
        auto batch_size = t_src.size(1);

        auto head_cnt = config.head_cnt;
        auto head_size = config.head_size;

        auto rotary_dims = config.rotary_ndims;

        if (!m_init) {
            m_init = true;
            jit_rotary_compile_params jcp;
            jcp.src_prc = precision_of<T>::value;
            jcp.dst_prc = precision_of<T>::value;
            jcp.rotary_ndims = rotary_dims;
            jcp.interleave = true;
            jcp.mix_cos_sin = true;
            if (mayiuse(cpu_isa_t::avx512_core)) {
                auto vec_size = jit_rotary_kernel<cpu_isa_t::avx512_core>::vec_size;
                // no tails will hit fast path
                if (rotary_dims % (2 * vec_size) == 0)
                    m_rotaryKernel.reset(new jit_rotary_kernel<cpu_isa_t::avx512_core>(jcp));
            } else if (mayiuse(cpu_isa_t::avx2)) {
                auto vec_size = jit_rotary_kernel<cpu_isa_t::avx2>::vec_size;
                if (rotary_dims % (2 * vec_size) == 0)
                    m_rotaryKernel.reset(new jit_rotary_kernel<cpu_isa_t::avx2>(jcp));
            }

            if (m_rotaryKernel)
                m_rotaryKernel->create_ker();
        }

        parallel_for3d(seq_len, batch_size, head_cnt, [&](size_t p, size_t b, size_t h) {
            auto* src = t_src.ptr<T>(p, b, h * head_size);
            // [length, batch_size, ndims//2, 2]
            auto* cos_sin = &t_cos_sin.at<float>({p, b, 0, 0}, true);
            auto* dst = t_dst.ptr<T>(p, b, h, 0);

            if (m_rotaryKernel) {
                jit_rotary_call_args call_args;
                call_args.src = src;
                call_args.cos = cos_sin;
                call_args.dst = dst;
                (*m_rotaryKernel)(&call_args);
            } else {
                size_t i = 0;
                for (; i < rotary_dims; i += 2) {
                    auto cosv = cos_sin[i];
                    auto sinv = cos_sin[i + 1];
                    dst[i] = cosv * src[i] - sinv * src[i + 1];
                    dst[i + 1] = sinv * src[i] + cosv * src[i + 1];
                }
            }

            memcpy(dst + rotary_dims, src + rotary_dims, (head_size - rotary_dims) * sizeof(T));
        });
    }
};

template <typename T>
struct RoPE::RoPEExecutorQwen : public RoPE::Executor {
    std::unique_ptr<jit_uni_rotary_kernel> m_rotaryKernel;
    bool m_init = false;

    void execute(dnnl::stream strm,
                 const op::internal::RoPE::Config& config,
                 const std::vector<MemoryPtr>& inputs,
                 const std::vector<MemoryPtr>& outputs) override {
        ov::intel_cpu::PlainTensor t_src(inputs[0]);    // [batch, length, head_cnt*head_size * 3]
        ov::intel_cpu::PlainTensor t_cos(inputs[1]);    // [1, present-kv-length, 1, rotary_dims]
        ov::intel_cpu::PlainTensor t_sin(inputs[2]);    // [1, present-kv-length, 1, rotary_dims]
        ov::intel_cpu::PlainTensor t_dst(outputs[0]);   // [batch, length, head_cnt, head_size]>
        auto rotary_dims = t_cos.size(3);

        if (!m_init) {
            m_init = true;
            jit_rotary_compile_params jcp;
            jcp.src_prc = precision_of<T>::value;
            jcp.dst_prc = precision_of<T>::value;
            jcp.rotary_ndims = rotary_dims;
            jcp.interleave = false;
            if (mayiuse(cpu_isa_t::avx512_core)) {
                m_rotaryKernel.reset(new jit_rotary_kernel<cpu_isa_t::avx512_core>(jcp));
            } else if (mayiuse(cpu_isa_t::avx2)) {
                m_rotaryKernel.reset(new jit_rotary_kernel<cpu_isa_t::avx2>(jcp));
            }

            if (m_rotaryKernel)
                m_rotaryKernel->create_ker();
        }

        bool can_inplace = true;
        if (config.slice_stop - config.slice_start > 0) {
            t_src = t_src.slice(2, config.slice_start, config.slice_stop);
            can_inplace = false;
        }

        auto batch_size = t_src.size(0);
        auto seq_len = t_src.size(1);
        auto head_cnt = config.head_cnt;
        auto head_size = config.head_size;
        auto present_kv_len = t_cos.size(1);

        parallel_for3d(batch_size, seq_len, head_cnt, [&](size_t b, size_t p, size_t h) {
            auto* src = t_src.ptr<T>(b, p, h * head_size);
            auto* cos = &t_cos.at<float>({b, present_kv_len - seq_len + p, h, 0}, true);
            auto* sin = &t_sin.at<float>({b, present_kv_len - seq_len + p, h, 0}, true);
            auto* dst = t_dst.ptr<T>(b, p, h);

            if (m_rotaryKernel) {
                jit_rotary_call_args call_args;
                call_args.src = src;
                call_args.cos = cos;
                call_args.sin = sin;
                call_args.dst = dst;
                (*m_rotaryKernel)(&call_args);
            } else {
                auto half_rotary_dims = rotary_dims / 2;
                size_t i = 0;
                for (; i < half_rotary_dims; i++) {
                    auto s0 = src[i];
                    auto s1 = src[i + half_rotary_dims];
                    dst[i] = cos[i] * s0 - sin[i] * s1;
                    dst[i + half_rotary_dims] = cos[i + half_rotary_dims] * s1 + sin[i + half_rotary_dims] * s0;
                }
            }
            if (!can_inplace) {
                memcpy(dst + rotary_dims, src + rotary_dims, (head_size - rotary_dims) * sizeof(T));
            }
        });
    }
};

void RoPE::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    auto srcPrecision = getOriginalInputPrecisionAtPort(0);

    auto rtPrecision = srcPrecision;
    auto CosSinPrecision = ov::element::f32;
    bool can_inplace = true;

    if (m_config.is_qwen) {
        if (rtPrecision == ov::element::bf16) {
            m_executor = std::make_shared<RoPEExecutorQwen<ov::bfloat16>>();
        } else {
            m_executor = std::make_shared<RoPEExecutorQwen<float>>();
            rtPrecision = ov::element::f32;
        }
        if (m_config.slice_stop - m_config.slice_start > 0)
            can_inplace = false;
    } else if (m_config.is_chatglm) {
        if (rtPrecision == ov::element::bf16) {
            m_executor = std::make_shared<RoPEExecutorChatGLM<ov::bfloat16>>();
        } else {
            m_executor = std::make_shared<RoPEExecutorChatGLM<float>>();
            rtPrecision = ov::element::f32;
        }
        can_inplace = false;
    } else if (m_config.is_interleaved) {
        OPENVINO_ASSERT(m_config.input_trans0213 == false);
        OPENVINO_ASSERT(m_config.slice_start == 0);
        OPENVINO_ASSERT(m_config.slice_stop == 0);
        OPENVINO_ASSERT(m_config.gather_position_arg_id == 0);
        if (rtPrecision == ov::element::bf16) {
            m_executor = std::make_shared<RoPEExecutorInterleaved<ov::bfloat16>>();
        } else {
            m_executor = std::make_shared<RoPEExecutorInterleaved<float>>();
            rtPrecision = ov::element::f32;
        }
        can_inplace = false;
    } else {
        if (rtPrecision == ov::element::bf16) {
            m_executor = std::make_shared<RoPEExecutorRotateHalf<ov::bfloat16>>();
        } else {
            m_executor = std::make_shared<RoPEExecutorRotateHalf<float>>();
            rtPrecision = ov::element::f32;
        }
        if (m_config.slice_stop - m_config.slice_start > 0 || m_config.input_trans0213)
            can_inplace = false;
    }

    // initialize input ports
    std::vector<PortConfigurator> inPortConfigs;
    inPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getInputShapeAtPort(0), false, -1);
    inPortConfigs.emplace_back(LayoutType::ncsp, CosSinPrecision, getInputShapeAtPort(1), false, -1);
    inPortConfigs.emplace_back(LayoutType::ncsp, CosSinPrecision, getInputShapeAtPort(2), false, -1);
    if (m_config.gather_position_arg_id > 0) {
        inPortConfigs.emplace_back(LayoutType::ncsp,
                                   ov::element::i32,
                                   getInputShapeAtPort(m_config.gather_position_arg_id),
                                   false,
                                   -1);
    }

    // initialize output port
    std::vector<PortConfigurator> outPortConfigs;
    outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(0), false, can_inplace ? 0 : -1);

    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref_any);
}

void RoPE::execute(dnnl::stream strm) {
    std::vector<MemoryPtr> inputs(getParentEdges().size()), outputs(getChildEdges().size());
    for (size_t i = 0; i < inputs.size(); i++) {
        inputs[i] = getSrcMemoryAtPort(i);
    }
    for (size_t i = 0; i < outputs.size(); i++) {
        outputs[i] = getDstMemoryAtPort(i);
    }
    m_executor->execute(strm, m_config, inputs, outputs);
}

bool RoPE::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto node = std::dynamic_pointer_cast<const op::internal::RoPE>(op);
        if (!node) {
            errorMessage = "Only RoPE operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
