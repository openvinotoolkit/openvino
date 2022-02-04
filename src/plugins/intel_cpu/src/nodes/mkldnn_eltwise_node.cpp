// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_eltwise_node.h"

#include <ie_parallel.hpp>

#include <mkldnn_types.h>
#include "cpu_types.h"
#include "utils/bfloat16.hpp"
#include <cpu/x64/injectors/jit_uni_quantization_injector.hpp>
#include <cpu/ref_eltwise.hpp>

#include "mkldnn_extension_utils.h"
#include "mkldnn_fake_quantize_node.h"
#include "mkldnn_pooling_node.h"
#include "mkldnn_input_node.h"
#include "common/cpu_convert.h"

#include "emitters/jit_emitter.hpp"
#include "emitters/jit_eltwise_emitters.hpp"
#include "emitters/jit_mkldnn_emitters.hpp"
#include "emitters/jit_bf16_emitters.hpp"
#include <mkldnn_selective_build.h>
#include "utils/general_utils.h"
#include "utils/cpu_utils.hpp"
#include <common/primitive_hashing_utils.hpp>

#include "ngraph/ngraph.hpp"
#include <ngraph/opsets/opset1.hpp>
#include "ngraph_transformations/op/power_static.hpp"
#include "ngraph_transformations/op/leaky_relu.hpp"
#include "ngraph_transformations/op/swish_cpu.hpp"

#include <oneapi/dnnl/dnnl.hpp>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <map>
#include <functional>
#include "memory_desc/dnnl_blocked_memory_desc.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::cpu;
using namespace mkldnn::impl::cpu::x64;
using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_eltwise_call_args_ptrs, field)

namespace {

template<typename T>
struct SupportedPrecisions {
    void operator()(std::set<Precision> &precisions) {
        precisions = T::get_supported_precisions();
    }
};

struct EltwiseEmitterContext {
    std::shared_ptr<jit_emitter> emitter;
    jit_generator *host;
    cpu_isa_t host_isa;
    const MKLDNNEltwiseNode::EltwiseData& opData;
    InferenceEngine::Precision exec_prc;
};

template<typename T>
struct EltwiseEmitter {
    void operator()(EltwiseEmitterContext & ctx) {
        ctx.emitter = std::make_shared<T>(ctx.host, ctx.host_isa, ctx.exec_prc);
    }
};

template<>
struct EltwiseEmitter<jit_mkldnn_aux_emitter> {
    void operator()(EltwiseEmitterContext & ctx) {
        auto algKind = static_cast<mkldnn_alg_kind_t>(ctx.opData.mkldnnAlgorithm);
        ctx.emitter = std::make_shared<jit_mkldnn_aux_emitter>(ctx.host, ctx.host_isa, algKind,
                                                               ctx.opData.alpha, ctx.opData.beta, ctx.exec_prc);
    }
};

template<>
struct EltwiseEmitter<jit_power_static_emitter> {
    void operator()(EltwiseEmitterContext & ctx) {
        ctx.emitter = std::make_shared<jit_power_static_emitter>(ctx.host, ctx.host_isa, ctx.opData.alpha,
                                                                 ctx.opData.beta, ctx.opData.gamma, ctx.exec_prc);
    }
};

}   // namespace

template <cpu_isa_t isa>
struct jit_uni_eltwise_generic : public MKLDNNPlugin::jit_uni_eltwise_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_eltwise_generic)

    explicit jit_uni_eltwise_generic(const jit_eltwise_params& jep,
                                     const std::vector<MKLDNNEltwiseNode::EltwiseData>& eltwise_data,
                                     const std::vector<MKLDNNPlugin::Type>& ops_list,
                                     const mkldnn::post_ops& post_ops)
    : jit_uni_eltwise_kernel(jep), jit_generator(), eltwise_data_(eltwise_data), ops_list_(ops_list), post_ops_(post_ops) {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        Precision exec_prc = Precision::UNSPECIFIED;

        std::set<Precision> supported_precision_intersection = get_supported_precisions(eltwise_data_.front().algo);
        for (size_t i = 1; i < eltwise_data_.size(); ++i) {
            std::set<Precision> prcs = get_supported_precisions(eltwise_data_[i].algo);
            std::set<Precision> prcs_intersect = {};

            std::set_intersection(supported_precision_intersection.begin(), supported_precision_intersection.end(),
                                  prcs.begin(), prcs.end(), std::inserter(prcs_intersect, prcs_intersect.begin()));

            supported_precision_intersection = prcs_intersect;
        }

        static const Precision exec_precisions_priority[] = {
                Precision::U8,
                Precision::I8,
                Precision::U16,
                Precision::I16,
                Precision::BF16,
                Precision::I32,
                Precision::FP32
        };

        for (auto prc : exec_precisions_priority) {
            if (std::find(supported_precision_intersection.begin(), supported_precision_intersection.end(), prc) != supported_precision_intersection.end()) {
                exec_prc = prc;
                break;
            }
        }

        for (int i = 0; i < jep_.inputs_number; i++) {
            if (jep_.src_prc[i] != exec_prc) {
                exec_prc = Precision::FP32;
                break;
            }
        }

        if (exec_prc == Precision::UNSPECIFIED) {
            IE_THROW() << "Eltwise jitter failed to specify execution precision for Eltwise node";
        }

        eltwise_emitter = create_eltwise_emitter(eltwise_data_.front(), exec_prc);
        for (size_t i = 1; i < eltwise_data_.size(); ++i) {
            post_op_emitters.push_back(create_eltwise_emitter(eltwise_data_[i], exec_prc));
        }

        const auto& p = post_ops_.get();
        for (int i = 0; i < post_ops_.len(); ++i) {
            if (!p->entry_[i].is_quantization()) {
                IE_THROW() << "Eltwise jitter error. Unsupported post op detected";
            }
            quantization_injectors.push_back(std::make_shared<jit_uni_quantization_injector_f32<isa>>(
                    this, p->entry_[i], vmm_d_weights, vmm_d_bias, reg_d_weights, reg_d_bias));
        }

        if (!mayiuse(avx512_core_bf16) && mayiuse(avx512_core))
            emu_vcvtneps2bf16.reset(new jit_emu_vcvtneps2bf16(this, isa));

        const auto &jep = jep_;

        this->preamble();

        const int offset_count = jep.input_size - 1;

        // ptrs initializing
        auto init_ptrs_with_offsets = [this, offset_count](Reg64 pointer, const std::vector<size_t>& offsets) {
            for (int j = 0; j < offset_count; j++) {
                if (jep_.dims[j] != 1 && offsets[j] != 0) {
                    mov(reg_tmp_64, offsets[j]);
                    imul(reg_tmp_64, ptr[reg_indexes + j * sizeof(size_t)]);
                    add(pointer, reg_tmp_64);
                }
            }
        };

        for (int i = 0; i < jep.inputs_number; i++) {
            mov(get_src_reg(i), ptr[reg_const_params + GET_OFF(src_ptr[0]) + i * sizeof(size_t)]);
            init_ptrs_with_offsets(get_src_reg(i), jep.src_offsets[i]);
        }

        mov(reg_dst, ptr[reg_const_params + GET_OFF(dst_ptr)]);
        init_ptrs_with_offsets(reg_dst, jep.dst_offsets);

        mov(reg_post_op_ptrs, ptr[reg_const_params + GET_OFF(post_op_data)]);

        xor_(reg_oc_off, reg_oc_off);
        init_ptrs_with_offsets(reg_oc_off, jep.oc_offsets);

        mov(reg_work_amount, jep.work_amount);

        Xbyak::Label unroll_loop_label;
        Xbyak::Label unroll_loop_end_label;
        Xbyak::Label main_loop_label;
        Xbyak::Label main_loop_end_label;
        Xbyak::Label tail_loop_label;
        Xbyak::Label tail_loop_end_label;

        if (isa == x64::avx512_common)
            vpxord(vmm_zero, vmm_zero, vmm_zero);

        for (int i = 0; i < jep.inputs_number; i++) {
            if (jep.src_size[i] == 1)
                load_vector(get_vmm_reg(i), ptr[get_src_reg(i)], jep.src_prc[i], exec_prc, true);
        }

        size_t min_src_size = jep.dst_size;
        for (int i = 0; i < jep.inputs_number; i++) {
            if (jep.src_size[i] != 1)
                min_src_size = std::min(min_src_size, jep.src_size[i]);
        }
        if (jep_.oc_size > 1)
            min_src_size = std::min(min_src_size, jep_.oc_size);

        if (min_src_size != jep.dst_size) {
            bool is_valid_configuration = true;
            if (jep.dst_size % min_src_size != 0)
                is_valid_configuration = false;

            for (int i = 0; i < jep.inputs_number; i++) {
                if (jep.src_size[i] != 1 && jep.src_size[i] != min_src_size && jep.src_size[i] != jep.dst_size)
                    is_valid_configuration = false;
            }

            if (jep_.oc_size > 1 && jep_.oc_size != min_src_size && jep_.oc_size != jep.dst_size)
                is_valid_configuration = false;

            if (!is_valid_configuration)
                IE_THROW() << "Eltwise jitter has invalid configuration for Eltwise node";

            L(unroll_loop_label);
            {
                size_t loop_step = min_src_size;
                size_t vec_step = cpu_isa_traits<isa>::vlen / exec_prc.size();

                cmp(reg_work_amount, loop_step);
                jl(unroll_loop_end_label, T_NEAR);

                for (int j = 0; j < min_src_size / vec_step; j++) {
                    for (int i = 0; i < jep.inputs_number; i++) {
                        if (jep.src_size[i] != 1)
                            load_vector(get_vmm_reg(i), ptr[get_src_reg(i) + j * vec_step * jep.src_prc[i].size()], jep.src_prc[i], exec_prc, false);
                    }

                    compute_eltwise_op();

                    apply_post_ops(false, jep_.oc_size > 1 ? j * vec_step * sizeof(float) : 0);

                    store_vector(ptr[reg_dst + j * vec_step * jep.dst_prc.size()], vmm_dst, exec_prc, jep.dst_prc);
                }

                int tail_start = min_src_size - min_src_size % vec_step;
                for (int j = tail_start; j < min_src_size; j++) {
                    for (int i = 0; i < jep.inputs_number; i++) {
                        if (jep.src_size[i] != 1)
                            load_scalar(get_xmm_reg(i), ptr[get_src_reg(i) + j * jep.src_prc[i].size()], jep.src_prc[i], exec_prc);
                    }

                    compute_eltwise_op();

                    apply_post_ops(true, jep_.oc_size > 1 ? j * sizeof(float) : 0);

                    store_scalar(ptr[reg_dst + j * jep.dst_prc.size()], xmm_dst, exec_prc, jep.dst_prc);
                }

                for (int i = 0; i < jep.inputs_number; i++)
                    if (jep.src_size[i] == jep.dst_size)
                        add(get_src_reg(i), jep.src_prc[i].size() * loop_step);

                add(reg_dst, jep.dst_prc.size() * loop_step);
                sub(reg_work_amount, loop_step);
                if (jep_.oc_size > 1 && jep_.oc_size != min_src_size)
                    add(reg_oc_off, loop_step * sizeof(float));

                jmp(unroll_loop_label, T_NEAR);
            }

            L(unroll_loop_end_label);
        }

        if (min_src_size == jep.dst_size) {
            L(main_loop_label);
            {
                size_t loop_step = cpu_isa_traits<isa>::vlen / exec_prc.size();

                cmp(reg_work_amount, loop_step);
                jl(main_loop_end_label, T_NEAR);

                for (int i = 0; i < jep.inputs_number; i++) {
                    if (jep.src_size[i] != 1)
                        load_vector(get_vmm_reg(i), ptr[get_src_reg(i)], jep.src_prc[i], exec_prc, false);
                }

                compute_eltwise_op();

                apply_post_ops(false);

                store_vector(ptr[reg_dst], vmm_dst, exec_prc, jep.dst_prc);

                for (int i = 0; i < jep.inputs_number; i++)
                    if (jep.src_size[i] != 1)
                        add(get_src_reg(i), jep.src_prc[i].size() * loop_step);

                add(reg_dst, jep.dst_prc.size() * loop_step);
                sub(reg_work_amount, loop_step);
                if (jep_.oc_size > 1)
                    add(reg_oc_off, loop_step * sizeof(float));

                jmp(main_loop_label, T_NEAR);
            }

            L(main_loop_end_label);
        }

        L(tail_loop_label);
        {
            size_t loop_step = 1;

            cmp(reg_work_amount, loop_step);
            jl(tail_loop_end_label, T_NEAR);

            for (int i = 0; i < jep.inputs_number; i++) {
                if (jep.src_size[i] != 1)
                    load_scalar(get_xmm_reg(i), ptr[get_src_reg(i)], jep.src_prc[i], exec_prc);
            }

            compute_eltwise_op();

            apply_post_ops(true);

            store_scalar(ptr[reg_dst], xmm_dst, exec_prc, jep.dst_prc);

            for (int i = 0; i < jep.inputs_number; i++)
                if (jep.src_size[i] != 1)
                    add(get_src_reg(i), jep.src_prc[i].size() * loop_step);

            add(reg_dst, jep.dst_prc.size() * loop_step);
            sub(reg_work_amount, loop_step);
            if (jep_.oc_size > 1)
                add(reg_oc_off, loop_step * sizeof(float));

            jmp(tail_loop_label, T_NEAR);
        }

        L(tail_loop_end_label);

        this->postamble();

        if (!mayiuse(avx512_core_bf16) && mayiuse(avx512_core))
            emu_vcvtneps2bf16->emit_data();

        eltwise_emitter->emit_data();
        for (int i = 0; i < post_op_emitters.size(); i++) {
            post_op_emitters[i]->emit_data();
        }
    }

private:
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;

    Reg64 get_src_reg(int idx) {
        return Reg64(r8.getIdx() + idx);
    }

    Vmm get_vmm_reg(int idx) {
        return Vmm(1 + idx);
    }

    Vmm get_aux_vmm(int idx) {
        return Vmm(10 + idx);
    }

    Xmm get_xmm_reg(int idx) {
        return Xmm(get_vmm_reg(idx).getIdx());
    }

    Reg64 reg_post_op_ptrs = rax;
    Reg64 reg_dst = rbx;
    Reg64 reg_work_amount = rdx;

    Reg64 reg_oc_off = abi_not_param1;
    Reg64 reg_const_params = abi_param1;
    Reg64 reg_indexes = abi_param2;  // reg_d_bias

    Reg8 reg_tmp_8 = Reg8(r15.getIdx());
    Reg32 reg_tmp_32 = Reg32(r15.getIdx());
    Reg64 reg_tmp_64 = Reg64(r15.getIdx());

    Reg64 reg_d_weights = rbp;
    Reg64 reg_d_bias = rsi;

    Vmm vmm_dst = Vmm(9);
    Xmm xmm_dst = Xmm(9);

    Vmm vmm_d_weights = Vmm(12);
    Vmm vmm_d_bias = Vmm(13);
    Vmm vmm_zero = Vmm(15);

    std::shared_ptr<jit_emu_vcvtneps2bf16> emu_vcvtneps2bf16;

    std::shared_ptr<jit_emitter> eltwise_emitter = nullptr;
    std::vector<std::shared_ptr<jit_emitter>> post_op_emitters = {};

    std::vector<std::shared_ptr<jit_uni_quantization_injector_f32<isa>>> quantization_injectors = {};

    const std::vector<MKLDNNEltwiseNode::EltwiseData>& eltwise_data_;
    const std::vector<MKLDNNPlugin::Type>& ops_list_;
    const mkldnn::post_ops& post_ops_;

    std::set<Precision> get_supported_precisions(Algorithm algo) {
        std::set<Precision> precisions;

        OV_SWITCH(MKLDNNPlugin, SupportedPrecisions, precisions, algo,
        OV_CASE(EltwiseRelu, jit_mkldnn_aux_emitter),
        OV_CASE(EltwiseGelu, jit_mkldnn_aux_emitter),
        OV_CASE(EltwiseElu, jit_mkldnn_aux_emitter),
        OV_CASE(EltwiseTanh, jit_mkldnn_aux_emitter),
        OV_CASE(EltwiseSigmoid, jit_mkldnn_aux_emitter),
        OV_CASE(EltwiseAbs, jit_mkldnn_aux_emitter),
        OV_CASE(EltwiseSqrt, jit_mkldnn_aux_emitter),
        OV_CASE(EltwiseSoftRelu, jit_mkldnn_aux_emitter),
        OV_CASE(EltwiseExp, jit_mkldnn_aux_emitter),
        OV_CASE(EltwiseClamp, jit_mkldnn_aux_emitter),
        OV_CASE(EltwiseSwish, jit_mkldnn_aux_emitter),
        OV_CASE(EltwiseHswish, jit_mkldnn_aux_emitter),
        OV_CASE(EltwiseMish, jit_mkldnn_aux_emitter),
        OV_CASE(EltwiseHsigmoid, jit_mkldnn_aux_emitter),
        OV_CASE(EltwiseRoundHalfToEven, jit_mkldnn_aux_emitter),
        OV_CASE(EltwiseRoundHalfAwayFromZero, jit_mkldnn_aux_emitter),
        OV_CASE(EltwiseAdd, jit_add_emitter),
        OV_CASE(EltwiseMulAdd, jit_mul_add_emitter),
        OV_CASE(EltwiseSubtract, jit_subtract_emitter),
        OV_CASE(EltwiseMultiply, jit_multiply_emitter),
        OV_CASE(EltwiseDivide, jit_divide_emitter),
        OV_CASE(EltwiseFloorMod, jit_floor_mod_emitter),
        OV_CASE(EltwiseMod, jit_mod_emitter),
        OV_CASE(EltwiseMaximum, jit_maximum_emitter),
        OV_CASE(EltwiseMinimum, jit_minimum_emitter),
        OV_CASE(EltwiseSquaredDifference, jit_squared_difference_emitter),
        OV_CASE(EltwisePowerDynamic, jit_power_dynamic_emitter),
        OV_CASE(EltwiseEqual, jit_equal_emitter),
        OV_CASE(EltwiseNotEqual, jit_not_equal_emitter),
        OV_CASE(EltwiseGreater, jit_greater_emitter),
        OV_CASE(EltwiseGreaterEqual, jit_greater_equal_emitter),
        OV_CASE(EltwiseLess, jit_less_emitter),
        OV_CASE(EltwiseLessEqual, jit_less_equal_emitter),
        OV_CASE(EltwiseLogicalAnd, jit_logical_and_emitter),
        OV_CASE(EltwiseLogicalOr, jit_logical_or_emitter),
        OV_CASE(EltwiseLogicalXor, jit_logical_xor_emitter),
        OV_CASE(EltwiseLogicalNot, jit_logical_not_emitter),
        OV_CASE(EltwisePowerStatic, jit_power_static_emitter),
        OV_CASE(EltwisePrelu, jit_prelu_emitter),
        OV_CASE(EltwiseErf, jit_erf_emitter));

        if (precisions.empty())
            IE_THROW() << "Unsupported operation type for Eltwise emitter";

        return precisions;
    }

    std::shared_ptr<jit_emitter> create_eltwise_emitter(const MKLDNNEltwiseNode::EltwiseData& data, Precision exec_prec) {
        EltwiseEmitterContext ctx = {
            nullptr,
            this,
            isa,
            data,
            exec_prec
        };

        OV_SWITCH(MKLDNNPlugin, EltwiseEmitter, ctx, data.algo,
        OV_CASE(EltwiseRelu, jit_mkldnn_aux_emitter),
        OV_CASE(EltwiseGelu, jit_mkldnn_aux_emitter),
        OV_CASE(EltwiseElu, jit_mkldnn_aux_emitter),
        OV_CASE(EltwiseTanh, jit_mkldnn_aux_emitter),
        OV_CASE(EltwiseSigmoid, jit_mkldnn_aux_emitter),
        OV_CASE(EltwiseAbs, jit_mkldnn_aux_emitter),
        OV_CASE(EltwiseSqrt, jit_mkldnn_aux_emitter),
        OV_CASE(EltwiseSoftRelu, jit_mkldnn_aux_emitter),
        OV_CASE(EltwiseExp, jit_mkldnn_aux_emitter),
        OV_CASE(EltwiseClamp, jit_mkldnn_aux_emitter),
        OV_CASE(EltwiseSwish, jit_mkldnn_aux_emitter),
        OV_CASE(EltwiseHswish, jit_mkldnn_aux_emitter),
        OV_CASE(EltwiseMish, jit_mkldnn_aux_emitter),
        OV_CASE(EltwiseHsigmoid, jit_mkldnn_aux_emitter),
        OV_CASE(EltwiseRoundHalfToEven, jit_mkldnn_aux_emitter),
        OV_CASE(EltwiseRoundHalfAwayFromZero, jit_mkldnn_aux_emitter),
        OV_CASE(EltwiseAdd, jit_add_emitter),
        OV_CASE(EltwiseMulAdd, jit_mul_add_emitter),
        OV_CASE(EltwiseSubtract, jit_subtract_emitter),
        OV_CASE(EltwiseMultiply, jit_multiply_emitter),
        OV_CASE(EltwiseDivide, jit_divide_emitter),
        OV_CASE(EltwiseFloorMod, jit_floor_mod_emitter),
        OV_CASE(EltwiseMod, jit_mod_emitter),
        OV_CASE(EltwiseMaximum, jit_maximum_emitter),
        OV_CASE(EltwiseMinimum, jit_minimum_emitter),
        OV_CASE(EltwiseSquaredDifference, jit_squared_difference_emitter),
        OV_CASE(EltwisePowerDynamic, jit_power_dynamic_emitter),
        OV_CASE(EltwiseEqual, jit_equal_emitter),
        OV_CASE(EltwiseNotEqual, jit_not_equal_emitter),
        OV_CASE(EltwiseGreater, jit_greater_emitter),
        OV_CASE(EltwiseGreaterEqual, jit_greater_equal_emitter),
        OV_CASE(EltwiseLess, jit_less_emitter),
        OV_CASE(EltwiseLessEqual, jit_less_equal_emitter),
        OV_CASE(EltwiseLogicalAnd, jit_logical_and_emitter),
        OV_CASE(EltwiseLogicalOr, jit_logical_or_emitter),
        OV_CASE(EltwiseLogicalXor, jit_logical_xor_emitter),
        OV_CASE(EltwiseLogicalNot, jit_logical_not_emitter),
        OV_CASE(EltwisePowerStatic, jit_power_static_emitter),
        OV_CASE(EltwisePrelu, jit_prelu_emitter),
        OV_CASE(EltwiseErf, jit_erf_emitter));

        if (!ctx.emitter)
            IE_THROW() << "Unsupported operation type for Eltwise emitter";

        return ctx.emitter;
    }

    inline void compute_eltwise_op() {
        std::vector<size_t> in_idxs;
        std::vector<size_t> aux_idxs;
        for (int i = 0; i < eltwise_emitter->get_inputs_num(); i++)
            in_idxs.push_back(get_vmm_reg(i).getIdx());
        for (int i = 0; i < eltwise_emitter->aux_vecs_count(); i++)
            aux_idxs.push_back(get_aux_vmm(i).getIdx());

        std::vector<size_t> out_idxs;
        out_idxs.push_back(vmm_dst.getIdx());

        eltwise_emitter->emit_code(in_idxs, out_idxs, aux_idxs);
    }

    inline void apply_post_ops(bool is_scalar, int offset = 0) {
        int input_idx = eltwise_emitter->get_inputs_num();
        int eltwise_post_op_idx = 0;
        int quantization_post_op_idx = 0;
        for (int i = 1; i < ops_list_.size(); i++) {
            if (ops_list_[i] == Eltwise) {
                std::vector<size_t> in_idxs;
                std::vector<size_t> aux_idxs;
                in_idxs.push_back(vmm_dst.getIdx());
                for (int j = 1; j < post_op_emitters[eltwise_post_op_idx]->get_inputs_num(); j++)
                    in_idxs.push_back(get_vmm_reg(input_idx++).getIdx());
                for (int j = 0; j < post_op_emitters[eltwise_post_op_idx]->aux_vecs_count(); j++)
                    aux_idxs.push_back(get_aux_vmm(j).getIdx());

                std::vector<size_t> out_idxs;
                out_idxs.push_back(vmm_dst.getIdx());

                post_op_emitters[eltwise_post_op_idx]->emit_code(in_idxs, out_idxs, aux_idxs);

                eltwise_post_op_idx++;
            } else if (ops_list_[i] == FakeQuantize) {
                auto& p = post_ops_.get()->entry_[quantization_post_op_idx];
                bool do_dequantization = p.quantization.alg == dnnl::impl::alg_kind::quantization_quantize_dequantize;
                bool do_rounding = do_dequantization || jep_.dst_prc == Precision::FP32 || i != ops_list_.size() - 1;
                int s_idx = vmm_dst.getIdx();

                size_t ptrs_table_off = quantization_post_op_idx * quantization_injectors[quantization_post_op_idx]->memoryStep();

                quantization_injectors[quantization_post_op_idx]->init_crop_ptrs(reg_post_op_ptrs + ptrs_table_off, reg_oc_off);
                quantization_injectors[quantization_post_op_idx]->compute_crop(s_idx, s_idx + 1, offset, is_scalar, jep_.oc_size == 1);

                quantization_injectors[quantization_post_op_idx]->init_input_scale_shift_ptrs(reg_post_op_ptrs + ptrs_table_off, reg_oc_off);
                quantization_injectors[quantization_post_op_idx]->compute_input_scale_shift(s_idx, s_idx + 1, offset, do_rounding,
                                                                                            is_scalar, jep_.oc_size == 1);

                quantization_injectors[quantization_post_op_idx]->init_output_scale_shift_ptrs(reg_post_op_ptrs + ptrs_table_off, reg_oc_off);
                quantization_injectors[quantization_post_op_idx]->compute_output_scale_shift(s_idx, s_idx + 1, offset, is_scalar, jep_.oc_size == 1);

                quantization_post_op_idx++;
            } else {
                IE_THROW(Unexpected) << "Eltwise jit kernel: unexpected operation type";
            }
        }
    }

    inline void load_vector(Vmm vmm_src, const Xbyak::Address &op, Precision src_prc, Precision dst_prc, bool broadcast) {
        Xmm xmm_src = Xmm(vmm_src.getIdx());

        if (broadcast) {
            load_scalar(xmm_src, op, src_prc, dst_prc);
            uni_vbroadcastss(vmm_src, xmm_src);
        } else {
            switch (src_prc) {
                case Precision::FP32:
                case Precision::I32:
                    uni_vmovups(vmm_src, op);
                    break;
                case Precision::BF16:
                    vpmovzxwd(vmm_src, op);
                    uni_vpslld(vmm_src, vmm_src, 16);
                    break;
                case Precision::U16:
                    uni_vpmovzxwd(vmm_src, op);
                    break;
                case Precision::I16:
                    uni_vpmovsxwd(vmm_src, op);
                    break;
                case Precision::I8:
                    uni_vpmovsxbd(vmm_src, op);
                    break;
                case Precision::U8:
                    uni_vpmovzxbd(vmm_src, op);
                    break;
                default:
                    assert(!"unknown src_prc");
            }

            switch (dst_prc) {
                case Precision::FP32:
                    if (src_prc != Precision::FP32 && src_prc != Precision::BF16)
                        uni_vcvtdq2ps(vmm_src, vmm_src);
                    break;
                case Precision::I32:
                    if (src_prc == Precision::FP32 || src_prc == Precision::BF16)
                        uni_vcvtps2dq(vmm_src, vmm_src);
                    break;
                default:
                    assert(!"unknown dst_prc");
            }
        }
    }

    inline void load_scalar(Xmm xmm_src, const Xbyak::Address &op, Precision src_prc, Precision dst_prc) {
        switch (src_prc) {
            case Precision::FP32:
            case Precision::I32:
                uni_vmovss(xmm_src, op);
                break;
            case Precision::BF16:
                uni_vpinsrw(xmm_src, xmm_src, op, 0);
                uni_vpslld(xmm_src, xmm_src, 16);
                break;
            case Precision::I16:
                uni_vpinsrw(xmm_src, xmm_src, op, 0);
                uni_vpmovsxwd(xmm_src, op);
                break;
            case Precision::U16:
                uni_vpinsrw(xmm_src, xmm_src, op, 0);
                uni_vpmovzxwd(xmm_src, op);
                break;
            case Precision::I8:
                movsx(reg_tmp_32, op);
                uni_vmovq(xmm_src, reg_tmp_64);
                break;
            case Precision::U8:
                movzx(reg_tmp_32, op);
                uni_vmovq(xmm_src, reg_tmp_64);
                break;
            default:
                assert(!"unknown src_prc");
        }

        switch (dst_prc) {
            case Precision::FP32:
                if (src_prc != Precision::FP32 && src_prc != Precision::BF16)
                    uni_vcvtdq2ps(xmm_src, xmm_src);
                break;
            case Precision::I32:
                if (src_prc == Precision::FP32 || src_prc == Precision::BF16)
                    uni_vcvtps2dq(xmm_src, xmm_src);
                break;
            default:
                assert(!"unknown dst_prc");
        }
    }

    inline void store_vector(const Xbyak::Address &op, Vmm vmm_dst, Precision src_prc, Precision dst_prc) {
        Xmm xmm_dst = Xmm(vmm_dst.getIdx());
        Ymm ymm_dst = Ymm(vmm_dst.getIdx());

        switch (src_prc) {
            case Precision::FP32:
                if (dst_prc != Precision::FP32 && dst_prc != Precision::BF16)
                    uni_vcvtps2dq(vmm_dst, vmm_dst);
                break;
            case Precision::I32:
                if (dst_prc == Precision::FP32 || dst_prc == Precision::BF16)
                    uni_vcvtdq2ps(vmm_dst, vmm_dst);
                break;
            default:
                assert(!"unknown src_prc");
        }

        switch (dst_prc) {
            case Precision::FP32:
            case Precision::I32:
                uni_vmovups(op, vmm_dst);
                break;
            case Precision::BF16:
                if (mayiuse(avx512_core_bf16))
                    vcvtneps2bf16(ymm_dst, vmm_dst);
                else
                    emu_vcvtneps2bf16->emit_code({static_cast<size_t>(vmm_dst.getIdx())}, {static_cast<size_t>(ymm_dst.getIdx())});
                vmovdqu16(op, ymm_dst);
                break;
            case Precision::I16:
                if (isa == x64::avx512_common) {
                    vpmovsdw(op, vmm_dst);
                } else {
                    uni_vpackssdw(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != x64::sse41) {
                        vpermq(ymm_dst, ymm_dst, 0x08);
                        uni_vmovdqu(op, xmm_dst);
                    } else {
                        movq(op, xmm_dst);
                    }
                }
                break;
            case Precision::U16:
                if (isa == x64::avx512_common) {
                    vmaxsd(vmm_dst, vmm_zero, vmm_dst);
                    vpmovusdw(op, vmm_dst);
                } else {
                    uni_vpackusdw(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != x64::sse41) {
                        vpermq(ymm_dst, ymm_dst, 0x08);
                        uni_vmovdqu(op, xmm_dst);
                    } else {
                        movq(op, xmm_dst);
                    }
                }
                break;
            case Precision::I8:
                if (isa == x64::avx512_common) {
                    vmaxps(vmm_dst, vmm_zero, vmm_dst);
                    vpmovsdb(op, vmm_dst);
                } else {
                    uni_vpackssdw(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != x64::sse41)
                        vpermq(ymm_dst, ymm_dst, 0x08);
                    uni_vpacksswb(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != x64::sse41)
                        vmovq(op, xmm_dst);
                    else
                        movd(op, xmm_dst);
                }
                break;
            case Precision::U8:
                if (isa == x64::avx512_common) {
                    vpmovusdb(op, vmm_dst);
                } else {
                    uni_vpackusdw(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != x64::sse41)
                        vpermq(ymm_dst, ymm_dst, 0x08);
                    uni_vpackuswb(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != x64::sse41)
                        vmovq(op, xmm_dst);
                    else
                        movd(op, xmm_dst);
                }
                break;
            default:
                assert(!"unknown dst_prc");
        }
    }

    inline void store_scalar(const Xbyak::Address &op, Xmm xmm_dst, Precision src_prc, Precision dst_prc) {
        switch (src_prc) {
            case Precision::FP32:
                if (dst_prc != Precision::FP32 && dst_prc != Precision::BF16)
                    uni_vcvtps2dq(xmm_dst, xmm_dst);
                break;
            case Precision::I32:
                if (dst_prc == Precision::FP32 || dst_prc == Precision::BF16)
                    uni_vcvtdq2ps(xmm_dst, xmm_dst);
                break;
            default:
                assert(!"unknown src_prc");
        }

        switch (dst_prc) {
            case Precision::FP32:
            case Precision::I32:
                uni_vmovss(op, xmm_dst);
                break;
            case Precision::BF16:
                uni_vpsrld(xmm_dst, xmm_dst, 16);
                uni_vpextrw(op, xmm_dst, 0x0);
                break;
            case Precision::I16:
                uni_vpackssdw(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
                break;
            case Precision::U16:
                uni_vpackusdw(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
                break;
            case Precision::I8:
                uni_vpackssdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpacksswb(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
                break;
            case Precision::U8:
                uni_vpackusdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpackuswb(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
                break;
            default:
                assert(!"unknown dst_prc");
        }
    }
};

MKLDNNEltwiseNode::BroadcastingPolicy MKLDNNEltwiseNode::determineBroadcastingPolicy(const std::shared_ptr<ngraph::Node>& op) {
    const auto const1 = std::dynamic_pointer_cast<ngraph::opset1::Constant>(op->get_input_node_shared_ptr(0));
    const auto const2 = std::dynamic_pointer_cast<ngraph::opset1::Constant>(op->get_input_node_shared_ptr(1));
    int constPort = -1;
    if (const2) {
        constPort = 1;
    } else if (const1) {
        constPort = 0;
    } else {
        return Undefined;
    }

    auto const_shape = op->get_input_shape(constPort);
    if (ngraph::shape_size(const_shape) == 1)
        return PerTensor;
    else
        return PerChannel;
}

const std::map<const ngraph::DiscreteTypeInfo, MKLDNNEltwiseNode::Initializer> MKLDNNEltwiseNode::initializers = {
    {ngraph::op::v1::Add::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseAdd;
        node.broadcastingPolicy = determineBroadcastingPolicy(op);
    }},
    {ngraph::op::v1::Subtract::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseSubtract;
        node.broadcastingPolicy = determineBroadcastingPolicy(op);
    }},
    {ngraph::op::v1::Multiply::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseMultiply;
        node.broadcastingPolicy = determineBroadcastingPolicy(op);
    }},
    {ngraph::op::v1::Divide::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseDivide;
        node.broadcastingPolicy = determineBroadcastingPolicy(op);
    }},
    {ngraph::op::v0::SquaredDifference::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseSquaredDifference;
    }},
    {ngraph::op::v1::Maximum::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseMaximum;
    }},
    {ngraph::op::v1::Minimum::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseMinimum;
    }},
    {ngraph::op::v1::Mod::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseMod;
    }},
    {ngraph::op::v1::FloorMod::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseFloorMod;
    }},
    {ngraph::op::v1::Power::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwisePowerDynamic;
    }},
    {PowerStaticNode::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        auto powerStatic = getNgraphOpAs<PowerStaticNode>(op);
        node.algorithm = EltwisePowerStatic;
        node.alpha = powerStatic->get_power();
        node.beta = powerStatic->get_scale();
        node.gamma = powerStatic->get_shift();
        node.broadcastingPolicy = PerTensor;
    }},
    {ngraph::op::v1::Equal::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseEqual;
    }},
    {ngraph::op::v1::NotEqual::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseNotEqual;
    }},
    {ngraph::op::v1::Greater::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseGreater;
    }},
    {ngraph::op::v1::GreaterEqual::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseGreaterEqual;
    }},
    {ngraph::op::v1::Less::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseLess;
    }},
    {ngraph::op::v1::LessEqual::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseLessEqual;
    }},
    {ngraph::op::v1::LogicalAnd::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseLogicalAnd;
    }},
    {ngraph::op::v1::LogicalOr::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseLogicalOr;
    }},
    {ngraph::op::v1::LogicalXor::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseLogicalXor;
    }},
    {ngraph::op::v1::LogicalNot::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseLogicalNot;
    }},
    {ngraph::op::v0::Relu::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseRelu;
        node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_relu;
    }},
    {LeakyReluNode::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        auto leakyRelu = getNgraphOpAs<LeakyReluNode>(op);
        node.algorithm = EltwiseRelu;
        node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_relu;
        node.alpha = leakyRelu->get_slope();
        node.beta = 0.0f;
    }},
    {ngraph::op::v0::Gelu::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseGelu;
        node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_gelu_erf;
    }},
    {ngraph::op::v7::Gelu::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        auto gelu = getNgraphOpAs<ngraph::op::v7::Gelu>(op);
        node.algorithm = EltwiseGelu;
        ngraph::op::GeluApproximationMode approximationMode = gelu->get_approximation_mode();
        if (approximationMode == ngraph::op::GeluApproximationMode::ERF)
            node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_gelu_erf;
        else if (approximationMode == ngraph::op::GeluApproximationMode::TANH)
            node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_gelu_tanh;
        else
            IE_THROW(NotImplemented) << "CPU Eltwise node doesn't support ngraph operation Gelu with approximation mode: " << approximationMode;
    }},
    {ngraph::op::v0::Elu::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        auto eluOp = getNgraphOpAs<ngraph::op::v0::Elu>(op);

        node.alpha = static_cast<float>(eluOp->get_alpha());
        node.algorithm = EltwiseElu;
        node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_elu;
    }},
    {ngraph::op::v0::Tanh::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseTanh;
        node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_tanh;
    }},
    {ngraph::op::v0::Sigmoid::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseSigmoid;
        node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_logistic;
    }},
    {ngraph::op::v0::Abs::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseAbs;
        node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_abs;
    }},
    {ngraph::op::v0::Sqrt::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseSqrt;
        node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_sqrt;
    }},
    {ngraph::op::v0::Clamp::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        auto clampOp = getNgraphOpAs<ngraph::op::v0::Clamp>(op);

        node.alpha = static_cast<float>(clampOp->get_min());
        node.beta = static_cast<float>(clampOp->get_max());
        node.algorithm = EltwiseClamp;
        node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_clip;
    }},
    {ngraph::op::v0::Exp::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseExp;
        node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_exp;
    }},
    {SwishNode::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        auto swishOp = getNgraphOpAs<SwishNode>(op);
        node.algorithm = EltwiseSwish;
        node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_swish;
        node.alpha = swishOp->get_alpha();
    }},
    {ngraph::op::v4::HSwish::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseHswish;
        node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_hardswish;
    }},
    {ngraph::op::v4::Mish::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseMish;
        node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_mish;
    }},
    {ngraph::op::v5::HSigmoid::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseHsigmoid;
        node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_hsigmoid;
    }},
    {ngraph::op::v5::Round::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        auto roundOp = getNgraphOpAs<ngraph::op::v5::Round>(op);

        switch (roundOp->get_mode()) {
            case ngraph::op::v5::Round::RoundMode::HALF_TO_EVEN:
                node.algorithm = EltwiseRoundHalfToEven;
                node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_round_half_to_even;
                break;
            case ngraph::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO:
                node.algorithm = EltwiseRoundHalfAwayFromZero;
                node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_round_half_away_from_zero;
                break;
        }
    }},
    {ngraph::op::v0::PRelu::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwisePrelu;
        node.broadcastingPolicy = determineBroadcastingPolicy(op);
    }},
    {ngraph::op::v0::Erf::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseErf;
    }},
    {ngraph::op::v4::SoftPlus::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseSoftRelu;
        node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_soft_relu;
    }},
};


namespace {
struct EltwiseKey {
    std::vector<MKLDNNEltwiseNode::EltwiseData> eltwise_data;
    std::vector<Type> ops_list;
    VectorDims outBlkDims;
    VectorDims outOrder;
    std::vector<VectorDims> inpDims;
    std::vector<InferenceEngine::Precision> inpPrc;
    InferenceEngine::Precision outPrc;
    mkldnn::post_ops postOps;
    bool useDynBatch;
    bool useJit;

    size_t hash() const {
        using namespace dnnl::impl;
        using namespace dnnl::impl::primitive_hashing;
        size_t seed = 0;
        auto hash_combine_eltwiseData = [](size_t seed, const MKLDNNEltwiseNode::EltwiseData& eltwiseData) {
            seed = hash_combine(seed, eltwiseData.algo);
            seed = hash_combine(seed, eltwiseData.mkldnnAlgorithm);
            seed = hash_combine(seed, eltwiseData.alpha);
            seed = hash_combine(seed, eltwiseData.beta);
            seed = hash_combine(seed, eltwiseData.gamma);
            return seed;
        };
        std::for_each(eltwise_data.begin(), eltwise_data.end(), [&](const MKLDNNEltwiseNode::EltwiseData& item) {
            seed = hash_combine_eltwiseData(seed, item);
        });
        seed = get_vector_hash(seed, ops_list);
        seed = get_vector_hash(seed, outBlkDims);
        seed = get_vector_hash(seed, outOrder);
        for (auto&& item : inpDims) {
            seed = get_vector_hash(seed, item);
        }
        std::for_each(inpPrc.begin(), inpPrc.end(), [&](const Precision& item) {
            seed = hash_combine(seed, item.getPrecVal());
        });
        seed = hash_combine(seed, outPrc.getPrecVal());
        seed = get_post_op_hash(seed, *postOps.get());
        seed = hash_combine(seed, useDynBatch);
        seed = hash_combine(seed, useJit);
        return seed;
    }

    bool operator==(const EltwiseKey& rhs) const {
        if (inpDims.size() != rhs.inpDims.size()) {
            return false;
        }

        bool result = eltwise_data == rhs.eltwise_data &&
                      ops_list == rhs.ops_list &&
                      outBlkDims == rhs.outBlkDims &&
                      outOrder == rhs.outOrder &&
                      inpPrc == rhs.inpPrc &&
                      outPrc == rhs.outPrc &&
                      *postOps.get() == *rhs.postOps.get() &&
                      useDynBatch == rhs.useDynBatch &&
                      useJit == rhs.useJit;

        for (size_t i = 0; i < inpDims.size() && result; ++i) {
            result = result && (inpDims[i] == rhs.inpDims[i]);
        }
        return result;
    }
};

class EltwiseJitExecutor : public MKLDNNEltwiseNode::IEltwiseExecutor {
public:
    static void offset_out_calc(VectorDims& offset, const VectorDims& dims) {
        int k = 1;
        for (int i = offset.size() - 1; i >= 0; i--) {
            offset[i] = k;
            k *= dims[i];
        }
    }

    static void offset_in_calc(VectorDims& offset, const VectorDims& dims_in, const VectorDims& dims_out) {
        int k = 1;
        for (int i = offset.size() - 1; i >= 0; i--) {
            offset[i] = (dims_in[i] == dims_out[i]) ? k : 0;
            k *= dims_in[i];
        }
    }

    EltwiseJitExecutor(const std::vector<MKLDNNEltwiseNode::EltwiseData>& eltwise_data,
                       const std::vector<Type>& ops_list,
                       const VectorDims& outBlkDims,
                       const VectorDims& outOrder,
                       std::vector<VectorDims> inpDims,
                       const std::vector<InferenceEngine::Precision>& inpPrc,
                       const InferenceEngine::Precision& outPrc,
                       const mkldnn::post_ops& post_ops,
                       bool useDynBatch) {
        auto collapseLastDims = [](std::vector<size_t>& dims, int dimsToCollapse) {
            for (int i = dims.size() - 2; i > dims.size() - dimsToCollapse - 2; i--) {
                dims[dims.size() - 1] *= dims[i];
            }

            for (int i = dims.size() - 2; i >= dimsToCollapse; i--) {
                dims[i] = dims[i - dimsToCollapse];
            }

            for (int i = dimsToCollapse - 1; i >= 0; i--) {
                dims[i] = 1;
            }
        };

        auto collapseLastOffsets = [](std::vector<size_t>& dims, int dimsToCollapse) {
            for (int i = dims.size() - 2; i > dims.size() - dimsToCollapse - 2; i--) {
                if (dims[dims.size() - 1] > 0 || dims[i] > 0)
                    dims[dims.size() - 1] = std::max(dims[dims.size() - 1], static_cast<size_t>(1)) * std::max(dims[i], static_cast<size_t>(1));
                else
                    dims[dims.size() - 1] *= dims[i];
            }

            for (int i = dims.size() - 2; i >= dimsToCollapse; i--) {
                dims[i] = dims[i - dimsToCollapse];
            }

            for (int i = dimsToCollapse - 1; i >= 0; i--) {
                dims[i] = 0;
            }
        };

        auto isFusedWith = [&](Type type_) {
            auto start_itr = ops_list.begin();
            std::advance(start_itr, 1); // apply offset since the first op in the list is the op itself
            return any_of(start_itr, ops_list.end(), [=](Type type) { return type == type_; });
        };

        if (inpDims.empty()) {
            IE_THROW() << "Can not make Eltwise executor from empty input dims array";
        } else if (inpDims.front().empty()) {
            IE_THROW() << "Can not make Eltwise executor from empty input dims members";
        }

        jit_eltwise_params jep = {};
        size_t inputsNumber = inpDims.size();

        jep.input_size = inpDims.front().size();

        jep.dims.resize(jep.input_size, 1);

        if (outBlkDims.empty()) {
            IE_THROW() << "Can not make Eltwise executor from empty block dims vector";
        }

        size_t outRank = outBlkDims.size();
        for (int i = 0; i < outRank; i++) {
            jep.dims[jep.dims.size() - 1 - i] = outBlkDims[outRank - 1 - i];
        }

        for (int i = 0; i < inpDims.size(); i++) {
            for (int j = 0; j < inpDims[i].size(); j++) {
                if (inpDims[i][j] != jep.dims[j] && inpDims[i][j] != 1)
                    IE_THROW() << "Eltwise executor got invalid input/output dims configuration.";
            }
        }

        if (outBlkDims.size() != outOrder.size()) {
            IE_THROW() << "Can not make Elwtise executor due to out blocked dims and out order vectors size mismatch.";
        }

        int lastUnchangedAxis = 0;
        size_t oc_size = 0;
        jep.oc_offsets.resize(jep.input_size, 0);
        std::fill(jep.oc_offsets.begin(), jep.oc_offsets.end(), 0);
        if (isFusedWith(FakeQuantize)) {
            size_t offset_oc = 1;
            for (int i = outOrder.size() - 1; i >= 0; i--) {
                if (outOrder[i] == 1) {
                    int oc_dim_idx = i + (jep.input_size - outOrder.size());
                    jep.oc_offsets[oc_dim_idx] = offset_oc;
                    offset_oc *= jep.dims[oc_dim_idx];
                    if (oc_dim_idx + 1 != jep.input_size) { // since in nspc case we can safely collapse the last axis
                        lastUnchangedAxis = oc_dim_idx;
                    }
                }
            }
            oc_size = jep.oc_offsets[jep.dims.size() - 1] != 0 ? jep.dims[jep.dims.size() - 1] : 1;
        }

        int maxCollapsedDims = static_cast<int>(jep.dims.size()) - lastUnchangedAxis - 2;

        size_t fullWorkAmount = 1;
        for (int i = 0; i < jep.dims.size(); i++) {
            fullWorkAmount *= jep.dims[i];
        }

        size_t minimalConcurrency = parallel_get_max_threads();
        size_t minimalJitWorkAmount = 256;
        size_t currentJitWorkAmount = jep.dims[jep.dims.size() - 1];
        int collapsedDims = 0;

        bool hasDifferentDims = false;
        while (currentJitWorkAmount < minimalJitWorkAmount && currentJitWorkAmount < fullWorkAmount &&
               // we shouldn't collapse batch dimension in case dynamic batch is enabled
               (!useDynBatch || (outBlkDims.size() - collapsedDims > 2))) {
            if (collapsedDims >= maxCollapsedDims)
                break;

            for (int j = 1; j < inpDims.size(); j++) {
                if (inpDims[j].back() != inpDims[0].back()) {
                    hasDifferentDims = true;
                }
            }

            if (oc_size > 1 && oc_size != inpDims[0][inpDims[0].size() - 1]) {
                hasDifferentDims = true;
            }

            bool canCollapse = true;
            for (int i = 0; i < inpDims.size(); i++) {
                if (inpDims[i][inpDims[i].size() - 2] != 1) {
                    if (hasDifferentDims) {
                        canCollapse = false;
                        break;
                    }
                }
            }

            if (!canCollapse) {
                break;
            }

            size_t nextJitWorkAmount = currentJitWorkAmount * jep.dims[jep.dims.size() - 2];
            if (fullWorkAmount / nextJitWorkAmount >= minimalConcurrency) {
                currentJitWorkAmount = nextJitWorkAmount;
                collapsedDims++;

                for (int i = 0; i < inpDims.size(); i++) {
                    collapseLastDims(inpDims[i], 1);
                }
                collapseLastDims(jep.dims, 1);

                if (isFusedWith(FakeQuantize)) {
                    collapseLastOffsets(jep.oc_offsets, 1);
                }
            } else {
                break;
            }
        }

        _batchDimIdx = jep.input_size - outBlkDims.size() + collapsedDims;
        _schedulerWorkAmount = fullWorkAmount / jep.dims[jep.dims.size() - 1];

        if (inpPrc.size() != inputsNumber) {
            IE_THROW() << "Can not make Elwtise executor. Wrong input precisions vector size.";
        }

        // init offset
        jep.dst_offsets.resize(jep.input_size, 1);
        offset_out_calc(jep.dst_offsets, jep.dims);
        for (int j = 0; j < jep.input_size; j++) {
            jep.dst_offsets[j] *= outPrc.size();
        }

        for (int i = 0; i < inputsNumber; i++) {
            jep.src_offsets[i].resize(jep.input_size, 1);
            offset_in_calc(jep.src_offsets[i], inpDims[i], jep.dims);
            for (int j = 0; j < jep.input_size; j++) {
                jep.src_offsets[i][j] *= inpPrc[i].size();
            }
        }

        jep.inputs_number = inputsNumber;

        for (int i = 0; i < inputsNumber; i++) {
            jep.src_prc[i] = inpPrc[i];
            jep.src_size[i] = inpDims[i][inpDims[i].size() - 1];
        }
        jep.dst_prc = outPrc;
        jep.work_amount = jep.dst_size = jep.dims.back();
        jep.oc_size = oc_size;

        std::transform(jep.oc_offsets.begin(), jep.oc_offsets.end(), jep.oc_offsets.begin(),
                       [](size_t& offset) { return offset * sizeof(float);});

        if (mayiuse(x64::avx512_common)) {
            _pKernel.reset(new jit_uni_eltwise_generic<x64::avx512_common>(jep, eltwise_data, ops_list, post_ops));
        } else if (mayiuse(x64::avx2)) {
            _pKernel.reset(new jit_uni_eltwise_generic<x64::avx2>(jep, eltwise_data, ops_list, post_ops));
        } else if (mayiuse(x64::sse41)) {
            _pKernel.reset(new jit_uni_eltwise_generic<x64::sse41>(jep, eltwise_data, ops_list, post_ops));
        } else {
            IE_THROW() << "Can't create jit eltwise kernel";
        }

        if (_pKernel)
            _pKernel->create_ker();
    }

    void exec(const jit_eltwise_call_args_ptrs &args_ptrs, const VectorDims &dims_out) override {
        if (!_pKernel)
            IE_THROW() << "Can't execute, kernel for eltwise node is not compiled";

        if (_pKernel->jep_.input_size == optimalTensorRank) {
            // execute Optimized 6D
            parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4],
                           [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
                               auto args = jit_eltwise_call_args_indexes();
                               args.indexes[0] = i0;
                               args.indexes[1] = i1;
                               args.indexes[2] = i2;
                               args.indexes[3] = i3;
                               args.indexes[4] = i4;

                               (*_pKernel)(&args_ptrs, &args);
                           });
        } else {
            // execute Optimized Generic
            parallel_nt(0, [&](const int ithr, const int nthr) {
                size_t start = 0, end = 0;
                splitter(_schedulerWorkAmount, nthr, ithr, start, end);

                std::vector<size_t> counters(dims_out.size() - 1, 0);
                auto args = jit_eltwise_call_args_indexes();
                for (size_t iwork = start; iwork < end; ++iwork) {
                    size_t tmp = iwork;
                    for (ptrdiff_t j = dims_out.size() - 2; j >= 0; j--) {
                        counters[j] = tmp % dims_out[j];
                        tmp /= dims_out[j];
                    }

                    for (size_t j = 0; j < counters.size(); j++)
                        args.indexes[j] = counters[j];

                    (*_pKernel)(&args_ptrs, &args);
                }
            });
        }
    }
    const VectorDims& getOutDims() const override {
        if (!_pKernel)
            IE_THROW() << "Can't get jit eltwise params, kernel for Eltwise executor is not compiled";
        return _pKernel->jep_.dims;
    }
    size_t getBatchDimIdx() const override {
        return _batchDimIdx;
    }

private:
    std::unique_ptr<jit_uni_eltwise_kernel> _pKernel;
    size_t _schedulerWorkAmount = 0;
    size_t _batchDimIdx = 0;

public:
    static const int optimalTensorRank = 6;
};

class EltwiseRefExecutor : public MKLDNNEltwiseNode::IEltwiseExecutor {
public:
    EltwiseRefExecutor(MKLDNNEltwiseNode::EltwiseData opData,
                       const VectorDims& outBlkDims,
                       std::vector<VectorDims> inpDims)
    : _opData(std::move(opData)) {
        if (inpDims.empty()) {
            IE_THROW() << "Can not make Eltwise executor from empty input dims array";
        } else if (inpDims.front().empty()) {
            IE_THROW() << "Can not make Eltwise executor from empty input dims array members";
        }

        if (outBlkDims.empty()) {
            IE_THROW() << "Can not make Elwtise executor from empty output blocked dims vector";
        }

        _inputNum = inpDims.size();
        size_t input_size = inpDims.front().size();
        _batchDimIdx = input_size - outBlkDims.size();

        _dims.resize(input_size, 1);
        for (int i = 0; i < outBlkDims.size(); i++) {
            _dims[_dims.size() - 1 - i] = outBlkDims[outBlkDims.size() - 1 - i];
        }

        _fullWorkAmount = 1;
        for (int i = 0; i < _dims.size(); i++) {
            _fullWorkAmount *= _dims[i];
        }

        // init offset
        _dst_offsets.resize(input_size, 1);
        EltwiseJitExecutor::offset_out_calc(_dst_offsets, _dims);
        for (int j = 0; j < input_size; j++) {
            _dst_offsets[j] *= sizeof(float); // only FP32 out prc is supported
        }

        for (int i = 0; i < _inputNum; i++) {
            _src_offsets[i].resize(input_size, 1);
            EltwiseJitExecutor::offset_in_calc(_src_offsets[i], inpDims[i], _dims);
            for (int j = 0; j < input_size; j++) {
                _src_offsets[i][j] *= sizeof(float); // only FP32 inp prcs are supported
            }
        }
    }

    void exec(const jit_eltwise_call_args_ptrs &args_ptrs, const VectorDims &dims_out) override {
        std::shared_ptr<ref_eltwise_scalar_fwd_t> ref_eltwise_injector = nullptr;
        if (_opData.mkldnnAlgorithm != mkldnn::algorithm::undef) {
            ref_eltwise_injector = std::make_shared<ref_eltwise_scalar_fwd_t>(
                    static_cast<mkldnn_alg_kind_t>(_opData.mkldnnAlgorithm), _opData.alpha, _opData.beta, 1.f);
        }

        parallel_nt(0, [&](const int ithr, const int nthr) {
            size_t start = 0, end = 0;
            splitter(_fullWorkAmount, nthr, ithr, start, end);

            std::vector<size_t> counters(dims_out.size(), 0);

            for (size_t iwork = start; iwork < end; ++iwork) {
                size_t tmp = iwork;
                for (ptrdiff_t j = dims_out.size() - 1; j >= 0; j--) {
                    counters[j] = tmp % dims_out[j];
                    tmp /= dims_out[j];
                }

                size_t index_in[MAX_ELTWISE_INPUTS] = {0};
                for (int i = 0; i < _inputNum; i++) {
                    index_in[i] = 0;
                    for (int j = 0; j < counters.size(); j++) {
                        index_in[i] += counters[j] * _src_offsets[i][j];
                    }
                    index_in[i] /= sizeof(float);
                }

                size_t index_out = 0;
                for (int j = 0; j < counters.size(); j++) {
                    index_out += counters[j] * _dst_offsets[j];
                }
                index_out /= sizeof(float);

                std::vector<float> src_f(_inputNum);
                for (int i = 0; i < _inputNum; i++) {
                    src_f[i] = (reinterpret_cast<const float*>(args_ptrs.src_ptr[i]) + index_in[i])[0];
                }
                float* dst_ptr_f = reinterpret_cast<float*>(args_ptrs.dst_ptr) + index_out;

                switch (_opData.algo) {
                    case EltwiseRelu: case EltwiseGelu: case EltwiseElu: case EltwiseTanh: case EltwiseSigmoid: case EltwiseAbs:
                    case EltwiseSqrt: case EltwiseSoftRelu: case EltwiseExp: case EltwiseClamp:
                    case EltwiseSwish: case EltwiseHswish: case EltwiseMish: case EltwiseHsigmoid:
                    case EltwiseRoundHalfToEven: case EltwiseRoundHalfAwayFromZero:
                        *dst_ptr_f = ref_eltwise_injector->compute_scalar(src_f[0]); break;
                    case EltwiseAdd:               *dst_ptr_f = src_f[0] + src_f[1]; break;
                    case EltwiseMulAdd:            *dst_ptr_f = src_f[0] * src_f[1] + src_f[2]; break;
                    case EltwiseSubtract:          *dst_ptr_f = src_f[0] - src_f[1]; break;
                    case EltwiseMultiply:          *dst_ptr_f = src_f[0] * src_f[1]; break;
                    case EltwiseDivide:            *dst_ptr_f = src_f[0] / src_f[1]; break;
                    case EltwiseFloorMod:          *dst_ptr_f = src_f[0] - floorf(src_f[0] / src_f[1]) * src_f[1]; break;
                    case EltwiseMod:               *dst_ptr_f = src_f[0] - truncf(src_f[0] / src_f[1]) * src_f[1]; break;
                    case EltwiseMaximum:           *dst_ptr_f = std::max(src_f[0], src_f[1]); break;
                    case EltwiseMinimum:           *dst_ptr_f = std::min(src_f[0], src_f[1]); break;
                    case EltwiseSquaredDifference: *dst_ptr_f = powf((src_f[0] - src_f[1]), 2.f); break;
                    case EltwisePowerDynamic:      *dst_ptr_f = powf(src_f[0], src_f[1]); break;
                    case EltwiseEqual:             *dst_ptr_f = src_f[0] == src_f[1]; break;
                    case EltwiseNotEqual:          *dst_ptr_f = src_f[0] != src_f[1]; break;
                    case EltwiseGreater:           *dst_ptr_f = src_f[0] > src_f[1]; break;
                    case EltwiseGreaterEqual:      *dst_ptr_f = src_f[0] >= src_f[1]; break;
                    case EltwiseLess:              *dst_ptr_f = src_f[0] < src_f[1]; break;
                    case EltwiseLessEqual:         *dst_ptr_f = src_f[0] <= src_f[1]; break;
                    case EltwiseLogicalAnd:        *dst_ptr_f = src_f[0] && src_f[1]; break;
                    case EltwiseLogicalOr:         *dst_ptr_f = src_f[0] || src_f[1]; break;
                    case EltwiseLogicalXor:        *dst_ptr_f = (src_f[0] || src_f[1]) - (src_f[0] && src_f[1]); break;
                    case EltwiseLogicalNot:        *dst_ptr_f = !src_f[0]; break;
                    case EltwisePowerStatic:       *dst_ptr_f = powf(_opData.beta * src_f[0] + _opData.gamma, _opData.alpha); break;
                    case EltwisePrelu:             *dst_ptr_f = src_f[0] > 0 ? src_f[0] : src_f[0] * src_f[1]; break;
                    case EltwiseErf:               *dst_ptr_f = std::erf(src_f[0]); break;
                    default: IE_THROW() << "Unsupported operation type for Eltwise executor";
                }
            }
        });
    }

    const VectorDims& getOutDims() const override {
        return _dims;
    }

    size_t getBatchDimIdx() const override {
        return _batchDimIdx;
    }

private:
    const MKLDNNEltwiseNode::EltwiseData _opData;
    VectorDims _dims;
    VectorDims _src_offsets[MAX_ELTWISE_INPUTS];
    VectorDims _dst_offsets;
    size_t _fullWorkAmount = 0;
    size_t _inputNum = 0;
    size_t _batchDimIdx = 0;
};

} // namespace

bool MKLDNNEltwiseNode::EltwiseData::operator==(const EltwiseData &rhs) const noexcept {
    return algo == rhs.algo &&
           mkldnnAlgorithm == rhs.mkldnnAlgorithm &&
           alpha == rhs.alpha &&
           beta == rhs.beta &&
           gamma == rhs.gamma;
}

static MKLDNNEltwiseNode::executorPtr buildExecutor(const EltwiseKey& key) {
    MKLDNNEltwiseNode::executorPtr execPtr;
    if (key.useJit) {
        execPtr = std::make_shared<EltwiseJitExecutor>(key.eltwise_data,
                                                       key.ops_list,
                                                       key.outBlkDims,
                                                       key.outOrder,
                                                       key.inpDims,
                                                       key.inpPrc,
                                                       key.outPrc,
                                                       key.postOps,
                                                       key.useDynBatch);
    } else {
        execPtr = std::make_shared<EltwiseRefExecutor>(key.eltwise_data.front(),
                                                       key.outBlkDims,
                                                       key.inpDims);
    }
    return execPtr;
}

bool MKLDNNEltwiseNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (initializers.find(op->get_type_info()) == initializers.end()) {
            errorMessage = "Doesn't support Eltwise algorithm: " +  std::string(op->get_type_name());
            return false;
        }
        if (const auto binOp = std::dynamic_pointer_cast<const ov::op::util::BinaryElementwiseArithmetic>(op)) {
            if (binOp->get_autob().m_type != ngraph::op::AutoBroadcastType::NONE &&
                binOp->get_autob().m_type != ngraph::op::AutoBroadcastType::NUMPY) {
                errorMessage = "Doesn't support broadcast type: " + ngraph::as_string(binOp->get_autob().m_type);
                return false;
            }
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNEltwiseNode::MKLDNNEltwiseNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
    MKLDNNNode(op, eng, cache), broadcastingPolicy(Undefined) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
    initializers.at(op->get_type_info())(op, *this);
}

size_t MKLDNNEltwiseNode::getOpInputsNum() const {
    switch (getAlgorithm()) {
        case EltwiseRelu: case EltwiseGelu: case EltwiseElu: case EltwiseTanh: case EltwiseSigmoid: case EltwiseAbs: case EltwiseSqrt:
        case EltwiseSoftRelu: case EltwiseExp: case EltwiseClamp: case EltwiseErf: case EltwiseLogicalNot: case EltwisePowerStatic:
        case EltwiseSwish: case EltwiseHswish: case EltwiseMish: case EltwiseHsigmoid: case EltwiseRoundHalfToEven: case EltwiseRoundHalfAwayFromZero:
            return 1;
        case EltwiseAdd: case EltwiseSubtract: case EltwiseMultiply: case EltwiseDivide: case EltwiseFloorMod: case EltwiseMod: case EltwiseMaximum:
        case EltwiseMinimum: case EltwiseSquaredDifference: case EltwisePowerDynamic: case EltwiseEqual: case EltwiseNotEqual: case EltwiseGreater:
        case EltwiseGreaterEqual: case EltwiseLess:  case EltwiseLessEqual: case EltwiseLogicalAnd: case EltwiseLogicalOr: case EltwiseLogicalXor:
        case EltwisePrelu:
            return 2;
        case EltwiseMulAdd:
            return 3;
        default: IE_THROW() << "Unsupported operation for Eltwise node with name `" << getName() << "`.";
    }
}

// TODO [DS]: used only in FuseConvolutionSumAndConvolutionSumActivation
// fix when reimplement this transformation for dynamic shapes
bool MKLDNNEltwiseNode::isWithBroadcast() {
    auto oDims = getOutputShapeAtPort(0).getStaticDims();
    for (size_t i = 0; i < inputShapes.size(); i++) {
        auto iDims = getInputShapeAtPort(i).getStaticDims();
        if (iDims != oDims)
            return true;
    }

    return false;
}

void MKLDNNEltwiseNode::getSupportedDescriptors() {
    if (getParentEdges().size() < 1)
        IE_THROW() << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        IE_THROW() << "Incorrect number of output edges for layer " << getName();
}

void MKLDNNEltwiseNode::initSupportedPrimitiveDescriptors() {
    std::vector<Precision> supportedPrecisions = {
            Precision::FP32,
            Precision::U8,
            Precision::I8,
            Precision::U16,
            Precision::I16,
            Precision::BF16,
            Precision::I32
    };

    if (!supportedPrimitiveDescriptors.empty())
        return;

    // if dim rank is greater than the maximum possible, we should use the reference execution
    canUseOptimizedImpl = mayiuse(x64::sse41) && getInputShapeAtPort(0).getRank() <= MAX_ELTWISE_DIM_RANK;

    if (!canUseOptimizedImpl && !fusedWith.empty()) {
        IE_THROW(Unexpected) << "Eltwise node with name '" << getName() << "' uses reference impl, but unexpectedly fused with other ops";
    }

    size_t expectedInputsNum = getOpInputsNum();
    for (auto& postOp : fusedWith) {
        auto* eltwiseNode = dynamic_cast<const MKLDNNEltwiseNode*>(postOp.get());
        if (eltwiseNode != nullptr) {
            expectedInputsNum += eltwiseNode->getOpInputsNum() - 1;
        }
    }
    if (getParentEdges().size() > MAX_ELTWISE_INPUTS)
        IE_THROW() << "Eltwise node with name `" << getName() << "` doesn't support more than " << MAX_ELTWISE_INPUTS
                           << " inputs (actual = " << getParentEdges().size() << ")";

    if (expectedInputsNum != getParentEdges().size())
        IE_THROW() << "Eltwise node with name `" << getName() << "` has invalid input number of inputs: expected = " << expectedInputsNum
                           << " (actual = " << getParentEdges().size() << ")";

    std::vector<InferenceEngine::Precision> inputPrecisions;
    for (const auto &prec : getOriginalInputPrecisions()) {
        inputPrecisions.push_back(prec);
    }

    for (auto& fusedNode : fusedWith) {
        if (fusedNode->getType() == Eltwise) {
            for (int i = 0; i < fusedNode->getOriginalInputsNumber(); i++) {
                if (fusedNode->getFusingPort() != i)
                    inputPrecisions.push_back(fusedNode->getOriginalInputPrecisionAtPort(i));
            }
        }
    }

    if (inputPrecisions.size() != getParentEdges().size())
        IE_THROW() << "Eltwise node with name `" << getName() << "` has invalid input precisions configuration.";

    InferenceEngine::Precision outputPrecision = getOriginalOutputPrecisionAtPort(0);
    if (!fusedWith.empty()) {
        outputPrecision = fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0);
    }

    if (!mayiuse(avx512_core)) {
        bool hasBF16 = false;
        for (auto &inPrc : inputPrecisions)
            if (inPrc == Precision::BF16)
                hasBF16 = true;

        if (outputPrecision == Precision::BF16 || hasBF16)
            IE_THROW() << "Eltwise node with name `" << getName() << "` doesn't support BF16 precision on this target.";
    }

    auto filterPrecision = [&](Precision& prc) {
        if (!canUseOptimizedImpl) {
            return Precision(Precision::FP32);
        } else if (std::find(supportedPrecisions.begin(), supportedPrecisions.end(), prc) == supportedPrecisions.end()) {
            if (prc == Precision::U32 || prc == Precision::I64 || prc == Precision::U64) {
                return Precision(Precision::I32);
            } else {
                IE_THROW() << "Eltwise node with name `" << getName() << "` doesn't support " << prc << " precision.";
            }
        } else {
            return prc;
        }
    };

    for (int i = 0; i < inputPrecisions.size(); i++) {
        inputPrecisions[i] = filterPrecision(inputPrecisions[i]);
    }
    outputPrecision = filterPrecision(outputPrecision);

    // TODO: delete after new LPT (ngraph based) is merged
    // WA is needed to handle bug in LPT that produces wrong precision after average pooling (I8/U8 instead of FP32)
    if ((getAlgorithm() == EltwiseMulAdd || getAlgorithm() == EltwisePowerStatic) &&
            (inputPrecisions[0] == Precision::U8 || inputPrecisions[0] == Precision::I8)) {
        auto parentNode = getParentEdgesAtPort(0)[0]->getParent();
        if (getParentEdgesAtPort(0)[0]->getParent()->getAlgorithm() == PoolingAvg) {
            inputPrecisions[0] = Precision::FP32;
        }
    }

    enum LayoutType {
        Planar,
        ChannelsFirst,
        Blocked
    };

    auto initDesc = [&] (LayoutType lt) -> NodeDesc {
        auto createMemoryDesc = [lt](const Shape &shape, Precision prc, size_t offset) -> std::shared_ptr<CpuBlockedMemoryDesc> {
            const auto &dims = shape.getDims();
            if (lt == ChannelsFirst && shape.getRank() != 1) {
                auto ndims = shape.getRank();
                VectorDims order(ndims);
                std::iota(order.begin(), order.end(), 0);
                if (ndims > 1) {
                    order.erase(order.begin() + 1);
                    order.push_back(1);
                }

                VectorDims blocks(ndims);
                for (size_t i = 0; i < order.size(); i++) {
                    blocks[i] = dims[order[i]];
                }

                return std::make_shared<CpuBlockedMemoryDesc>(prc, shape, blocks, order, offset);
            // TODO: need investigate
            // bad accuracy for shape {1, 1, 4, 11}, {2, 5, 1, 1}
            // same for disabled collapse dims
            } else if (lt == Blocked && shape.getRank() != 1 && (shape.getMinDims()[1] != Shape::UNDEFINED_DIM && shape.getMinDims()[1] > 1)) {
                size_t blockSize = mayiuse(x64::avx512_common) ? 16 : 8;

                VectorDims blocks = dims;
                VectorDims order(blocks.size());
                std::iota(order.begin(), order.end(), 0);

                blocks[1] = dims[1] != Shape::UNDEFINED_DIM ? div_up(blocks[1], blockSize) : Shape::UNDEFINED_DIM;
                blocks.push_back(blockSize);
                order.push_back(1);

                return std::make_shared<CpuBlockedMemoryDesc>(prc, shape, blocks, order, offset);
            } else {
                VectorDims blocks = dims;
                VectorDims order(blocks.size());
                std::iota(order.begin(), order.end(), 0);

                return std::make_shared<CpuBlockedMemoryDesc>(prc, shape, blocks, order, offset);
            }
        };

        // TODO [DS]: inplace
        size_t offset = isDynamicNode() ? 0 : std::numeric_limits<size_t>::max();
        NodeConfig config;
        if (!isDynamicNode()) {
            config.dynBatchSupport = getOutputShapeAtPort(0).getRank() > 1 && getOutputShapeAtPort(0) ==
                                                                                    getInputShapeAtPort(0);
        }

        for (size_t i = 0; i < getParentEdges().size(); i++) {
            PortConfig portConfig;
            // TODO [DS]: inplace
            if (!isDynamicNode())
                portConfig.inPlace = (!i && canBeInPlace() && inputPrecisions[i] == outputPrecision) ? 0 : -1;
            portConfig.constant = false;

            const auto &srcShape = getInputShapeAtPort(i);
            portConfig.desc = createMemoryDesc(srcShape, inputPrecisions[i], offset);
            if (!isDynamicNode() && srcShape.getDims()[0] == 1) {
                const auto denseDesc = portConfig.desc->as<BlockedMemoryDesc>();
                auto strides = denseDesc->getStrides();
                strides[0] = Shape::UNDEFINED_DIM;
                portConfig.desc = std::make_shared<CpuBlockedMemoryDesc>(denseDesc->getPrecision(),
                                                                         denseDesc->getShape(),
                                                                         denseDesc->getBlockDims(),
                                                                         denseDesc->getOrder(),
                                                                         denseDesc->getOffsetPadding(),
                                                                         denseDesc->getOffsetPaddingToData(),
                                                                         strides);
            }

            config.inConfs.push_back(portConfig);
        }

        PortConfig portConfig;
        portConfig.inPlace = -1;
        portConfig.constant = false;

        const auto &dstShape = getOutputShapeAtPort(0);
        portConfig.desc = createMemoryDesc(dstShape, outputPrecision, offset);
        if (!isDynamicNode() && dstShape.getDims()[0] == 1) {
            const auto denseDesc = portConfig.desc->as<BlockedMemoryDesc>();
            auto strides = denseDesc->getStrides();
            strides[0] = Shape::UNDEFINED_DIM;
            portConfig.desc = std::make_shared<CpuBlockedMemoryDesc>(denseDesc->getPrecision(),
                                                                     denseDesc->getShape(),
                                                                     denseDesc->getBlockDims(),
                                                                     denseDesc->getOrder(),
                                                                     denseDesc->getOffsetPadding(),
                                                                     denseDesc->getOffsetPaddingToData(),
                                                                     strides);
        }

        config.outConfs.push_back(portConfig);

        impl_desc_type impl_type;
        if (mayiuse(x64::avx512_common)) {
            impl_type = impl_desc_type::jit_avx512;
        } else if (mayiuse(x64::avx2)) {
            impl_type = impl_desc_type::jit_avx2;
        } else if (mayiuse(x64::sse41)) {
            impl_type = impl_desc_type::jit_sse42;
        } else {
            impl_type = impl_desc_type::ref;
        }

        return {config, impl_type};
    };

    bool isChannelsFirstApplicable = one_of(getOutputShapeAtPort(0).getRank(), 1, 2, 3, 4, 5);
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        isChannelsFirstApplicable = isChannelsFirstApplicable && one_of(getInputShapeAtPort(i).getRank(), 1, 2, 3, 4, 5);
        isChannelsFirstApplicable = isChannelsFirstApplicable && implication(getInputShapeAtPort(i).getRank() != 1,
                                                                             getOutputShapeAtPort(0).getRank() ==
                                                                                     getInputShapeAtPort(i).getRank());
    }

    bool isBlockedApplicable = one_of(getOutputShapeAtPort(0).getRank(), 1, 3, 4, 5);
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        const auto &inShape = getInputShapeAtPort(i);
        isBlockedApplicable = isBlockedApplicable && one_of(inShape.getRank(), 1, 3, 4, 5);
        isBlockedApplicable = isBlockedApplicable && implication(inShape.getRank() != 1,
                                                                 getOutputShapeAtPort(0).getRank() ==
                                                                 inShape.getRank());
        if (isDynamicNode() && inShape.getRank() != 1)
            isBlockedApplicable = isBlockedApplicable && inShape.getMinDims()[1] != Shape::UNDEFINED_DIM && inShape.getMinDims()[1] > 1;
    }

    if (isChannelsFirstApplicable)
        supportedPrimitiveDescriptors.emplace_back(initDesc(ChannelsFirst));
    if (isBlockedApplicable)
        supportedPrimitiveDescriptors.emplace_back(initDesc(Blocked));
    supportedPrimitiveDescriptors.emplace_back(initDesc(Planar));

    inputNum = getParentEdges().size();
    currentInBlkDims.resize(inputNum);
}

std::vector<VectorDims> MKLDNNEltwiseNode::shapeInfer() const {
    ov::PartialShape outShape = getParentEdgesAtPort(0)[0]->getMemory().GetShape().toPartialShape();
    for (size_t i = 1; i < getParentEdges().size(); i++) {
        ov::PartialShape::broadcast_merge_into(outShape, getParentEdgesAtPort(i)[0]->getMemory().GetShape().toPartialShape(),
        ov::op::AutoBroadcastType::NUMPY);
    }
    return {outShape.get_shape()};
}

void MKLDNNEltwiseNode::prepareParams() {
    if (memPtrs.empty()) {
        for (auto i = 0; i < inputNum; i++)
            memPtrs.push_back(getParentEdgeAt(i)->getMemoryPtr());
        memPtrs.push_back(getChildEdgeAt(0)->getMemoryPtr());
    }

    auto outBlockingDesc = getChildEdgeAt(0)->getMemory().GetDescWithType<BlockedMemoryDesc>();
    const auto &outOrder = outBlockingDesc->getOrder();
    const auto &currentOutBlkDims = outBlockingDesc->getBlockDims();
    isDynBatchEnabled = getSelectedPrimitiveDescriptor()->getConfig().dynBatchSupport;

    size_t input_size = std::max(static_cast<size_t>(EltwiseJitExecutor::optimalTensorRank), currentOutBlkDims.size());

    std::vector<VectorDims> dims_in;
    // init dims
    dims_in.resize(inputNum);
    for (int i = 0; i < inputNum; i++) {
        dims_in[i].resize(input_size, 1);
    }

    size_t outRank = currentOutBlkDims.size();

    for (int i = 0; i < inputNum; i++) {
        auto inBlockingDesc = getParentEdgeAt(i)->getMemory().GetDescWithType<BlockedMemoryDesc>();
        currentInBlkDims[i] = inBlockingDesc->getBlockDims();
        size_t inRank = currentInBlkDims[i].size();

        // WA to normalize blocked and planar layouts
        const auto &inOrder = inBlockingDesc->getOrder();
        size_t startOff = outOrder.size() != outBlockingDesc->getShape().getRank() &&
                          outOrder[outOrder.size() - 1] != inOrder[inOrder.size() - 1] ? 1 : 0;

        // WA to handle nspc layout with 1D tensors
        if (1 == inRank) {
            if (outRank > 2 && 1 == outOrder.back()) startOff = 1;
        }

        for (int j = 0; j < inRank; j++) {
            dims_in[i][dims_in[i].size() - 1 - j - startOff] = currentInBlkDims[i][inRank - 1 - j];
        }
    }

    start_offset_in.resize(inputNum);
    for (size_t i = 0; i < inputNum; i++) {
        const auto desc = getParentEdgeAt(i)->getMemory().GetDescWithType<BlockedMemoryDesc>();
        start_offset_in[i] = desc->getOffsetPadding() * desc->getPrecision().size();
    }
    const auto desc = getChildEdgeAt(0)->getMemory().GetDescWithType<BlockedMemoryDesc>();
    start_offset_out = desc->getOffsetPadding() * desc->getPrecision().size();

    std::vector<InferenceEngine::Precision> inpPrc;
    for (size_t i = 0; i < inputNum; ++i) {
        inpPrc.push_back(getParentEdgeAt(i)->getMemory().getDesc().getPrecision());
    }

    auto outPrc = getChildEdgeAt(0)->getMemory().getDesc().getPrecision();

    EltwiseData thisOp{getAlgorithm(), getMKLDNNAlgorithm(), getAlpha(), getBeta(), getGamma()};

    EltwiseKey key = {{thisOp}, {getType()}, currentOutBlkDims, outOrder, dims_in, inpPrc, outPrc, mkldnn::post_ops(), isDynBatchEnabled, canUseOptimizedImpl};

    for (const auto &node : fusedWith) {
        key.ops_list.push_back(node->getType());
        if (node->getType() == Eltwise) {
            if (auto eltwise = std::dynamic_pointer_cast<MKLDNNEltwiseNode>(node)) {
                key.eltwise_data.push_back({eltwise->getAlgorithm(), eltwise->getMKLDNNAlgorithm(), eltwise->getAlpha(),
                                            eltwise->getBeta(), eltwise->getGamma()});
            }
        } else if (node->getType() == FakeQuantize) {
            node->appendPostOps(key.postOps, {});
        } else {
            IE_THROW(Unexpected) << "Eltwise node with name '" << getName() << "' has unexpected fused op of type '" << node->getTypeStr() << "'";
        }
    }

    // TODO: we need to rewrite quantization_t to remove the pointers from its content and update all the jit kernels at once
    // together with the corresponding appendPostOps method to pass the scales and shifts pointers at runtime.
    // Until then we have to read them from the quantization_t directly, store them somewhere
    // and nullify them to get read of the address dependency in the key structure
    fqDataPtrs.clear();
    for (int i = 0; i < key.postOps.len(); ++i) {
        auto &data = key.postOps.get()->entry_[i].quantization.data;
        fqDataPtrs.insert(fqDataPtrs.end(), std::begin(data), std::end(data));
        memset(data, 0, sizeof(data));
    }
    // end of TODO

    auto cache = getRuntimeCache();
    auto result = cache->getOrCreate(key, buildExecutor);
    execPtr = result.first;
}

bool MKLDNNEltwiseNode::needPrepareParams() const {
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        if (getParentEdgesAtPort(i)[0]->getMemory().GetDescWithType<BlockedMemoryDesc>()->getBlockDims() != currentInBlkDims[i])
            return true;
    }
    return false;
}

void MKLDNNEltwiseNode::selectOptimalPrimitiveDescriptor() {
    selectPreferPrimitiveDescriptor(getPrimitivesPriority(), true);
}

void MKLDNNEltwiseNode::initOptimalPrimitiveDescriptor() {
    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set.";
    auto config = selected_pd->getConfig();
    if (!isConfigDefined(config)) {
        for (size_t i = 0; i < config.inConfs.size(); i++) {
            config.inConfs[i].desc = getDefinedInputDesc(config, i);
        }

        for (size_t i = 0; i < config.outConfs.size(); i++) {
            config.outConfs[i].desc = getDefinedOutputDesc(config, i);
        }

        initDescriptor(config);
    } else {
        initDescriptor(config);
    }
}

void MKLDNNEltwiseNode::execute(mkldnn::stream strm) {
    if (execPtr) {
        jit_eltwise_call_args_ptrs args_ptrs = {};
        auto batchDimIdx = execPtr->getBatchDimIdx();
        VectorDims dims_out = execPtr->getOutDims();
        for (int i = 0; i < memPtrs.size() - 1; i++)
            args_ptrs.src_ptr[i] = reinterpret_cast<const uint8_t*>(memPtrs[i]->GetData()) + start_offset_in[i];
        args_ptrs.dst_ptr = reinterpret_cast<uint8_t*>(memPtrs.back()->GetData()) + start_offset_out;

        // In general case we need to recompute offsets as well but currently all supported layout assumes batch to be outermost dimension
        if (isDynBatchEnabled) {
            if (dims_out.size() <= batchDimIdx)
                IE_THROW() << "Can't set batch dims for eltwise node with rank: " << dims_out.size() << " and batch idx: " << batchDimIdx;
            dims_out[batchDimIdx] = static_cast<size_t>(batchToProcess());
        }

        args_ptrs.post_op_data = fqDataPtrs.data();

        execPtr->exec(args_ptrs, dims_out);
    } else {
        IE_THROW() << "Can't execute eltwise node with name: " << getName() << ". Primitive isn't created";
    }
}

void MKLDNNEltwiseNode::executeDynamicImpl(mkldnn::stream strm) {
    execute(strm);
}

bool MKLDNNEltwiseNode::created() const {
    return getType() == Eltwise;
}

bool MKLDNNEltwiseNode::canBeInPlace() const {
    if (getParentEdgesAtPort(0)[0]->getParent()->getType() == Input) {
        return false;
    }

    for (auto& parentEdge : getParentEdges()) {
        auto parent = parentEdge.lock()->getParent();
        if (parent->getChildEdges().size() != 1)
            return false;

        // WA to prevent memory corruption caused by inplace feature
        if (parent->getType() == Concatenation) {
            for (auto& parentParentEdge : parent->getParentEdges()) {
                auto parentParent = parentParentEdge.lock()->getParent();
                if (parentParent->getChildEdges().size() != 1)
                    return false;
            }
        }
    }

    return getInputShapeAtPort(0) == getOutputShapeAtPort(0);
}

void MKLDNNEltwiseNode::fuseInto(MKLDNNNodePtr& parentNode) {
    // Handling Convolution custom Add node fusing case which is processed via dnnl append_sum() API.
    // TODO [DS]: at this moment this transformation prohibit for dynamic case
    specialConvolutionAddFusing = (parentNode->getType() == Convolution || parentNode->getType() == BinaryConvolution) && getAlgorithm() == EltwiseAdd &&
            getInputShapeAtPort(0) == getInputShapeAtPort(1);
    if (!specialConvolutionAddFusing && canBePerformedAsScaleShift(parentNode.get())) {
        std::tie(scales, shifts) = getScalesAndShifts(parentNode.get());
        if ((parentNode->getType() == FullyConnected || parentNode->getType() == MatMul) && one_of(getAlgorithm(), EltwiseAdd, EltwiseSubtract,
                EltwiseMultiply, EltwiseDivide, EltwiseMulAdd, EltwisePowerStatic, EltwisePrelu)) {
            std::tie(scales, shifts) = getScalesAndShifts(parentNode.get());
        }
    }
    MKLDNNNode::fuseInto(parentNode);
}

void MKLDNNEltwiseNode::appendPostOps(mkldnn::post_ops& ops, const VectorDims &postOpDims) {
    const std::string errorPrefix = "Appending Eltwise node with name '" + getName() + "' ";

    if (getMKLDNNAlgorithm() != mkldnn::algorithm::undef) {
        switch (getMKLDNNAlgorithm()) {
        case mkldnn::algorithm::eltwise_relu:
        case mkldnn::algorithm::eltwise_tanh:
        case mkldnn::algorithm::eltwise_elu:
        case mkldnn::algorithm::eltwise_square:
        case mkldnn::algorithm::eltwise_abs:
        case mkldnn::algorithm::eltwise_sqrt:
        case mkldnn::algorithm::eltwise_linear:
        case mkldnn::algorithm::eltwise_bounded_relu:
        case mkldnn::algorithm::eltwise_soft_relu:
        case mkldnn::algorithm::eltwise_logistic:
        case mkldnn::algorithm::eltwise_exp:
        case mkldnn::algorithm::eltwise_gelu_erf:
        case mkldnn::algorithm::eltwise_gelu_tanh:
        case mkldnn::algorithm::eltwise_clip:
        case mkldnn::algorithm::eltwise_swish:
        case mkldnn::algorithm::eltwise_hardswish:
        case mkldnn::algorithm::eltwise_mish:
        case mkldnn::algorithm::eltwise_hsigmoid:
        case mkldnn::algorithm::eltwise_round_half_to_even:
        case mkldnn::algorithm::eltwise_round_half_away_from_zero:
            ops.append_eltwise(1.0, getMKLDNNAlgorithm(), getAlpha(), getBeta());
            break;
        default: IE_THROW() << errorPrefix << "as post operation is not supported";
        }
    } else {
        const size_t chIdx = postOpDims.size() > 1 ? getFusingAxis() : 0;
        constexpr int align = 16; // always align for legacy scale/shift post ops
        scalesBuffer = makeAlignedBuffer(postOpDims[chIdx], scales, align);
        if (getAlgorithm() != EltwisePrelu) {
            shiftsBuffer = makeAlignedBuffer(postOpDims[chIdx], shifts, align);
        }
        /* @todo legacy depthwise post ops are kept for now
         * for performance reasons
         */
        switch (getAlgorithm()) {
        case EltwiseAdd:
        case EltwiseSubtract:
        case EltwiseMultiply:
        case EltwiseDivide:
        case EltwiseMulAdd:
        case EltwisePowerStatic:
            if (scales.empty() || shifts.empty())
                IE_THROW() << errorPrefix << "cannot be performed since buffers are not allocated";
            ops.append_depthwise(mkldnn::algorithm::depthwise_scale_shift, &scalesBuffer[0], &shiftsBuffer[0]);
            break;
        case EltwisePrelu:
            if (scales.empty())
                IE_THROW() << errorPrefix << "cannot be performed since buffers are not allocated";
            ops.append_depthwise(mkldnn::algorithm::depthwise_prelu, &scalesBuffer[0], nullptr);
            break;
        default:
            IE_THROW() << errorPrefix << "as post operation is not supported";
        }
    }
}

void MKLDNNEltwiseNode::appendBinPostOps(mkldnn::post_ops& ops, const VectorDims& postOpDims, std::vector<MKLDNNMemoryPtr>& binaryPostOpsMem) {
    const std::string errorPrefix = "Appending Eltwise node with name '" + getName() + "' as binary post op ";
    VectorDims broadcastBinaryShape(postOpDims.size(), 1);

    auto appendBinary = [&](const mkldnn::algorithm alg, MKLDNNMemoryPtr &memPtr, const std::vector<float> &data) {
        if (data.empty())
            IE_THROW() << errorPrefix << "cannot be performed since buffers are not allocated";
        if (broadcastingPolicy == Undefined)
            IE_THROW() << errorPrefix << "cannot be performed since policy is Undefined";

        DnnlBlockedMemoryDesc memoryDesc(Precision::FP32, broadcastingPolicy == PerTensor ? Shape(broadcastBinaryShape) : Shape(postOpDims));

        ops.append_binary(alg, memoryDesc.getDnnlDesc());

        if (!memPtr) {
            memPtr.reset(new MKLDNNMemory(getEngine()));
            memPtr->Create(memoryDesc, &data[0]);

            binaryPostOpsMem.push_back(memPtr);
        }
    };

    switch (getAlgorithm()) {
    case EltwiseAdd:
    case EltwiseSubtract:
        appendBinary(mkldnn::algorithm::binary_add, shiftsMemory, shifts);
        break;
    case EltwiseDivide:
    case EltwiseMultiply:
        appendBinary(mkldnn::algorithm::binary_mul, scalesMemory, scales);
        break;
    case EltwiseMulAdd:
        appendBinary(mkldnn::algorithm::binary_mul, scalesMemory, scales);
        appendBinary(mkldnn::algorithm::binary_add, shiftsMemory, shifts);
        break;
    case EltwisePowerStatic:
        if (beta != 1.0f) // Multiply if has scales
            appendBinary(mkldnn::algorithm::binary_mul, scalesMemory, scales);
        if (gamma != 0.0f) // Add only if has shifts
            appendBinary(mkldnn::algorithm::binary_add, shiftsMemory, shifts);
        break;
    case EltwisePrelu:
        appendBinary(mkldnn::algorithm::binary_prelu, scalesMemory, scales);
        break;
    default:
        IE_THROW() << errorPrefix << "as post operation is not supported";
    }
}

bool MKLDNNEltwiseNode::canFuse(const MKLDNNNodePtr& node) const {
    auto isSuitableNode = [this](const MKLDNNEltwiseNode* node) {
        // [WA] Since execution precision change from I32 to FP32 for Divide operation may lead to incorrect results
        // we disable its fusing otherwise there is no guarantee it will be executed it I32
        // [TODO] We need to rewrite support for different precisions at all to avoid implicit conversions to FP32
        // (all should be handled via explicit convert operations)
        if (node->getAlgorithm() == EltwiseDivide) {
            for (const auto &originalInputPrecision : getOriginalInputPrecisions()) {
                if (originalInputPrecision == Precision::I32) {
                    return false;
                }
            }
        }

        return true;
    };

    if (!mayiuse(x64::sse41) || getInputShapeAtPort(0).getRank() > MAX_ELTWISE_DIM_RANK)
        return false;

    if (!isSuitableNode(this)) {
        return false;
    }

    // FQ inputs with quantization parameters will be hided inside post_op object, so will not increase inputs number
    size_t addedInputEdgesNum = node->getType() != FakeQuantize ? (node->getParentEdges().size() - 1) : 0;
    if (getParentEdges().size() + addedInputEdgesNum > MAX_ELTWISE_INPUTS)
        return false;

    if (node->getType() == Eltwise) {
        if (node->getParentEdgesAtPort(0)[0]->getParent().get() != this) {
            // Eltwise jitter doesn't respect commutative property, so fusing is disabled in case it applied not for 0-th port.
            if (one_of(node->getAlgorithm(), EltwiseSubtract, EltwiseDivide, EltwiseFloorMod, EltwiseMod, EltwisePowerDynamic, EltwiseGreater,
                                             EltwiseGreaterEqual, EltwiseLess, EltwiseLessEqual, EltwiseMulAdd)) {
                return false;
            }

            // Limitation: inputs precision definition inside Eltwise node assumes fusing is applied for 0-th port,
            // otherwise we need identical precision on all inputs of fused node
            for (int i = 1; i < getOriginalInputsNumber(); i++) {
                if (getOriginalInputPrecisionAtPort(0) != getOriginalInputPrecisionAtPort(i)) {
                    return false;
                }
            }
        }

        // We can use optimized execution with fusions only in cases when dim rank is less or equal to the maximum possible
        if (node->getInputShapeAtPort(0).getRank() > MAX_ELTWISE_DIM_RANK)
            return false;

        return true;
    }

    if (node->getType() == FakeQuantize) {
        return node->getAlgorithm() != FQBinarization;
    }

    return false;
}

InferenceEngine::Precision MKLDNNEltwiseNode::getRuntimePrecision() const {
    std::vector<InferenceEngine::Precision> inputPrecisions;
    // Don't take bias precision into account
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (parentEdge && parentEdge->getStatus() == MKLDNNEdge::Status::Validated && !parentEdge->getParent()->isConstant()) {
            inputPrecisions.emplace_back(MKLDNNExtensionUtils::DataTypeToIEPrecision((parentEdge->getMemoryPtr()->GetDataType())));
        }
    }

    return getMaxPrecision(inputPrecisions);
}

REG_MKLDNN_PRIM_FOR(MKLDNNEltwiseNode, Eltwise);
