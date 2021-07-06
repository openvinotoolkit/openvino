// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_eltwise_node.h"

#include <ie_parallel.hpp>

#include <mkldnn_types.h>
#include "utils/bfloat16.hpp"
#include <cpu/x64/jit_uni_quantization_injector.hpp>
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

#include "ngraph/ngraph.hpp"
#include <ngraph/opsets/opset1.hpp>
#include "ngraph_transformations/op/power_static.hpp"
#include "ngraph_transformations/op/leaky_relu.hpp"
#include "ngraph_transformations/op/swish_cpu.hpp"
#include <cpu_memory_desc_utils.h>

#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <map>
#include <functional>

using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::cpu;
using namespace mkldnn::impl::cpu::x64;
using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_eltwise_call_args, field)

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
    const MKLDNNNode *node;
    InferenceEngine::Precision exec_prc;
};

template<typename T>
struct EltwiseEmitter {
    void operator()(EltwiseEmitterContext & ctx) {
        ctx.emitter = std::make_shared<T>(ctx.host, ctx.host_isa, ctx.node, ctx.exec_prc);
    }
};

}   // namespace

template <cpu_isa_t isa>
struct jit_uni_eltwise_generic : public MKLDNNPlugin::jit_uni_eltwise_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_eltwise_generic)

    explicit jit_uni_eltwise_generic(jit_eltwise_params jep, MKLDNNEltwiseNode& eltwiseNode) : jit_uni_eltwise_kernel(jep, eltwiseNode), jit_generator() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        Precision exec_prc = Precision::UNSPECIFIED;

        std::set<Precision> supported_precision_intersection = get_supported_precisions(eltwiseNode);
        for (int i = 0; i < eltwiseNode.getFusedWith().size(); i++) {
            if (eltwiseNode.getFusedWith()[i].get()->getType() == Eltwise) {
                std::set<Precision> prcs = get_supported_precisions(*eltwiseNode.getFusedWith()[i].get());
                std::set<Precision> prcs_intersect = {};

                std::set_intersection(supported_precision_intersection.begin(), supported_precision_intersection.end(),
                                      prcs.begin(), prcs.end(), std::inserter(prcs_intersect, prcs_intersect.begin()));

                supported_precision_intersection = prcs_intersect;
            }
        }

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
            IE_THROW() << "Eltwise jitter failed to specify execution precision for Eltwise node with name `" << eltwiseNode.getName() << "`";
        }

        eltwise_emitter = create_eltwise_emitter(eltwiseNode, exec_prc);

        mkldnn::post_ops post_ops;
        for (int i = 0; i < eltwiseNode.getFusedWith().size(); i++) {
            if (eltwiseNode.getFusedWith()[i].get()->getType() == Eltwise) {
                post_op_emitters.push_back(create_eltwise_emitter(*eltwiseNode.getFusedWith()[i].get(), exec_prc));
            } else if (eltwiseNode.getFusedWith()[i].get()->getType() == FakeQuantize) {
                IE_THROW() << "[DS] Unimplemented";
//                auto fakeQuantizeNode = dynamic_cast<MKLDNNFakeQuantizeNode*>(eltwiseNode.getFusedWith()[i].get());
//                fakeQuantizeNode->appendPostOps(post_ops);
//
//                quantization_injectors.push_back(std::make_shared<jit_uni_quantization_injector_f32<isa>>(
//                        this, post_ops.get()->entry_[post_ops.len() - 1], vmm_d_weights, vmm_d_bias, reg_d_weights, reg_d_bias));
            }
        }

        if (!mayiuse(avx512_core_bf16) && mayiuse(avx512_core))
            emu_vcvtneps2bf16.reset(new jit_emu_vcvtneps2bf16(this, isa, nullptr));

        const auto &jep = jep_;

        this->preamble();

        for (int i = 0; i < jep.inputs_number; i++)
            mov(get_src_reg(i), ptr[reg_params + GET_OFF(src_ptr[0]) + i * sizeof(size_t)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);
        mov(reg_oc_off, ptr[reg_params + GET_OFF(oc_off)]);

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
                IE_THROW() << "Eltwise jitter has invalid configuration for Eltwise node with name `" << eltwiseNode.getName() << "`";

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

    Reg64 reg_dst = rbx;
    Reg64 reg_work_amount = rdx;

    Reg64 reg_oc_off = abi_not_param1;
    Reg64 reg_params = abi_param1;

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

    std::unique_ptr<jit_emu_vcvtneps2bf16> emu_vcvtneps2bf16;

    std::shared_ptr<jit_emitter> eltwise_emitter = nullptr;
    std::vector<std::shared_ptr<jit_emitter>> post_op_emitters = {};

    std::vector<std::shared_ptr<jit_uni_quantization_injector_f32<isa>>> quantization_injectors = {};

    std::vector<Precision> exec_precisions_priority = {
        Precision::U8,
        Precision::I8,
        Precision::U16,
        Precision::I16,
        Precision::BF16,
        Precision::I32,
        Precision::FP32
    };

    std::set<Precision> get_supported_precisions(MKLDNNNode& node) {
        std::set<Precision> precisions;

        OV_SWITCH(MKLDNNPlugin, SupportedPrecisions, precisions, node.getAlgorithm(),
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

    std::shared_ptr<jit_emitter> create_eltwise_emitter(MKLDNNNode& node, Precision exec_prec) {
        const auto& eltwiseNode = dynamic_cast<const MKLDNNEltwiseNode&>(node);

        EltwiseEmitterContext ctx = {
            nullptr,
            this,
            isa,
            &node,
            exec_prec
        };

        OV_SWITCH(MKLDNNPlugin, EltwiseEmitter, ctx, eltwiseNode.getAlgorithm(),
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
        for (int i = 0; i < eltwiseNode.getFusedWith().size(); i++) {
            if (eltwiseNode.getFusedWith()[i].get()->getType() == Eltwise) {
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
            } else {
                bool do_dequantization = eltwiseNode.getFusedWith()[i]->getAlgorithm() == FQCommon;
                bool do_rounding = do_dequantization || jep_.dst_prc == Precision::FP32 || i != eltwiseNode.getFusedWith().size() - 1;
                int s_idx = vmm_dst.getIdx();

                quantization_injectors[quantization_post_op_idx]->init_crop_ptrs(reg_oc_off);
                quantization_injectors[quantization_post_op_idx]->compute_crop(s_idx, s_idx + 1, offset, is_scalar, jep_.oc_size == 1);

                quantization_injectors[quantization_post_op_idx]->init_input_scale_shift_ptrs(reg_oc_off);
                quantization_injectors[quantization_post_op_idx]->compute_input_scale_shift(s_idx, s_idx + 1, offset, do_rounding,
                                                                                            is_scalar, jep_.oc_size == 1);

                quantization_injectors[quantization_post_op_idx]->init_output_scale_shift_ptrs(reg_oc_off);
                quantization_injectors[quantization_post_op_idx]->compute_output_scale_shift(s_idx, s_idx + 1, offset, is_scalar, jep_.oc_size == 1);

                quantization_post_op_idx++;
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
                movss(xmm_src, op);
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
                movq(xmm_src, reg_tmp_64);
                break;
            case Precision::U8:
                movzx(reg_tmp_32, op);
                movq(xmm_src, reg_tmp_64);
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
                movss(op, xmm_dst);
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

std::map<const ngraph::DiscreteTypeInfo, std::function<void(const std::shared_ptr<ngraph::Node>&, MKLDNNEltwiseNode& node)>> MKLDNNEltwiseNode::initializers = {
    {ngraph::op::v1::Add::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseAdd;
    }},
    {ngraph::op::v1::Subtract::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseSubtract;
    }},
    {ngraph::op::v1::Multiply::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseMultiply;
    }},
    {ngraph::op::v1::Divide::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseDivide;
    }},
    {ngraph::op::v0::SquaredDifference::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseSquaredDifference;
    }},
    {ngraph::op::v1::Maximum::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseMaximum;
    }},
    {ngraph::op::v1::Minimum::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseMinimum;
    }},
    {ngraph::op::v1::Mod::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseMod;
    }},
    {ngraph::op::v1::FloorMod::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseFloorMod;
    }},
    {ngraph::op::v1::Power::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwisePowerDynamic;
    }},
    {PowerStaticNode::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        auto powerStatic = getNgraphOpAs<PowerStaticNode>(op);
        node.algorithm = EltwisePowerStatic;
        node.alpha = powerStatic->get_power();
        node.beta = powerStatic->get_scale();
        node.gamma = powerStatic->get_shift();
    }},
    {ngraph::op::v1::Equal::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseEqual;
    }},
    {ngraph::op::v1::NotEqual::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseNotEqual;
    }},
    {ngraph::op::v1::Greater::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseGreater;
    }},
    {ngraph::op::v1::GreaterEqual::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseGreaterEqual;
    }},
    {ngraph::op::v1::Less::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseLess;
    }},
    {ngraph::op::v1::LessEqual::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseLessEqual;
    }},
    {ngraph::op::v1::LogicalAnd::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseLogicalAnd;
    }},
    {ngraph::op::v1::LogicalOr::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseLogicalOr;
    }},
    {ngraph::op::v1::LogicalXor::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseLogicalXor;
    }},
    {ngraph::op::v1::LogicalNot::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseLogicalNot;
    }},
    {ngraph::op::v0::Relu::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseRelu;
        node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_relu;
    }},
    {LeakyReluNode::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        auto leakyRelu = getNgraphOpAs<LeakyReluNode>(op);
        node.algorithm = EltwiseRelu;
        node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_relu;
        node.alpha = leakyRelu->get_slope();
        node.beta = 0.0f;
    }},
    {ngraph::op::v0::Gelu::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseGelu;
        node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_gelu_erf;
    }},
    {ngraph::op::v7::Gelu::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
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
    {ngraph::op::v0::Elu::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        auto eluOp = getNgraphOpAs<ngraph::op::v0::Elu>(op);

        node.alpha = static_cast<float>(eluOp->get_alpha());
        node.algorithm = EltwiseElu;
        node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_elu;
    }},
    {ngraph::op::v0::Tanh::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseTanh;
        node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_tanh;
    }},
    {ngraph::op::v0::Sigmoid::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseSigmoid;
        node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_logistic;
    }},
    {ngraph::op::v0::Abs::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseAbs;
        node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_abs;
    }},
    {ngraph::op::v0::Sqrt::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseSqrt;
        node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_sqrt;
    }},
    {ngraph::op::v0::Clamp::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        auto clampOp = getNgraphOpAs<ngraph::op::v0::Clamp>(op);

        node.alpha = static_cast<float>(clampOp->get_min());
        node.beta = static_cast<float>(clampOp->get_max());
        node.algorithm = EltwiseClamp;
        node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_clip;
    }},
    {ngraph::op::v0::Exp::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseExp;
        node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_exp;
    }},
    {SwishNode::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        auto swishOp = getNgraphOpAs<SwishNode>(op);
        node.algorithm = EltwiseSwish;
        node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_swish;
        node.alpha = swishOp->get_alpha();
    }},
    {ngraph::op::v4::HSwish::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseHswish;
        node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_hswish;
    }},
    {ngraph::op::v4::Mish::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseMish;
        node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_mish;
    }},
    {ngraph::op::v5::HSigmoid::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseHsigmoid;
        node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_hsigmoid;
    }},
    {ngraph::op::v5::Round::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
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
    {ngraph::op::v0::PRelu::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwisePrelu;
    }},
    {ngraph::op::v0::Erf::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseErf;
    }},
    {ngraph::op::v4::SoftPlus::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNEltwiseNode& node) {
        node.algorithm = EltwiseSoftRelu;
        node.mkldnnAlgorithm = mkldnn::algorithm::eltwise_soft_relu;
    }},
};

MKLDNNEltwiseNode::MKLDNNEltwiseNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(op, eng, cache) {
    if (initializers.find(op->get_type_info()) != initializers.end()) {
        initializers[op->get_type_info()](op, *this);
    } else {
        IE_THROW(NotImplemented)
            << "CPU Eltwise node doesn't support ngraph operation " << op->get_type_name() << " with name " << op->get_friendly_name();
    }
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

bool MKLDNNEltwiseNode::isWithBroadcast() {
    auto oDims = outputShapes[0].getStaticDims();
    for (size_t i = 0; i < inputShapes.size(); i++) {
        auto iDims = inputShapes[i].getStaticDims();
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

    canUseOptimizedImpl = mayiuse(x64::sse41);

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
    for (const auto &i : getOriginalInputPrecisions()) {
        inputPrecisions.push_back(i);
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
        auto createMemoryDesc = [lt](MKLDNNEdgePtr edge, Precision prc, size_t offset) -> std::unique_ptr<BlockedMemoryDesc> {
            if (lt == ChannelsFirst && edge->getShape().getRank() != 1) {
                auto dims = edge->getShape().getStaticDims();
                auto ndims = dims.size();
                std::vector<size_t> order(ndims);
                std::iota(order.begin(), order.end(), 0);
                if (ndims > 1) {
                    order.erase(order.begin() + 1);
                    order.push_back(1);
                }

                std::vector<size_t> blocks(ndims);
                for (size_t i = 0; i < order.size(); i++) {
                    blocks[i] = dims[order[i]];
                }

                return make_unique<BlockedMemoryDesc>(prc, edge->getShape().getStaticDims(), blocks, order, offset);
            } else if (lt == Blocked && edge->getShape().getRank() != 1 && edge->getShape().getStaticDims()[1] != 1) {
                size_t blockSize = mayiuse(x64::avx512_common) ? 16 : 8;

                std::vector<size_t> blocks = edge->getShape().getStaticDims();
                std::vector<size_t> order(blocks.size());
                std::iota(order.begin(), order.end(), 0);

                blocks[1] = div_up(blocks[1], blockSize);
                blocks.push_back(blockSize);
                order.push_back(1);

                return make_unique<BlockedMemoryDesc>(prc, edge->getShape().getStaticDims(), blocks, order, offset);
            } else {
                std::vector<size_t> blocks = edge->getShape().getStaticDims();
                std::vector<size_t> order(blocks.size());
                std::iota(order.begin(), order.end(), 0);

                return make_unique<BlockedMemoryDesc>(prc, edge->getShape().getStaticDims(), blocks, order, offset);
            }
        };

        size_t offset = std::numeric_limits<size_t>::max();
        NodeConfig config;
        config.dynBatchSupport = getChildEdgeAt(0)->getShape().getRank() > 1 && getChildEdgeAt(0)->getShape() ==
                                                                                getParentEdgeAt(0)->getShape();

        for (size_t i = 0; i < getParentEdges().size(); i++) {
            PortConfig portConfig;
            portConfig.inPlace = (!i && canBeInPlace() && inputPrecisions[i] == outputPrecision) ? 0 : -1;
            portConfig.constant = false;

            portConfig.desc = createMemoryDesc(getParentEdgeAt(i), inputPrecisions[i], offset);

            config.inConfs.push_back(portConfig);
        }

        PortConfig portConfig;
        portConfig.inPlace = -1;
        portConfig.constant = false;

        portConfig.desc = createMemoryDesc(getChildEdgeAt(0), outputPrecision, offset);

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

    bool isChannelsFirstApplicable = one_of(getChildEdgeAt(0)->getShape().getRank(), 1, 2, 4, 5);
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        isChannelsFirstApplicable = isChannelsFirstApplicable && one_of(getParentEdgeAt(i)->getShape().getRank(), 1, 2, 4, 5);
        isChannelsFirstApplicable = isChannelsFirstApplicable && implication(getParentEdgeAt(i)->getShape().getRank() != 1,
                                                                             getChildEdgeAt(0)->getShape().getRank() ==
                                                                                     getParentEdgeAt(i)->getShape().getRank());
    }

    bool isBlockedApplicable = one_of(getChildEdgeAt(0)->getShape().getRank(), 1, 4, 5);
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        isBlockedApplicable = isBlockedApplicable && one_of(getParentEdgeAt(i)->getShape().getRank(), 1, 4, 5);
        isBlockedApplicable = isBlockedApplicable && implication(getParentEdgeAt(i)->getShape().getRank() != 1,
                                                                 getChildEdgeAt(0)->getShape().getRank() ==
                                                                 getParentEdgeAt(i)->getShape().getRank());
    }

    if (isChannelsFirstApplicable)
        supportedPrimitiveDescriptors.emplace_back(initDesc(ChannelsFirst));
    if (isBlockedApplicable)
        supportedPrimitiveDescriptors.emplace_back(initDesc(Blocked));
    supportedPrimitiveDescriptors.emplace_back(initDesc(Planar));
}

void MKLDNNEltwiseNode::createPrimitive() {
    auto config = getSelectedPrimitiveDescriptor()->getConfig();

    auto initDims = [this, config](size_t maxInputSize) {
        size_t inputNum = getParentEdges().size();

        dims_in.resize(inputNum);
        for (int i = 0; i < inputNum; i++) {
            dims_in[i].resize(maxInputSize, 1);
        }

        dims_out.resize(maxInputSize, 1);

        auto outBlockingDesc = MemoryDescUtils::convertToBlockedDescriptor(*config.outConfs[0].desc);
        std::vector<size_t> order(maxInputSize);
        auto outOrder = outBlockingDesc.getOrder();
        for (size_t i = 0; i < order.size(); i++) {
            if (i < order.size() - outOrder.size())
                order[i] = i;
            else
                order[i] = outOrder[i - (order.size() - outOrder.size())] + (order.size() - outOrder.size());
        }

        size_t outRank = outBlockingDesc.getBlockDims().size();
        for (int i = 0; i < outRank; i++) {
            dims_out[dims_out.size() - 1 - i] = outBlockingDesc.getBlockDims()[outRank - 1 - i];
        }

        for (int i = 0; i < inputNum; i++) {
            auto inBlockingDesc = MemoryDescUtils::convertToBlockedDescriptor(*config.inConfs[i].desc);
            size_t inRank = inBlockingDesc.getBlockDims().size();

            // WA to normalize blocked and planar layouts
            auto inOrder = inBlockingDesc.getOrder();
            size_t startOff = outOrder.size() != outBlockingDesc.getShape().getRank() &&
                              outOrder[outOrder.size() - 1] != inOrder[inOrder.size() - 1] ? 1 : 0;

            // WA to handle nspc layout with 1D tensors
            if (1 == inRank) {
                if (outRank > 2 && 1 == outOrder.back()) startOff = 1;
            }

            for (int j = 0; j < inRank; j++) {
                dims_in[i][dims_in[i].size() - 1 - j - startOff] = inBlockingDesc.getBlockDims()[inRank - 1 - j];
            }
        }

        for (int i = 0; i < dims_in.size(); i++) {
            for (int j = 0; j < dims_in[i].size(); j++) {
                if (dims_in[i][j] != dims_out[j] && dims_in[i][j] != 1)
                    IE_THROW() << "Eltwise node with name `" << getName() << "` has invalid input/output dims configuration.";
            }
        }
    };

    auto initOffsets = [this, config](size_t maxInputSize) {
        size_t inputNum = getParentEdges().size();

        offsets_out.resize(maxInputSize, 1);
        offset_out_calc(offsets_out, dims_out);
        for (int j = 0; j < maxInputSize; j++) {
            offsets_out[j] *= config.outConfs[0].desc->getPrecision().size();
        }

        offsets_in.resize(inputNum);
        for (int i = 0; i < inputNum; i++) {
            offsets_in[i].resize(maxInputSize, 1);
            offset_in_calc(offsets_in[i], dims_in[i], dims_out);
            for (int j = 0; j < maxInputSize; j++) {
                offsets_in[i][j] *= config.inConfs[i].desc->getPrecision().size();
            }
        }

        start_offset_in.resize(inputNum);
        for (size_t i = 0; i < inputNum; i++) {
            start_offset_in[i] = getParentEdgeAt(i)->getMemory().GetDescriptor().data.offset0 *
                               MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(getParentEdgeAt(i)->getMemory().GetDescriptor().data.data_type));
        }
        start_offset_out = getChildEdgeAt(0)->getMemory().GetDescriptor().data.offset0 *
                         MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(getChildEdgeAt(0)->getMemory().GetDescriptor().data.data_type));
    };

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

    auto outBlockingDesc = MemoryDescUtils::convertToBlockedDescriptor(*config.outConfs[0].desc);
    tensorRank = std::max(static_cast<size_t>(optimalTensorRank), outBlockingDesc.getBlockDims().size());
    initDims(tensorRank);

    auto outOrder = outBlockingDesc.getOrder();
    size_t oc_size = 0;
    offsets_oc.resize(tensorRank, 0);
    if (isFusedWith(FakeQuantize)) {
        size_t offset_oc = 1;
        for (int i = outOrder.size() - 1; i >= 0; i--) {
            if (outOrder[i] == 1) {
                int oc_dim_idx = i + (tensorRank - outOrder.size());
                offsets_oc[oc_dim_idx] = offset_oc;
                offset_oc *= dims_out[oc_dim_idx];
            }
        }
        oc_size = offsets_oc[dims_out.size() - 1] != 0 ? dims_out[dims_out.size() - 1] : 1;
    }

    fullWorkAmount = 1;
    for (int i = 0; i < dims_out.size(); i++) {
        fullWorkAmount *= dims_out[i];
    }

    isDynBatchEnabled = config.dynBatchSupport;

    size_t minimalConcurrency = parallel_get_max_threads();
    size_t minimalJitWorkAmount = 256;
    size_t currentJitWorkAmount = dims_out[dims_out.size() - 1];
    int collapsedDims = 0;
    if (canUseOptimizedImpl) {
        bool hasDifferentDims = false;
        while (currentJitWorkAmount < minimalJitWorkAmount && currentJitWorkAmount < fullWorkAmount &&
               // we shouldn't collapse batch dimension in case dynamic batch is enabled
               (!isDynBatchEnabled || (outBlockingDesc.getBlockDims().size() - collapsedDims > 2))) {
            if (dims_out.size() - collapsedDims - 2 < 0)
                break;

            for (int j = 1; j < dims_in.size(); j++) {
                if (dims_in[j][dims_in[j].size() - 1] != dims_in[0][dims_in[0].size() - 1]) {
                    hasDifferentDims = true;
                }
            }

            if (oc_size > 1 && oc_size != dims_in[0][dims_in[0].size() - 1]) {
                hasDifferentDims = true;
            }

            bool canCollapse = true;
            for (int i = 0; i < dims_in.size(); i++) {
                if (dims_in[i][dims_in[i].size() - 2] != 1) {
                    if (dims_in[i][dims_in[i].size() - 1] == 1) {
                        canCollapse = false;
                        break;
                    }

                    if (hasDifferentDims) {
                        canCollapse = false;
                        break;
                    }
                }
            }

            if (!canCollapse) {
                break;
            }

            size_t nextJitWorkAmount = currentJitWorkAmount * dims_out[dims_out.size() - 2];
            if (fullWorkAmount / nextJitWorkAmount >= minimalConcurrency) {
                currentJitWorkAmount = nextJitWorkAmount;
                collapsedDims++;

                for (int i = 0; i < dims_in.size(); i++) {
                    collapseLastDims(dims_in[i], 1);
                }
                collapseLastDims(dims_out, 1);

                if (isFusedWith(FakeQuantize)) {
                    collapseLastOffsets(offsets_oc, 1);
                }
            } else {
                break;
            }
        }
    }

    batchDimIdx = tensorRank - outBlockingDesc.getBlockDims().size() + collapsedDims;
    schedulerWorkAmount = fullWorkAmount / dims_out[dims_out.size() - 1];

    initOffsets(tensorRank);

    jep.inputs_number = config.inConfs.size();
    jep.input_size = tensorRank;

    for (int i = 0; i < config.inConfs.size(); i++) {
        jep.src_size[i] = dims_in[i][dims_in[i].size() - 1];
        jep.src_prc[i] = config.inConfs[i].desc->getPrecision();
    }
    jep.dst_size = dims_out[dims_out.size() - 1];
    jep.dst_prc = config.outConfs[0].desc->getPrecision();

    for (int i = 0; i < config.inConfs.size(); i++) {
        jep.src_offsets[i] = offsets_in[i];
    }
    jep.dst_offsets = offsets_out;

    jep.oc_size = oc_size;

    if (mayiuse(x64::avx512_common)) {
        eltwise_kernel.reset(new jit_uni_eltwise_generic<x64::avx512_common>(jep, *this));
    } else if (mayiuse(x64::avx2)) {
        eltwise_kernel.reset(new jit_uni_eltwise_generic<x64::avx2>(jep, *this));
    } else if (mayiuse(x64::sse41)) {
        eltwise_kernel.reset(new jit_uni_eltwise_generic<x64::sse41>(jep, *this));
    }

    if (eltwise_kernel)
        eltwise_kernel->create_ker();
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
            config.inConfs[i].desc = std::move(getDefinedInputDesc(config, i));
        }

        for (size_t i = 0; i < config.outConfs.size(); i++) {
            config.outConfs[i].desc = std::move(getDefinedOutputDesc(config, i));
        }

        initDescriptor(config);
    } else {
        initDescriptor(config);
    }
}

void MKLDNNEltwiseNode::offset_out_calc(std::vector<size_t>& offset, std::vector<size_t>& dims) {
    int k = 1;
    for (int i = offset.size() - 1; i >= 0; i--) {
        offset[i] = k;
        k *= dims[i];
    }
}

void MKLDNNEltwiseNode::offset_in_calc(std::vector<size_t>& offset, std::vector<size_t>& dims_in, std::vector<size_t>& dims_out) {
    int k = 1;
    for (int i = offset.size() - 1; i >= 0; i--) {
        offset[i] = (dims_in[i] == dims_out[i]) ? k : 0;
        k *= dims_in[i];
    }
}

void MKLDNNEltwiseNode::executeOptimized6D(const std::vector<const uint8_t *>& src_ptrs, uint8_t *dst_ptr) {
    size_t inputNum = src_ptrs.size();

    parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4],
        [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
            // TODO: reimplement initializer via jit approach
            size_t index_in[MAX_ELTWISE_INPUTS] = {0};
            for (int i = 0; i < inputNum; i++) {
                index_in[i] = i0 * offsets_in[i][0] + i1 * offsets_in[i][1] + i2 * offsets_in[i][2] +
                              i3 * offsets_in[i][3] + i4 * offsets_in[i][4];
            }
            size_t index_out = i0 * offsets_out[0] + i1 * offsets_out[1] + i2 * offsets_out[2] +
                               i3 * offsets_out[3] + i4 * offsets_out[4];

            auto arg = jit_eltwise_call_args();
            for (int i = 0; i < inputNum; i++) {
                arg.src_ptr[i] = src_ptrs[i] + index_in[i];
            }
            arg.dst = dst_ptr + index_out;
            arg.work_amount = static_cast<size_t>(dims_out[dims_out.size() - 1]);
            arg.oc_off = (i0 * offsets_oc[0] + i1 * offsets_oc[1] + i2 * offsets_oc[2] +
                          i3 * offsets_oc[3] + i4 * offsets_oc[4]) * sizeof(float);

            (*eltwise_kernel)(&arg);
        });
}

void MKLDNNEltwiseNode::executeOptimizedGeneric(const std::vector<const uint8_t *>& src_ptrs, uint8_t *dst_ptr) {
    size_t inputNum = src_ptrs.size();

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        splitter(schedulerWorkAmount, nthr, ithr, start, end);

        std::vector<size_t> counters(dims_out.size() - 1, 0);

        for (size_t iwork = start; iwork < end; ++iwork) {
            size_t tmp = iwork;
            for (ptrdiff_t j = dims_out.size() - 2; j >= 0; j--) {
                counters[j] = tmp % dims_out[j];
                tmp /= dims_out[j];
            }

            size_t index_in[MAX_ELTWISE_INPUTS] = {0};
            for (int i = 0; i < inputNum; i++) {
                index_in[i] = 0;
                for (int j = 0; j < counters.size(); j++) {
                    index_in[i] += counters[j] * offsets_in[i][j];
                }
            }

            size_t index_out = 0;
            for (int j = 0; j < counters.size(); j++) {
                index_out += counters[j] * offsets_out[j];
            }

            auto arg = jit_eltwise_call_args();
            for (int i = 0; i < inputNum; i++) {
                arg.src_ptr[i] = src_ptrs[i] + index_in[i];
            }
            arg.dst = dst_ptr + index_out;
            arg.work_amount = static_cast<size_t>(dims_out[dims_out.size() - 1]);

            arg.oc_off = 0;
            for (int j = 0; j < counters.size(); j++) {
                arg.oc_off += counters[j] * offsets_oc[j] * sizeof(float);
            }

            (*eltwise_kernel)(&arg);
        }
    });
}

void MKLDNNEltwiseNode::executeReference(const std::vector<const uint8_t *>& src_ptrs, uint8_t *dst_ptr) {
    size_t inputNum = src_ptrs.size();

    std::shared_ptr<ref_eltwise_scalar_fwd_t> ref_eltwise_injector = nullptr;
    if (getMKLDNNAlgorithm() != mkldnn::algorithm::undef) {
        ref_eltwise_injector = std::make_shared<ref_eltwise_scalar_fwd_t>(static_cast<mkldnn_alg_kind_t>(getMKLDNNAlgorithm()), alpha, beta, 1.f);
    }

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        splitter(fullWorkAmount, nthr, ithr, start, end);

        std::vector<size_t> counters(dims_out.size(), 0);

        for (size_t iwork = start; iwork < end; ++iwork) {
            size_t tmp = iwork;
            for (ptrdiff_t j = dims_out.size() - 1; j >= 0; j--) {
                counters[j] = tmp % dims_out[j];
                tmp /= dims_out[j];
            }

            size_t index_in[MAX_ELTWISE_INPUTS] = {0};
            for (int i = 0; i < inputNum; i++) {
                index_in[i] = 0;
                for (int j = 0; j < counters.size(); j++) {
                    index_in[i] += counters[j] * offsets_in[i][j];
                }
            }

            size_t index_out = 0;
            for (int j = 0; j < counters.size(); j++) {
                index_out += counters[j] * offsets_out[j];
            }

            std::vector<float> src_f(inputNum);
            for (int i = 0; i < inputNum; i++) {
                src_f[i] = reinterpret_cast<const float *>(src_ptrs[i] + index_in[i])[0];
            }
            float* dst_ptr_f = reinterpret_cast<float *>(dst_ptr + index_out);

            switch (getAlgorithm()) {
                case EltwiseRelu: case EltwiseGelu: case EltwiseElu: case EltwiseTanh: case EltwiseSigmoid: case EltwiseAbs:
                case EltwiseSqrt: case EltwiseSoftRelu: case EltwiseExp: case EltwiseClamp:
                case EltwiseSwish: case EltwiseHswish: case EltwiseMish: case EltwiseHsigmoid: case EltwiseRoundHalfToEven: case EltwiseRoundHalfAwayFromZero:
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
                case EltwisePowerStatic:       *dst_ptr_f = powf(beta * src_f[0] + gamma, alpha); break;
                case EltwisePrelu:             *dst_ptr_f = src_f[0] > 0 ? src_f[0] : src_f[0] * src_f[1]; break;
                default: IE_THROW() << "Unsupported operation type for Eltwise node with name `" << getName() << "`";
            }
        }
    });
}

void MKLDNNEltwiseNode::execute(mkldnn::stream strm) {
    size_t inputNum = getParentEdges().size();

    std::vector<const uint8_t *> src_ptrs(inputNum);
    for (int i = 0; i < inputNum; i++) {
        src_ptrs[i] = reinterpret_cast<const uint8_t*>(getParentEdgeAt(i)->getMemory().GetData()) + start_offset_in[i];
    }
    uint8_t *dst_ptr = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemory().GetData()) + start_offset_out;

    // In general case we need to recompute offsets as well but currently all supported layout assumes batch to be outermost dimension
    if (isDynBatchEnabled)
        dims_out[batchDimIdx] = static_cast<size_t>(batchToProcess());

    if (eltwise_kernel) {
        if (tensorRank == optimalTensorRank) {
            executeOptimized6D(src_ptrs, dst_ptr);
        } else {
            executeOptimizedGeneric(src_ptrs, dst_ptr);
        }
    } else {
        executeReference(src_ptrs, dst_ptr);
    }
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

    return getParentEdgesAtPort(0)[0].get()->getShape() == getChildEdgesAtPort(0)[0].get()->getShape();
}

void MKLDNNEltwiseNode::fuseInto(MKLDNNNodePtr& parentNode) {
    // Handling Convolution custom Add node fusing case which is processed via dnnl append_sum() API.
    specialConvolutionAddFusing = (parentNode->getType() == Convolution || parentNode->getType() == BinaryConvolution) && getAlgorithm() == EltwiseAdd &&
            getParentEdgesAtPort(0)[0]->getShape() == getParentEdgesAtPort(1)[0]->getShape();
    if (!specialConvolutionAddFusing && canBePerformedAsScaleShift(parentNode.get())) {
        fillScalesAndShifts(parentNode.get(), scales, shifts, 16);
    }
    MKLDNNNode::fuseInto(parentNode);
}

void MKLDNNEltwiseNode::appendPostOps(mkldnn::post_ops& ops) {
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
            case mkldnn::algorithm::eltwise_hswish:
            case mkldnn::algorithm::eltwise_mish:
            case mkldnn::algorithm::eltwise_hsigmoid:
            case mkldnn::algorithm::eltwise_round_half_to_even:
            case mkldnn::algorithm::eltwise_round_half_away_from_zero:
                ops.append_eltwise(1.0, getMKLDNNAlgorithm(), getAlpha(), getBeta());
                break;
            default: IE_THROW() << errorPrefix << "as post operation is not supported";
        }
    } else {
        switch (getAlgorithm()) {
            case EltwiseAdd:
            case EltwiseSubtract:
            case EltwiseMultiply:
            case EltwiseDivide:
            case EltwiseMulAdd:
            case EltwisePowerStatic:
                if (scales.empty() || shifts.empty())
                    IE_THROW() << errorPrefix << "cannot be performed since buffers are not allocated";
                ops.append_depthwise(mkldnn::algorithm::depthwise_scale_shift, &scales[0], &shifts[0]);
                break;
            case EltwisePrelu:
                if (scales.empty())
                    IE_THROW() << errorPrefix << "cannot be performed since buffers are not allocated";
                ops.append_depthwise(mkldnn::algorithm::depthwise_prelu, &scales[0], nullptr);
                break;
            default: IE_THROW() << errorPrefix << "as post operation is not supported";
        }
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

    if (!mayiuse(x64::sse41))
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

    return MKLDNNExtensionUtils::getMaxPrecision(inputPrecisions);
}

REG_MKLDNN_PRIM_FOR(MKLDNNEltwiseNode, Eltwise);
