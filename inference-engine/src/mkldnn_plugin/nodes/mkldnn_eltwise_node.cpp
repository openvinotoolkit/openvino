// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_eltwise_node.h"

#include <legacy/ie_layers.h>
#include <ie_parallel.hpp>

#include <mkldnn_types.h>
#include "utils/bfloat16.hpp"
#include <cpu/x64/jit_uni_quantization_injector.hpp>
#include <cpu/ref_eltwise.hpp>

#include "mkldnn_extension_utils.h"
#include "mkldnn_quantize_node.h"
#include "mkldnn_pooling_node.h"

#include "emitters/jit_emitter.hpp"
#include "emitters/jit_eltwise_emitters.hpp"
#include "emitters/jit_mkldnn_emitters.hpp"
#include "emitters/jit_bf16_emitters.hpp"
#include <mkldnn_selective_build.h>

#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <map>

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
            THROW_IE_EXCEPTION << "Eltwise jitter failed to specify execution precision for Eltwise node with name `" << eltwiseNode.getName() << "`";
        }

        eltwise_emitter = create_eltwise_emitter(eltwiseNode, exec_prc);

        mkldnn::post_ops post_ops;
        for (int i = 0; i < eltwiseNode.getFusedWith().size(); i++) {
            if (eltwiseNode.getFusedWith()[i].get()->getType() == Eltwise) {
                post_op_emitters.push_back(create_eltwise_emitter(*eltwiseNode.getFusedWith()[i].get(), exec_prc));
            } else if (eltwiseNode.getFusedWith()[i].get()->getType() == Quantize) {
                auto quantizeNode = dynamic_cast<MKLDNNQuantizeNode*>(eltwiseNode.getFusedWith()[i].get());
                quantizeNode->appendPostOps(post_ops);

                quantization_injectors.push_back(std::make_shared<jit_uni_quantization_injector_f32<isa>>(
                        this, post_ops.get()->entry_[post_ops.len() - 1], vmm_d_weights, vmm_d_bias, reg_d_weights, reg_d_bias));
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
                THROW_IE_EXCEPTION << "Eltwise jitter has invalid configuration for Eltwise node with name `" << eltwiseNode.getName() << "`";

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
        auto& eltwiseNode = dynamic_cast<const MKLDNNEltwiseNode&>(node);

        std::set<Precision> precisions;

        OV_SWITCH(MKLDNNPlugin, SupportedPrecisions, precisions, eltwiseNode.getOpType(),
        OV_CASE(Relu, jit_mkldnn_aux_emitter),
        OV_CASE(Gelu, jit_mkldnn_aux_emitter),
        OV_CASE(Elu, jit_mkldnn_aux_emitter),
        OV_CASE(Tanh, jit_mkldnn_aux_emitter),
        OV_CASE(Logistic, jit_mkldnn_aux_emitter),
        OV_CASE(Square, jit_mkldnn_aux_emitter),
        OV_CASE(Abs, jit_mkldnn_aux_emitter),
        OV_CASE(Sqrt, jit_mkldnn_aux_emitter),
        OV_CASE(Linear, jit_mkldnn_aux_emitter),
        OV_CASE(BoundedRelu, jit_mkldnn_aux_emitter),
        OV_CASE(SoftRelu, jit_mkldnn_aux_emitter),
        OV_CASE(Relu6, jit_mkldnn_aux_emitter),
        OV_CASE(Exp, jit_mkldnn_aux_emitter),
        OV_CASE(Clamp, jit_mkldnn_aux_emitter),
        OV_CASE(Swish, jit_mkldnn_aux_emitter),
        OV_CASE(Hswish, jit_mkldnn_aux_emitter),
        OV_CASE(Mish, jit_mkldnn_aux_emitter),
        OV_CASE(Hsigmoid, jit_mkldnn_aux_emitter),
        OV_CASE(Round, jit_mkldnn_aux_emitter),
        OV_CASE(Add, jit_add_emitter),
        OV_CASE(MulAdd, jit_mul_add_emitter),
        OV_CASE(Subtract, jit_subtract_emitter),
        OV_CASE(Multiply, jit_multiply_emitter),
        OV_CASE(Divide, jit_divide_emitter),
        OV_CASE(FloorMod, jit_floor_mod_emitter),
        OV_CASE(Mod, jit_mod_emitter),
        OV_CASE(Maximum, jit_maximum_emitter),
        OV_CASE(Minimum, jit_minimum_emitter),
        OV_CASE(SquaredDifference, jit_squared_difference_emitter),
        OV_CASE(PowerDynamic, jit_power_dynamic_emitter),
        OV_CASE(Equal, jit_equal_emitter),
        OV_CASE(NotEqual, jit_not_equal_emitter),
        OV_CASE(Greater, jit_greater_emitter),
        OV_CASE(GreaterEqual, jit_greater_equal_emitter),
        OV_CASE(Less, jit_less_emitter),
        OV_CASE(LessEqual, jit_less_equal_emitter),
        OV_CASE(LogicalAnd, jit_logical_and_emitter),
        OV_CASE(LogicalOr, jit_logical_or_emitter),
        OV_CASE(LogicalXor, jit_logical_xor_emitter),
        OV_CASE(LogicalNot, jit_logical_not_emitter),
        OV_CASE(PowerStatic, jit_power_static_emitter),
        OV_CASE(Prelu, jit_prelu_emitter));

        if (precisions.empty())
            THROW_IE_EXCEPTION << "Unsupported operation type for Eltwise emitter";

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

        OV_SWITCH(MKLDNNPlugin, EltwiseEmitter, ctx, eltwiseNode.getOpType(),
        OV_CASE(Relu, jit_mkldnn_aux_emitter),
        OV_CASE(Gelu, jit_mkldnn_aux_emitter),
        OV_CASE(Elu, jit_mkldnn_aux_emitter),
        OV_CASE(Tanh, jit_mkldnn_aux_emitter),
        OV_CASE(Logistic, jit_mkldnn_aux_emitter),
        OV_CASE(Square, jit_mkldnn_aux_emitter),
        OV_CASE(Abs, jit_mkldnn_aux_emitter),
        OV_CASE(Sqrt, jit_mkldnn_aux_emitter),
        OV_CASE(Linear, jit_mkldnn_aux_emitter),
        OV_CASE(BoundedRelu, jit_mkldnn_aux_emitter),
        OV_CASE(SoftRelu, jit_mkldnn_aux_emitter),
        OV_CASE(Relu6, jit_mkldnn_aux_emitter),
        OV_CASE(Exp, jit_mkldnn_aux_emitter),
        OV_CASE(Clamp, jit_mkldnn_aux_emitter),
        OV_CASE(Swish, jit_mkldnn_aux_emitter),
        OV_CASE(Hswish, jit_mkldnn_aux_emitter),
        OV_CASE(Mish, jit_mkldnn_aux_emitter),
        OV_CASE(Hsigmoid, jit_mkldnn_aux_emitter),
        OV_CASE(Round, jit_mkldnn_aux_emitter),
        OV_CASE(Add, jit_add_emitter),
        OV_CASE(MulAdd, jit_mul_add_emitter),
        OV_CASE(Subtract, jit_subtract_emitter),
        OV_CASE(Multiply, jit_multiply_emitter),
        OV_CASE(Divide, jit_divide_emitter),
        OV_CASE(FloorMod, jit_floor_mod_emitter),
        OV_CASE(Mod, jit_mod_emitter),
        OV_CASE(Maximum, jit_maximum_emitter),
        OV_CASE(Minimum, jit_minimum_emitter),
        OV_CASE(SquaredDifference, jit_squared_difference_emitter),
        OV_CASE(PowerDynamic, jit_power_dynamic_emitter),
        OV_CASE(Equal, jit_equal_emitter),
        OV_CASE(NotEqual, jit_not_equal_emitter),
        OV_CASE(Greater, jit_greater_emitter),
        OV_CASE(GreaterEqual, jit_greater_equal_emitter),
        OV_CASE(Less, jit_less_emitter),
        OV_CASE(LessEqual, jit_less_equal_emitter),
        OV_CASE(LogicalAnd, jit_logical_and_emitter),
        OV_CASE(LogicalOr, jit_logical_or_emitter),
        OV_CASE(LogicalXor, jit_logical_xor_emitter),
        OV_CASE(LogicalNot, jit_logical_not_emitter),
        OV_CASE(PowerStatic, jit_power_static_emitter),
        OV_CASE(Prelu, jit_prelu_emitter));

        if (!ctx.emitter)
            THROW_IE_EXCEPTION << "Unsupported operation type for Eltwise emitter";

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
                auto quantizeNode = dynamic_cast<MKLDNNQuantizeNode*>(eltwiseNode.getFusedWith()[i].get());

                bool do_dequantization = quantizeNode->getOpType() == QuantizeOpType::FakeQuantization;
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

MKLDNNEltwiseNode::MKLDNNEltwiseNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(layer, eng, cache) {
}

InferenceEngine::details::caseless_map<std::string, std::function<void(GenericLayer*, EltwiseOpType&, mkldnn::algorithm&, float&, float&)>>
MKLDNNEltwiseNode::initializers = {
        {"relu", [](GenericLayer* activationLayer, EltwiseOpType& opType, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = activationLayer->GetParamAsFloat("negative_slope", 0.0f);
            beta = 0.0f;
            opType = Relu;
            algorithm = mkldnn::algorithm::eltwise_relu;
        }},
        {"gelu", [](GenericLayer* activationLayer, EltwiseOpType& opType, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = 0.0f;
            beta = 0.0f;
            opType = Gelu;
            algorithm = mkldnn::algorithm::eltwise_gelu;
        }},
        {"elu", [](GenericLayer* activationLayer, EltwiseOpType& opType, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = activationLayer->GetParamAsFloat("alpha", 1.0f);
            beta = 0.0f;
            opType = Elu;
            algorithm = mkldnn::algorithm::eltwise_elu;
        }},
        {"tanh", [](GenericLayer* activationLayer, EltwiseOpType& opType, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = 0.0f;
            beta = 0.0f;
            opType = Tanh;
            algorithm = mkldnn::algorithm::eltwise_tanh;
        }},
        {"sigmoid", [](GenericLayer* activationLayer, EltwiseOpType& opType, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = 0.0f;
            beta = 0.0f;
            opType = Logistic;
            algorithm = mkldnn::algorithm::eltwise_logistic;
        }},
        {"logistic", [](GenericLayer* activationLayer, EltwiseOpType& opType, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = 0.0f;
            beta = 0.0f;
            opType = Logistic;
            algorithm = mkldnn::algorithm::eltwise_logistic;
        }},
        {"square", [](GenericLayer* activationLayer, EltwiseOpType& opType, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = 0.0f;
            beta = 0.0f;
            opType = Square;
            algorithm = mkldnn::algorithm::eltwise_square;
        }},
        {"abs", [](GenericLayer* activationLayer, EltwiseOpType& opType, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = 0.0f;
            beta = 0.0f;
            opType = Abs;
            algorithm = mkldnn::algorithm::eltwise_abs;
        }},
        {"sqrt", [](GenericLayer* activationLayer, EltwiseOpType& opType, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = 0.0f;
            beta = 0.0f;
            opType = Sqrt;
            algorithm = mkldnn::algorithm::eltwise_sqrt;
        }},
        {"linear", [](GenericLayer* activationLayer, EltwiseOpType& opType, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = activationLayer->GetParamAsFloat("alpha", 1.0f);
            beta = activationLayer->GetParamAsFloat("beta", 0.0f);
            opType = Linear;
            algorithm = mkldnn::algorithm::eltwise_linear;
        }},
        {"bounded_relu", [](GenericLayer* activationLayer, EltwiseOpType& opType, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = activationLayer->GetParamAsFloat("alpha", 0.0f);
            beta = 0.0f;
            opType = BoundedRelu;
            algorithm = mkldnn::algorithm::eltwise_bounded_relu;
        }},
        {"soft_relu", [](GenericLayer* activationLayer, EltwiseOpType& opType, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = 0.0f;
            beta = 0.0f;
            opType = SoftRelu;
            algorithm = mkldnn::algorithm::eltwise_soft_relu;
        }},
        {"relu6", [](GenericLayer* activationLayer, EltwiseOpType& opType, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = activationLayer->GetParamAsFloat("n", 6.0f);
            beta = 0.0f;
            opType = Relu6;
            algorithm = mkldnn::algorithm::eltwise_bounded_relu;
        }},
        {"clamp", [](GenericLayer* activationLayer, EltwiseOpType& opType, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = activationLayer->GetParamAsFloat("min", 1.0f);
            beta = activationLayer->GetParamAsFloat("max", 0.0f);
            opType = Clamp;
            algorithm = mkldnn::algorithm::eltwise_clip;
        }},
        {"exp", [](GenericLayer* activationLayer, EltwiseOpType& opType, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = 0.0f;
            beta = 0.0f;
            opType = Exp;
            algorithm = mkldnn::algorithm::eltwise_exp;
        }},
        {"not", [](GenericLayer* activationLayer, EltwiseOpType& opType, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = 0.0f;
            beta = 0.0f;
            opType = LogicalNot;
        }},
        {"swish", [](GenericLayer* activationLayer, EltwiseOpType& opType, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = activationLayer->GetParamAsFloat("alpha", 1.0f);
            beta = 0.0f;
            opType = Swish;
            algorithm = mkldnn::algorithm::eltwise_swish;
        }},
        {"hswish", [](GenericLayer* activationLayer, EltwiseOpType& opType, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = 0.0f;
            beta = 0.0f;
            opType = Hswish;
            algorithm = mkldnn::algorithm::eltwise_hswish;
        }},
        {"mish", [](GenericLayer* activationLayer, EltwiseOpType& opType, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = 0.0f;
            beta = 0.0f;
            opType = Mish;
            algorithm = mkldnn::algorithm::eltwise_mish;
        }},
        {"hsigmoid", [](GenericLayer* activationLayer, EltwiseOpType& opType, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = 0.0f;
            beta = 0.0f;
            opType = Hsigmoid;
            algorithm = mkldnn::algorithm::eltwise_hsigmoid;
        }},
        {"round", [](GenericLayer* activationLayer, EltwiseOpType& opType, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = 0.0f;
            beta = 0.0f;
            opType = Round;
            std::string mode = activationLayer->GetParamAsString("mode", "half_to_even");
            if (mode == "half_to_even")
                algorithm = mkldnn::algorithm::eltwise_round_half_to_even;
            else if (mode == "half_away_from_zero")
                algorithm = mkldnn::algorithm::eltwise_round_half_away_from_zero;
            else
                THROW_IE_EXCEPTION << "Round layer with name " << activationLayer->name << " doesn't support mode " << mode;
        }},
};

void MKLDNNEltwiseNode::init() {
    InferenceEngine::details::CaselessEq<std::string> comparator;
    auto layerType = getCnnLayer().get()->type;

    auto * eltwiseLayer = dynamic_cast<EltwiseLayer*>(getCnnLayer().get());
    if (eltwiseLayer) {
        if (!eltwiseLayer->coeff.empty())
            THROW_IE_EXCEPTION << "Eltwise node with name `" << getName() << "` doesn't support input coefficients.";

        switch (eltwiseLayer->_operation) {
            case EltwiseLayer::Sum: eltwiseOp = Add; break;
            case EltwiseLayer::Prod: eltwiseOp = Multiply; break;
            case EltwiseLayer::Max: eltwiseOp = Maximum; break;
            case EltwiseLayer::Sub: eltwiseOp = Subtract; break;
            case EltwiseLayer::Min: eltwiseOp = Minimum; break;
            case EltwiseLayer::Div: eltwiseOp = Divide; break;
            case EltwiseLayer::Squared_diff: eltwiseOp = SquaredDifference; break;
            case EltwiseLayer::Floor_mod: eltwiseOp = FloorMod; break;
            case EltwiseLayer::Pow: eltwiseOp = PowerDynamic; break;
            case EltwiseLayer::Equal: eltwiseOp = Equal; break;
            case EltwiseLayer::Not_equal: eltwiseOp = NotEqual; break;
            case EltwiseLayer::Greater: eltwiseOp = Greater; break;
            case EltwiseLayer::Greater_equal: eltwiseOp = GreaterEqual; break;
            case EltwiseLayer::Less: eltwiseOp = Less; break;
            case EltwiseLayer::Less_equal: eltwiseOp = LessEqual; break;
            case EltwiseLayer::Logical_AND: eltwiseOp = LogicalAnd; break;
            case EltwiseLayer::Logical_OR: eltwiseOp = LogicalOr; break;
            case EltwiseLayer::Logical_XOR: eltwiseOp = LogicalXor; break;
            default: THROW_IE_EXCEPTION << "Unsupported algorithm for Eltwise node with name `" << getName() << "`.";
        }
    } else if (comparator(layerType, "mod")) {
        eltwiseOp = Mod;
    } else if (comparator(layerType, "power")) {
        eltwiseOp = PowerStatic;

        auto *powerLayer = dynamic_cast<InferenceEngine::PowerLayer *>(getCnnLayer().get());
        if (powerLayer == nullptr)
            THROW_IE_EXCEPTION << "Cannot convert power layer.";

        alpha = powerLayer->power;
        beta = powerLayer->scale;
        gamma = powerLayer->offset;
    } else if (comparator(layerType, "scaleshift")) {
        if (getCnnLayer().get()->blobs.size() == 2) {
            eltwiseOp = MulAdd;
            eltwiseAlgorithm = mkldnn::algorithm::depthwise_scale_shift;
        } else {
            eltwiseOp = Multiply;
        }
    } else if (comparator(layerType, "prelu")) {
        eltwiseOp = Prelu;
        eltwiseAlgorithm = mkldnn::algorithm::depthwise_prelu;
    } else if (comparator(layerType, "activation") && initializers.find(getCnnLayer().get()->GetParamAsString("type")) != initializers.end()) {
        initializers[getCnnLayer().get()->GetParamAsString("type")](getCnnLayer().get(), eltwiseOp, eltwiseAlgorithm, alpha, beta);
    } else if (comparator(layerType, "relu") ||
               comparator(layerType, "gelu") ||
               comparator(layerType, "elu") ||
               comparator(layerType, "sigmoid") ||
               comparator(layerType, "logistic") ||
               comparator(layerType, "tanh") ||
               comparator(layerType, "relu6") ||
               comparator(layerType, "exp") ||
               comparator(layerType, "not") ||
               comparator(layerType, "clamp") ||
               comparator(layerType, "swish") ||
               comparator(layerType, "hswish") ||
               comparator(layerType, "mish") ||
               comparator(layerType, "hsigmoid") ||
               comparator(layerType, "round")) {
        initializers[layerType](getCnnLayer().get(), eltwiseOp, eltwiseAlgorithm, alpha, beta);
    } else {
        THROW_IE_EXCEPTION << "Unsupported algorithm for Eltwise node with name `" << getName() << "`.";
    }
}

size_t MKLDNNEltwiseNode::getOpInputsNum() const {
    switch (getOpType()) {
        case Relu: case Gelu: case Elu: case Tanh: case Logistic: case Square: case Abs: case Sqrt: case PowerStatic:
        case Linear: case BoundedRelu: case SoftRelu: case Relu6: case Exp: case Clamp: case Swish: case Hswish:
        case Mish: case Hsigmoid: case Round:
        case LogicalNot:
            return 1;
        case Add: case Subtract: case Multiply: case Divide: case FloorMod: case Mod: case Maximum: case Minimum: case SquaredDifference:
        case PowerDynamic: case Equal: case NotEqual: case Greater: case GreaterEqual: case Less: case LessEqual: case LogicalAnd:
        case LogicalOr: case LogicalXor: case Prelu:
            return 2;
        case MulAdd:
            return 3;
        default: THROW_IE_EXCEPTION << "Unsupported operation for Eltwise node with name `" << getName() << "`.";
    }
}

bool MKLDNNEltwiseNode::isSum() {
    return eltwiseOp == Add;
}

bool MKLDNNEltwiseNode::isWithBroadcast() {
    auto oDims = outDims[0].ToSizeVector();
    for (size_t i = 0; i < inDims.size(); i++) {
        auto iDims = inDims[i].ToSizeVector();
        if (iDims != oDims)
            return true;
    }

    return false;
}

void MKLDNNEltwiseNode::getSupportedDescriptors() {
    if (getParentEdges().size() < 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();
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
        THROW_IE_EXCEPTION << "Eltwise node with name `" << getName() << "` doesn't support more than " << MAX_ELTWISE_INPUTS
                           << " inputs (actual = " << getParentEdges().size() << ")";

    if (expectedInputsNum != getParentEdges().size())
        THROW_IE_EXCEPTION << "Eltwise node with name `" << getName() << "` has invalid input number of inputs: expected = " << expectedInputsNum
                           << " (actual = " << getParentEdges().size() << ")";

    std::vector<InferenceEngine::Precision> inputPrecisions;
    for (int i = 0; i < getCnnLayer()->insData.size(); i++) {
        inputPrecisions.push_back(getCnnLayer()->insData[i].lock()->getPrecision());
    }

    for (auto& fusedNode : fusedWith) {
        if (fusedNode->getType() == Eltwise) {
            for (int i = 1; i < fusedNode->getCnnLayer()->insData.size(); i++) {
                inputPrecisions.push_back(fusedNode->getCnnLayer()->insData[i].lock()->getPrecision());
            }
        }
    }

    if (inputPrecisions.size() != getParentEdges().size())
        THROW_IE_EXCEPTION << "Eltwise node with name `" << getName() << "` has invalid input precisions configuration.";

    InferenceEngine::Precision outputPrecision = getCnnLayer()->outData[0]->getPrecision();
    if (!fusedWith.empty()) {
        auto lastFusedLayer = fusedWith[fusedWith.size() - 1].get()->getCnnLayer();
        if (lastFusedLayer) {
            outputPrecision = lastFusedLayer->outData[0]->getPrecision();
        }
    }

    if (!mayiuse(avx512_core)) {
        bool hasBF16 = false;
        for (auto &inPrc : inputPrecisions)
            if (inPrc == Precision::BF16)
                hasBF16 = true;

        if (outputPrecision == Precision::BF16 || hasBF16)
            THROW_IE_EXCEPTION << "Eltwise node with name `" << getName() << "` doesn't support BF16 precision on this target.";
    }

    auto filterPrecision = [&](Precision& prc) {
        if (!canUseOptimizedImpl) {
            return Precision(Precision::FP32);
        } else if (std::find(supportedPrecisions.begin(), supportedPrecisions.end(), prc) == supportedPrecisions.end()) {
            if (prc == Precision::U32 || prc == Precision::I64 || prc == Precision::U64) {
                return Precision(Precision::I32);
            } else {
                THROW_IE_EXCEPTION << "Eltwise node with name `" << getName() << "` doesn't support " << prc << " precision.";
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
    if (eltwiseOp == MulAdd && (inputPrecisions[0] == Precision::U8 || inputPrecisions[0] == Precision::I8)) {
        auto poolingLayer = dynamic_cast<PoolingLayer*>(getParentEdgesAtPort(0)[0]->getParent()->getCnnLayer().get());
        if (poolingLayer && poolingLayer->_type == PoolingLayer::AVG) {
            inputPrecisions[0] = Precision::FP32;
        }
    }

    enum LayoutType {
        Planar,
        ChannelsFirst,
        Blocked
    };

    auto initDesc = [&] (LayoutType lt) -> PrimitiveDescInfo {
        auto createMemoryDesc = [lt](MKLDNNEdgePtr edge, Precision prc, size_t offset) -> TensorDesc {
            if (lt == ChannelsFirst) {
                auto dims = edge->getDims().ToSizeVector();
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

                return TensorDesc(prc, edge->getDims().ToSizeVector(), {blocks, order, offset});
            } else if (lt == Blocked && edge->getDims()[1] != 1) {
                size_t blockSize = mayiuse(x64::avx512_common) ? 16 : 8;

                std::vector<size_t> blocks = edge->getDims().ToSizeVector();
                std::vector<size_t> order(blocks.size());
                std::iota(order.begin(), order.end(), 0);

                blocks[1] = div_up(blocks[1], blockSize);
                blocks.push_back(blockSize);
                order.push_back(1);

                return TensorDesc(prc, edge->getDims().ToSizeVector(), {blocks, order, offset});
            } else {
                std::vector<size_t> blocks = edge->getDims().ToSizeVector();
                std::vector<size_t> order(blocks.size());
                std::iota(order.begin(), order.end(), 0);

                return TensorDesc(prc, edge->getDims().ToSizeVector(), {blocks, order, offset});
            }
        };

        size_t offset = std::numeric_limits<size_t>::max();
        InferenceEngine::LayerConfig config;
        config.dynBatchSupport = getChildEdgeAt(0)->getDims().ndims() > 1 && getChildEdgeAt(0)->getDims() == getParentEdgeAt(0)->getDims();

        for (size_t i = 0; i < getParentEdges().size(); i++) {
            InferenceEngine::DataConfig dataConfig;
            dataConfig.inPlace = (!i && canBeInPlace() && inputPrecisions[i] == outputPrecision) ? 0 : -1;
            dataConfig.constant = false;


            dataConfig.desc = createMemoryDesc(getParentEdgeAt(i), inputPrecisions[i], offset);

            config.inConfs.push_back(dataConfig);
        }

        InferenceEngine::DataConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;

        dataConfig.desc = createMemoryDesc(getChildEdgeAt(0), outputPrecision, offset);

        config.outConfs.push_back(dataConfig);

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

    bool isChannelsFirstApplicable = one_of(getChildEdgeAt(0)->getDims().ndims(), 1, 2, 4, 5);
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        isChannelsFirstApplicable = isChannelsFirstApplicable && one_of(getParentEdgeAt(i)->getDims().ndims(), 1, 2, 4, 5);
        isChannelsFirstApplicable = isChannelsFirstApplicable && getChildEdgeAt(0)->getDims().ndims() == getParentEdgeAt(i)->getDims().ndims();
    }

    bool isBlockedApplicable = one_of(getChildEdgeAt(0)->getDims().ndims(), 4, 5);
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        isBlockedApplicable = isBlockedApplicable && one_of(getParentEdgeAt(i)->getDims().ndims(), 4, 5);
        isBlockedApplicable = isBlockedApplicable && getChildEdgeAt(0)->getDims().ndims() == getParentEdgeAt(i)->getDims().ndims();
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

        std::vector<size_t> order(maxInputSize);
        auto outOrder = config.outConfs[0].desc.getBlockingDesc().getOrder();
        for (size_t i = 0; i < order.size(); i++) {
            if (i < order.size() - outOrder.size())
                order[i] = i;
            else
                order[i] = outOrder[i - (order.size() - outOrder.size())] + (order.size() - outOrder.size());
        }

        size_t outRank = config.outConfs[0].desc.getBlockingDesc().getBlockDims().size();
        for (int i = 0; i < outRank; i++) {
            dims_out[dims_out.size() - 1 - i] = config.outConfs[0].desc.getBlockingDesc().getBlockDims()[outRank - 1 - i];
        }

        for (int i = 0; i < inputNum; i++) {
            size_t inRank = config.inConfs[i].desc.getBlockingDesc().getBlockDims().size();

            // WA to normalize blocked and planar layouts
            auto inOrder = config.inConfs[i].desc.getBlockingDesc().getOrder();
            size_t startOff = outOrder.size() != config.outConfs[0].desc.getDims().size() &&
                              outOrder[outOrder.size() - 1] != inOrder[inOrder.size() - 1] ? 1 : 0;

            for (int j = 0; j < inRank; j++) {
                dims_in[i][dims_in[i].size() - 1 - j - startOff] = config.inConfs[i].desc.getBlockingDesc().getBlockDims()[inRank - 1 - j];
            }
        }

        for (int i = 0; i < dims_in.size(); i++) {
            for (int j = 0; j < dims_in[i].size(); j++) {
                if (dims_in[i][j] != dims_out[j] && dims_in[i][j] != 1)
                    THROW_IE_EXCEPTION << "Eltwise node with name `" << getName() << "` has invalid input/output dims configuration.";
            }
        }
    };

    auto initOffsets = [this, config](size_t maxInputSize) {
        size_t inputNum = getParentEdges().size();

        offsets_out.resize(maxInputSize, 1);
        offset_out_calc(offsets_out, dims_out);
        for (int j = 0; j < maxInputSize; j++) {
            offsets_out[j] *= config.outConfs[0].desc.getPrecision().size();
        }

        offsets_in.resize(inputNum);
        for (int i = 0; i < inputNum; i++) {
            offsets_in[i].resize(maxInputSize, 1);
            offset_in_calc(offsets_in[i], dims_in[i], dims_out);
            for (int j = 0; j < maxInputSize; j++) {
                offsets_in[i][j] *= config.inConfs[i].desc.getPrecision().size();
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

    tensorRank = std::max(static_cast<size_t>(optimalTensorRank), config.outConfs[0].desc.getBlockingDesc().getBlockDims().size());
    initDims(tensorRank);

    auto outOrder = config.outConfs[0].desc.getBlockingDesc().getOrder();
    size_t oc_size = 0;
    offsets_oc.resize(tensorRank, 0);
    if (isFusedWith(Quantize)) {
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
               (!isDynBatchEnabled || (config.outConfs[0].desc.getBlockingDesc().getBlockDims().size() - collapsedDims > 2))) {
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

                if (isFusedWith(Quantize)) {
                    collapseLastOffsets(offsets_oc, 1);
                }
            } else {
                break;
            }
        }
    }

    batchDimIdx = tensorRank - config.outConfs[0].desc.getBlockingDesc().getBlockDims().size() + collapsedDims;
    schedulerWorkAmount = fullWorkAmount / dims_out[dims_out.size() - 1];

    initOffsets(tensorRank);

    jep.inputs_number = config.inConfs.size();
    jep.input_size = tensorRank;

    for (int i = 0; i < config.inConfs.size(); i++) {
        jep.src_size[i] = dims_in[i][dims_in[i].size() - 1];
        jep.src_prc[i] = config.inConfs[i].desc.getPrecision();
    }
    jep.dst_size = dims_out[dims_out.size() - 1];
    jep.dst_prc = config.outConfs[0].desc.getPrecision();

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
    for (auto& type : getPrimitivesPriority()) {
        int selectedPrimitive = -1;
        int equalsFormatCount = -1;
        for (size_t i = 0; i < getSupportedPrimitiveDescriptors().size(); i++) {
            impl_desc_type supportedType = getSupportedPrimitiveDescriptors()[i].getImplementationType();
            if (type == supportedType) {
                int equalsLocalFormatCount = 0;
                if (getSupportedPrimitiveDescriptors()[i].getConfig().inConfs.size() > getParentEdges().size())
                    continue;
                for (size_t j = 0; j < getSupportedPrimitiveDescriptors()[i].getConfig().inConfs.size(); j++) {
                    auto parentEdge = getParentEdgeAt(j);
                    auto parentPtr = parentEdge->getParent();
                    // We don't take into account constant edges since reorders on them will be executed on load network stage
                    if (j > 0 && parentPtr->isConstant()) {
                        equalsLocalFormatCount++;
                        continue;
                    }

                    auto parent_spd = parentPtr->getSelectedPrimitiveDescriptor();

                    if (parent_spd != nullptr && !parent_spd->getConfig().outConfs.empty()) {
                        int inNum = parentEdge->getInputNum();
                        if (inNum < 0 || inNum >= parent_spd->getConfig().outConfs.size()) {
                            inNum = 0;
                        }
                        if (MKLDNNExtensionUtils::initTensorsAreEqual(
                                getSupportedPrimitiveDescriptors()[i].getConfig().inConfs[j].desc,
                                parent_spd->getConfig().outConfs[inNum].desc)) {
                            equalsLocalFormatCount++;
                        }
                    }
                }
                if (equalsLocalFormatCount > equalsFormatCount) {
                    equalsFormatCount = equalsLocalFormatCount;
                    selectedPrimitive = static_cast<int>(i);
                }
            }
        }
        if (selectedPrimitive >= 0) {
            selectPrimitiveDescriptorByIndex(selectedPrimitive);
            return;
        }
    }

    if (getSupportedPrimitiveDescriptors().empty())
        THROW_IE_EXCEPTION << "Supported primitive descriptors list is empty for node: " << getName();
    // fallback. If there are no primitives from priority list just select a first
    selectPrimitiveDescriptorByIndex(0);
}

void MKLDNNEltwiseNode::initOptimalPrimitiveDescriptor() {
    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";
    auto config = selected_pd->getConfig();
    if (!isInitConfig(config)) {
        for (size_t i = 0; i < config.inConfs.size(); i++) {
            config.inConfs[i].desc = getConfiguredInputDesc(config, i);
        }

        for (size_t i = 0; i < config.outConfs.size(); i++) {
            config.outConfs[i].desc = getConfiguredOutputDesc(config, i);
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
    if (eltwiseAlgorithm != mkldnn::algorithm::undef) {
        ref_eltwise_injector = std::make_shared<ref_eltwise_scalar_fwd_t>(static_cast<mkldnn_alg_kind_t>(eltwiseAlgorithm), alpha, beta, 1.f);
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

            switch (getOpType()) {
                case Relu: case Gelu: case Elu: case Tanh: case Logistic: case Square: case Abs: case Sqrt:
                case Linear: case BoundedRelu: case SoftRelu: case Relu6: case Exp: case Clamp: case Swish: case Hswish:
                case Mish: case Hsigmoid: case Round:
                    *dst_ptr_f = ref_eltwise_injector->compute_scalar(src_f[0]); break;
                case Add:               *dst_ptr_f = src_f[0] + src_f[1]; break;
                case MulAdd:            *dst_ptr_f = src_f[0] * src_f[1] + src_f[2]; break;
                case Subtract:          *dst_ptr_f = src_f[0] - src_f[1]; break;
                case Multiply:          *dst_ptr_f = src_f[0] * src_f[1]; break;
                case Divide:            *dst_ptr_f = src_f[0] / src_f[1]; break;
                case FloorMod:          *dst_ptr_f = src_f[0] - floorf(src_f[0] / src_f[1]) * src_f[1]; break;
                case Mod:               *dst_ptr_f = src_f[0] - truncf(src_f[0] / src_f[1]) * src_f[1]; break;
                case Maximum:           *dst_ptr_f = std::max(src_f[0], src_f[1]); break;
                case Minimum:           *dst_ptr_f = std::min(src_f[0], src_f[1]); break;
                case SquaredDifference: *dst_ptr_f = powf((src_f[0] - src_f[1]), 2.f); break;
                case PowerDynamic:      *dst_ptr_f = powf(src_f[0], src_f[1]); break;
                case Equal:             *dst_ptr_f = src_f[0] == src_f[1]; break;
                case NotEqual:          *dst_ptr_f = src_f[0] != src_f[1]; break;
                case Greater:           *dst_ptr_f = src_f[0] > src_f[1]; break;
                case GreaterEqual:      *dst_ptr_f = src_f[0] >= src_f[1]; break;
                case Less:              *dst_ptr_f = src_f[0] < src_f[1]; break;
                case LessEqual:         *dst_ptr_f = src_f[0] <= src_f[1]; break;
                case LogicalAnd:        *dst_ptr_f = src_f[0] && src_f[1]; break;
                case LogicalOr:         *dst_ptr_f = src_f[0] || src_f[1]; break;
                case LogicalXor:        *dst_ptr_f = (src_f[0] || src_f[1]) - (src_f[0] && src_f[1]); break;
                case LogicalNot:        *dst_ptr_f = !src_f[0]; break;
                case PowerStatic:       *dst_ptr_f = powf(beta * src_f[0] + gamma, alpha); break;
                case Prelu:             *dst_ptr_f = src_f[0] > 0 ? src_f[0] : src_f[0] * src_f[1]; break;
                default: THROW_IE_EXCEPTION << "Unsupported operation type for Eltwise node with name `" << getName() << "`";
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

    return getParentEdgesAtPort(0)[0].get()->getDims() == getChildEdgesAtPort(0)[0].get()->getDims();
}

void MKLDNNEltwiseNode::appendPostOps(mkldnn::post_ops& ops) {
    switch (getAlgorithm()) {
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
        case mkldnn::algorithm::eltwise_gelu:
        case mkldnn::algorithm::eltwise_clip:
        case mkldnn::algorithm::eltwise_swish:
        case mkldnn::algorithm::eltwise_hswish:
        case mkldnn::algorithm::eltwise_mish:
        case mkldnn::algorithm::eltwise_hsigmoid:
        case mkldnn::algorithm::eltwise_round_half_to_even:
        case mkldnn::algorithm::eltwise_round_half_away_from_zero:
            ops.append_eltwise(1.0, getAlgorithm(), getAlpha(), getBeta());
            break;
        case mkldnn::algorithm::depthwise_scale_shift:
        case mkldnn::algorithm::depthwise_prelu:
            if (scales.empty() && shifts.empty()) {
                size_t bufferSize = static_cast<size_t>(outDims[0][outDims[0].size() > 1 ? 1 : 0]);
                size_t bufferSizeAligned = rnd_up(bufferSize, 16);

                Blob::Ptr scalesBlob = getCnnLayer()->blobs["weights"];
                if (scalesBlob == nullptr)
                    THROW_IE_EXCEPTION << "Cannot get weights blob in Eltwise node with name `" << getName() << "`";
                scales.resize(bufferSizeAligned, 0);
                const float *scalesBufferPtr = scalesBlob->buffer().as<float *>();
                for (int i = 0; i < bufferSize; i++) {
                    scales[i] = scalesBufferPtr[scalesBlob->size() == 1 ? 0 : i];
                }

                Blob::Ptr shiftsBlob = getCnnLayer()->blobs["biases"];
                if (shiftsBlob != nullptr) {
                    shifts.resize(bufferSizeAligned, 0);
                    const float *shiftsBufferPtr = shiftsBlob->buffer().as<float *>();
                    for (int i = 0; i < bufferSize; i++) {
                        shifts[i] = shiftsBufferPtr[shiftsBlob->size() == 1 ? 0 : i];
                    }
                }
            }

            ops.append_depthwise(getAlgorithm(), &scales[0], shifts.empty() ? nullptr : &shifts[0]);
            break;
        default: THROW_IE_EXCEPTION << "Appending Eltwise node with name `" << getName() << "` as post operation is not supported";
    }
}

bool MKLDNNEltwiseNode::canFuse(const MKLDNNNodePtr& node) const {
    auto isOneOf = [](EltwiseOpType alg, std::vector<EltwiseOpType> algs) {
        for (auto a : algs) {
            if (alg == a) {
                return true;
            }
        }
        return false;
    };

    auto isSuitableNode = [](const MKLDNNEltwiseNode* node) {
        // [WA] Since execution precision change from I32 to FP32 for Divide operation may lead to incorrect results
        // we disable its fusing otherwise there is no guarantee it will be executed it I32
        // [TODO] We need to rewrite support for different precisions at all to avoid implicit conversions to FP32
        // (all should be handled via explicit convert operations)
        if (node->getOpType() == Divide) {
            for (int i = 0; i < node->getCnnLayer()->insData.size(); i++) {
                if (node->getCnnLayer()->insData[i].lock()->getPrecision() == Precision::I32) {
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
    size_t addedInputEdgesNum = node->getType() != Quantize ? (node->getParentEdges().size() - 1) : 0;
    if (getParentEdges().size() + addedInputEdgesNum > MAX_ELTWISE_INPUTS)
        return false;

    if (node->getType() == Eltwise) {
        auto eltwiseNode = dynamic_cast<MKLDNNEltwiseNode*>(node.get());
        if (eltwiseNode->getParentEdgesAtPort(0)[0]->getParent().get() != this) {
            if (!isSuitableNode(this)) {
                return false;
            }

            // Eltwise jitter doesn't respect commutative property, so fusing is disabled in case it applied not for 0-th port.
            if (isOneOf(eltwiseNode->getOpType(), {Subtract, Divide, FloorMod, Mod, PowerDynamic, Greater, GreaterEqual, Less, LessEqual})) {
                return false;
            }

            // Limitation: inputs precision definition inside Eltwise node assumes fusing is applied for 0-th port,
            // otherwise we need identical precision on all inputs of fused node
            for (int i = 1; i < eltwiseNode->getCnnLayer()->insData.size(); i++) {
                if (eltwiseNode->getCnnLayer()->insData[0].lock()->getPrecision() != eltwiseNode->getCnnLayer()->insData[i].lock()->getPrecision()) {
                    return false;
                }
            }
        }

        return true;
    }

    if (node->getType() == Quantize) {
        auto *quantizeNode = dynamic_cast<MKLDNNQuantizeNode *>(node.get());
        if (quantizeNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot get quantize layer " << node->getName();
        return !quantizeNode->isBinarization();
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
