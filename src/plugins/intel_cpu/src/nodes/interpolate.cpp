// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "interpolate.h"

#include "common/cpu_memcpy.h"
#include "cpu/x64/injectors/jit_uni_depthwise_injector.hpp"
#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/injectors/jit_uni_quantization_injector.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_uni_eltwise.hpp"
#include "dnnl_extension_utils.h"
#include "eltwise.h"
#include "emitters/plugin/x64/jit_bf16_emitters.hpp"
#include "emitters/plugin/x64/jit_load_store_emitters.hpp"
#include "fake_quantize.h"
#include "onednn/dnnl.h"
#include "openvino/core/parallel.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/opsets/opset4.hpp"
#include "shape_inference/shape_inference.hpp"
#include "shape_inference/shape_inference_ngraph.hpp"
#include "shape_inference/static_shape.hpp"
#include "utils/bfloat16.hpp"
#include "utils/cpu_utils.hpp"
#include "utils/ngraph_utils.hpp"

#include <algorithm>
#include <string>
#include <vector>

using namespace dnnl;

using namespace dnnl::impl;
using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
using namespace Xbyak;


#define GET_OFF(field) offsetof(jit_interpolate_call_args, field)

namespace ov {
namespace intel_cpu {
namespace node {

static inline bool isFloatCompatible(ov::element::Type prc) {
    return one_of(prc, ov::element::f32, ov::element::bf16, ov::element::f16, ov::element::f64);
}

#if defined(OPENVINO_ARCH_X86_64)

template <cpu_isa_t isa>
struct jit_uni_interpolate_kernel_f32 : public jit_uni_interpolate_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_interpolate_kernel_f32)

    explicit jit_uni_interpolate_kernel_f32(jit_interpolate_config_params jcp, const dnnl_primitive_attr &attr)
    : jit_uni_interpolate_kernel(jcp, attr), jit_generator(jit_name()) {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        // dummy second reg_tmp_64 as no fill needed
        load_pool_gpr_idxs = {static_cast<size_t>(reg_tmp_64.getIdx()), static_cast<size_t>(reg_tmp_64.getIdx())};
        store_pool_gpr_idxs = {static_cast<size_t>(reg_tmp_64.getIdx())};
        store_pool_vec_idxs = {static_cast<size_t>(vmm_zero.getIdx())};

        const auto &p = attr_.post_ops_;
        for (int i = 0; i < p.len(); i++) {
            auto &post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors.push_back(std::make_shared<jit_uni_eltwise_injector_f32<isa>>(
                        this,
                        post_op.eltwise.alg,
                        post_op.eltwise.alpha,
                        post_op.eltwise.beta,
                        1.f));
            } else if (post_op.is_depthwise()) {
                depthwise_injectors.push_back(std::make_shared<jit_uni_depthwise_injector_f32<isa>>(
                        this,
                        post_op));
            } else if (post_op.is_quantization()) {
                quantization_injectors.push_back(std::make_shared<jit_uni_quantization_injector_f32<isa>>(
                        this, post_op, vmm_d_weights, vmm_d_bias, reg_d_weights, reg_d_bias));
            }
        }

        this->preamble();

        if (attr_.post_ops_.len() != 0) {
            mov(reg_post_ops_data, ptr[reg_params + GET_OFF(post_op_data)]);
            mov(reg_oc_off, ptr[reg_params + GET_OFF(oc_off)]);
        }
        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        switch (jcp_.mode) {
            case InterpolateMode::nearest: {
                mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
                mov(reg_src, ptr[reg_params + GET_OFF(src_ptr[0])]);
                mov(reg_index, ptr[reg_params + GET_OFF(index)]);
                mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);

                switch (jcp_.layout) {
                    case InterpolateLayoutType::planar: {
                        nn_planar();
                        break;
                    }
                    case InterpolateLayoutType::block: {
                        nn_blk();
                        break;
                    }
                    case InterpolateLayoutType::by_channel: {
                        nn_by_channel();
                        break;
                    }
                    default:
                        assert(!"unsupported memory layout for interpolate layer with nearest neighbor mode.");
                }
                break;
            }
            case InterpolateMode::linear_onnx: {
                switch (jcp_.layout) {
                    case InterpolateLayoutType::planar: {
                        linear_onnx_planar();
                        break;
                    }
                    case InterpolateLayoutType::block:
                    case InterpolateLayoutType::by_channel: {
                        linear_onnx_c_gathered();
                        break;
                    }
                    default:
                        assert(!"unsupported memory layout for interpolate layer with linear_onnx mode.");
                }
                break;
            }
            case InterpolateMode::cubic: {
                switch (jcp_.layout) {
                    case InterpolateLayoutType::planar: {
                        cubic_planar();
                        break;
                    }
                    case InterpolateLayoutType::block:
                    case InterpolateLayoutType::by_channel: {
                        cubic_c_gathered();
                        break;
                    }
                    default:
                        assert(!"unsupported memory layout for interpolate layer with cubic mode.");
                }
                break;
            }
            case InterpolateMode::bilinear_pillow:
            case InterpolateMode::bicubic_pillow: {
                switch (jcp_.layout) {
                    case InterpolateLayoutType::by_channel: {
                        pillow_by_channel();
                        break;
                    }
                    default:
                        assert(!"unsupported memory layout for interpolate layer with bilinear_pillow and bicubic_pillow modes.");
                }
                break;
            }
            case InterpolateMode::linear: {
                assert(!"unsupported mode for interpolate layer with JITTED implimentation.");
                break;
            }
            default: {
                assert(!"unsupported mode for interpolate layer.");
            }
        }

        this->postamble();

        emit_emitters_data();
        for (auto& inj : eltwise_injectors)
            inj->prepare_table();
        if ((jcp_.mode == InterpolateMode::cubic) && (jcp_.layout == InterpolateLayoutType::planar)) {
            prepare_cubic_planar_table();
        }
    }

private:
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;

    const int vlen = cpu_isa_traits<isa>::vlen;
    const int vector_step = vlen / sizeof(float);
    const int tail_step = jcp_.C % vector_step;
    const int scalar_step = 1;

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_src_aux = r15;
    Xbyak::Reg64 reg_src_aux1 = r11;
    Xbyak::Reg64 reg_src_aux2 = r12;
    Xbyak::Reg64 reg_dst = r9;
    Xbyak::Reg64 reg_work_amount = r13;
    Xbyak::Reg64 reg_index = r14;
    Xbyak::Reg64 reg_params = abi_param1;

    Reg8 reg_tmp_8 = r10b;
    Reg32 reg_tmp_32 = r10d;
    Reg64 reg_tmp_64 = r10;

    Xbyak::Reg64 reg_oc_off = rax;
    Xbyak::Reg64 reg_post_ops_data = rbx;
    Xbyak::Reg64 reg_d_weights = reg_tmp_64;
    Xbyak::Reg64 reg_d_bias = rcx;
    Xbyak::Reg32 reg_index_offset = edx;

    // for cubic planar
    Xbyak::Reg64 reg_tbl_y = rsi;
    Xbyak::Reg64 reg_tbl_x = rbp;
    Xbyak::Reg64 reg_table = rdx;   // do not need reg_index_offset in this mode, so use rdx

    Vmm vmm_val = Vmm(1);
    Vmm vmm_index = Vmm(0);
    Vmm vmm_zero = Vmm(2);
    Vmm vmm_mask = Vmm(3);
    Vmm vmm_d_weights = Vmm(4);
    Vmm vmm_d_bias = Vmm(5);

    // for linear
    Vmm vmm_weightT = Vmm(15);
    Vmm vmm_weightB = Vmm(14);
    Vmm vmm_weightL = Vmm(13);
    Vmm vmm_weightR = Vmm(12);
    Vmm vmm_weightF = Vmm(6);
    Vmm vmm_weightE = Vmm(7);
    Vmm vmm_valTL = Vmm(11);
    Vmm vmm_valTR = vmm_val;
    Vmm vmm_valBL = Vmm(9);
    Vmm vmm_valBR = Vmm(8);

    // for cubic
    Vmm vmm_src = Vmm(6);
    Xmm xmm_src = Xmm(6);
    Vmm vmm_dstX = Vmm(7);

    Vmm vmm_weightX0 = vmm_weightT;
    Vmm vmm_weightX1 = vmm_weightB;
    Vmm vmm_weightX2 = vmm_weightL;
    Vmm vmm_weightX3 = vmm_weightR;
    Vmm vmm_weightY0 = vmm_valTL;
    Vmm vmm_weightY1 = Vmm(10);  // vmm_valTR is vmm_val, need reserved
    Vmm vmm_weightY2 = vmm_valBL;
    Vmm vmm_weightY3 = vmm_valBR;
    // cubic planar
    Vmm vmm_one = vmm_index;
    Vmm vmm_weightY = vmm_weightY0;
    Vmm vmm_index_y_itr = vmm_weightY1;
    Vmm vmm_index_x_itr = vmm_weightY2;
    Vmm vmm_tbl_y = vmm_weightY3;
    // temporally used. when post ops, value in vmm_d_weights and vmm_d_bias is re-loaded(init) each time.
    Vmm vmm_index_in_y = vmm_d_weights;
    Vmm vmm_index_in_x = vmm_d_bias;

    // pillow
    Vmm vmm_weight = Vmm(15);
    Vmm vmm_dst = Vmm(14);

    Xbyak::Label l_table_constant;
    Opmask k_mask = Xbyak::Opmask(1);

    std::unordered_map<size_t, std::unique_ptr<jit_emitter>> emitters;

    std::vector<size_t> store_pool_gpr_idxs;
    std::vector<size_t> store_pool_vec_idxs;
    std::vector<size_t> load_pool_gpr_idxs;

    std::vector<std::shared_ptr<jit_uni_eltwise_injector_f32<isa>>> eltwise_injectors;
    std::vector<std::shared_ptr<jit_uni_depthwise_injector_f32<isa>>> depthwise_injectors;
    std::vector<std::shared_ptr<jit_uni_quantization_injector_f32<isa>>> quantization_injectors;

    void emit_emitters_data() {
        for (const auto& emitter : emitters) {
            if (emitter.second)
                emitter.second->emit_data();
        }
    }

    inline void load(Xbyak::Reg64 reg_src, Vmm vmm_src, const int elt_num, const int offset = 0) {
        emit_load(reg_src, vmm_src, jcp_.src_prc, ov::element::f32, elt_num, offset);
    }

    inline void load_weights(Xbyak::Reg64 reg_src, Vmm vmm_src, const int elt_num, const int offset = 0) {
        emit_load(reg_src, vmm_src, ov::element::f32, ov::element::f32, elt_num, offset);
    }

    inline void emit_load(Xbyak::Reg64 reg_src, Vmm vmm_src, ov::element::Type src_prc, ov::element::Type dst_prc, const int elt_num, const int offset = 0) {
        const auto seed = load_emitter_params(src_prc, dst_prc, elt_num).hash();
        if (!emitters[seed]) {
            emitters[seed].reset(new jit_load_emitter(this, isa, src_prc, dst_prc, elt_num));
        }

        emitters[seed]->emit_code({static_cast<size_t>(reg_src.getIdx()), static_cast<size_t>(offset)},
                                  {static_cast<size_t>(vmm_src.getIdx())}, {}, {load_pool_gpr_idxs});
    }

    inline void store(Vmm vmm_dst, Xbyak::Reg64 reg_dst, const int elt_num, const int offset = 0) {
        const auto seed = store_emitter_params(ov::element::f32, jcp_.dst_prc, elt_num).hash();
        if (!emitters[seed]) {
            emitters[seed].reset(new jit_store_emitter(this, isa, ov::element::f32, jcp_.dst_prc, elt_num));
        }

        // for cases when Store emitter need 2 aux vmm we can use vmm_dst as second aux vmm
        std::vector<size_t> local_store_pool_vec_idxs = { static_cast<size_t>(vmm_dst.getIdx()) };
        local_store_pool_vec_idxs.insert(local_store_pool_vec_idxs.begin(), store_pool_vec_idxs.begin(), store_pool_vec_idxs.end());

        emitters[seed]->emit_code({static_cast<size_t>(vmm_dst.getIdx())},
                                  {static_cast<size_t>(reg_dst.getIdx()), static_cast<size_t>(offset)},
                                  {local_store_pool_vec_idxs}, {store_pool_gpr_idxs});
    }

    // kernel for OH * OW * C
    void pillow_by_channel() {
        Xbyak::Reg64 reg_src = r8;
        Xbyak::Reg64 reg_src_aux = r9;
        Xbyak::Reg64 reg_src_aux1 = rbp;
        Xbyak::Reg64 reg_weights = r11;
        Xbyak::Reg64 reg_weights_bk = rdx;
        Xbyak::Reg64 reg_dst = r12;
        Xbyak::Reg64 reg_dst_xpass = r13;
        Xbyak::Reg64 reg_src_ypass = r14;
        Xbyak::Reg64 reg_dst_aux = r15;
        Xbyak::Reg64 reg_params = abi_param1;

        mov(reg_src, ptr[reg_params + GET_OFF(src_ptr[0])]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_weights, ptr[reg_params + GET_OFF(weight_ptr[0])]);
        mov(reg_weights_bk, reg_weights);

        bool xPass = (jcp_.IW != jcp_.OW);
        bool yPass = (jcp_.IH != jcp_.OH);
        if (xPass && yPass) {
            mov(reg_dst_xpass, ptr[reg_params + GET_OFF(src_ptr[0]) + sizeof(size_t)]);
            mov(reg_src_ypass, reg_dst_xpass);
        } else if (xPass && !yPass) {
            mov(reg_dst_xpass, reg_dst);
        } else if (!xPass && yPass) {
            mov(reg_src_ypass, reg_src);
        } else if (!xPass && !yPass) {
            for (int blk = 0; blk < (jcp_.OH * jcp_.OW * jcp_.C) / vector_step; blk++) {
                load(reg_src, vmm_val, vector_step);
                add(reg_src, vector_step * jcp_.src_data_size);
                store(vmm_val, reg_dst, vector_step);
                add(reg_dst, vector_step * jcp_.dst_data_size);
            }
            int tail_num = (jcp_.OH * jcp_.OW * jcp_.C) % vector_step;
            if (tail_num) {
                load(reg_src, vmm_val, tail_num);
                store(vmm_val, reg_dst, tail_num);
            }
        }
        // /       /   /  /
        // --------    ----
        // |      |    |  |    ..... -> .
        // |      |--> |  |
        // |      |    |  |    .
        // |      |    |  |    .
        // --------    ----    .
        //              \|/
        //                    \|/
        //             /  /
        //             ----
        //             |  |
        //             |  |    .
        //             ----
        int f, filterS, filterL;
        int tail_num = jcp_.C % vector_step;
        // xpass
        if (xPass) {
            mov(reg_dst_aux, reg_dst_xpass);
            for (size_t ih = 0; ih < static_cast<size_t>(jcp_.IH); ih++) {
                // reg_dst_xpass: point to start of this dst height
                // reset reg_dst_aux to start of this height
                mov(reg_weights, reg_weights_bk);
                for (size_t ow = 0; ow < static_cast<size_t>(jcp_.OW); ow++) {
                    // reg_src: point to start of this src height src
                    // reset reg_src_aux to reg_src
                    mov(reg_src_aux, reg_src);
                    filterS = jcp_.bound[ow * 2];
                    filterL = jcp_.bound[ow * 2 + 1];
                    for (int blk = 0; blk < jcp_.C / vector_step; blk++) {
                        uni_vpxor(vmm_dst, vmm_dst, vmm_dst);
                        for (f = 0; f < filterL; f++) {
                            mov(reg_src_aux1, reg_src_aux);
                            add(reg_src_aux1, (f + filterS) * jcp_.C * jcp_.src_data_size);
                            load(reg_src_aux1, vmm_val, vector_step);
                            uni_vbroadcastss(vmm_weight, ptr[reg_weights + f * sizeof(float)]);
                            uni_vfmadd231ps(vmm_dst, vmm_val, vmm_weight);
                        }
                        // if int, round
                        if (!isFloatCompatible(jcp_.src_prc)) {
                            uni_vroundps(vmm_dst, vmm_dst, 0x0); // Round near
                        }
                        // src_prc, dst_prc and buf ov::element::Type is the same, otherwise need another store with buf(src) precision
                        store(vmm_dst, reg_dst_aux, vector_step);
                        add(reg_dst_aux, vector_step * jcp_.src_data_size);
                        // advance 8/16 faciliate next block
                        add(reg_src_aux, vector_step * jcp_.src_data_size);
                    }
                    if (tail_num) {
                        uni_vpxor(vmm_dst, vmm_dst, vmm_dst);
                        for (f = 0; f < filterL; f++) {
                            mov(reg_src_aux1, reg_src_aux);
                            add(reg_src_aux1, (f + filterS) * jcp_.C * jcp_.src_data_size);
                            load(reg_src_aux1, vmm_val, tail_num);
                            uni_vbroadcastss(vmm_weight, ptr[reg_weights + f * sizeof(float)]);
                            uni_vfmadd231ps(vmm_dst, vmm_val, vmm_weight);
                        }
                        if (!isFloatCompatible(jcp_.src_prc)) {
                            uni_vroundps(vmm_dst, vmm_dst, 0x0); // Round near
                        }
                        store(vmm_dst, reg_dst_aux, tail_num);
                        add(reg_dst_aux, tail_num * jcp_.src_data_size);
                        add(reg_src_aux, tail_num * jcp_.src_data_size);  // remove
                    }
                    add(reg_weights, jcp_.filterLenX * sizeof(float));
                }
                // reg_src: point to start of this height
                add(reg_src, jcp_.IW * jcp_.C * jcp_.src_data_size);
            }
        }
        if (yPass) {
            add(reg_weights_bk, jcp_.OW * jcp_.filterLenX * sizeof(float));
            mov(reg_weights, reg_weights_bk);
            size_t bound_offset_y = jcp_.OW * 2;
            for (size_t oh = 0; oh < static_cast<size_t>(jcp_.OH); oh++) {
                filterS = jcp_.bound[bound_offset_y + oh * 2];
                filterL = jcp_.bound[bound_offset_y + oh * 2 + 1];
                for (size_t ow = 0; ow < static_cast<size_t>(jcp_.OW); ow++) {
                    mov(reg_src_aux, reg_src_ypass);  // reg_src_aux to advance block
                    for (int blk = 0; blk < jcp_.C / vector_step; blk++) {
                        uni_vpxor(vmm_dst, vmm_dst, vmm_dst);
                        for (f = 0; f < filterL; f++) {
                            // shared weight
                            uni_vbroadcastss(vmm_weight, ptr[reg_weights + f * sizeof(float)]);
                            mov(reg_src_aux1, reg_src_aux);
                            add(reg_src_aux1, ((f + filterS) * jcp_.OW + ow) * jcp_.C * jcp_.src_data_size);
                            load(reg_src_aux1, vmm_val, vector_step);
                            uni_vfmadd231ps(vmm_dst, vmm_val, vmm_weight);
                        }
                        if (!isFloatCompatible(jcp_.src_prc)) {
                            uni_vroundps(vmm_dst, vmm_dst, 0x0); // Round near
                        }
                        store(vmm_dst, reg_dst, vector_step);
                        add(reg_dst, vector_step * jcp_.dst_data_size);
                        add(reg_src_aux, vector_step * jcp_.src_data_size);
                    }
                    if (tail_num) {
                        uni_vpxor(vmm_dst, vmm_dst, vmm_dst);
                        for (f = 0; f < filterL; f++) {
                            uni_vbroadcastss(vmm_weight, ptr[reg_weights + f * sizeof(float)]);
                            mov(reg_src_aux1, reg_src_aux);
                            add(reg_src_aux1, ((f + filterS) * jcp_.OW + ow) * jcp_.C * jcp_.src_data_size);
                            load(reg_src_aux1, vmm_val, tail_num);
                            uni_vfmadd231ps(vmm_dst, vmm_val, vmm_weight);
                        }
                        if (!isFloatCompatible(jcp_.src_prc)) {
                            uni_vroundps(vmm_dst, vmm_dst, 0x0); // Round near
                        }
                        store(vmm_dst, reg_dst, tail_num);
                        add(reg_dst, tail_num * jcp_.dst_data_size);
                        add(reg_src_aux, tail_num * jcp_.src_data_size);
                    }
                }
                add(reg_weights, jcp_.filterLenY * sizeof(float));
            }
        }
    }

    void nn_planar() {
        Xbyak::Reg64 reg_index_h = reg_src_aux1;
        Xbyak::Reg64 reg_index_w = reg_src_aux2;
        mov(reg_index_h, reg_index);
        // reg_index represent reg_index_w
        add(reg_index, jcp_.OH * jcp_.indices_size);
        // bk for reset to reg_index_w
        mov(reg_index_w, reg_index);

        Xbyak::Label out_loop_label;
        Xbyak::Label out_loop_end;

        Xbyak::Reg64 reg_work_amount_oh = rdi;
        mov(reg_work_amount_oh, jcp_.OH);
        L(out_loop_label);
        {
            // outloop status
            cmp(reg_work_amount_oh, 1);
            jl(out_loop_end, T_NEAR);

            //reset work_amount to OW
            mov(reg_work_amount, jcp_.OW);

            Xbyak::Reg64 reg_src_h = rsi;
            mov(reg_src_h, reg_src);
            // index_h * IW * dataSize done when built to avoid redundent compute
            mov(reg_index_offset, dword[reg_index_h]);
            add(reg_src_h, reg_index_offset);  // reg_src_h now point to begin of row

            // reset index_w, index_w * dataSize done when built to avoid redundent compute
            mov(reg_index, reg_index_w);

            Xbyak::Label nn_loop_label;
            Xbyak::Label nn_loop_end_label;
            Xbyak::Label nn_tail_loop_label;
            Xbyak::Label nn_tail_loop_end_label;

            L(nn_loop_label);   // inner loop
            {
                cmp(reg_work_amount, vector_step);
                jl(nn_loop_end_label, T_NEAR);

                uni_vmovdqu(vmm_index, ptr[reg_index]);
                uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
                vgatherdps(vmm_val, ptr[reg_src_h + vmm_index], vmm_mask);
                if (attr_.post_ops_.len() != 0)
                    apply_post_ops(jcp_.dst_prc, 1);
                store(vmm_val, reg_dst, vector_step);

                add(reg_dst, vector_step * jcp_.dst_data_size);
                add(reg_index, vector_step * jcp_.indices_size);
                sub(reg_work_amount, vector_step);

                jmp(nn_loop_label, T_NEAR);
            }
            L(nn_loop_end_label);

            L(nn_tail_loop_label);
            {
                cmp(reg_work_amount, 1);
                jl(nn_tail_loop_end_label, T_NEAR);

                mov(reg_src_aux, reg_src_h);
                mov(reg_index_offset, dword[reg_index]);
                add(reg_src_aux, reg_index_offset);

                load(reg_src_aux, vmm_val, scalar_step);
                if (attr_.post_ops_.len() != 0)
                    apply_post_ops(jcp_.dst_prc, 1);
                store(vmm_val, reg_dst, scalar_step);

                add(reg_dst, scalar_step * jcp_.dst_data_size);
                add(reg_index, scalar_step * jcp_.indices_size);
                sub(reg_work_amount, scalar_step);

                jmp(nn_tail_loop_label, T_NEAR);
            }
            L(nn_tail_loop_end_label);    // inner loop end

            //increment index_h to next row
            add(reg_index_h, jcp_.indices_size);

            sub(reg_work_amount_oh, 1);
            jmp(out_loop_label, T_NEAR);
        }
        L(out_loop_end);
    }

    void nn_blk() {
        Xbyak::Label nn_loop_label;
        Xbyak::Label nn_loop_end_label;
        L(nn_loop_label);
        {
            cmp(reg_work_amount, 0);
            jle(nn_loop_end_label, T_NEAR);

            mov(reg_src_aux, reg_src);
            mov(reg_index_offset, dword[reg_index]);
            add(reg_src_aux, reg_index_offset);

            load(reg_src_aux, vmm_val, vector_step);
            if (attr_.post_ops_.len() != 0)
                apply_post_ops(jcp_.dst_prc, 0);
            store(vmm_val, reg_dst, vector_step);
            add(reg_dst, vector_step * jcp_.dst_data_size);

            if (isa == cpu::x64::sse41) {
                add(reg_src_aux, vector_step * jcp_.src_data_size);
                load(reg_src_aux, vmm_val, vector_step);
                if (attr_.post_ops_.len() != 0) {
                    add(reg_oc_off, vector_step * sizeof(float));
                    apply_post_ops(jcp_.dst_prc, 0);
                    sub(reg_oc_off, vector_step * sizeof(float));
                }
                store(vmm_val, reg_dst, vector_step);
                add(reg_dst, vector_step * jcp_.dst_data_size);
            }

            add(reg_index, jcp_.indices_size);
            sub(reg_work_amount, 1);

            jmp(nn_loop_label, T_NEAR);
        }
        L(nn_loop_end_label);
    }

    void nn_by_channel() {
        // kernel for C * OW
        Xbyak::Label out_loop_label;
        Xbyak::Label out_loop_end;

        Xbyak::Reg64 reg_work_amount_bk = reg_src_aux2;
        Xbyak::Reg64 reg_oc_off_bk = rsi;
        mov(reg_work_amount_bk, ptr[reg_params + GET_OFF(work_amount)]);
        if (attr_.post_ops_.len() != 0) {
            mov(reg_oc_off_bk, ptr[reg_params + GET_OFF(oc_off)]);
        }

        Xbyak::Reg64 reg_work_amount_out = reg_src_aux1;
        mov(reg_work_amount_out, jcp_.OW);
        L(out_loop_label);
        {
            cmp(reg_work_amount_out, 1);
            jl(out_loop_end, T_NEAR);

            //inner loop for C
            Xbyak::Label nn_loop_label;
            Xbyak::Label nn_loop_end_label;
            Xbyak::Label nn_tail_loop_label;
            Xbyak::Label nn_tail_loop_end_label;

            // inner loop for C
            // get current loop address reg_src_aux, from reg_src which is unchange, point this C * OW.
            // reset offset and work_amount.
            // dst and index address is continous, advanced each interator.
            mov(reg_src_aux, reg_src);
            // index*C*dataSize done when built to avoid redundent compute
            mov(reg_index_offset, dword[reg_index]);
            add(reg_src_aux, reg_index_offset);

            mov(reg_work_amount, reg_work_amount_bk);
            if (attr_.post_ops_.len() != 0)
                mov(reg_oc_off, reg_oc_off_bk);

            L(nn_loop_label);
            {
                cmp(reg_work_amount, vector_step);
                jl(nn_loop_end_label, T_NEAR);

                load(reg_src_aux, vmm_val, vector_step);
                if (attr_.post_ops_.len() != 0)
                    apply_post_ops(jcp_.dst_prc, 0);
                store(vmm_val, reg_dst, vector_step);

                add(reg_dst, vector_step * jcp_.dst_data_size);
                add(reg_src_aux, vector_step * jcp_.src_data_size);
                add(reg_oc_off, vector_step * sizeof(float));
                sub(reg_work_amount, vector_step);

                jmp(nn_loop_label, T_NEAR);
            }
            L(nn_loop_end_label);

            if (tail_step != 0) {
                load(reg_src_aux, vmm_val, tail_step);
                if (attr_.post_ops_.len() != 0)
                    apply_post_ops(jcp_.dst_prc, 0);
                store(vmm_val, reg_dst, tail_step);

                // check to remove below
                add(reg_dst, tail_step * jcp_.dst_data_size);
                add(reg_src_aux, tail_step * jcp_.src_data_size);
                add(reg_oc_off, tail_step * sizeof(float));
                sub(reg_work_amount, tail_step);
            }
            add(reg_index, jcp_.indices_size);
            sub(reg_work_amount_out, 1);
            jmp(out_loop_label, T_NEAR);
        }
        L(out_loop_end);
    }

    void linear_onnx_c_gathered() {
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        // load weight
        mov(reg_src, ptr[reg_params + GET_OFF(weight_ptr[0])]);
        mov(reg_src_aux, ptr[reg_params + GET_OFF(weight_ptr[0]) + sizeof(size_t)]);
        uni_vbroadcastss(vmm_weightL, ptr[reg_src]);
        uni_vbroadcastss(vmm_weightR, ptr[reg_src_aux]);
        if (jcp_.spatial_dim_size > 1) {
            mov(reg_src_aux1, ptr[reg_params + GET_OFF(weight_ptr[0]) + 2 * sizeof(size_t)]);
            mov(reg_src_aux2, ptr[reg_params + GET_OFF(weight_ptr[0]) + 3 * sizeof(size_t)]);
            uni_vbroadcastss(vmm_weightT, ptr[reg_src_aux1]);
            uni_vbroadcastss(vmm_weightB, ptr[reg_src_aux2]);
        }
        if (jcp_.spatial_dim_size > 2) {
            mov(reg_src, ptr[reg_params + GET_OFF(weight_ptr[0]) + 4 * sizeof(size_t)]);
            mov(reg_src_aux, ptr[reg_params + GET_OFF(weight_ptr[0]) + 5 * sizeof(size_t)]);
            uni_vbroadcastss(vmm_weightF, ptr[reg_src]);
            uni_vbroadcastss(vmm_weightE, ptr[reg_src_aux]);
        }
        // load src
        mov(reg_src, ptr[reg_params + GET_OFF(src_ptr[0])]);
        mov(reg_src_aux, ptr[reg_params + GET_OFF(src_ptr[0]) + sizeof(size_t)]);
        if (jcp_.spatial_dim_size > 1) {
            mov(reg_src_aux1, ptr[reg_params + GET_OFF(src_ptr[0]) + 2 * sizeof(size_t)]);
            mov(reg_src_aux2, ptr[reg_params + GET_OFF(src_ptr[0]) + 3 * sizeof(size_t)]);
        }
        Xbyak::Reg64 reg_src_aux4 = r14;
        Xbyak::Reg64 reg_src_aux5 = rdx;
        Xbyak::Reg64 reg_src_aux6 = rsi;
        Xbyak::Reg64 reg_src_aux7 = rbp;
        if (jcp_.spatial_dim_size > 2) {
            mov(reg_src_aux4, ptr[reg_params + GET_OFF(src_ptr[0]) + 4 * sizeof(size_t)]);
            mov(reg_src_aux5, ptr[reg_params + GET_OFF(src_ptr[0]) + 5 * sizeof(size_t)]);
            mov(reg_src_aux6, ptr[reg_params + GET_OFF(src_ptr[0]) + 6 * sizeof(size_t)]);
            mov(reg_src_aux7, ptr[reg_params + GET_OFF(src_ptr[0]) + 7 * sizeof(size_t)]);
        }
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);

        int blk = (isa == cpu::x64::sse41) ? (2 * vector_step) : vector_step;
        int dst_stride = (jcp_.layout == InterpolateLayoutType::by_channel) ? (vector_step * jcp_.dst_data_size) :
                                            (blk * jcp_.OW * jcp_.OH * jcp_.OD * jcp_.dst_data_size);
        int src_stride = (jcp_.layout == InterpolateLayoutType::by_channel) ? (vector_step * jcp_.src_data_size) :
                                            (blk * jcp_.IW * jcp_.IH * jcp_.ID * jcp_.src_data_size);

        Xbyak::Label main_loop_label;
        Xbyak::Label main_loop_end_label;
        Xbyak::Label blk_tail_loop_label;
        Xbyak::Label blk_tail_loop_end_label;
        Xbyak::Label tail_loop_label;
        Xbyak::Label tail_loop_end_label;
        L(main_loop_label);
        {
            if (jcp_.layout == InterpolateLayoutType::by_channel) {
                cmp(reg_work_amount, vector_step);
                jl(main_loop_end_label, T_NEAR);
            } else {
                cmp(reg_work_amount, 1);
                jl(main_loop_end_label, T_NEAR);
            }
            // progressive manner
            load(reg_src, vmm_valTL, vector_step);
            load(reg_src_aux, vmm_valTR, vector_step);
            if (jcp_.spatial_dim_size == 1) {
                linear_onnx_worker_1d();
            }
            if (jcp_.spatial_dim_size > 1) {
                load(reg_src_aux1, vmm_valBL, vector_step);
                load(reg_src_aux2, vmm_valBR, vector_step);
                linear_onnx_worker_2d();
            }
            if (jcp_.spatial_dim_size > 2) {
                uni_vmovups(vmm_d_bias, vmm_valTR);  // temporally save front result to temp_vmm
                load(reg_src_aux4, vmm_valTL, vector_step);
                load(reg_src_aux5, vmm_valTR, vector_step);
                load(reg_src_aux6, vmm_valBL, vector_step);
                load(reg_src_aux7, vmm_valBR, vector_step);

                // 2d for end depth
                linear_onnx_worker_2d();
                // 3th dimension
                uni_vmulps(vmm_valTR, vmm_valTR, vmm_weightE); // end_value * end_weight
                uni_vfmadd231ps(vmm_valTR, vmm_d_bias, vmm_weightF); // start_value * start_weight + end_value * end_weight
            }

            if (attr_.post_ops_.len() != 0) {
                apply_post_ops(jcp_.dst_prc, false);  // vmm_val is vmm_valTR
                add(reg_oc_off, vector_step * sizeof(float));
            }
            store(vmm_valTR, reg_dst, vector_step);

            if ((isa == cpu::x64::sse41) && (jcp_.layout == InterpolateLayoutType::block)) {
                int offset_src = vector_step * jcp_.src_data_size;
                load(reg_src, vmm_valTL, vector_step, offset_src);
                load(reg_src_aux, vmm_valTR, vector_step, offset_src);
                if (jcp_.spatial_dim_size == 1) {
                    linear_onnx_worker_1d();
                }
                if (jcp_.spatial_dim_size > 1) {
                    load(reg_src_aux1, vmm_valBL, vector_step, offset_src);
                    load(reg_src_aux2, vmm_valBR, vector_step, offset_src);
                    linear_onnx_worker_2d();
                }
                if (jcp_.spatial_dim_size > 2) {
                    uni_vmovups(vmm_d_bias, vmm_valTR);  // temporally save front result to temp_vmm
                    load(reg_src_aux4, vmm_valTL, vector_step, offset_src);
                    load(reg_src_aux5, vmm_valTR, vector_step, offset_src);
                    load(reg_src_aux6, vmm_valBL, vector_step, offset_src);
                    load(reg_src_aux7, vmm_valBR, vector_step, offset_src);
                    // 2d for end depth
                    linear_onnx_worker_2d();
                    // 3th dimension
                    uni_vmulps(vmm_valTR, vmm_valTR, vmm_weightE); // end_value * end_weight
                    uni_vfmadd231ps(vmm_valTR, vmm_d_bias, vmm_weightF); // start_value * start_weight + end_value * end_weight
                }

                if (attr_.post_ops_.len() != 0) {
                    apply_post_ops(jcp_.dst_prc, false);
                    add(reg_oc_off, vector_step * sizeof(float));
                }
                int offset_dst = vector_step * jcp_.dst_data_size;
                store(vmm_valTR, reg_dst, vector_step, offset_dst);
            }
            add(reg_dst, dst_stride);
            add(reg_src, src_stride);
            add(reg_src_aux, src_stride);
            if (jcp_.spatial_dim_size > 1) {
                add(reg_src_aux1, src_stride);
                add(reg_src_aux2, src_stride);
            }
            if (jcp_.spatial_dim_size > 2) {
                add(reg_src_aux4, src_stride);
                add(reg_src_aux5, src_stride);
                add(reg_src_aux6, src_stride);
                add(reg_src_aux7, src_stride);
            }
            if (jcp_.layout == InterpolateLayoutType::by_channel) {
                sub(reg_work_amount, vector_step);    // work_amount is c
            } else {
                sub(reg_work_amount, 1);       // work_amount = div_up(c, blk), no tails
            }

            jmp(main_loop_label, T_NEAR);
        }
        L(main_loop_end_label);

        if ((jcp_.layout == InterpolateLayoutType::by_channel) && (tail_step != 0)) {
            load(reg_src, vmm_valTL, tail_step);
            load(reg_src_aux, vmm_valTR, tail_step);
            if (jcp_.spatial_dim_size == 1) {
                linear_onnx_worker_1d();
            }
            if (jcp_.spatial_dim_size > 1) {
                load(reg_src_aux1, vmm_valBL, tail_step);
                load(reg_src_aux2, vmm_valBR, tail_step);
                linear_onnx_worker_2d();
            }
            if (jcp_.spatial_dim_size > 2) {
                uni_vmovups(vmm_d_bias, vmm_valTR);  // temporally save front result to temp_vmm

                load(reg_src_aux4, vmm_valTL, tail_step);
                load(reg_src_aux5, vmm_valTR, tail_step);
                load(reg_src_aux6, vmm_valBL, tail_step);
                load(reg_src_aux7, vmm_valBR, tail_step);
                // 2d for end depth
                linear_onnx_worker_2d();
                // 3th dimension
                uni_vmulps(vmm_valTR, vmm_valTR, vmm_weightE); // end_value * end_weight
                uni_vfmadd231ps(vmm_valTR, vmm_d_bias, vmm_weightF); // start_value * start_weight + end_value * end_weight
            }

            if (attr_.post_ops_.len() != 0) {
                apply_post_ops(jcp_.dst_prc, false);  // vmm_val is vmm_valTR
                add(reg_oc_off, tail_step * sizeof(float));
            }

            store(vmm_valTR, reg_dst, tail_step);
        }
    }

    void linear_onnx_planar() {
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_src, ptr[reg_params + GET_OFF(src_ptr[0])]);
        mov(reg_index, ptr[reg_params + GET_OFF(index)]);
        mov(reg_src_aux, ptr[reg_params + GET_OFF(weight_ptr[0])]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);

        int index_stride = jcp_.OW * jcp_.OH * jcp_.OD * jcp_.indices_size;
        int weight_stride = jcp_.OW * jcp_.OH * jcp_.OD * sizeof(float);

        Xbyak::Label main_loop_label;
        Xbyak::Label main_loop_end_label;
        Xbyak::Label tail_loop_label;
        Xbyak::Label tail_loop_end_label;
        L(main_loop_label);
        {
            cmp(reg_work_amount, vector_step);
            jl(main_loop_end_label, T_NEAR);

            uni_vmovdqu(vmm_index, ptr[reg_index]);
            uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
            vgatherdps(vmm_valTL, ptr[reg_src + vmm_index], vmm_mask);

            uni_vmovdqu(vmm_index, ptr[reg_index + index_stride]);
            uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
            vgatherdps(vmm_valTR, ptr[reg_src + vmm_index], vmm_mask);

            load_weights(reg_src_aux, vmm_weightL, vector_step);
            load_weights(reg_src_aux, vmm_weightR, vector_step, weight_stride);

            // progressive manner
            if (jcp_.spatial_dim_size == 1) {
                linear_onnx_worker_1d();
            }
            if (jcp_.spatial_dim_size > 1) {
                uni_vmovdqu(vmm_index, ptr[reg_index + 2 * index_stride]);
                uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
                vgatherdps(vmm_valBL, ptr[reg_src + vmm_index], vmm_mask);

                uni_vmovdqu(vmm_index, ptr[reg_index + 3 * index_stride]);
                uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
                vgatherdps(vmm_valBR, ptr[reg_src + vmm_index], vmm_mask);

                load_weights(reg_src_aux, vmm_weightT, vector_step, 2 * weight_stride);
                load_weights(reg_src_aux, vmm_weightB, vector_step, 3 * weight_stride);

                linear_onnx_worker_2d();
            }
            if (jcp_.spatial_dim_size > 2) {
                uni_vmovups(vmm_d_bias, vmm_valTR);  // temporally save front result to temp_vmm

                // for end depth
                uni_vmovdqu(vmm_index, ptr[reg_index + 4 * index_stride]);
                uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
                vgatherdps(vmm_valTL, ptr[reg_src + vmm_index], vmm_mask);

                uni_vmovdqu(vmm_index, ptr[reg_index + 5 * index_stride]);
                uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
                vgatherdps(vmm_valTR, ptr[reg_src + vmm_index], vmm_mask);

                uni_vmovdqu(vmm_index, ptr[reg_index + 6 * index_stride]);
                uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
                vgatherdps(vmm_valBL, ptr[reg_src + vmm_index], vmm_mask);

                uni_vmovdqu(vmm_index, ptr[reg_index + 7 * index_stride]);
                uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
                vgatherdps(vmm_valBR, ptr[reg_src + vmm_index], vmm_mask);

                linear_onnx_worker_2d();

                load_weights(reg_src_aux, vmm_weightE, vector_step, 5 * weight_stride);
                load_weights(reg_src_aux, vmm_weightF, vector_step, 4 * weight_stride);

                uni_vmulps(vmm_valTR, vmm_valTR, vmm_weightE); // end_value * end_weight
                uni_vfmadd231ps(vmm_valTR, vmm_d_bias, vmm_weightF); // start_value * start_weight + end_value * end_weight
            }

            if (attr_.post_ops_.len() != 0) {
                apply_post_ops(jcp_.dst_prc, true);  // vmm_val is vmm_valTR, broadcase is true
            }
            store(vmm_valTR, reg_dst, vector_step);

            add(reg_dst, vector_step * jcp_.dst_data_size);
            add(reg_src_aux, vector_step * sizeof(float));
            add(reg_index, vector_step * jcp_.indices_size);
            sub(reg_work_amount, vector_step);

            jmp(main_loop_label, T_NEAR);
        }
        L(main_loop_end_label);

        L(tail_loop_label);
        {
            cmp(reg_work_amount, 1);
            jl(tail_loop_end_label, T_NEAR);

            mov(reg_src_aux1, reg_src);
            mov(reg_index_offset, dword[reg_index]);
            add(reg_src_aux1, reg_index_offset);
            load(reg_src_aux1, vmm_valTL, scalar_step);

            mov(reg_src_aux1, reg_src);
            mov(reg_index_offset, dword[reg_index + index_stride]);
            add(reg_src_aux1, reg_index_offset);
            load(reg_src_aux1, vmm_valTR, scalar_step);

            load_weights(reg_src_aux, vmm_weightL, scalar_step, 0);
            load_weights(reg_src_aux, vmm_weightR, scalar_step, weight_stride);

            if (jcp_.spatial_dim_size == 1) {
                linear_onnx_worker_1d();
            }
            if (jcp_.spatial_dim_size > 1) {
                mov(reg_src_aux1, reg_src);
                mov(reg_index_offset, dword[reg_index + 2 * index_stride]);
                add(reg_src_aux1, reg_index_offset);
                load(reg_src_aux1, vmm_valBL, scalar_step);

                mov(reg_src_aux1, reg_src);
                mov(reg_index_offset, dword[reg_index + 3 * index_stride]);
                add(reg_src_aux1, reg_index_offset);
                load(reg_src_aux1, vmm_valBR, scalar_step);

                load_weights(reg_src_aux, vmm_weightT, scalar_step, 2 * weight_stride);
                load_weights(reg_src_aux, vmm_weightB, scalar_step, 3 * weight_stride);

                linear_onnx_worker_2d();
            }
            if (jcp_.spatial_dim_size > 2) {
                uni_vmovups(vmm_d_bias, vmm_valTR);  // save from front result to temp_vmm

                // for end depth
                mov(reg_src_aux1, reg_src);
                mov(reg_index_offset, dword[reg_index + 4 * index_stride]);
                add(reg_src_aux1, reg_index_offset);
                load(reg_src_aux1, vmm_valTL, scalar_step);

                mov(reg_src_aux1, reg_src);
                mov(reg_index_offset, dword[reg_index + 5 * index_stride]);
                add(reg_src_aux1, reg_index_offset);
                load(reg_src_aux1, vmm_valTR, scalar_step);

                mov(reg_src_aux1, reg_src);
                mov(reg_index_offset, dword[reg_index + 6 * index_stride]);
                add(reg_src_aux1, reg_index_offset);
                load(reg_src_aux1, vmm_valBL, scalar_step);

                mov(reg_src_aux1, reg_src);
                mov(reg_index_offset, dword[reg_index + 7 * index_stride]);
                add(reg_src_aux1, reg_index_offset);
                load(reg_src_aux1, vmm_valBR, scalar_step);

                linear_onnx_worker_2d();

                load_weights(reg_src_aux, vmm_weightE, scalar_step, 5 * weight_stride);
                load_weights(reg_src_aux, vmm_weightF, scalar_step, 4 * weight_stride);

                uni_vmulps(vmm_valTR, vmm_valTR, vmm_weightE); // end_value * end_weight
                uni_vfmadd231ps(vmm_valTR, vmm_d_bias, vmm_weightF); // start_value * start_weight + end_value * end_weight
            }

            if (attr_.post_ops_.len() != 0) {
                apply_post_ops(jcp_.dst_prc, true);  // process on vmm_val, vmm_val is vmm_valTR, and bc
            }
            store(vmm_valTR, reg_dst, scalar_step);

            add(reg_dst, scalar_step * jcp_.dst_data_size);
            add(reg_src_aux, scalar_step * sizeof(float));
            add(reg_index, scalar_step * jcp_.indices_size);
            sub(reg_work_amount, scalar_step);

            jmp(tail_loop_label, T_NEAR);
        }
        L(tail_loop_end_label);
    }

    inline void linear_onnx_worker_1d() {
        uni_vmulps(vmm_valTR, vmm_valTR, vmm_weightR);
        uni_vfmadd231ps(vmm_valTR, vmm_valTL, vmm_weightL);
    }

    // weightT * (srcTL * weightL + srcTR * weightR) +
    // weightB * (srcBL * weightL + srcBR * weightR)
    inline void linear_onnx_worker_2d() {
        uni_vmulps(vmm_valTR, vmm_valTR, vmm_weightR);
        uni_vmulps(vmm_valBR, vmm_valBR, vmm_weightR);
        uni_vfmadd231ps(vmm_valTR, vmm_valTL, vmm_weightL);
        uni_vfmadd231ps(vmm_valBR, vmm_valBL, vmm_weightL);
        uni_vmulps(vmm_valTR, vmm_valTR, vmm_weightT);
        uni_vfmadd231ps(vmm_valTR, vmm_valBR, vmm_weightB);
    }

    void cubic_c_gathered() {
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_src, ptr[reg_params + GET_OFF(src_ptr[0])]);
        mov(reg_index, ptr[reg_params + GET_OFF(index)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);

        // weight_ptr[0] point to weightX
        mov(reg_src_aux1, ptr[reg_params + GET_OFF(weight_ptr[0])]);
        uni_vbroadcastss(vmm_weightX0, ptr[reg_src_aux1]);
        uni_vbroadcastss(vmm_weightX1, ptr[reg_src_aux1 + 1 * sizeof(float)]);
        uni_vbroadcastss(vmm_weightX2, ptr[reg_src_aux1 + 2 * sizeof(float)]);
        uni_vbroadcastss(vmm_weightX3, ptr[reg_src_aux1 + 3 * sizeof(float)]);

        // weight_ptr[1] point to weightY
        mov(reg_src_aux1, ptr[reg_params + GET_OFF(weight_ptr[0]) + sizeof(size_t)]);
        uni_vbroadcastss(vmm_weightY0, ptr[reg_src_aux1]);
        uni_vbroadcastss(vmm_weightY1, ptr[reg_src_aux1 + 1 * sizeof(float)]);
        uni_vbroadcastss(vmm_weightY2, ptr[reg_src_aux1 + 2 * sizeof(float)]);
        uni_vbroadcastss(vmm_weightY3, ptr[reg_src_aux1 + 3 * sizeof(float)]);

        int blk = (isa == cpu::x64::sse41) ? (2 * vector_step) : vector_step;

        Xbyak::Label main_loop_label;
        Xbyak::Label main_loop_end_label;
        Xbyak::Label tail_loop_label;
        Xbyak::Label tail_loop_end_label;
        L(main_loop_label);
        {
            if (jcp_.layout == InterpolateLayoutType::by_channel) {
                cmp(reg_work_amount, vector_step);
                jl(main_loop_end_label, T_NEAR);
            } else {
                cmp(reg_work_amount, 1);
                jl(tail_loop_end_label, T_NEAR);
            }

            uni_vpxor(vmm_val, vmm_val, vmm_val);

            cubic_c_gathered_matrix(false);

            if (attr_.post_ops_.len() != 0) {
                apply_post_ops(jcp_.dst_prc, false);     // vmm_val is default dst value to post_ops and store
                add(reg_oc_off, vector_step * sizeof(float));
            }
            store(vmm_val, reg_dst, vector_step);

            if ((isa == cpu::x64::sse41) && (jcp_.layout == InterpolateLayoutType::block)) {
                // vmm is xmm here
                add(reg_src, vector_step * jcp_.src_data_size);
                add(reg_dst, vector_step * jcp_.dst_data_size);

                uni_vpxor(vmm_val, vmm_val, vmm_val);

                cubic_c_gathered_matrix(false);

                if (attr_.post_ops_.len() != 0) {
                    apply_post_ops(jcp_.dst_prc, false);
                    add(reg_oc_off, vector_step * sizeof(float));  // second vector_step for one blk
                }
                store(vmm_val, reg_dst, vector_step);

                sub(reg_src, vector_step * jcp_.src_data_size);
                sub(reg_dst, vector_step * jcp_.dst_data_size);
            }
            if (jcp_.layout == InterpolateLayoutType::by_channel) {
                int dst_stride = vector_step * jcp_.dst_data_size;
                int src_stride = vector_step * jcp_.src_data_size;
                add(reg_dst, dst_stride);
                add(reg_src, src_stride);
                sub(reg_work_amount, vector_step);    // work_amount is c
            } else {
                int dst_stride = blk * jcp_.OW * jcp_.OH * jcp_.dst_data_size;
                int src_stride = blk * jcp_.IW * jcp_.IH * jcp_.src_data_size;
                add(reg_dst, dst_stride);
                add(reg_src, src_stride);
                sub(reg_work_amount, 1);  // work_amount = div_up(c, blk), no tails
            }

            jmp(main_loop_label, T_NEAR);
        }
        L(main_loop_end_label);

        // only for by_channel layout for tails.
        L(tail_loop_label);
        {
            cmp(reg_work_amount, 1);
            jl(tail_loop_end_label, T_NEAR);

            // store final computed value
            uni_vpxor(vmm_val, vmm_val, vmm_val);

            cubic_c_gathered_matrix(true);

            if (attr_.post_ops_.len() != 0) {
                apply_post_ops(jcp_.dst_prc, false);     // vmm_val is default dst value
                add(reg_oc_off, scalar_step * sizeof(float));
            }
            store(vmm_val, reg_dst, scalar_step);

            int dst_stride = scalar_step * jcp_.dst_data_size;
            int src_stride = scalar_step * jcp_.src_data_size;
            add(reg_dst, dst_stride);
            add(reg_src, src_stride);
            sub(reg_work_amount, scalar_step);    // work_amount is c

            jmp(tail_loop_label, T_NEAR);
        }
        L(tail_loop_end_label);
    }

    inline void cubic_c_gathered_matrix(bool is_scalar) {
        // y0:  (x0 * weightX0 + x1 * weightX1 + x2 * weightX2 + x3 * weightX3) * weightY0
        cubic_c_gathered_line(0, vmm_weightY0, is_scalar);
        // y1
        cubic_c_gathered_line(4, vmm_weightY1, is_scalar);
        // y2
        cubic_c_gathered_line(8, vmm_weightY2, is_scalar);
        // y3
        cubic_c_gathered_line(12, vmm_weightY3, is_scalar);
    }

    inline void cubic_c_gathered_line(int index_start, Vmm vmm_weight, bool is_scalar) {
        uni_vpxor(vmm_dstX, vmm_dstX, vmm_dstX);
        cubic_c_gathered_pixel(index_start, vmm_weightX0, is_scalar);
        cubic_c_gathered_pixel(index_start + 1, vmm_weightX1, is_scalar);
        cubic_c_gathered_pixel(index_start + 2, vmm_weightX2, is_scalar);
        cubic_c_gathered_pixel(index_start + 3, vmm_weightX3, is_scalar);
        uni_vfmadd231ps(vmm_val, vmm_dstX, vmm_weight);
    }

    inline void cubic_c_gathered_pixel(int i, Vmm vmm_weight, bool is_scalar) {
        mov(reg_src_aux, reg_src);
        mov(reg_index_offset, dword[reg_index + i * jcp_.indices_size]);
        add(reg_src_aux, reg_index_offset);
        int step = is_scalar ? 1 : vlen / sizeof(float);
        load(reg_src_aux, vmm_src, step);
        uni_vfmadd231ps(vmm_dstX, vmm_src, vmm_weight);
    }

    void cubic_planar() {
        mov(reg_table, l_table_constant);
        // src_ptr[2] for oh sequence, src_ptr[3] for ow sequence
        mov(reg_tbl_y, ptr[reg_params + GET_OFF(src_ptr[0]) + 2 * sizeof(size_t)]);
        mov(reg_tbl_x, ptr[reg_params + GET_OFF(src_ptr[0]) + 3 * sizeof(size_t)]);
        uni_vmovdqu(vmm_one, cubic_planar_table_val(0));
        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_src, ptr[reg_params + GET_OFF(src_ptr[0])]);
        // index_OW
        mov(reg_index, ptr[reg_params + GET_OFF(index)]);
        // index_OH from src_ptr[1]
        Xbyak::Reg64 reg_index_y = reg_src_aux;
        mov(reg_index_y, ptr[reg_params + GET_OFF(src_ptr[0]) + sizeof(size_t)]);
        // weight_OW
        Xbyak::Reg64 reg_weight_x = reg_src_aux1;
        mov(reg_weight_x, ptr[reg_params + GET_OFF(weight_ptr[0])]);
        // weight_OH
        Xbyak::Reg64 reg_weight_y = reg_src_aux2;
        mov(reg_weight_y, ptr[reg_params + GET_OFF(weight_ptr[0]) + sizeof(size_t)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);

        int grid_len = 4;

        // 0   1   2   3   4   5   6   7   8   9   10   11   12   13   14   15   16   17   18   19
        // 20  21  22  23  24  25  26  27  28  29  30   31   32   33   34   35   36   37   38   39
        // for 3th step(8): 16  17  18  19  20  21  22  23
        //               y: 0   0   0   0   1   1   1   1
        //               x: 16  17  18  19  0   1   2   3

        Xbyak::Label main_loop_label;
        Xbyak::Label main_loop_end_label;
        Xbyak::Label tail_loop_label;
        Xbyak::Label tail_loop_end_label;
        L(main_loop_label);
        {
            cmp(reg_work_amount, vector_step);
            jl(main_loop_end_label, T_NEAR);

            // vmm_tbl_y: (0 0 0 0 1 1 1 1 * index_size) --> (0 0 0 0 4 4 4 4)
            uni_vmovdqu(vmm_tbl_y, ptr[reg_tbl_y]);
            uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
            // vmm_index_in_y: 0 0 0 0 2 2 2 2
            vpgatherdd(vmm_index_in_y, ptr[reg_index_y + vmm_tbl_y], vmm_mask);

            // use vmm_val temporally for value in reg_tbl_x: 16  17  18  19  0   1   2   3
            uni_vmovdqu(vmm_val, ptr[reg_tbl_x]);
            uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
            // e.g. vmm_index_in_x: 32 34 36 38 0 2 4 6, now save src index.
            vpgatherdd(vmm_index_in_x, ptr[reg_index + vmm_val], vmm_mask);

            // build weightX used in y0-y3
            // weight format: w0_0 w1_0 w2_0 w3_0 w0_1 w1_1 w2_1 w3_1 ...
            uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
            vgatherdps(vmm_weightX0, ptr[reg_weight_x + vmm_val * grid_len], vmm_mask);  // 4 in vmm_val for weight_size, another 4 for grid_len

            uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
            // shift weight_size then gather second weight
            vgatherdps(vmm_weightX1, ptr[reg_weight_x + sizeof(float) + (vmm_val * grid_len)], vmm_mask);

            uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
            vgatherdps(vmm_weightX2, ptr[reg_weight_x + 2 * sizeof(float) + (vmm_val * grid_len)], vmm_mask);

            uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
            vgatherdps(vmm_weightX3, ptr[reg_weight_x + 3 * sizeof(float) + (vmm_val * grid_len)], vmm_mask);
            // vmm_val is now relieved and used for dst_value

            uni_vpxor(vmm_val, vmm_val, vmm_val);
            // y0
            vpsubd(vmm_index_y_itr, vmm_index_in_y, vmm_one);
            // crop to [0, IH - 1]
            vpminsd(vmm_index_y_itr, vmm_index_y_itr, cubic_planar_table_val(1));
            vpmaxsd(vmm_index_y_itr, vmm_index_y_itr, vmm_zero);

            // weight y0
            uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
            vgatherdps(vmm_weightY, ptr[reg_weight_y + (vmm_tbl_y * grid_len)], vmm_mask);
            cubic_planar_line(false);

            // y1
            // crop to [0, IH - 1]
            vpminsd(vmm_index_y_itr, vmm_index_in_y, cubic_planar_table_val(1));
            vpmaxsd(vmm_index_y_itr, vmm_index_y_itr, vmm_zero);
            // weight y1: shift weight_size
            uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
            vgatherdps(vmm_weightY, ptr[reg_weight_y + sizeof(float) + (vmm_tbl_y * grid_len)], vmm_mask);
            cubic_planar_line(false);

            // y2
            vpaddd(vmm_index_y_itr, vmm_index_in_y, vmm_one);
            // crop to [0, IH - 1]
            vpminsd(vmm_index_y_itr, vmm_index_y_itr, cubic_planar_table_val(1));
            vpmaxsd(vmm_index_y_itr, vmm_index_y_itr, vmm_zero);
            // weight y2
            uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
            vgatherdps(vmm_weightY, ptr[reg_weight_y + 2 * sizeof(float) + (vmm_tbl_y * grid_len)], vmm_mask);
            cubic_planar_line(false);

            // y3
            vpaddd(vmm_index_y_itr, vmm_index_in_y, vmm_one);
            vpaddd(vmm_index_y_itr, vmm_index_y_itr, vmm_one);
            // crop to [0, IH - 1]
            vpminsd(vmm_index_y_itr, vmm_index_y_itr, cubic_planar_table_val(1));
            vpmaxsd(vmm_index_y_itr, vmm_index_y_itr, vmm_zero);
            // weight y3
            uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
            vgatherdps(vmm_weightY, ptr[reg_weight_y + 3 * sizeof(float) + (vmm_tbl_y * grid_len)], vmm_mask);
            cubic_planar_line(false);

            if (attr_.post_ops_.len() != 0) {
                apply_post_ops(jcp_.dst_prc, true);  // oc_off is broadcast and always the same value for this channel
            }
            store(vmm_val, reg_dst, vector_step);

            add(reg_tbl_y, vector_step * sizeof(int));  // sizeof(int): sequence by dd()
            add(reg_tbl_x, vector_step * sizeof(int));
            add(reg_dst, vector_step * jcp_.dst_data_size);

            sub(reg_work_amount, vector_step);

            jmp(main_loop_label, T_NEAR);
        }
        L(main_loop_end_label);

        L(tail_loop_label);
        {
            cmp(reg_work_amount, 1);
            jl(tail_loop_end_label, T_NEAR);

            // get idx for input
            uni_vmovss(Xmm(vmm_tbl_y.getIdx()), ptr[reg_tbl_y]);
            gather_i32_indices(vmm_index_in_y, reg_index_y, 0, vmm_tbl_y, 1, ov::element::i32, true);

            uni_vmovss(Xmm(vmm_val.getIdx()), ptr[reg_tbl_x]);
            gather_i32_indices(vmm_index_in_x, reg_index, 0, vmm_val, 1, ov::element::i32, true);
            // gather weightX by input idx, used in y0-y3
            gather_i32_indices(vmm_weightX0, reg_weight_x, 0, vmm_val, grid_len, ov::element::f32, true);
            gather_i32_indices(vmm_weightX1, reg_weight_x, sizeof(float), vmm_val, grid_len, ov::element::f32, true);
            gather_i32_indices(vmm_weightX2, reg_weight_x, 2 * sizeof(float), vmm_val, grid_len, ov::element::f32, true);
            gather_i32_indices(vmm_weightX3, reg_weight_x, 3 * sizeof(float), vmm_val, grid_len, ov::element::f32, true);
            // vmm_val is now relieved and used for dst_value

            uni_vpxor(vmm_val, vmm_val, vmm_val);
            // y0
            vpsubd(vmm_index_y_itr, vmm_index_in_y, vmm_one);
            // crop to [0, IH - 1]
            vpminsd(vmm_index_y_itr, vmm_index_y_itr, cubic_planar_table_val(1));
            vpmaxsd(vmm_index_y_itr, vmm_index_y_itr, vmm_zero);

            gather_i32_indices(vmm_weightY, reg_weight_y, 0, vmm_tbl_y, grid_len, ov::element::f32, true);
            cubic_planar_line(true);

            // y1
            // crop to [0, IH - 1]
            vpminsd(vmm_index_y_itr, vmm_index_in_y, cubic_planar_table_val(1));
            vpmaxsd(vmm_index_y_itr, vmm_index_y_itr, vmm_zero);
            // weight y1: shift weight_size
            gather_i32_indices(vmm_weightY, reg_weight_y, sizeof(float), vmm_tbl_y, grid_len, ov::element::f32, true);
            cubic_planar_line(true);

            // y2
            vpaddd(vmm_index_y_itr, vmm_index_in_y, vmm_one);
            // crop to [0, IH - 1]
            vpminsd(vmm_index_y_itr, vmm_index_y_itr, cubic_planar_table_val(1));
            vpmaxsd(vmm_index_y_itr, vmm_index_y_itr, vmm_zero);
            // weight y2
            gather_i32_indices(vmm_weightY, reg_weight_y, 2 * sizeof(float), vmm_tbl_y, grid_len, ov::element::f32, true);
            cubic_planar_line(true);

            // y3
            vpaddd(vmm_index_y_itr, vmm_index_in_y, vmm_one);
            vpaddd(vmm_index_y_itr, vmm_index_y_itr, vmm_one);
            // crop to [0, IH - 1]
            vpminsd(vmm_index_y_itr, vmm_index_y_itr, cubic_planar_table_val(1));
            vpmaxsd(vmm_index_y_itr, vmm_index_y_itr, vmm_zero);
            // weight y3
            gather_i32_indices(vmm_weightY, reg_weight_y, 3 * sizeof(float), vmm_tbl_y, grid_len, ov::element::f32, true);
            cubic_planar_line(true);

            if (attr_.post_ops_.len() != 0) {
                apply_post_ops(jcp_.dst_prc, true);  // oc_off is broadcast and always the same value for this channel
            }
            store(vmm_val, reg_dst, scalar_step);

            add(reg_tbl_y, scalar_step * sizeof(int));  // sizeof(int): sequence with dd()
            add(reg_tbl_x, scalar_step * sizeof(int));
            add(reg_dst, scalar_step * jcp_.dst_data_size);

            sub(reg_work_amount, scalar_step);

            jmp(tail_loop_label, T_NEAR);
        }
        L(tail_loop_end_label);
    }

    inline void cubic_planar_line(bool is_scalar) {
        uni_vpxor(vmm_dstX, vmm_dstX, vmm_dstX);
        cubic_planar_pixel(0, is_scalar);
        cubic_planar_pixel(1, is_scalar);
        cubic_planar_pixel(2, is_scalar);
        cubic_planar_pixel(3, is_scalar);
        uni_vfmadd231ps(vmm_val, vmm_dstX, vmm_weightY);
    }

    inline void cubic_planar_pixel(int itr, bool is_scalar) {
        // vmm_index_in_x have index for src
        if (itr == 0) {
            vpsubd(vmm_index_x_itr, vmm_index_in_x, vmm_one);
        } else if (itr == 1) {
            vpaddd(vmm_index_x_itr, vmm_index_in_x, vmm_zero);
        } else if (itr == 2) {
            vpaddd(vmm_index_x_itr, vmm_index_in_x, vmm_one);
        } else if (itr == 3) {
            vpaddd(vmm_index_x_itr, vmm_index_in_x, vmm_one);
            vpaddd(vmm_index_x_itr, vmm_index_x_itr, vmm_one);
        }

        // crop to [0, IW - 1]
        vpminsd(vmm_index_x_itr, vmm_index_x_itr, cubic_planar_table_val(2));
        vpmaxsd(vmm_index_x_itr, vmm_index_x_itr, vmm_zero);

        // value
        // index is: ptr[reg_src + (vmm_index_y_itr * jcp_.IW + vmm_index_x_itr) * jcp_.src_data_size]
        uni_vmovdqu(vmm_mask, cubic_planar_table_val(2));
        vpaddd(vmm_mask, vmm_mask, vmm_one);  // (IW - 1) + 1 = IW
        uni_vpmulld(vmm_mask, vmm_mask, vmm_index_y_itr);
        uni_vpaddd(vmm_index_x_itr, vmm_index_x_itr, vmm_mask);
        gather_i32_indices(vmm_src, reg_src, 0, vmm_index_x_itr, jcp_.src_data_size, ov::element::f32, is_scalar);

        if (itr == 0) {
            uni_vfmadd231ps(vmm_dstX, vmm_src, vmm_weightX0);
        } else if (itr == 1) {
            uni_vfmadd231ps(vmm_dstX, vmm_src, vmm_weightX1);
        } else if (itr == 2) {
            uni_vfmadd231ps(vmm_dstX, vmm_src, vmm_weightX2);
        } else if (itr == 3) {
            uni_vfmadd231ps(vmm_dstX, vmm_src, vmm_weightX3);
        }
    }

    inline void prepare_cubic_planar_table() {
        auto broadcast_int = [&](int val) {
            for (size_t d = 0; d < vlen / sizeof(int); ++d) {
                dd(val);
            }
        };

        align(64);
        L(l_table_constant);
        broadcast_int(vals_for_cubic_planar.int_one);
        broadcast_int(jcp_.IH - 1);
        broadcast_int(jcp_.IW - 1);
        dd(vals_for_cubic_planar.mask_gather_avx512);
    }

    struct vals_for_cubic_planar_type {
        int int_one = 0x00000001;
        int mask_gather_avx512 = 0x0000ffff;  // 00000000000000001111111111111111
    } vals_for_cubic_planar;

    inline Xbyak::Address cubic_planar_table_val(int index) {
        return ptr[reg_table + index * vlen];
    }

    // always gather to Vmm, compute with Vmm, store with Xmm if scalar_step
    inline void gather_i32_indices(Vmm vmm_src, const Xbyak::Reg64 &base, int offset, Vmm vmm_indices, int scale,
                                ov::element::Type src_prc, bool is_scalar) {
        Xbyak::Address table_idx = ptr[base + offset + vmm_indices * scale];
        if ((isa == cpu::x64::avx512_core) && !is_scalar) {
            // [0-15] bit of int to mask
            kmovw(k_mask, cubic_planar_table_val(3));
            if (src_prc == ov::element::f32) {
                vgatherdps(vmm_src | k_mask, table_idx);  // dword index, packed single data
            } else if (src_prc == ov::element::i32) {
                vpgatherdd(vmm_src | k_mask, table_idx);  // dword index, dword data
            }
        } else if ((isa == cpu::x64::avx2) && !is_scalar) {
            uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
            if (src_prc == ov::element::f32) {
                vgatherdps(vmm_src, table_idx, vmm_mask);
            } else if (src_prc == ov::element::i32) {
                vpgatherdd(vmm_src, table_idx, vmm_mask);
            }
        } else {
            const int gpr_size = 8;
            sub(rsp, gpr_size);
            // move content in register to content in address(ptr[])
            mov(ptr[rsp], reg_tmp_64);

            // replace index with value in stack
            sub(rsp, vlen);
            uni_vmovdqu(ptr[rsp], vmm_indices);

            int repeats = is_scalar ? 1 : vlen / sizeof(float);
            for (int i = 0; i < repeats; ++i) {
                mov(reg_tmp_64.cvt32(), ptr[rsp + i * sizeof(int)]);       // sizeof(int)  index_size
                table_idx = ptr[base + offset + reg_tmp_64 * scale];       // scale: sizeof(float)   value_size
                mov(reg_tmp_64.cvt32(), table_idx);
                mov(ptr[rsp + i * sizeof(int)], reg_tmp_64.cvt32());
            }

            uni_vmovups(vmm_src, ptr[rsp]);
            add(rsp, vlen);
            // restore GPR state
            mov(reg_tmp_64, ptr[rsp]);
            add(rsp, gpr_size);
        }
    }

    // is_broadcast for broadcasting param for depth_wise and quantize(channel-sensitive post-ops), for fusion with plain layout.
    void apply_post_ops(ov::element::Type dst_prc, bool is_broadcast) {
        const auto &p = attr_.post_ops_;
        int eltwise_inj_idx = 0;
        int depthwise_inj_idx = 0;
        int quantization_inj_idx = 0;
        int post_ops_data_offset = 0;
        for (int i = 0; i < p.len(); i++) {
            auto& post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors[eltwise_inj_idx]->compute_vector_range(vmm_val.getIdx(), vmm_val.getIdx() + 1);
                eltwise_inj_idx++;
            } else if (post_op.is_depthwise()) {
                mov(reg_d_weights, ptr[reg_post_ops_data + post_ops_data_offset]);
                add(reg_d_weights, reg_oc_off);

                // weight and bias is padded. scalar as vector.
                depthwise_injectors[depthwise_inj_idx]->compute_vector_range(
                        vmm_val.getIdx(), vmm_val.getIdx() + 1, reg_d_weights, reg_d_weights, is_broadcast);

                post_ops_data_offset += depthwise_injectors[depthwise_inj_idx]->memoryStep();
                depthwise_inj_idx++;
            } else if (post_op.is_quantization()) {
                bool do_dequantization = post_op.quantization.alg == alg_kind::quantization_quantize_dequantize;
                bool do_rounding = do_dequantization || dst_prc == ov::element::f32 || i != p.len() - 1;

                int s_idx = vmm_val.getIdx();

                quantization_injectors[quantization_inj_idx]->init_crop_ptrs(reg_post_ops_data + post_ops_data_offset, reg_oc_off);
                quantization_injectors[quantization_inj_idx]->compute_crop(s_idx, s_idx + 1, 0, 0, is_broadcast);

                quantization_injectors[quantization_inj_idx]->init_input_scale_shift_ptrs(reg_post_ops_data + post_ops_data_offset, reg_oc_off);
                quantization_injectors[quantization_inj_idx]->compute_input_scale_shift(s_idx, s_idx + 1, 0, do_rounding, 0, is_broadcast);

                if (do_dequantization) {
                    quantization_injectors[quantization_inj_idx]->init_output_scale_shift_ptrs(reg_post_ops_data + post_ops_data_offset, reg_oc_off);
                    quantization_injectors[quantization_inj_idx]->compute_output_scale_shift(s_idx, s_idx + 1, 0, 0, is_broadcast);
                }

                post_ops_data_offset += quantization_injectors[quantization_inj_idx]->memoryStep();
                quantization_inj_idx++;
            }
        }
    }
};

#endif // OPENVINO_ARCH_X86_64

namespace {
struct InterpolateKey {
    InterpolateAttrs nodeAttrs;
    VectorDims srcDims;
    VectorDims dstDims;
    std::vector<float> dataScales;
    dnnl::primitive_attr attr;

    size_t hash() const;
    bool operator==(const InterpolateKey& rhs) const;
};

size_t InterpolateKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;

    seed = hash_combine(seed, nodeAttrs.mode);
    seed = hash_combine(seed, nodeAttrs.coordTransMode);
    seed = hash_combine(seed, nodeAttrs.nearestMode);
    seed = hash_combine(seed, nodeAttrs.layout);

    seed = hash_combine(seed, nodeAttrs.antialias);
    seed = hash_combine(seed, nodeAttrs.cubeCoeff);

    seed = get_vector_hash(seed, nodeAttrs.padBegin);
    seed = get_vector_hash(seed, nodeAttrs.padEnd);

    seed = hash_combine(seed, nodeAttrs.inPrc.hash());
    seed = hash_combine(seed, nodeAttrs.outPrc.hash());

    seed = get_vector_hash(seed, srcDims);
    seed = get_vector_hash(seed, dstDims);
    seed = get_vector_hash(seed, dataScales);

    seed = hash_combine(seed, get_attr_hash(*attr.get()));
    return seed;
}

bool InterpolateKey::operator==(const InterpolateKey &rhs) const {
    if (nodeAttrs.mode != rhs.nodeAttrs.mode)
        return false;
    if (nodeAttrs.coordTransMode != rhs.nodeAttrs.coordTransMode)
        return false;
    if (nodeAttrs.nearestMode != rhs.nodeAttrs.nearestMode)
        return false;
    if (nodeAttrs.layout != rhs.nodeAttrs.layout)
        return false;
    if (nodeAttrs.antialias != rhs.nodeAttrs.antialias)
        return false;
    if (nodeAttrs.cubeCoeff != rhs.nodeAttrs.cubeCoeff)
        return false;
    if (nodeAttrs.padBegin != rhs.nodeAttrs.padBegin)
        return false;
    if (nodeAttrs.padEnd != rhs.nodeAttrs.padEnd)
        return false;
    if (nodeAttrs.inPrc != rhs.nodeAttrs.inPrc)
        return false;
    if (nodeAttrs.outPrc != rhs.nodeAttrs.outPrc)
        return false;

    if (srcDims != rhs.srcDims)
        return false;
    if (dstDims != rhs.dstDims)
        return false;
    if (dataScales != rhs.dataScales)
        return false;
    if (!(*attr.get() == *rhs.attr.get()))
        return false;

    return true;
}

} // namespace

// shapeND: n     c     d     h    w
// blockND: ncdhw cdhw  dhw   hw   w    1
// index  : 0      1    2     3    4    5
inline VectorDims getBlockND(const VectorDims& shape) {
    int shapeRank = shape.size();
    VectorDims blockND(shapeRank + 1, 1);
    for (int i = shapeRank - 1; i >= 0; i--) {
        blockND[i] = shape[i] * blockND[i+1];
    }
    return blockND;
}
// w/hw/ncw/nchw/ncdhw to ncdhw
inline VectorDims to5Dim(VectorDims casesDim) {
    size_t caseSize = casesDim.size();
    VectorDims dim5(5, 1lu);
    dim5[4] = casesDim[caseSize - 1];
    if (caseSize > 1) {
        dim5[3] = casesDim[caseSize - 2];
    }
    if (caseSize > 2) {
        dim5[0] = casesDim[0];
    }
    if (caseSize > 3) {
        dim5[1] = casesDim[1];
    }
    if (caseSize > 4) {
        dim5[2] = casesDim[2];
    }
    if (caseSize == 3) {  // nhw -> ncw
        dim5[1] = dim5[3];
        dim5[3] = 1lu;
    }
    return dim5;
}

using ngInterpMode = ov::op::v4::Interpolate::InterpolateMode;
using ngInterpCoordTransf = ov::op::v4::Interpolate::CoordinateTransformMode;
using ngInterpNearMode = ov::op::v4::Interpolate::NearestMode;
using ngInterpShapeCalcMode = ov::op::v4::Interpolate::ShapeCalcMode;

bool Interpolate::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (const auto interp = std::dynamic_pointer_cast<const ov::op::v4::Interpolate>(op)) {
            const auto &interpAttr = interp->get_attrs();
            const auto &interpMode = interpAttr.mode;
            if (!one_of(interpMode, ngInterpMode::NEAREST, ngInterpMode::LINEAR, ngInterpMode::LINEAR_ONNX, ngInterpMode::CUBIC)) {
                errorMessage = "Interpolate-4 does not support interpolate mode: " + ov::as_string(interpMode);
                return false;
            }

            const auto &interpCoordTransMode = interpAttr.coordinate_transformation_mode;
            if (!one_of(interpCoordTransMode, ngInterpCoordTransf::HALF_PIXEL, ngInterpCoordTransf::PYTORCH_HALF_PIXEL, ngInterpCoordTransf::ASYMMETRIC,
                                              ngInterpCoordTransf::TF_HALF_PIXEL_FOR_NN, ngInterpCoordTransf::ALIGN_CORNERS)) {
                errorMessage = "Interpolate-4 does not support coordinate transformation mode: " + ov::as_string(interpCoordTransMode);
                return false;
            }

            if (interpMode == ngInterpMode::NEAREST) {
                const auto &interpNearestMode = interpAttr.nearest_mode;
                if (!one_of(interpNearestMode, ngInterpNearMode::ROUND_PREFER_FLOOR, ngInterpNearMode::ROUND_PREFER_CEIL, ngInterpNearMode::FLOOR,
                                               ngInterpNearMode::CEIL, ngInterpNearMode::SIMPLE)) {
                    errorMessage = "Interpolate-4 does not support nearest round mode: " + ov::as_string(interpNearestMode);
                    return false;
                }
            }

            const auto &interpShapeCalcMode = interpAttr.shape_calculation_mode;
            if (!one_of(interpShapeCalcMode, ngInterpShapeCalcMode::SCALES, ngInterpShapeCalcMode::SIZES)) {
                errorMessage = "Interpolate-4 does not support shape_calculation_mode: " + ov::as_string(interpShapeCalcMode);
                return false;
            }

            const size_t dataRank = interp->get_input_partial_shape(DATA_ID).rank().get_length();
            if (dataRank < 1 || dataRank > 5) {
                errorMessage = "Interpolate-4 does not support input tensor of rank : " + std::to_string(dataRank);
                return false;
            }

            if (dataRank == 5 && interpMode == ngInterpMode::CUBIC) {
                errorMessage = "Interpolate-4 doesn't support input tensor with rank: " + std::to_string(dataRank) + " for 'cubic' mode ";
                return false;
            }

            if (!isDynamicNgraphNode(op) && interpShapeCalcMode == ngInterpShapeCalcMode::SCALES &&
                !ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(SCALES_ID))) {
                errorMessage = "Only const 'scales' input is supported for static shapes in Interpolate-4";
                return false;
            }

            if (interp->get_input_size() > 3 &&
                std::dynamic_pointer_cast<const ov::op::v0::Constant>(interp->get_input_node_shared_ptr(AXES_ID)) == nullptr) {
                errorMessage = "Only const 'axes' input is supported in Interpolate-4";
                return false;
            }
        } else if (const auto interp = std::dynamic_pointer_cast<const ov::op::v11::Interpolate>(op)) {
            const auto &interpAttr = interp->get_attrs();
            const auto &interpMode = interpAttr.mode;
            if (!one_of(interpMode, ngInterpMode::BILINEAR_PILLOW, ngInterpMode::BICUBIC_PILLOW)) {
                errorMessage = "Interpolate-11 does not support interpolate mode: " + ov::as_string(interpMode);
                return false;
            }
            const auto &interpShapeCalcMode = interpAttr.shape_calculation_mode;
            if (!one_of(interpShapeCalcMode, ngInterpShapeCalcMode::SCALES, ngInterpShapeCalcMode::SIZES)) {
                errorMessage = "Interpolate-11 does not support shape_calculation_mode: " + ov::as_string(interpShapeCalcMode);
                return false;
            }
            const size_t dataRank = interp->get_input_partial_shape(DATA_ID).rank().get_length();
            if (dataRank < 2 || dataRank > 4) {
                // pillow only resize on H and W. resize on D(depth) is not defined.
                errorMessage = "Interpolate-11 does not support input tensor of rank : " + std::to_string(dataRank);
                return false;
            }
            if (!isDynamicNgraphNode(op) &&
                    !ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(SIZE_OR_SCALE_ID_V11))) {
                errorMessage = "Only const 'scales_or_sizes' input is supported for static shapes in Interpolate-11";
                return false;
            }
            if (interp->get_input_size() > 2 &&
                std::dynamic_pointer_cast<const ov::op::v0::Constant>(interp->get_input_node_shared_ptr(AXES_ID_V11)) == nullptr) {
                errorMessage = "Only const 'axes' input is supported in Interpolate-11";
                return false;
            }
        } else {
            errorMessage = "Only opset4 and opset11 interpolate operation are supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

namespace {
/**
 * Interpolate shape inference factory. It defines the input mask depending on the shape calculation mode.
 *
 */
class InterpolateShapeInferFactory : public ShapeInferFactory {
public:
    InterpolateShapeInferFactory(std::shared_ptr<ov::Node> op) : m_op(op) {}
    ShapeInferPtr makeShapeInfer() const override {
        IShapeInfer::port_mask_t port_mask = 0x00;
        if (auto interp4 = ov::as_type_ptr<ov::opset4::Interpolate>(m_op)) {
            const auto &attr = interp4->get_attrs();

            if (attr.shape_calculation_mode == ngInterpShapeCalcMode::SCALES) {
                port_mask = PortMask(Interpolate::SCALES_ID, Interpolate::AXES_ID);
            } else if (attr.shape_calculation_mode == ngInterpShapeCalcMode::SIZES) {
                port_mask = PortMask(Interpolate::TARGET_SHAPE_ID, Interpolate::AXES_ID);
            } else {
                OPENVINO_ASSERT(false, "Unsupported interpolate shape calculation mode");
            }
        } else if (auto interp11 = ov::as_type_ptr<ov::op::v11::Interpolate>(m_op)) {
            port_mask = PortMask(Interpolate::SIZE_OR_SCALE_ID_V11, Interpolate::AXES_ID_V11);
        } else {
            OPENVINO_THROW("Shape infer factory cannot be created for ",
                           m_op->get_type_name(),
                           " node with name: ",
                           m_op->get_friendly_name(),
                           ", only versions 4 and 11 are supported.");
        }
        return std::make_shared<NgraphShapeInfer>(make_shape_inference(m_op), port_mask);
    }

private:
    std::shared_ptr<ov::Node> m_op;
};
} // namespace

Interpolate::Interpolate(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
        : Node(op, context, InterpolateShapeInferFactory(op)) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "Interpolate node with name '" + getName() + "'";
        dataRank = getInputShapeAtPort(DATA_ID).getRank();
        if (const auto interp = std::dynamic_pointer_cast<const ov::opset4::Interpolate>(op)) {
            is_version11 = false;
            const auto numInputs = inputShapes.size();
            if (numInputs != 3 && numInputs != 4)
                OPENVINO_THROW(errorPrefix, " has incorrect number of input edges");
            if (outputShapes.size() != 1)
                OPENVINO_THROW(errorPrefix, " has incorrect number of output edges");
            isAxesSpecified = numInputs != 3;

            const auto &interpAttr = interp->get_attrs();

            const auto &interpMode = interpAttr.mode;
            if (interpMode == ngInterpMode::NEAREST) {
                interpAttrs.mode = InterpolateMode::nearest;
            } else if (interpMode == ngInterpMode::LINEAR) {
                if (dataRank < 5) {
                    interpAttrs.mode = InterpolateMode::linear_onnx;
                } else {
                    interpAttrs.mode = InterpolateMode::linear;
                }
            } else if (interpMode == ngInterpMode::LINEAR_ONNX) {
                interpAttrs.mode = InterpolateMode::linear_onnx;
            } else if (interpMode == ngInterpMode::CUBIC) {
                interpAttrs.mode = InterpolateMode::cubic;
            } else {
                OPENVINO_THROW(errorPrefix, " has unsupported interpolate mode");
            }

            const auto &interpCoordTransMode = interpAttr.coordinate_transformation_mode;
            if (interpCoordTransMode == ngInterpCoordTransf::HALF_PIXEL) {
                interpAttrs.coordTransMode = InterpolateCoordTransMode::half_pixel;
            } else if (interpCoordTransMode == ngInterpCoordTransf::PYTORCH_HALF_PIXEL) {
                interpAttrs.coordTransMode = InterpolateCoordTransMode::pytorch_half_pixel;
            } else if (interpCoordTransMode == ngInterpCoordTransf::ASYMMETRIC) {
                interpAttrs.coordTransMode = InterpolateCoordTransMode::asymmetric;
            } else if (interpCoordTransMode == ngInterpCoordTransf::TF_HALF_PIXEL_FOR_NN) {
                interpAttrs.coordTransMode = InterpolateCoordTransMode::tf_half_pixel_for_nn;
            } else if (interpCoordTransMode == ngInterpCoordTransf::ALIGN_CORNERS) {
                interpAttrs.coordTransMode = InterpolateCoordTransMode::align_corners;
            } else {
                OPENVINO_THROW(errorPrefix, " has unsupported coordination transformation mode");
            }

            if (interpAttrs.mode == InterpolateMode::nearest) {
                const auto &interpNearestMode = interpAttr.nearest_mode;
                if (interpNearestMode == ngInterpNearMode::ROUND_PREFER_FLOOR) {
                    interpAttrs.nearestMode = InterpolateNearestMode::round_prefer_floor;
                } else if (interpNearestMode == ngInterpNearMode::ROUND_PREFER_CEIL) {
                    interpAttrs.nearestMode = InterpolateNearestMode::round_prefer_ceil;
                } else if (interpNearestMode == ngInterpNearMode::FLOOR) {
                    interpAttrs.nearestMode = InterpolateNearestMode::floor;
                } else if (interpNearestMode == ngInterpNearMode::CEIL) {
                    interpAttrs.nearestMode = InterpolateNearestMode::ceil;
                } else if (interpNearestMode == ngInterpNearMode::SIMPLE) {
                    interpAttrs.nearestMode = InterpolateNearestMode::simple;
                } else {
                    OPENVINO_THROW(errorPrefix, " has unsupported nearest mode");
                }
            } else if (interpAttrs.mode == InterpolateMode::cubic) {
                interpAttrs.cubeCoeff = static_cast<float>(interpAttr.cube_coeff);
            }
            interpAttrs.antialias = interpAttr.antialias;

            const auto &interpShapeCalcMode = interpAttr.shape_calculation_mode;
            if (interpShapeCalcMode == ngInterpShapeCalcMode::SCALES) {
                interpAttrs.shapeCalcMode = InterpolateShapeCalcMode::scales;
            } else if (interpShapeCalcMode == ngInterpShapeCalcMode::SIZES) {
                interpAttrs.shapeCalcMode = InterpolateShapeCalcMode::sizes;
            } else {
                OPENVINO_THROW(errorPrefix, " has unsupported shape calculation mode");
            }

            if (interpAttr.pads_begin.empty()) {
                interpAttrs.padBegin.resize(dataRank, 0);
            } else {
                interpAttrs.padBegin.resize(interpAttr.pads_begin.size());
                for (size_t i = 0; i < interpAttr.pads_begin.size(); i++)
                    interpAttrs.padBegin[i] = static_cast<int>(interpAttr.pads_begin[i]);
            }

            if (interpAttr.pads_end.empty()) {
                interpAttrs.padEnd.resize(dataRank, 0);
            } else {
                interpAttrs.padEnd.resize(interpAttr.pads_end.size());
                for (size_t i = 0; i < interpAttr.pads_end.size(); i++)
                    interpAttrs.padEnd[i] = static_cast<int>(interpAttr.pads_end[i]);
            }

            const auto scalesNode = std::dynamic_pointer_cast<const ov::op::v0::Constant>(interp->get_input_node_shared_ptr(SCALES_ID));
            if (scalesNode) {
                scales = scalesNode->cast_vector<float>();
                isScaleConstant = true;
            }

            if (isAxesSpecified) {
                axes = std::dynamic_pointer_cast<const ov::op::v0::Constant>(interp->get_input_node_shared_ptr(AXES_ID))->cast_vector<int>();
            } else {
                axes.resize(dataRank);
                for (int i = 0; i < static_cast<int>(dataRank); i++) {
                    axes[i] = i;
                }
            }
        } else if (const auto interp = std::dynamic_pointer_cast<const ov::op::v11::Interpolate>(op)) {
            is_version11 = true;
            const auto numInputs = inputShapes.size();
            if (numInputs != 2 && numInputs != 3)
                OPENVINO_THROW(errorPrefix, " has incorrect number of input edges");
            if (outputShapes.size() != 1)
                OPENVINO_THROW(errorPrefix, " has incorrect number of output edges");
            isAxesSpecified = numInputs != 2;

            const auto &interpAttr = interp->get_attrs();
            const auto &interpMode = interpAttr.mode;
            if (interpMode == ngInterpMode::BILINEAR_PILLOW) {
                interpAttrs.mode = InterpolateMode::bilinear_pillow;
            } else if (interpMode == ngInterpMode::BICUBIC_PILLOW) {
                interpAttrs.mode = InterpolateMode::bicubic_pillow;
                interpAttrs.cubeCoeff = static_cast<float>(interpAttr.cube_coeff); // fixed to be -0.5
            } else {
                OPENVINO_THROW(errorPrefix, " has unsupported interpolate mode");
            }

            // pillow use fixed tf_half_pixel_for_nn style mode for coodinate transformation
            interpAttrs.coordTransMode = InterpolateCoordTransMode::tf_half_pixel_for_nn;
            interpAttrs.antialias = interpAttr.antialias;

            const auto &interpShapeCalcMode = interpAttr.shape_calculation_mode;
            if (interpShapeCalcMode == ngInterpShapeCalcMode::SCALES) {
                interpAttrs.shapeCalcMode = InterpolateShapeCalcMode::scales;
                const auto scalesNode = std::dynamic_pointer_cast<const ov::op::v0::Constant>(interp->get_input_node_shared_ptr(SIZE_OR_SCALE_ID_V11));
                if (scalesNode) {
                    scales = scalesNode->cast_vector<float>();
                    isScaleConstant = true;
                }
            } else if (interpShapeCalcMode == ngInterpShapeCalcMode::SIZES) {
                interpAttrs.shapeCalcMode = InterpolateShapeCalcMode::sizes;
            } else {
                OPENVINO_THROW(errorPrefix, " has unsupported shape calculation mode");
            }

            if (interpAttr.pads_begin.empty()) {
                interpAttrs.padBegin.resize(dataRank, 0);
            } else {
                interpAttrs.padBegin.resize(interpAttr.pads_begin.size());
                for (size_t i = 0; i < interpAttr.pads_begin.size(); i++)
                    interpAttrs.padBegin[i] = static_cast<int>(interpAttr.pads_begin[i]);
            }

            if (interpAttr.pads_end.empty()) {
                interpAttrs.padEnd.resize(dataRank, 0);
            } else {
                interpAttrs.padEnd.resize(interpAttr.pads_end.size());
                for (size_t i = 0; i < interpAttr.pads_end.size(); i++)
                    interpAttrs.padEnd[i] = static_cast<int>(interpAttr.pads_end[i]);
            }

            if (isAxesSpecified) {
                axes = std::dynamic_pointer_cast<const ov::op::v0::Constant>(interp->get_input_node_shared_ptr(AXES_ID_V11))->cast_vector<int>();
                if (dataRank == 4 && axes.size() == 2 && axes[0] == 1 && axes[1] == 2 && mayiuse(cpu::x64::sse41)) {
                    NCHWAsNHWC = true;
                    axes[0] = 2;
                    axes[1] = 3;
                }
            } else {
                axes.resize(dataRank);
                for (int i = 0; i < static_cast<int>(dataRank); i++) {
                    axes[i] = i;
                }
            }
        }
    } else {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
}

void Interpolate::getSupportedDescriptors() {
    if (getParentEdges().size() != 2 && getParentEdges().size() != 3 && getParentEdges().size() != 4)
        // v4: data, target_shape, scale, axis(optional).
        // v11: data, size_or_scale, axis(optional)
        OPENVINO_THROW(errorPrefix, " has incorrect number of input edges");
    if (getChildEdges().empty())
        OPENVINO_THROW(errorPrefix, " has incorrect number of output edges");

    // get pad
    for (size_t i = 0; i < interpAttrs.padBegin.size(); i++) {
        if (interpAttrs.padBegin[i] != 0) {
            hasPad = true;
            break;
        }
    }
    for (size_t i = 0; i < interpAttrs.padEnd.size(); i++) {
        if (interpAttrs.padEnd[i] != 0) {
            hasPad = true;
            break;
        }
    }
    //correct pad
    if (hasPad) {
        NCHWAsNHWC = false;
        auto correctPad = [&](std::vector<int> pad, int rank) {
            int padLen = pad.size();
            if (padLen == rank) {
                return pad;
            }
            std::vector<int> result;
            if (padLen > rank) {
                result.insert(result.end(), pad.begin(), pad.begin() + rank);
            } else {
                result = pad;
                result.insert(result.end(), rank - padLen, 0);
            }
            return result;
        };

        interpAttrs.padBegin = correctPad(interpAttrs.padBegin, dataRank);
        interpAttrs.padEnd = correctPad(interpAttrs.padEnd, dataRank);
    }
}

void Interpolate::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    ov::element::Type inputPrecision = getOriginalInputPrecisionAtPort(DATA_ID);

#if defined(OV_CPU_WITH_ACL)
    bool isInputPrecisionSupported = one_of(inputPrecision, ov::element::i8, ov::element::u8, ov::element::f16);
#else
    bool isInputPrecisionSupported = one_of(inputPrecision, ov::element::i8, ov::element::u8, ov::element::bf16);
#endif
    if (!isInputPrecisionSupported) {
        inputPrecision = ov::element::f32;
    }

    if (!hasHardwareSupport(inputPrecision))
        inputPrecision = ov::element::f32;

    // support input with rank<=3 only with float precision and planar layout.
    // Jit for avx2(gather is available) and ref for no-avx2 machine.
    if (!one_of(dataRank, 4u, 5u)) {
        inputPrecision = ov::element::f32;
    }
    ov::element::Type outputPrecision = inputPrecision;

    if (!fusedWith.empty()) {
        outputPrecision = fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(DATA_ID);
    }

#if !defined(OV_CPU_WITH_ACL)
    if (!mayiuse(cpu::x64::sse41)) {
        inputPrecision = outputPrecision = ov::element::f32;
    }
#endif

    auto targetShapeType = ov::element::i32;
    auto scalesType = ov::element::f32;
    auto axesType = ov::element::i32;

    NodeConfig config;
    config.outConfs.resize(1);
    if (is_version11) {
        if (isAxesSpecified) {
            config.inConfs.resize(3);
        } else {
            config.inConfs.resize(2);
        }
    } else {
        if (isAxesSpecified) {
            config.inConfs.resize(4);
        } else {
            config.inConfs.resize(3);
        }
    }
    auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    auto pushDesc = [&](LayoutType dataFormat, impl_desc_type implDetail, bool is_version11, bool useAclExecutor = false) {
        config.inConfs[DATA_ID].setMemDesc(creatorsMap.at(dataFormat)->createSharedDesc(inputPrecision, getInputShapeAtPort(DATA_ID)));
        if (is_version11) {
            if (interpAttrs.shapeCalcMode == InterpolateShapeCalcMode::sizes) {
                config.inConfs[SIZE_OR_SCALE_ID_V11].setMemDesc(
                    creatorsMap.at(LayoutType::ncsp)->createSharedDesc(targetShapeType, getInputShapeAtPort(SIZE_OR_SCALE_ID_V11)));
            } else {
                config.inConfs[SIZE_OR_SCALE_ID_V11].setMemDesc(
                    creatorsMap.at(LayoutType::ncsp)->createSharedDesc(scalesType, getInputShapeAtPort(SIZE_OR_SCALE_ID_V11)));
            }

            if (isAxesSpecified)
                config.inConfs[AXES_ID_V11].setMemDesc(
                    creatorsMap.at(LayoutType::ncsp)->createSharedDesc(axesType, getInputShapeAtPort(AXES_ID_V11)));
        } else {
            config.inConfs[TARGET_SHAPE_ID].setMemDesc(
                creatorsMap.at(LayoutType::ncsp)->createSharedDesc(targetShapeType, getInputShapeAtPort(TARGET_SHAPE_ID)));
            config.inConfs[get_scale_id()].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(scalesType, getInputShapeAtPort(get_scale_id())));

            if (isAxesSpecified)
                config.inConfs[get_axis_id()].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(axesType, getInputShapeAtPort(get_axis_id())));
        }

        config.outConfs[0].setMemDesc(creatorsMap.at(dataFormat)->createSharedDesc(outputPrecision, getOutputShapeAtPort(0)));

        if (useAclExecutor) {
            std::vector<MemoryDescPtr> srcMemoryDescs;
            for (size_t i = 0; i < config.inConfs.size(); i++) {
                srcMemoryDescs.push_back(config.inConfs[i].getMemDesc());
            }
            std::vector<MemoryDescPtr> dstMemoryDescs;
            for (size_t i = 0; i < config.outConfs.size(); i++) {
                dstMemoryDescs.push_back(config.outConfs[i].getMemDesc());
            }

            auto factory = std::make_shared<InterpolateExecutorFactory>(interpAttrs, srcMemoryDescs, dstMemoryDescs,
                                                                    std::make_shared<ExecutorContext>(context, getImplPriority()));
            if (!factory->isEmpty()) {
                supportedPrimitiveDescriptors.push_back({config, implDetail, factory});
            }
        } else {
            supportedPrimitiveDescriptors.push_back({config, implDetail});
        }
    };
    if (is_version11) {
#if defined (OV_CPU_WITH_ACL)
        interpAttrs.hasPad = hasPad;
        pushDesc(LayoutType::nspc, undef, true, true);
        pushDesc(LayoutType::ncsp, undef, true, true);
        canUseAclExecutor = !supportedPrimitiveDescriptors.empty();
        if (canUseAclExecutor)
            return;
        //fallback to f32 if ref is used
        inputPrecision = outputPrecision = ov::element::f32;
#endif

        if (dataRank == 4) {
            if (mayiuse(cpu::x64::avx512_core)) {
                if (NCHWAsNHWC)
                    pushDesc(LayoutType::ncsp, jit_avx512, true);
                else
                    pushDesc(LayoutType::nspc, jit_avx512, true);
            } else if (mayiuse(cpu::x64::avx2)) {
                if (NCHWAsNHWC)
                    pushDesc(LayoutType::ncsp, jit_avx2, true);
                else
                    pushDesc(LayoutType::nspc, jit_avx2, true);
            } else if (mayiuse(cpu::x64::sse41)) {
                if (NCHWAsNHWC)
                    pushDesc(LayoutType::ncsp, jit_sse42, true);
                else
                    pushDesc(LayoutType::nspc, jit_sse42, true);
            }
        }
        pushDesc(LayoutType::ncsp, ref, true);
    } else {
        const auto &dataMinDims = getInputShapeAtPort(DATA_ID).getMinDims();
        bool isBlkApplied = dataRank > 1 && dataMinDims[1] != Shape::UNDEFINED_DIM && dataMinDims[1] > 1;

#if defined (OV_CPU_WITH_ACL)
        interpAttrs.hasPad = hasPad;
        pushDesc(LayoutType::nspc, undef, false, true);
        pushDesc(LayoutType::ncsp, undef, false, true);
        canUseAclExecutor = !supportedPrimitiveDescriptors.empty();
        if (canUseAclExecutor)
            return;
        //fallback to f32 if ref is used
        inputPrecision = outputPrecision = ov::element::f32;
#endif

        if (!mayiuse(cpu::x64::sse41) || interpAttrs.mode == InterpolateMode::linear) {
            pushDesc(LayoutType::ncsp, ref, false);
        } else {
            // blk and by_channel JIT kernel on sse41 or above machine
            if (dataRank == 4 || (dataRank == 5 && interpAttrs.mode != InterpolateMode::cubic)) {
                if (mayiuse(cpu::x64::avx512_core)) {
                    pushDesc(LayoutType::nspc, jit_avx512, false);
                    if (isBlkApplied)
                        pushDesc(LayoutType::nCsp16c, jit_avx512, false);
                } else if (mayiuse(cpu::x64::avx2)) {
                    pushDesc(LayoutType::nspc, jit_avx2, false);
                    if (isBlkApplied)
                        pushDesc(LayoutType::nCsp8c, jit_avx2, false);
                } else {
                    pushDesc(LayoutType::nspc, jit_sse42, false);
                    if (isBlkApplied)
                        pushDesc(LayoutType::nCsp8c, jit_sse42, false);
                }
            }

            // planar is only for float precision.
            // 1.ref on machine w/o avx2(no fuse)
            // 2.JIT kernel for avx2(gatherps is available).(with fuse)
            if (inputPrecision == ov::element::f32) {
                if (mayiuse(cpu::x64::avx2))
                    pushDesc(LayoutType::ncsp, jit_avx2, false);
                else
                    pushDesc(LayoutType::ncsp, ref, false);
            }
        }
    }
}

bool Interpolate::needShapeInfer() const {
    if (Node::inputShapesModified()) {
        return true;
    }
    if (interpAttrs.shapeCalcMode == InterpolateShapeCalcMode::scales) {
        if (lastScales.empty()) {
            return true;
        }
        const float *scales = getSrcDataAtPortAs<const float>(get_scale_id());
        for (size_t i = 0; i < lastScales.size(); i++) {
            if (lastScales[i] != scales[i]) {
                return true;
            }
        }
    } else {
        if (lastSizes.empty()) {
            return true;
        }
        const int32_t *sizes = getSrcDataAtPortAs<const int32_t>(TARGET_SHAPE_ID);
        for (size_t i = 0; i < lastSizes.size(); i++) {
            if (sizes[i] != lastSizes[i]) {
                return true;
            }
        }
    }
    return false;
}

void Interpolate::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);

    const size_t port = interpAttrs.shapeCalcMode == InterpolateShapeCalcMode::sizes ? TARGET_SHAPE_ID : get_scale_id();
    const auto &memory = getParentEdgeAt(port)->getMemory();
    if (interpAttrs.shapeCalcMode == InterpolateShapeCalcMode::scales) {
        const float *scales = memory.getDataAs<const float>();
        lastScales.assign(scales, scales + memory.getDesc().getShape().getElementsCount());
    } else {
        const int32_t *sizes = memory.getDataAs<const int32_t>();
        lastSizes.assign(sizes, sizes + memory.getDesc().getShape().getElementsCount());
    }
}

bool Interpolate::needPrepareParams() const {
    return (inputShapesModified() || lastOutputDims != getChildEdgeAt(0)->getMemory().getStaticDims());
}

inline int Interpolate::get_scale_id() const {
    if (is_version11)
        return SIZE_OR_SCALE_ID_V11;
    else
        return SCALES_ID;
}
inline int Interpolate::get_axis_id() const {
    if (is_version11)
        return AXES_ID_V11;
    else
        return AXES_ID;
}

void Interpolate::prepareParams() {
    if (!shapesDefined()) {
        OPENVINO_THROW("Can't prepare params for Interpolate node with name: ",
                       getName(),
                       ", because input/output dims aren't defined");
    }

    auto dstMemPtr = getDstMemoryAtPort(0);
    if (!dstMemPtr || !dstMemPtr->isDefined())
        OPENVINO_THROW(errorPrefix, " has undefined destination memory");

    auto srcMemPtr = getSrcMemoryAtPort(DATA_ID);
    if (!srcMemPtr || !srcMemPtr->isDefined())
        OPENVINO_THROW(errorPrefix, " has undefined input memory");

    if (interpAttrs.shapeCalcMode == InterpolateShapeCalcMode::sizes) {
        auto tsMemPtr = getSrcMemoryAtPort(TARGET_SHAPE_ID);
        if (!tsMemPtr || !tsMemPtr->isDefined())
            OPENVINO_THROW(errorPrefix, " has undefined target shape memory");
    } else {
        auto scaleMemPtr = getSrcMemoryAtPort(get_scale_id());
        if (!scaleMemPtr || !scaleMemPtr->isDefined())
            OPENVINO_THROW(errorPrefix, " has undefined scales memory");
    }

    if (isAxesSpecified) {
        auto axesMemPtr = getSrcMemoryAtPort(get_axis_id());
        if (!axesMemPtr || !axesMemPtr->isDefined())
            OPENVINO_THROW(errorPrefix, " has undefined axes memory");
    }

    const NodeDesc *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        OPENVINO_THROW(errorPrefix, " did not set preferable primitive descriptor");

    const auto &srcDimsOrign = srcMemPtr->getStaticDims();
    const auto &dstDimsOrign = dstMemPtr->getStaticDims();

    VectorDims srcDims = srcDimsOrign;
    VectorDims dstDims = dstDimsOrign;

    // layoutAlignment
    if (NCHWAsNHWC && srcMemPtr->getDesc().hasLayoutType(LayoutType::ncsp)) {
        auto logicalShapeAlign = [] (VectorDims& Dims) {
            size_t C = Dims[3];
            Dims[3] = Dims[2];
            Dims[2] = Dims[1];
            Dims[1] = C;
        };
        logicalShapeAlign(srcDims);
        logicalShapeAlign(dstDims);
        interpAttrs.layout = InterpolateLayoutType::by_channel;
    }

    if (interpAttrs.shapeCalcMode == InterpolateShapeCalcMode::scales) {
        if (!isScaleConstant) {
            const auto& scalesMem = getParentEdgeAt(get_scale_id())->getMemory();
            const float* scalesData = scalesMem.getDataAs<const float>();
            scales.assign(scalesData, scalesData + scalesMem.getStaticDims()[0]);
        }
    }

    std::vector<float> dataScales = getScales(getPaddedInputShape(srcDims, interpAttrs.padBegin, interpAttrs.padEnd), dstDims);
    if (!NCHWAsNHWC && (getOutputShapeAtPort(0).getRank() > 2 && (dataScales[0] != 1.f || dataScales[1] != 1.f))) {
        OPENVINO_THROW("Interpolate layer only supports resize on spatial dimensions(depth, height and width)");
    }

    if (canUseAclExecutor) {
        interpAttrs.dataScales = dataScales;

        std::vector<MemoryDescPtr> srcMemoryDescs;
        for (size_t i = 0; i < getParentEdges().size(); i++) {
            srcMemoryDescs.push_back(getSrcMemoryAtPort(i)->getDescPtr());
        }
        std::vector<MemoryDescPtr> dstMemoryDescs;
        dstMemoryDescs.push_back(getDstMemoryAtPort(0)->getDescPtr());

        auto selectedPD = getSelectedPrimitiveDescriptor();
        aclExecPtr = selectedPD->getExecutorFactoryAs<InterpolateExecutorFactory>()->makeExecutor(interpAttrs, srcMemoryDescs, dstMemoryDescs, {});
        selectedPD->setImplementationType(aclExecPtr->getImplType());

        return;
    }

    InterpolateKey key = {interpAttrs, srcDims, dstDims, dataScales, dnnl::primitive_attr()};
    setPostOps(key.attr, dstDims);

    auto buildExecutor = [&](const InterpolateKey& key) -> std::shared_ptr<InterpolateExecutorBase> {
        std::shared_ptr<InterpolateExecutorBase> executor;
        if ((key.nodeAttrs.mode == InterpolateMode::nearest || key.nodeAttrs.mode == InterpolateMode::linear_onnx ||
            key.nodeAttrs.mode == InterpolateMode::cubic) &&
            ((key.nodeAttrs.layout != InterpolateLayoutType::planar && mayiuse(cpu::x64::sse41)) ||
                (mayiuse(cpu::x64::avx2) && key.nodeAttrs.inPrc == ov::element::f32))) {
            executor = std::make_shared<InterpolateJitExecutor>(key.nodeAttrs,
                                                               key.srcDims,
                                                               key.dstDims,
                                                               key.dataScales,
                                                               key.attr);
        } else if ((key.nodeAttrs.mode == InterpolateMode::bilinear_pillow || key.nodeAttrs.mode == InterpolateMode::bicubic_pillow) &&
            (key.nodeAttrs.layout == InterpolateLayoutType::by_channel)) {
            executor = std::make_shared<InterpolateJitExecutor>(key.nodeAttrs,
                                                               key.srcDims,
                                                               key.dstDims,
                                                               key.dataScales,
                                                               key.attr);
        } else {
            executor = std::make_shared<InterpolateRefExecutor>(key.nodeAttrs,
                                                               key.srcDims,
                                                               key.dstDims,
                                                               key.dataScales);
        }
        return executor;
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, buildExecutor);
    execPtr = result.first;

    lastOutputDims = dstDimsOrign;
}

void Interpolate::createPrimitive() {
    auto srcMemPtr = getSrcMemoryAtPort(DATA_ID);
    auto dstMemPtr = getDstMemoryAtPort(0);
    if (!srcMemPtr)
        OPENVINO_THROW(errorPrefix, " has null input memory");
    if (!dstMemPtr)
        OPENVINO_THROW(errorPrefix, " has null destination memory");

    if (dstMemPtr->getDesc().hasLayoutType(LayoutType::ncsp)) {
        interpAttrs.layout = InterpolateLayoutType::planar;
    } else if (dstMemPtr->getDesc().hasLayoutType(LayoutType::nCsp8c) ||
               dstMemPtr->getDesc().hasLayoutType(LayoutType::nCsp16c)) {
        interpAttrs.layout = InterpolateLayoutType::block;
    } else {
        interpAttrs.layout = InterpolateLayoutType::by_channel;
    }

    interpAttrs.inPrc = srcMemPtr->getDesc().getPrecision();
    interpAttrs.outPrc = dstMemPtr->getDesc().getPrecision();

    if (shapesDefined() && isExecutable()) {
        if (needPrepareParams())
            prepareParams();
        updateLastInputDims();
    }
}

inline int clipCoord(int pos, int length) {
    return std::max(static_cast<int>(0), std::min(pos, length - 1));
}

static inline float triangleCoeff(float x) {
    return (std::max)(0.0f, 1 - std::abs(x));
}

void Interpolate::setPostOps(dnnl::primitive_attr &attr, const VectorDims &dims) {
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
            eltwiseNode->appendPostOps(ops, dims, postOpsDataPtrs);
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

VectorDims Interpolate::getPaddedInputShape(const VectorDims &srcDims,
                                                      const std::vector<int> &padBegin,
                                                      const std::vector<int> &padEnd) {
    VectorDims paddedShape;
    int dataRank = srcDims.size();
    for (int i = 0; i < dataRank; i++) {
        paddedShape.push_back(srcDims[i] + padBegin[i] + padEnd[i]);
    }
    return paddedShape;
}

// get scales of data rank size
// if "scale" version: set scales with input scales, 1.f for other dims not in axis
// if "size" version: scales = shape[target] / shape[input].pad, 1.f for other dims not in axis
// scales is a required input, but should not use input scales when "size" case, which may added eps or is a dummy value, recalculate scales instead.
std::vector<float> Interpolate::getScales(const VectorDims &srcDimPad, const VectorDims &dstDim) {
    std::vector<float> fullScales(dataRank, 1.f);
    const size_t axesRank = axes.size();
    for (size_t i = 0; i < axesRank; i++) {
        int axis = axes[i];
        // pillow always re-generate scales with input and output shape
        if (interpAttrs.mode == InterpolateMode::bilinear_pillow || interpAttrs.mode == InterpolateMode::bicubic_pillow) {
            fullScales[axis] = static_cast<float>(dstDim[axis]) / static_cast<float>(srcDimPad[axis]);
        } else {
            fullScales[axis] = (interpAttrs.shapeCalcMode == InterpolateShapeCalcMode::scales) ? scales[i] :
                                                                                     static_cast<float>(dstDim[axis]) / static_cast<float>(srcDimPad[axis]);
        }
    }
    return fullScales;
}

void Interpolate::execute(dnnl::stream strm) {
    auto dstMemPtr = getDstMemoryAtPort(0);
    auto srcMemPtr = getSrcMemoryAtPort(DATA_ID);

    if (execPtr) {
        uint8_t *dst_data = dstMemPtr->getDataAs<uint8_t>();
        const uint8_t *src_data_origin = srcMemPtr->getDataAs<uint8_t>();
        const uint8_t *src_data = nullptr;
        std::vector<uint8_t> srcPadded;
        if (hasPad) {
            const auto &srcDim = srcMemPtr->getStaticDims();
            auto srcDimPad = execPtr->getSrcDimPad5d();
            size_t dimSize = srcDim.size();

            const auto srcDim5d = to5Dim(srcDim);
            const auto srcDimPad5d = to5Dim(srcDimPad);
            const auto srcDataSize = srcMemPtr->getDesc().getPrecision().size();

            int padB0 = (dimSize > 2) ? interpAttrs.padBegin[0] : 0;
            int padB1 = (dimSize > 2) ? interpAttrs.padBegin[1] : 0;
            int padB2 = (dimSize == 5) ? interpAttrs.padBegin[dimSize - 3] : 0;
            int padB3 = interpAttrs.padBegin[dimSize - 2];
            int padB4 = interpAttrs.padBegin[dimSize - 1];

            VectorDims inShapeBlock = getBlockND(srcDim5d);
            VectorDims inShapePadBlock = getBlockND(srcDimPad5d);

            if (interpAttrs.layout == InterpolateLayoutType::planar) {
                srcPadded.resize(inShapePadBlock[0] * srcDataSize, 0);
                uint8_t *src_data_pad = static_cast<uint8_t *>(&srcPadded[0]);
                parallel_for4d(srcDim5d[0], srcDim5d[1], srcDim5d[2], srcDim5d[3], [&](int n, int c, int d, int h) {
                    const uint8_t *src = src_data_origin +
                        (inShapeBlock[1] * n + inShapeBlock[2] * c + inShapeBlock[3] * d + inShapeBlock[4] * h) * srcDataSize;
                    uint8_t *srcPad = src_data_pad + (inShapePadBlock[1] * (n + padB0) + inShapePadBlock[2] * (c + padB1) +
                                inShapePadBlock[3] * (d + padB2) + inShapePadBlock[4] * (h + padB3) + padB4) * srcDataSize;
                    cpu_memcpy(srcPad, src, srcDim5d[4] * srcDataSize);
                });
                src_data = src_data_pad;
            } else if (interpAttrs.layout == InterpolateLayoutType::by_channel) {
                srcPadded.resize(inShapePadBlock[0] * srcDataSize, 0);
                uint8_t *src_data_pad = static_cast<uint8_t *>(&srcPadded[0]);
                parallel_for4d(srcDim5d[0], srcDim5d[2], srcDim5d[3], srcDim5d[4], [&](int n, int d, int h, int w) {
                    const uint8_t *src = src_data_origin + (inShapeBlock[1] * n +
                                    (inShapeBlock[3] * d + inShapeBlock[4] * h + inShapeBlock[5] * w) * srcDim5d[1]) * srcDataSize;
                    uint8_t *srcPad = src_data_pad + (inShapePadBlock[1] * (n + padB0) + (inShapePadBlock[3] * (d + padB2) +
                                    inShapePadBlock[4] * (h + padB3) + inShapePadBlock[5] * (w + padB4)) * srcDimPad5d[1] + padB1) * srcDataSize;
                    cpu_memcpy(srcPad, src, srcDim5d[1] * srcDataSize);
                });
                src_data = src_data_pad;
            } else if (interpAttrs.layout == InterpolateLayoutType::block) {
                size_t blkSize = mayiuse(cpu::x64::avx512_core) ? 16 : 8;
                size_t CB = div_up(srcDimPad5d[1], blkSize);
                size_t eltsTotal = srcDimPad5d[0] * CB * srcDimPad5d[2] * srcDimPad5d[3] * srcDimPad5d[4] * blkSize;
                srcPadded.resize(eltsTotal * srcDataSize, 0x0);
                uint8_t *src_data_pad = static_cast<uint8_t *>(&srcPadded[0]);
                if ((srcDim5d[0] != srcDimPad5d[0]) || (srcDim5d[1] != srcDimPad5d[1])) {
                    OPENVINO_THROW("Interpolate layer with name '",
                                   getName(),
                                   "' does not support padding on batch and channel dimensions");
                }
                parallel_for5d(srcDim5d[0], CB, srcDim5d[2], srcDim5d[3], srcDim5d[4], [&](int n, int cb, int d, int h, int w) {
                    const uint8_t *src = src_data_origin + (n * CB * srcDim5d[2] * srcDim5d[3] * srcDim5d[4] * blkSize) * srcDataSize
                                                + (cb * srcDim5d[2] * srcDim5d[3] * srcDim5d[4] * blkSize) * srcDataSize
                                                + (d * srcDim5d[3] * srcDim5d[4] * blkSize) * srcDataSize
                                                + (h * srcDim5d[4] * blkSize) * srcDataSize
                                                + (w * blkSize) * srcDataSize;
                    uint8_t *srcPad = src_data_pad + (n * CB * srcDimPad5d[2] * srcDimPad5d[3] * srcDimPad5d[4] * blkSize) * srcDataSize
                                                + (cb * srcDimPad5d[2] * srcDimPad5d[3] * srcDimPad5d[4] * blkSize) * srcDataSize
                                                + ((d + padB2) * srcDimPad5d[3] * srcDimPad5d[4] * blkSize) * srcDataSize
                                                + ((h + padB3) * srcDimPad5d[4] * blkSize) * srcDataSize
                                                + ((w + padB4) * blkSize) * srcDataSize;
                    cpu_memcpy(srcPad, src, blkSize * srcDataSize);
                });
                src_data = src_data_pad;
            }
        } else {
            src_data = src_data_origin;
        }

        execPtr->exec(src_data, dst_data, postOpsDataPtrs.data());
    } else if (aclExecPtr) {
        aclExecPtr->exec({srcMemPtr}, {dstMemPtr}, postOpsDataPtrs.data());
    } else {
        OPENVINO_THROW("Can't execute Interpolate node. Primitive didn't created");
    }
}

// for ndhwc and nCdhw8c[16c]
// input may be f32/bf16/int8, fused->output varies
void Interpolate::InterpolateJitExecutor::NNCGathered(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_,
                                                                int B, int C, int ID, int IH, int IW, int OD, int OH, int OW) {
    int *index_d = static_cast<int*>(&auxTable[0]);
    int *index_h = static_cast<int*>(&auxTable[OD]);
    int *index_w = static_cast<int*>(&auxTable[OD + OH]);

    bool is_nhwc = (configured_for_layout == by_channel);

    for (int b = 0; b < B; b++) {
        if (is_nhwc) {
            const uint8_t *in_ptr = in_ptr_ + (IW * IH * ID * C * b) * srcDataSize;
            uint8_t *out_ptr = out_ptr_ + (OW * OH * OD * C * b) * dstDataSize;
            std::vector<int> index_w_kernel(OW);
            for (int ox = 0; ox < OW; ox++) {
                index_w_kernel[ox] = index_w[ox] * C * srcDataSize;
            }
            parallel_for2d(OD, OH, [&](size_t d, size_t h) {
                // kernel for C * OW
                uint8_t *out_ptr_dh = out_ptr + (C * OW * OH * d + C * OW * h) * dstDataSize;
                const uint8_t *in_ptr_dh = in_ptr + (C * IW * IH * index_d[d] + C * IW * index_h[h]) * srcDataSize;
                auto arg = jit_interpolate_call_args();
                arg.dst = out_ptr_dh;
                arg.src_ptr[0] = in_ptr_dh;
                arg.index = static_cast<int*>(&(index_w_kernel[0]));
                arg.work_amount = C;
                arg.oc_off = 0;
                arg.post_op_data = post_ops_data_;
                (*interpolateKernel)(&arg);
            });
        } else {  // for blk
            int blk_size = mayiuse(cpu::x64::avx512_core) ? 16 : 8;
            int CB = div_up(C, blk_size);
            const uint8_t *in_ptr = in_ptr_ + (IW * IH * ID * CB * blk_size * b) * srcDataSize;
            uint8_t *out_ptr = out_ptr_ + (OW * OH * OD * CB * blk_size * b) * dstDataSize;
            std::vector<int> index_w_kernel(OW);
            for (int ox = 0; ox < OW; ox++) {
                index_w_kernel[ox] = index_w[ox] * blk_size * srcDataSize;
            }
            parallel_for2d(CB, OD, [&](size_t cb, size_t d) {
                uint8_t *out_ptr_cbd = out_ptr + (blk_size * OW * OH * OD * cb + blk_size * OW * OH * d) * dstDataSize;
                const uint8_t *in_ptr_cbd = in_ptr + (blk_size * IW * IH * ID * cb + blk_size * IW * IH * index_d[d]) * srcDataSize;
                auto arg = jit_interpolate_call_args();
                for (int h = 0; h < OH; h++) {  // kernel for blk_size * OW
                    arg.dst = out_ptr_cbd + blk_size * OW * h * dstDataSize;
                    arg.src_ptr[0] = in_ptr_cbd + blk_size * IW * index_h[h] * srcDataSize;
                    arg.index = static_cast<int*>(&(index_w_kernel[0]));
                    arg.work_amount = static_cast<size_t>(OW);
                    arg.oc_off = cb * blk_size * sizeof(float);
                    arg.post_op_data = post_ops_data_;
                    (*interpolateKernel)(&arg);
                }
            });
        }
    }  // batch end
}

void Interpolate::InterpolateJitExecutor::NNPlanar(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_,
                                                             int B, int C, int ID, int IH, int IW, int OD, int OH, int OW) {
    int *index_d = static_cast<int*>(&auxTable[0]);
    int *index_h = static_cast<int*>(&auxTable[OD]);
    int *index_w = static_cast<int*>(&auxTable[OD + OH]);

    std::vector<int> index_kernel(OH + OW);
    // index_h * IW * srcDataSize to reduce and simplify redundant compute
    for (int oh = 0; oh < OH; oh++) {
        index_kernel[oh] = index_h[oh] * IW * srcDataSize;
    }
    // index_w * srcDataSize
    for (int ow = 0; ow < OW; ow++) {
        index_kernel[OH + ow] = index_w[ow] * srcDataSize;
    }

    parallel_for3d(B, C, OD, [&](size_t b, size_t c, size_t od) {
        const uint8_t *in_ptr = in_ptr_ + (IW * IH * ID * C * b + IW * IH * ID * c + IW * IH * index_d[od]) * srcDataSize;
        uint8_t *out_ptr = out_ptr_ + (OW * OH * OD * C * b + OW * OH * OD * c + OW * OH * od) * dstDataSize;

        auto arg = jit_interpolate_call_args();
        arg.src_ptr[0] = in_ptr;
        arg.dst = out_ptr;
        arg.index = static_cast<int*>(&index_kernel[0]);  // need index_h and index_w in kernel, it's in continous memory so one param
        arg.oc_off = static_cast<size_t>(c * sizeof(float));
        // work_amount is OH(out loop) and OW(inner loop), can get in kernel from jcp.
        arg.post_op_data = post_ops_data_;
        (*interpolateKernel)(&arg);
    });
}

void Interpolate::InterpolateJitExecutor::linearOnnxPlanar(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_, int B, int C,
                                                                     int ID, int IH, int IW, int OD, int OH, int OW) {
    // FrontTopLeft:0, FrontTopRight:1, FrontBottomLeft:2, FrontBottomRight:3, EndTopLeft:4,   EndTopRight:5,   EndBottomLeft:6,   EndBottomRight:7
    // weight: Left:0, ritht:1, top:2, bottom:3, front:4, end:5
    int *index = static_cast<int*>(&auxTable[0]);
    int eltInGrid = (spatialDimSize > 2) ? MAX_INPUT_INTERPOLATE : ((spatialDimSize > 1) ? 4 : 2);
    int scratchLen = rnd_up(eltInGrid * OW * OH * OD, 16);
    float *weight = reinterpret_cast<float*>(&auxTable[scratchLen]);

    parallel_for2d(B, C, [&](size_t b, size_t c) {
        uint8_t *out_ptr_nc = out_ptr_ + (OH * OW * OD * C * b + OH * OW * OD * c) * dstDataSize;
        const uint8_t *in_ptr_nc = in_ptr_ + (IH * IW * ID * C * b + IH * IW * ID * c) * srcDataSize;
        auto arg = jit_interpolate_call_args();
        arg.src_ptr[0] = in_ptr_nc;
        arg.index = static_cast<int*>(&index[0]);
        arg.weight_ptr[0] = static_cast<float*>(&weight[0]);
        arg.dst = out_ptr_nc;
        arg.work_amount = OW * OH * OD;
        arg.oc_off = static_cast<size_t>(c * sizeof(float));
        arg.post_op_data = post_ops_data_;
        (*interpolateKernel)(&arg);
    });
}

void Interpolate::InterpolateJitExecutor::linearOnnxCGathered(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_,
                                                                        int B, int C, int ID, int IH, int IW, int OD, int OH, int OW) {
    // left:OW right:OW Top:OH Bottom:OH Front:OD End:OD
    std::vector<int*> indexPtr(MAX_INPUT_INTERPOLATE, 0);
    std::vector<float*> weightPtr(MAX_INPUT_INTERPOLATE, 0);
    size_t scratchLen = rnd_up(OW + OW + OH + OH + OD + OD, 16);
    indexPtr[0] = static_cast<int*>(&auxTable[0]);
    indexPtr[1] = static_cast<int*>(&auxTable[OW]);
    indexPtr[2] = static_cast<int*>(&auxTable[2 * OW]);
    indexPtr[3] = static_cast<int*>(&auxTable[2 * OW + OH]);
    indexPtr[4] = static_cast<int*>(&auxTable[2 * OW + 2 * OH]);
    indexPtr[5] = static_cast<int*>(&auxTable[2 * OW + 2 * OH + OD]);

    weightPtr[0] = reinterpret_cast<float*>(&auxTable[scratchLen]);
    weightPtr[1] = reinterpret_cast<float*>(&auxTable[scratchLen + OW]);
    weightPtr[2] = reinterpret_cast<float*>(&auxTable[scratchLen + 2 * OW]);
    weightPtr[3] = reinterpret_cast<float*>(&auxTable[scratchLen + 2 * OW + OH]);
    weightPtr[4] = reinterpret_cast<float*>(&auxTable[scratchLen + 2 * OW + 2 * OH]);
    weightPtr[5] = reinterpret_cast<float*>(&auxTable[scratchLen + 2 * OW + 2 * OH + OD]);

    bool isByChannel = (configured_for_layout == by_channel) ? true : false;

    int blkSize = mayiuse(cpu::x64::avx512_core) ? 16 : 8;
    int CB = isByChannel ? 1 : div_up(C, blkSize);
    int CGatherLen = isByChannel ? C : blkSize;
    int workAmount = isByChannel ? C : CB;
    // n_CB(1)_d_h_w_8[16](c), () for by-channel
    int C0 = OW * CGatherLen;
    int C1 = OH * C0;
    int C2 = OD * C1;
    int C3 = CB * C2;
    int I0 = IW * CGatherLen;
    int I1 = IH * I0;
    int I2 = ID * I1;
    int I3 = CB * I2;
    parallel_for3d(B, OD, OH, [&](size_t b, size_t d, size_t h) {
        uint8_t *out_ptr_ndh = out_ptr_ + (C3 * b + C1 * d + C0 * h) * dstDataSize;

        const uint8_t *in_ptr_n = in_ptr_ + (I3 * b) * srcDataSize;
        const uint8_t *in_ptr_nf = in_ptr_n + (indexPtr[4][d] * I1) * srcDataSize;
        const uint8_t *in_ptr_nft = in_ptr_nf + (indexPtr[2][h] * I0) * srcDataSize;
        const uint8_t *in_ptr_nfb = in_ptr_nf + (indexPtr[3][h] * I0) * srcDataSize;
        const uint8_t *in_ptr_ne = in_ptr_n + (indexPtr[5][d] * I1) * srcDataSize;
        const uint8_t *in_ptr_net = in_ptr_ne + (indexPtr[2][h] * I0) * srcDataSize;
        const uint8_t *in_ptr_neb = in_ptr_ne + (indexPtr[3][h] * I0) * srcDataSize;
        auto arg = jit_interpolate_call_args();
        for (int w = 0; w < OW; ++w) {
            uint8_t *out_ptr_ndhw = out_ptr_ndh + CGatherLen * w * dstDataSize;

            arg.src_ptr[0] = in_ptr_nft + (indexPtr[0][w] * CGatherLen) * srcDataSize;
            arg.src_ptr[1] = in_ptr_nft + (indexPtr[1][w] * CGatherLen) * srcDataSize;
            arg.src_ptr[2] = in_ptr_nfb + (indexPtr[0][w] * CGatherLen) * srcDataSize;
            arg.src_ptr[3] = in_ptr_nfb + (indexPtr[1][w] * CGatherLen) * srcDataSize;
            arg.src_ptr[4] = in_ptr_net + (indexPtr[0][w] * CGatherLen) * srcDataSize;
            arg.src_ptr[5] = in_ptr_net + (indexPtr[1][w] * CGatherLen) * srcDataSize;
            arg.src_ptr[6] = in_ptr_neb + (indexPtr[0][w] * CGatherLen) * srcDataSize;
            arg.src_ptr[7] = in_ptr_neb + (indexPtr[1][w] * CGatherLen) * srcDataSize;
            arg.weight_ptr[0] = static_cast<float*>(&weightPtr[0][w]);
            arg.weight_ptr[1] = static_cast<float*>(&weightPtr[1][w]);
            arg.weight_ptr[2] = static_cast<float*>(&weightPtr[2][h]);
            arg.weight_ptr[3] = static_cast<float*>(&weightPtr[3][h]);
            arg.weight_ptr[4] = static_cast<float*>(&weightPtr[4][d]);
            arg.weight_ptr[5] = static_cast<float*>(&weightPtr[5][d]);
            arg.dst = out_ptr_ndhw;
            arg.work_amount = workAmount;
            arg.oc_off = 0;
            arg.post_op_data = post_ops_data_;
            (*interpolateKernel)(&arg);
        }
    });
}

void Interpolate::InterpolateJitExecutor::cubicCGathered(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_,
                                                                   int B, int C, int IH, int IW, int OH, int OW) {
    const int idxNum = 1;
    int *xOrigin = static_cast<int*>(&auxTable[0]);
    float *xFactor = reinterpret_cast<float*>(&auxTable[OW]);
    int *yOrigin = static_cast<int*>(&auxTable[(CUBIC_GRID_LEN + idxNum) * OW]);
    float *yFactor = reinterpret_cast<float*>(&auxTable[(CUBIC_GRID_LEN + idxNum) * OW + OH]);

    int blkSize = mayiuse(cpu::x64::avx512_core) ? 16 : 8;
    int CB = div_up(C, blkSize);
    int CSize = configured_for_layout == InterpolateLayoutType::by_channel ? C : blkSize * CB;
    int CGatherLen = configured_for_layout == InterpolateLayoutType::by_channel ? C : blkSize;
    int workAmount = configured_for_layout == InterpolateLayoutType::by_channel ? C : CB;

    parallel_for3d(B, OH, OW, [&](size_t b, size_t h, size_t w) {
        uint8_t *out_ptr_nhw = out_ptr_ + (OH * OW * CSize * b + OW * CGatherLen * h + CGatherLen * w) * dstDataSize;
        const uint8_t *in_ptr_n = in_ptr_ + (IH * IW * CSize * b) * srcDataSize;

        std::vector<int> kernelIndex(CUBIC_GRID_LEN * CUBIC_GRID_LEN);  // 16 address offset to src(batch) or src(CB)
        int iy = yOrigin[h];
        int ix = xOrigin[w];
        for (int y = iy - 1, i = 0; y <= iy + 2; y++, i++) {
            int yInRange = std::max(0, std::min(y, IH - 1));
            yInRange = yInRange * CGatherLen * IW * srcDataSize;
            for (int x = ix - 1, j = 0; x <= ix + 2; x++, j++) {
                int xInRange = std::max(0, std::min(x, IW - 1));
                xInRange = yInRange + xInRange * CGatherLen * srcDataSize;
                kernelIndex[i * CUBIC_GRID_LEN + j] = xInRange;
            }
        }
        auto arg = jit_interpolate_call_args();
            arg.dst = out_ptr_nhw;
            arg.src_ptr[0] = in_ptr_n;
            arg.index = static_cast<int*>(&kernelIndex[0]);
            // 0 for weight_W, 1 for weight_H
            arg.weight_ptr[0] = static_cast<float*>(&xFactor[w * CUBIC_GRID_LEN]);
            arg.weight_ptr[1] = static_cast<float*>(&yFactor[h * CUBIC_GRID_LEN]);

            // for by channel, src + step, dst + step, process next step on continuous memory
            // for blk, src + IW*IH*blkSize, dst + OW*OH*blkSize, process the blkSize on next CB
            arg.work_amount = workAmount;
            arg.oc_off = 0;
            arg.post_op_data = post_ops_data_;
            (*interpolateKernel)(&arg);
    });
}

void Interpolate::InterpolateJitExecutor::cubicPlanar(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_,
                                                                int B, int C, int IH, int IW, int OH, int OW) {
    int tblAdvance = 0;
    int *xOrigin = static_cast<int*>(&auxTable[tblAdvance]);
    tblAdvance += OW;
    float *xFactor = reinterpret_cast<float*>(&auxTable[tblAdvance]);
    tblAdvance += CUBIC_GRID_LEN * OW;
    int *yOrigin = static_cast<int*>(&auxTable[tblAdvance]);
    tblAdvance += OH;
    float *yFactor = reinterpret_cast<float*>(&auxTable[tblAdvance]);

    tblAdvance += CUBIC_GRID_LEN * OH;
    int *sequenceOH = static_cast<int*>(&auxTable[tblAdvance]);
    tblAdvance += OW * OH;
    int *sequenceOW = static_cast<int*>(&auxTable[tblAdvance]);

    parallel_for2d(B, C, [&](size_t n, size_t c) {
        const uint8_t *in_ptr_nc = in_ptr_ + (IW * IH * C * n + IW * IH * c) * srcDataSize;
        uint8_t *out_ptr_nc = out_ptr_ + (OW * OH * C * n + OW * OH * c) * dstDataSize;

        auto arg = jit_interpolate_call_args();
        arg.dst = out_ptr_nc;
        arg.src_ptr[0] = in_ptr_nc;
        arg.index = xOrigin;
        arg.src_ptr[1] = yOrigin;
        arg.src_ptr[2] = static_cast<int*>(&sequenceOH[0]);
        arg.src_ptr[3] = static_cast<int*>(&sequenceOW[0]);
        arg.weight_ptr[0] = xFactor;
        arg.weight_ptr[1] = yFactor;
        arg.work_amount = static_cast<size_t>(OW * OH);
        arg.oc_off = static_cast<size_t>(c * sizeof(float));
        arg.post_op_data = post_ops_data_;
        (*interpolateKernel)(&arg);
    });
}

void Interpolate::InterpolateJitExecutor::pillowCGathered(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_,
                                                                        int B, int C, int IH, int IW, int OH, int OW) {
    // workBuffer needed when both pass are true
    bool xPass = IW != OW;
    bool yPass = IH != OH;

    auto b_loop = [&](size_t b) {
        auto arg = jit_interpolate_call_args();
        arg.src_ptr[0] = in_ptr_ + (IW * IH * C * b) * srcDataSize;
        if (xPass && yPass) {
            size_t parallel_num = B;
            // IH * OW * C buf needed
            size_t buffer_size = static_cast<size_t>(OW * IH * C);
            if (parallel_num < m_threads_num) {
                arg.src_ptr[1] = static_cast<uint8_t*>(&pillow_working_buf[b * buffer_size * srcDataSize]);
            } else {
                size_t threads_idx = parallel_get_thread_num();
                arg.src_ptr[1] = static_cast<uint8_t*>(&pillow_working_buf[threads_idx * buffer_size * srcDataSize]);
            }
        }
        arg.dst = out_ptr_ + (OW * OH * C * b) * dstDataSize;
        arg.weight_ptr[0] = reinterpret_cast<float*>(&auxTable[2]);
        (*interpolateKernel)(&arg);
    };

    parallel_nt_static(m_threads_num, [&](const int ithr, const int nthr) {
        for_1d(ithr, nthr, B, b_loop);
    });
}

// =====================================================================================================================
// index layout:
// d_0............d_OD-1, h_0..............h_OH-1, w_0................w_OW-1
void Interpolate::InterpolateExecutorBase::buildTblNN(const VectorDims& srcDimPad5d, const VectorDims& dstDim5d,
                                        const std::vector<float>& dataScales, InterpolateLayoutType layout, InterpolateNearestMode nearestMode) {
    const int dimSize = dataRank;
    float fz = (dimSize == 5) ? dataScales[dimSize - 3] : 1.f;
    float fy = dataScales[dimSize - 2];
    float fx = dataScales[dimSize - 1];
    size_t ID = srcDimPad5d[2], IH = srcDimPad5d[3], IW = srcDimPad5d[4];
    size_t OD = dstDim5d[2], OH = dstDim5d[3], OW = dstDim5d[4];

    auxTable.resize(OD + OH + OW);
    bool isDDownsample = (fz < 1) ? true : false;
    bool isHDownsample = (fy < 1) ? true : false;
    bool isWDownsample = (fx < 1) ? true : false;
    for (size_t oz = 0; oz < OD; oz++) {
        float iz = coordTransToInput(oz, fz, ID, OD);
        auxTable[oz] = nearestRound(iz, isDDownsample, nearestMode);
        auxTable[oz] = clipCoord(auxTable[oz], ID);
    }
    for (size_t oy = 0; oy < OH; oy++) {
        float iy = coordTransToInput(oy, fy, IH, OH);
        auxTable[OD + oy] = nearestRound(iy, isHDownsample, nearestMode);
        auxTable[OD + oy] = clipCoord(auxTable[OD + oy], IH);
    }
    for (size_t ox = 0; ox < OW; ox++) {
        float ix = coordTransToInput(ox, fx, IW, OW);
        auxTable[OD + OH + ox] = nearestRound(ix, isWDownsample, nearestMode);
        auxTable[OD + OH + ox] = clipCoord(auxTable[OD + OH + ox], IW);
    }
}

// scale is float(outShape) / float(inShape)
// strictly consistent with onnx calc manner(div scale, not multiply inverse), given this is done offline
// the slight precison diff can produce obvious wrong value due to "nearest round" behavior for NN mode
float Interpolate::InterpolateExecutorBase::coordTransToInput(int outCoord, float scale, int inShape, int outShape) const {
    if (scale == 1.0f || (inShape == outShape)) {
        return outCoord;
    }
    switch (coordTransMode) {
        case InterpolateCoordTransMode::half_pixel: {
            return (outCoord + 0.5f) / scale - 0.5f;
            break;
        }
        case InterpolateCoordTransMode::pytorch_half_pixel: {
            if (outShape > 1)
                return (outCoord + 0.5f) / scale - 0.5f;
            else
                return 0;
            break;
        }
        case InterpolateCoordTransMode::asymmetric: {
            return static_cast<float>(outCoord) / scale;
            break;
        }
        case InterpolateCoordTransMode::tf_half_pixel_for_nn: {
            return (outCoord + 0.5f) / scale;
            break;
        }
        case InterpolateCoordTransMode::align_corners: {
            if (outShape > 1)
                return outCoord * (static_cast<float>(inShape - 1) / static_cast<float>(outShape - 1));
            else
                return 0;
            break;
        }
        default: {
            OPENVINO_THROW("errorPrefix", " does not support specified coordinate transformation mode");
            break;
        }
    }
}

int Interpolate::InterpolateExecutorBase::nearestRound(float originCoord, bool isDownsample, InterpolateNearestMode nearestMode) const {
    switch (nearestMode) {
        case InterpolateNearestMode::round_prefer_floor: {
            if (originCoord == (static_cast<int>(originCoord) + 0.5f))
                return static_cast<int>(std::floor(originCoord));
            else
                return static_cast<int>(std::round(originCoord));
            break;
        }
        case InterpolateNearestMode::round_prefer_ceil: {
            return static_cast<int>(std::round(originCoord));
            break;
        }
        case InterpolateNearestMode::floor: {
            return static_cast<int>(std::floor(originCoord));
            break;
        }
        case InterpolateNearestMode::ceil: {
            return static_cast<int>(std::ceil(originCoord));
            break;
        }
        case InterpolateNearestMode::simple: {
            if (isDownsample)
                return static_cast<int>(std::ceil(originCoord));
            else
                return static_cast<int>(originCoord);
        }
        default: {
            OPENVINO_THROW("errorPrefix", " does not support specified nearest round mode");
            break;
        }
    }
}

void Interpolate::InterpolateExecutorBase::linearOnnxCF(int outCoord, float scale, int inShape, int outShape,
                int& index0, int& index1, float& weight0, float& weight1) {
    float inCoord = coordTransToInput(outCoord, scale, inShape, outShape);
    inCoord = std::max(0.0f, std::min(inCoord, static_cast<float>(inShape - 1)));
    index0 = std::min(static_cast<int>(inCoord), inShape - 1);
    index1 = std::min(index0 + 1, inShape - 1);

    weight1 = std::fabs(inCoord - index0);
    weight0 = std::fabs(inCoord - index1);
    if (index0 == index1) {
        weight0 = 0.5f;
        weight1 = 0.5f;
    }
}

void Interpolate::InterpolateExecutorBase::buildTblLinearOnnx(const VectorDims& srcDimPad5d, const VectorDims& dstDim5d,
                                                const std::vector<float>& dataScales, InterpolateLayoutType layout) {
    int dimSize = dataRank;
    float fz = (spatialDimSize > 2) ? dataScales[dimSize - 3] : 1.f;
    float fy = (spatialDimSize > 1) ? dataScales[dimSize - 2] : 1.f;
    float fx = dataScales[dimSize - 1];
    int ID = srcDimPad5d[2], IH = srcDimPad5d[3], IW = srcDimPad5d[4];
    int OD = dstDim5d[2], OH = dstDim5d[3], OW = dstDim5d[4];

    std::vector<int*> indexPtr(MAX_INPUT_INTERPOLATE, 0);
    std::vector<float*> weightPtr(MAX_INPUT_INTERPOLATE, 0);
    if (layout == InterpolateLayoutType::planar) {
        // FrontTopLeft:0, FrontTopRight:1, FrontBottomLeft:2, FrontBottomRight:3,
        // EndTopLeft:4,   EndTopRight:5,   EndBottomLeft:6,   EndBottomRight:7
        // weight: Left:0, ritht:1, top:2, bottom:3, front:4, end:5
        int eltInGrid = (spatialDimSize > 2) ? MAX_INPUT_INTERPOLATE : ((spatialDimSize > 1) ? 4 : 2);
        int idxType = 2;
        int scratchLen = rnd_up(eltInGrid * OW * OH * OD, 16);
        auxTable.resize(idxType * scratchLen);

        indexPtr[0] = static_cast<int*>(&auxTable[0]);
        indexPtr[1] = static_cast<int*>(&auxTable[OW * OH * OD]);
        weightPtr[0] = reinterpret_cast<float*>(&auxTable[scratchLen]);
        weightPtr[1] = reinterpret_cast<float*>(&auxTable[scratchLen + OW * OH * OD]);
        if (spatialDimSize > 1) {
            indexPtr[2] = static_cast<int*>(&auxTable[2 * OW * OH * OD]);
            indexPtr[3] = static_cast<int*>(&auxTable[3 * OW * OH * OD]);
            weightPtr[2] = reinterpret_cast<float*>(&auxTable[scratchLen + 2 * OW * OH * OD]);
            weightPtr[3] = reinterpret_cast<float*>(&auxTable[scratchLen + 3 * OW * OH * OD]);
        }
        if (spatialDimSize > 2) {
            indexPtr[4] = static_cast<int*>(&auxTable[4 * OW * OH * OD]);
            indexPtr[5] = static_cast<int*>(&auxTable[5 * OW * OH * OD]);
            indexPtr[6] = static_cast<int*>(&auxTable[6 * OW * OH * OD]);
            indexPtr[7] = static_cast<int*>(&auxTable[7 * OW * OH * OD]);
            weightPtr[4] = reinterpret_cast<float*>(&auxTable[scratchLen + 4 * OW * OH * OD]);
            weightPtr[5] = reinterpret_cast<float*>(&auxTable[scratchLen + 5 * OW * OH * OD]);
        }
        int scale = mayiuse(cpu::x64::sse41) ? srcDataSize : 1;

        for (int oz = 0; oz < OD; oz++) {
            int izF, izE;
            float weightF, weightE;
            linearOnnxCF(oz, fz, ID, OD, izF, izE, weightF, weightE);
            int idxOz = oz * OH * OW;
            for (int oy = 0; oy < OH; oy++) {
                int iyT, iyB;
                float weightT, weightB;
                linearOnnxCF(oy, fy, IH, OH, iyT, iyB, weightT, weightB);
                int idxOzOy = idxOz + oy * OW;
                for (int ox = 0; ox < OW; ox++) {
                    int ixL, ixR;
                    float weightL, weightR;
                    linearOnnxCF(ox, fx, IW, OW, ixL, ixR, weightL, weightR);

                    int idxOzOyOx = idxOzOy + ox;
                    indexPtr[0][idxOzOyOx] = (izF * IH * IW + iyT * IW + ixL) * scale;
                    indexPtr[1][idxOzOyOx] = (izF * IH * IW + iyT * IW + ixR) * scale;
                    weightPtr[0][idxOzOyOx] = weightL;
                    weightPtr[1][idxOzOyOx] = weightR;
                    if (spatialDimSize  > 1) {
                        indexPtr[2][idxOzOyOx] = (izF * IH * IW + iyB * IW + ixL) * scale;
                        indexPtr[3][idxOzOyOx] = (izF * IH * IW + iyB * IW + ixR) * scale;
                        weightPtr[2][idxOzOyOx] = weightT;
                        weightPtr[3][idxOzOyOx] = weightB;
                    }
                    if (spatialDimSize > 2) {
                        indexPtr[4][idxOzOyOx] = (izE * IH * IW + iyT * IW + ixL) * scale;
                        indexPtr[5][idxOzOyOx] = (izE * IH * IW + iyT * IW + ixR) * scale;
                        indexPtr[6][idxOzOyOx] = (izE * IH * IW + iyB * IW + ixL) * scale;
                        indexPtr[7][idxOzOyOx] = (izE * IH * IW + iyB * IW + ixR) * scale;
                        weightPtr[4][idxOzOyOx] = weightF;
                        weightPtr[5][idxOzOyOx] = weightE;
                    }
                }
            }
        }
    } else {
        // index: left:OW right:OW Top:OH Bottom:OH, Front:OD, End:OD
        // weight:same as index
        size_t scratchLen = rnd_up(OW + OW + OH + OH + OD + OD, 16);
        int idxType = 2;
        auxTable.resize(idxType * scratchLen);
        indexPtr[0] = static_cast<int*>(&auxTable[0]);
        indexPtr[1] = static_cast<int*>(&auxTable[OW]);
        indexPtr[2] = static_cast<int*>(&auxTable[2 * OW]);
        indexPtr[3] = static_cast<int*>(&auxTable[2 * OW + OH]);
        indexPtr[4] = static_cast<int*>(&auxTable[2 * OW + 2 * OH]);
        indexPtr[5] = static_cast<int*>(&auxTable[2 * OW + 2 * OH + OD]);

        weightPtr[0] = reinterpret_cast<float*>(&auxTable[scratchLen]);
        weightPtr[1] = reinterpret_cast<float*>(&auxTable[scratchLen + OW]);
        weightPtr[2] = reinterpret_cast<float*>(&auxTable[scratchLen + 2 * OW]);
        weightPtr[3] = reinterpret_cast<float*>(&auxTable[scratchLen + 2 * OW + OH]);
        weightPtr[4] = reinterpret_cast<float*>(&auxTable[scratchLen + 2 * OW + 2 * OH]);
        weightPtr[5] = reinterpret_cast<float*>(&auxTable[scratchLen + 2 * OW + 2 * OH + OD]);

        for (int ox = 0; ox < OW; ox++) {
            linearOnnxCF(ox, fx, IW, OW, indexPtr[0][ox], indexPtr[1][ox], weightPtr[0][ox], weightPtr[1][ox]);
        }
        for (int oy = 0; oy < OH; oy++) {
            linearOnnxCF(oy, fy, IH, OH, indexPtr[2][oy], indexPtr[3][oy], weightPtr[2][oy], weightPtr[3][oy]);
        }
        for (int oz = 0; oz < OD; oz++) {
            linearOnnxCF(oz, fz, ID, OD, indexPtr[4][oz], indexPtr[5][oz], weightPtr[4][oz], weightPtr[5][oz]);
        }
    }
}

// table layout:
// wd .........wd, wh............wh, ww.............ww, id...........id, ih............ih, iw..............iw
//                        |                                                      |
//                   wh0.....wh_diameter                                    ih0.....ih_diameter
void Interpolate::InterpolateExecutorBase::buildTblLinear(const VectorDims& srcDimPad5d, const VectorDims& dstDim5d,
                                            const std::vector<float>& dataScales, int kernel_width, bool antialias) {
    int dimSize = dataRank;
    float fz = (dimSize == 5) ? dataScales[dimSize - 3] : 1.f;
    float fy = dataScales[dimSize - 2];
    float fx = dataScales[dimSize - 1];
    size_t ID = srcDimPad5d[2], IH = srcDimPad5d[3], IW = srcDimPad5d[4];
    size_t OD = dstDim5d[2], OH = dstDim5d[3], OW = dstDim5d[4];

    if (!(IW == OW && IH == OH && ID == OD)) {
        float ax = antialias ? fx : 1.0f;
        float ay = antialias ? fy : 1.0f;
        float az = antialias ? fz : 1.0f;

        int rx = (fx > 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / ax));
        int ry = (fy > 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / ay));
        int rz = (fz > 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / az));

        int diaOD = 2 * rz + 1;
        int diaOH = 2 * ry + 1;
        int diaOW = 2 * rx + 1;
        int sizeOD = OD * diaOD;
        int sizeOH = OH * diaOH;
        int sizeOW = OW * diaOW;
        auxTable.resize((sizeOD + sizeOH + sizeOW) * 2);
        float *weightTable = reinterpret_cast<float*>(&auxTable[0]);
        float *weightOD = static_cast<float*>(&weightTable[0]);
        float *weightOH = static_cast<float*>(&weightTable[sizeOD]);
        float *weightOW = static_cast<float*>(&weightTable[sizeOD + sizeOH]);

        int *idxTable = static_cast<int*>(&auxTable[sizeOD + sizeOH + sizeOW]);
        int *idxOD = static_cast<int*>(&idxTable[0]);
        int *idxOH = static_cast<int*>(&idxTable[sizeOD]);
        int *idxOW = static_cast<int*>(&idxTable[sizeOD + sizeOH]);

        for (size_t oz = 0; oz < OD; oz++) {
            float iz = coordTransToInput(oz, fz, ID, OD);
            int iz_r = static_cast<int>(std::round(iz));
            for (int r = iz_r - rz, i = 0; r <= iz_r + rz; r++, i++) {
                idxOD[oz * diaOD + i] = r;
                if (r < 0 || r >= static_cast<int>(ID)) {
                    weightOD[oz * diaOD + i] = 0.f;
                } else {
                    float dz = iz - r;
                    weightOD[oz * diaOD + i] = az * triangleCoeff(az * dz);
                }
            }
        }
        for (size_t oy = 0; oy < OH; oy++) {
            float iy = coordTransToInput(oy, fy, IH, OH);
            int iy_r = static_cast<int>(std::round(iy));
            for (int r = iy_r - ry, i = 0; r <= iy_r + ry; r++, i++) {
                idxOH[oy * diaOH + i] = r;
                if (r < 0 || r >= static_cast<int>(IH)) {
                    weightOH[oy * diaOH + i] = 0.f;
                } else {
                    float dy = iy - r;
                    weightOH[oy * diaOH + i] = ay * triangleCoeff(ay * dy);
                }
            }
        }
        for (size_t ox = 0; ox < OW; ox++) {
            float ix = coordTransToInput(ox, fx, IW, OW);
            int ix_r = static_cast<int>(std::round(ix));
            for (int r = ix_r - rx, i = 0; r <= ix_r + rx; r++, i++) {
                idxOW[ox * diaOW + i] = r;
                if (r < 0 || r >= static_cast<int>(IW)) {
                    weightOW[ox * diaOW + i] = 0.f;
                } else {
                    float dx = ix - r;
                    weightOW[ox * diaOW + i] = ax * triangleCoeff(ax * dx);
                }
            }
        }
    }
}

std::vector<float> Interpolate::InterpolateExecutorBase::getCubicCoeffs(float mantissa, float a) {
    float m = std::fabs(mantissa);
    std::vector<float> coeffs(4, 0.f);

    coeffs[0] = a * (m - 1.0) * (m - 1.0) * m;
    coeffs[1] = ((a + 2.0) * m - (a + 3.0)) * m * m + 1.0;
    coeffs[2] = (((-a - 2.0) * m + (2.0 * a + 3.0)) * m - a) * m;
    coeffs[3] = -a * m * m * (m - 1.0);
    return coeffs;
}

// table layout:
// OW      OW         OW         OW         OW          OH       OH           OH           OH           OH
// x_idx   x_weight0  x_weight1  x_weight2  x_weight3   y_idx    y_weight0    y_weight1    y_weight2    y_weight3
void Interpolate::InterpolateExecutorBase::buildTblCubic(const VectorDims& srcDimPad5d, const VectorDims& dstDim5d, const std::vector<float>& dataScales,
                                        float cubicCoeff, InterpolateLayoutType layout) {
    int dimSize = dataRank;
    float fy = dataScales[dimSize - 2];
    float fx = dataScales[dimSize - 1];
    int IH = srcDimPad5d[3], IW = srcDimPad5d[4];
    int OH = dstDim5d[3], OW = dstDim5d[4];

    // idxNum for index, CUBIC_GRID_LEN for weight
    const int idxNum = 1;
    size_t idxWeightSize = (CUBIC_GRID_LEN + idxNum) * OW + (CUBIC_GRID_LEN + idxNum) * OH;
    if (layout != InterpolateLayoutType::planar) {
        auxTable.resize(idxWeightSize);
    } else {
        size_t sequenceSize = 2 * OH * OW;
        auxTable.resize(idxWeightSize + sequenceSize);
    }

    int tblAdvance = 0;
    int *xOrigin = static_cast<int*>(&auxTable[tblAdvance]);
    tblAdvance += OW;
    float *xFactor = reinterpret_cast<float*>(&auxTable[tblAdvance]);
    for (int ox = 0; ox < OW; ox++) {
        float ix = coordTransToInput(ox, fx, IW, OW);
        int ix_r = static_cast<int>(std::floor(ix));
        xOrigin[ox] = ix_r;
        float m = ix - ix_r;
        std::vector<float> coffes = getCubicCoeffs(m, cubicCoeff);
        xFactor[CUBIC_GRID_LEN * ox] = coffes[0];
        xFactor[CUBIC_GRID_LEN * ox + 1] = coffes[1];
        xFactor[CUBIC_GRID_LEN * ox + 2] = coffes[2];
        xFactor[CUBIC_GRID_LEN * ox + 3] = coffes[3];
    }

    tblAdvance += CUBIC_GRID_LEN * OW;
    int *yOrigin = static_cast<int*>(&auxTable[tblAdvance]);
    tblAdvance += OH;
    float *yFactor = reinterpret_cast<float*>(&auxTable[tblAdvance]);
    for (int oy = 0; oy < OH; oy++) {
        float iy = coordTransToInput(oy, fy, IH, OH);
        int iy_r = static_cast<int>(std::floor(iy));
        yOrigin[oy] = iy_r;
        float m = iy - iy_r;
        std::vector<float> coffes = getCubicCoeffs(m, cubicCoeff);
        yFactor[CUBIC_GRID_LEN * oy] = coffes[0];
        yFactor[CUBIC_GRID_LEN * oy + 1] = coffes[1];
        yFactor[CUBIC_GRID_LEN * oy + 2] = coffes[2];
        yFactor[CUBIC_GRID_LEN * oy + 3] = coffes[3];
    }

    if (layout == InterpolateLayoutType::planar) {
        tblAdvance += CUBIC_GRID_LEN * OH;
        int *sequenceOH = static_cast<int*>(&auxTable[tblAdvance]);
        tblAdvance += OH * OW;
        int *sequenceOW = static_cast<int*>(&auxTable[tblAdvance]);
        for (int h = 0; h < OH; ++h) {
            int offset = h * OW;
            for (int w = 0; w < OW; ++w) {
                sequenceOH[offset + w] = h * sizeof(int);
                sequenceOW[offset + w] = w * sizeof(int);
            }
        }
    }
}

float Interpolate::InterpolateExecutorBase::getPillowBilinearCoeffs(float m) {
    if (m < 0.0f)
        m = -m;
    if (m < 1.0)
        return 1.0f - m;
    return 0.0f;
}

float Interpolate::InterpolateExecutorBase::getPillowBicubicCoeffs(float m) {
    float a = -0.5f;
    if (m < 0.0f)
        m = -m;
    if (m < 1.0)
        return ((a + 2.0) * m - (a + 3.0)) * m * m + 1.0;
    if (m < 2.0f)
        return (((m - 5) * m + 8) * m - 4) * a;
    return 0.0f;
}

void Interpolate::InterpolateExecutorBase::buildTblPillow(const VectorDims& srcDimPad5d, const VectorDims& dstDim5d, const std::vector<float>& dataScales,
                                        float cubicCoeff, InterpolateLayoutType layout) {
    int dimSize = dataRank;
    float fy = dataScales[dimSize - 2];
    float fx = dataScales[dimSize - 1];
    int IH = srcDimPad5d[3], IW = srcDimPad5d[4];
    int OH = dstDim5d[3], OW = dstDim5d[4];

    struct filterArgs {
        float (*weightGen)(float m);
        float ScaleClipReciprocal;
        float filterRadius;
        float filterLen;
    };

    // pillowScale: e.g. 2.0 means down sample 2 times
    auto generateArgs = [&] (float pillowScale) -> filterArgs {
        filterArgs args;
        float scaleClip = pillowScale < 1.0f ? 1.0f : pillowScale;
        args.ScaleClipReciprocal = 1.0f / scaleClip;
        args.filterRadius = (mode == InterpolateMode::bilinear_pillow) ? PILLOW_BILINEAR_WINDOW_SCALE * scaleClip :
                                                                       PILLOW_BICUBIC_WINDOW_SCALE * scaleClip;
        args.filterLen = static_cast<int>(std::ceil(args.filterRadius) * 2 + 1);
        args.weightGen = (mode == InterpolateMode::bilinear_pillow) ? this->getPillowBilinearCoeffs:
                                                                       this->getPillowBicubicCoeffs;
        return args;
    };

    filterArgs filterArgsX = generateArgs(1.0f / fx);
    filterArgs filterArgsY = generateArgs(1.0f / fy);

    // index with Run Length Coding(start+len for each ow/oh)
    size_t weightLen = filterArgsX.filterLen * OW + filterArgsY.filterLen * OH;
    size_t boundLen = 2 * OW + 2 * OH;
    auxTable.resize(2 + weightLen + boundLen);
    size_t offset = 0;
    auxTable[offset] = filterArgsX.filterLen;
    auxTable[offset + 1] = filterArgsY.filterLen;
    offset += 2;
    float *weightX = reinterpret_cast<float*>(&auxTable[offset]);
    offset += filterArgsX.filterLen * OW;
    float *weightY = reinterpret_cast<float*>(&auxTable[offset]);
    offset += filterArgsY.filterLen * OH;
    int *indexX = static_cast<int*>(&auxTable[offset]);
    offset += 2 * OW;
    int *indexY = static_cast<int*>(&auxTable[offset]);

    auto generateTbl = [&] (int inLen, int outLen, float fScale, filterArgs args, float* weightTbl, int* idxTbl) {
        int min = 0;
        int max = 0;
        for (int ox = 0; ox < outLen; ox++) {
            float ixCenter = coordTransToInput(ox, fScale, inLen, outLen);
            min = static_cast<int>(ixCenter - args.filterRadius + 0.5f);
            if (min < 0) {
                min = 0;
            }
            max = static_cast<int>(ixCenter + args.filterRadius + 0.5f);
            if (max > inLen) {
                max = inLen;
            }
            // use [min, max) range of input to get output
            // below let max become len
            max -= min;
            idxTbl[2 * ox] = min;
            idxTbl[2 * ox + 1] = max;

            size_t offset = ox * args.filterLen;
            float weightSum = 0;
            int ix = 0;
            for (ix = 0; ix < max; ix++) {
                // use distance to center as a parameter to compute weight
                float w = args.weightGen((ix + min - ixCenter + 0.5) * args.ScaleClipReciprocal);
                weightTbl[offset + ix] = w;
                weightSum += w;
            }
            if (weightSum != 0) {
                for (ix = 0; ix < max; ix++) {
                    weightTbl[offset + ix] /= weightSum;
                }
            }

            // filterlen is maximum possible len, set others to 0 for possible uniform process(vector)
            for (; ix < args.filterLen; ix++)
                weightTbl[offset + ix] = 0.f;
        }
    };

    generateTbl(IW, OW, fx, filterArgsX, weightX, indexX);
    generateTbl(IH, OH, fy, filterArgsY, weightY, indexY);
}

void Interpolate::InterpolateRefExecutor::NNRef(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int ID, int IH, int IW,
                                                          int OD, int OH, int OW) {
    int *index_d = static_cast<int*>(&auxTable[0]);
    int *index_h = static_cast<int*>(&auxTable[OD]);
    int *index_w = static_cast<int*>(&auxTable[OD + OH]);

    const float *in_ptr_f32 = reinterpret_cast<const float *>(in_ptr_);
    float *out_ptr_f32 = reinterpret_cast<float *>(out_ptr_);

    parallel_for3d(B, C, OD, [&](size_t b, size_t c, size_t od) {
        const float *in_ptr = in_ptr_f32 + (IW * IH * ID * C * b + IW * IH * ID * c + IW * IH * index_d[od]);
        float *out_ptr = out_ptr_f32 + (OW * OH * OD * C * b + OW * OH * OD * c + OW * OH * od);
        for (int oh = 0; oh < OH; oh++) {
            const float *in_ptr_h = in_ptr + (IW * index_h[oh]);
            float *out_ptr_h = out_ptr + (OW * oh);
            for (int ow = 0; ow < OW; ow++) {
                out_ptr_h[ow] = in_ptr_h[index_w[ow]];
            }
        }
    });
}

void Interpolate::InterpolateRefExecutor::linearOnnxRef(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int ID, int IH, int IW,
                                                                  int OD, int OH, int OW) {
    std::vector<int*> indexPtr(MAX_INPUT_INTERPOLATE, 0);
    std::vector<float*> weightPtr(MAX_INPUT_INTERPOLATE, 0);
    // FrontTopLeft:0, FrontTopRight:1, FrontBottomLeft:2, FrontBottomRight:3,
    // EndTopLeft:4,   EndTopRight:5,   EndBottomLeft:6,   EndBottomRight:7
    // weight: Left:0, ritht:1, top:2, bottom:3, front:4, end:5

    int eltInGrid = (spatialDimSize > 2) ? MAX_INPUT_INTERPOLATE : ((spatialDimSize > 1) ? 4 : 2);
    int scratchLen = rnd_up(eltInGrid * OW * OH * OD, 16);

    indexPtr[0] = static_cast<int*>(&auxTable[0]);
    indexPtr[1] = static_cast<int*>(&auxTable[OW * OH * OD]);
    weightPtr[0] = reinterpret_cast<float*>(&auxTable[scratchLen]);
    weightPtr[1] = reinterpret_cast<float*>(&auxTable[scratchLen + OW * OH * OD]);
    if (spatialDimSize > 1) {
        indexPtr[2] = static_cast<int*>(&auxTable[2 * OW * OH * OD]);
        indexPtr[3] = static_cast<int*>(&auxTable[3 * OW * OH * OD]);
        weightPtr[2] = reinterpret_cast<float*>(&auxTable[scratchLen + 2 * OW * OH * OD]);
        weightPtr[3] = reinterpret_cast<float*>(&auxTable[scratchLen + 3 * OW * OH * OD]);
    }
    if (spatialDimSize > 2) {
        indexPtr[4] = static_cast<int*>(&auxTable[4 * OW * OH * OD]);
        indexPtr[5] = static_cast<int*>(&auxTable[5 * OW * OH * OD]);
        indexPtr[6] = static_cast<int*>(&auxTable[6 * OW * OH * OD]);
        indexPtr[7] = static_cast<int*>(&auxTable[7 * OW * OH * OD]);
        weightPtr[4] = reinterpret_cast<float*>(&auxTable[scratchLen + 4 * OW * OH * OD]);
        weightPtr[5] = reinterpret_cast<float*>(&auxTable[scratchLen + 5 * OW * OH * OD]);
    }

    const float *in_ptr_f32 = reinterpret_cast<const float *>(in_ptr_);
    float *out_ptr_f32 = reinterpret_cast<float *>(out_ptr_);

    parallel_for2d(B, C, [&](size_t b, size_t c) {
        float *out_ptr_nc = out_ptr_f32 + (OD * OH * OW * C * b + OD * OH * OW * c);
        const float *in_ptr_nc = in_ptr_f32 + (ID * IH * IW * C * b + ID * IH * IW * c);
        // do not combined 1d/2d to 3d unified process to get rid of invalid computing.
        switch (spatialDimSize) {
            case 1:
                for (int i = 0; i < OW; i++) {
                    float src0 = in_ptr_nc[indexPtr[0][i]];
                    float src1 = in_ptr_nc[indexPtr[1][i]];

                    out_ptr_nc[i] = src0 * weightPtr[0][i] +
                                    src1 * weightPtr[1][i];
                }
                break;
            case 2:
                for (int i = 0; i < OH * OW; i++) {
                    float src00 = in_ptr_nc[indexPtr[0][i]];
                    float src01 = in_ptr_nc[indexPtr[1][i]];
                    float src10 = in_ptr_nc[indexPtr[2][i]];
                    float src11 = in_ptr_nc[indexPtr[3][i]];

                    out_ptr_nc[i] = src00 * weightPtr[2][i] * weightPtr[0][i] +
                                    src01 * weightPtr[2][i] * weightPtr[1][i] +
                                    src10 * weightPtr[3][i] * weightPtr[0][i] +
                                    src11 * weightPtr[3][i] * weightPtr[1][i];
                }
                break;
            case 3:
                for (int i = 0; i < OD * OH * OW; i++) {
                    float src000 = in_ptr_nc[indexPtr[0][i]];
                    float src001 = in_ptr_nc[indexPtr[1][i]];
                    float src010 = in_ptr_nc[indexPtr[2][i]];
                    float src011 = in_ptr_nc[indexPtr[3][i]];
                    float src100 = in_ptr_nc[indexPtr[4][i]];
                    float src101 = in_ptr_nc[indexPtr[5][i]];
                    float src110 = in_ptr_nc[indexPtr[6][i]];
                    float src111 = in_ptr_nc[indexPtr[7][i]];

                    // float dstValue =
                    // weightPtr[4][i] * weightPtr[2][i] * weightPtr[0][i] * src000 +
                    // weightPtr[4][i] * weightPtr[2][i] * weightPtr[1][i] * src001 +
                    // weightPtr[4][i] * weightPtr[3][i] * weightPtr[0][i] * src010 +
                    // weightPtr[4][i] * weightPtr[3][i] * weightPtr[1][i] * src011 +
                    // weightPtr[5][i] * weightPtr[2][i] * weightPtr[0][i] * src100 +
                    // weightPtr[5][i] * weightPtr[2][i] * weightPtr[1][i] * src101 +
                    // weightPtr[5][i] * weightPtr[3][i] * weightPtr[0][i] * src110 +
                    // weightPtr[5][i] * weightPtr[3][i] * weightPtr[1][i] * src111;

                    out_ptr_nc[i] =
                    weightPtr[4][i] * (weightPtr[2][i] * (weightPtr[0][i] * src000 +
                                                          weightPtr[1][i] * src001) +
                                       weightPtr[3][i] * (weightPtr[0][i] * src010 +
                                                          weightPtr[1][i] * src011)) +
                    weightPtr[5][i] * (weightPtr[2][i] * (weightPtr[0][i] * src100 +
                                                          weightPtr[1][i] * src101) +
                                       weightPtr[3][i] * (weightPtr[0][i] * src110 +
                                                          weightPtr[1][i] * src111));
                }
                break;
            default:
                break;
        }
    });
}

void Interpolate::InterpolateRefExecutor::cubicRef(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int IH, int IW, int OH, int OW) {
    const int idxNum = 1;
    int *xOrigin = static_cast<int*>(&auxTable[0]);
    float *xFactor = reinterpret_cast<float*>(&auxTable[OW]);
    int *yOrigin = static_cast<int*>(&auxTable[(CUBIC_GRID_LEN + idxNum) * OW]);
    float *yFactor = reinterpret_cast<float*>(&auxTable[(CUBIC_GRID_LEN + idxNum) * OW + OH]);

    const float *in_ptr_f32 = reinterpret_cast<const float *>(in_ptr_);
    float *out_ptr_f32 = reinterpret_cast<float *>(out_ptr_);

    parallel_for4d(B, C, OH, OW, [&](size_t n, size_t c, size_t oy, size_t ox) {
        const float *in_ptr_nc = in_ptr_f32 + (IW * IH * C * n + IW * IH * c);
        float *out_ptr_nc = out_ptr_f32 + (OW * OH * C * n + OW * OH * c);

        int iy = yOrigin[oy];
        int ix = xOrigin[ox];

        float retY = 0.f;
        for (int y = iy - 1, i = 0; y <= iy + 2; y++, i++) {
            int yInRange = std::max(0, std::min(y, IH - 1));
            const float *in_ptr_nch = in_ptr_nc + IW * yInRange;
            float retX = 0.f;
            for (int x = ix - 1, j = 0; x <= ix + 2; x++, j++) {
                int xInRange = std::max(0, std::min(x, IW - 1));
                retX += xFactor[ox * CUBIC_GRID_LEN + j] * in_ptr_nch[xInRange];
            }
            retY += yFactor[oy * CUBIC_GRID_LEN + i] * retX;
        }
        out_ptr_nc[oy * OW + ox] = retY;
    });
}

float Interpolate::InterpolateRefExecutor::getValue(const uint8_t *base, size_t offset, ov::element::Type prec) {
    const uint8_t *baseOffset = base + offset;
    switch (prec) {
        case ov::element::u8: {
            return static_cast<float>(*baseOffset);
            break;
        }
        case ov::element::i8: {
            const int8_t *valuePtr = reinterpret_cast<const int8_t *>(baseOffset);
            return static_cast<float>(*valuePtr);
            break;
        }
        case ov::element::bf16: {
            const uint16_t *valuePtr = reinterpret_cast<const uint16_t *>(baseOffset);
            return bfloat16_t::from_bits(*valuePtr);
            break;
        }
        case ov::element::f32: {
            const float *valuePtr = reinterpret_cast<const float *>(baseOffset);
            return *valuePtr;
            break;
        }
        default: {
            OPENVINO_THROW("Interpolate layer does not support precision: ", prec);
            break;
        }
    }
}

void Interpolate::InterpolateRefExecutor::setValue(uint8_t *base, size_t offset, float value, ov::element::Type prec) {
    uint8_t *baseOffset = base + offset;
    switch (prec) {
        case ov::element::u8: {
            uint8_t data = static_cast<uint8_t>(value < 0 ? 0 : value);
            cpu_memcpy(baseOffset, &data, 1);
            break;
        }
        case ov::element::i8: {
            int8_t data = static_cast<int8_t>(value);
            cpu_memcpy(baseOffset, &data, 1);
            break;
        }
        case ov::element::bf16: {
            uint16_t data = bfloat16_t(value).to_bits();
            cpu_memcpy(baseOffset, &data, 2);
            break;
        }
        case ov::element::f32: {
            cpu_memcpy(baseOffset, &value, sizeof(float));
            break;
        }
        default: {
            OPENVINO_THROW("Interpolate layer does not support precision: ", prec);
            break;
        }
    }
}

void Interpolate::InterpolateRefExecutor::linearInterpolation(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int ID, int IH, int IW,
                                          float fx, float fy, float fz, int OD, int OH, int OW, int kernel_width, bool antialias) {
    if (IW == OW && IH == OH && ID == OD) {
        size_t spatialDimSize = IW * IH * ID;
        // TODO: enable when fusing into interp with linear mode will support
        if (/*fusedWith.empty() &&*/ inputPrec == outputPrec) {
            size_t size = B * C * spatialDimSize * srcDataSize;
            cpu_memcpy(out_ptr_, in_ptr_, size);
        } else {
            parallel_for2d(B, C, [&](size_t b, size_t c) {
                const uint8_t *in_ptr_nc = in_ptr_ + (spatialDimSize * C * b + spatialDimSize * c) * srcDataSize;
                uint8_t *out_ptr_nc = out_ptr_ + (spatialDimSize * C * b + spatialDimSize * c) * dstDataSize;
                for (size_t i = 0; i < spatialDimSize; i++) {
                    float dstValue = getValue(in_ptr_nc, i * srcDataSize, inputPrec);
                    setValue(out_ptr_nc, i * dstDataSize, dstValue, outputPrec);
                }
            });
        }
        return;
    }

    float ax = antialias ? fx : 1.0f;
    float ay = antialias ? fy : 1.0f;
    float az = antialias ? fz : 1.0f;

    int rx = (fx > 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / ax));
    int ry = (fy > 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / ay));
    int rz = (fz > 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / az));

    int diaOD = 2 * rz + 1;
    int diaOH = 2 * ry + 1;
    int diaOW = 2 * rx + 1;
    int sizeOD = OD * diaOD;
    int sizeOH = OH * diaOH;
    int sizeOW = OW * diaOW;

    float *weightTable = reinterpret_cast<float*>(&auxTable[0]);
    float *weightOD = static_cast<float*>(&weightTable[0]);
    float *weightOH = static_cast<float*>(&weightTable[sizeOD]);
    float *weightOW = static_cast<float*>(&weightTable[sizeOD + sizeOH]);

    int *idxTable = static_cast<int*>(&auxTable[sizeOD + sizeOH + sizeOW]);
    int *idxOD = static_cast<int*>(&idxTable[0]);
    int *idxOH = static_cast<int*>(&idxTable[sizeOD]);
    int *idxOW = static_cast<int*>(&idxTable[sizeOD + sizeOH]);

    parallel_for2d(B, C, [&](size_t b, size_t c) {
        const uint8_t *in_ptr_nc = in_ptr_ + (IW * IH * ID * C * b + IW * IH * ID * c) * srcDataSize;
        uint8_t *out_ptr_nc = out_ptr_ + (OW * OH * OD * C * b + OW * OH * OD * c) * dstDataSize;
        for (int oz = 0; oz < OD; oz++) {
            uint8_t *out_ptr_ncd = out_ptr_nc + (OW * OH * oz) * dstDataSize;
            for (int oy = 0; oy < OH; oy++) {
                uint8_t *out_ptr_ncdh = out_ptr_ncd + (OW * oy) * dstDataSize;
                for (int ox = 0; ox < OW; ox++) {
                    float sum = 0.f;
                    float wsum = 0.f;

                    // this comment explains the original algo.
                    // for (int z = iz_r - rz; z <= iz_r + rz; z++) {
                    //    for (int y = iy_r - ry; y <= iy_r + ry; y++) {
                    //        for (int x = ix_r - rx; x <= ix_r + rx; x++) {
                    //            bool is_continue =  z < 0                     ||
                    //                                y < 0                     ||
                    //                                x < 0                     ||
                    //                                z >= static_cast<int>(ID) ||
                    //                                y >= static_cast<int>(IH) ||
                    //                                x >= static_cast<int>(IW);
                    //            if (is_continue)
                    //                continue;

                    //            float dx = ix - x;
                    //            float dy = iy - y;
                    //            float dz = iz - z;

                    //            float w = ax * triangleCoeff(ax * dx) *
                    //                      ay * triangleCoeff(ay * dy) *
                    //                      az * triangleCoeff(az * dz);

                    //            sum += w * getValue(in_ptr_nc, (z * IH * IW + y * IW + x) * srcDataSize, inputPrec);
                    //            wsum += w;
                    //        }
                    //    }
                    //}

                    for (int iz = 0; iz < diaOD; iz++) {
                        if (weightOD[oz * diaOD + iz] == 0.f)
                            continue;
                        for (int iy = 0; iy < diaOH; iy++) {
                            if (weightOH[oy * diaOH + iy] == 0.f) {
                                continue;
                            }
                            for (int ix = 0; ix < diaOW; ix++) {
                                if (weightOW[ox * diaOW + ix] == 0.f) {
                                    continue;
                                }
                                float w = weightOD[oz * diaOD + iz] * weightOH[oy * diaOH + iy] * weightOW[ox * diaOW + ix];
                                float value = getValue(in_ptr_nc,
                                    (idxOD[oz * diaOD + iz] * IH * IW + idxOH[oy * diaOH + iy] * IW + idxOW[ox * diaOW + ix]) * srcDataSize, inputPrec);

                                sum += w * value;
                                wsum += w;
                            }
                        }
                    }

                    if (!wsum) {
                        setValue(out_ptr_ncdh, ox * dstDataSize, 0.f, outputPrec);
                    } else {
                        float dst_value = sum / wsum;
                        setValue(out_ptr_ncdh, ox * dstDataSize, dst_value, outputPrec);
                    }
                }
            }
        }
    });
}

void Interpolate::InterpolateRefExecutor::pillowRef(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int IH, int IW, int OH, int OW) {
    size_t offset = 0;
    int filterLenX = auxTable[offset];
    int filterLenY = auxTable[offset + 1];
    offset += 2;
    float *weightX = reinterpret_cast<float*>(&auxTable[offset]);
    offset += filterLenX * OW;
    float *weightY = reinterpret_cast<float*>(&auxTable[offset]);
    offset += filterLenY * OH;
    int *indexX = static_cast<int*>(&auxTable[offset]);
    offset += 2 * OW;
    int *indexY = static_cast<int*>(&auxTable[offset]);

    // workBuffer needed when both pass is true
    bool xPass = IW != OW;
    bool yPass = IH != OH;

    // --------    ----
    // |      |    |  |
    // |      |--> |  |
    // |      |    |  |
    // |      |    |  |
    // --------    ----
    //              \|/
    //             ----
    //             |  |
    //             |  |
    //             ----
    auto bc_loop = [&](size_t b, size_t c) {
        const uint8_t *in_ptr_nc = in_ptr_ + (IW * IH * C * b + IW * IH * c) * srcDataSize;
        uint8_t *out_ptr_nc = out_ptr_ + (OW * OH * C * b + OW * OH * c) * dstDataSize;
        uint8_t *xpass_out_ptr_nc = nullptr;
        const uint8_t *ypass_in_ptr_nc = nullptr;
        if (xPass && yPass) {
            size_t parallel_num = B * C;
            // IH * OW buf needed
            if (parallel_num < m_threads_num) {
                xpass_out_ptr_nc = static_cast<uint8_t*>(&pillow_working_buf[(OW * IH * C * b + OW * IH * c) * srcDataSize]);
                ypass_in_ptr_nc = static_cast<const uint8_t*>(&pillow_working_buf[(OW * IH * C * b + OW * IH * c) * srcDataSize]);
            } else {
                size_t threadsIdx = parallel_get_thread_num();
                size_t buffer_size = static_cast<size_t>(OW * IH);
                xpass_out_ptr_nc = static_cast<uint8_t*>(&pillow_working_buf[threadsIdx * buffer_size * srcDataSize]);
                ypass_in_ptr_nc = static_cast<const uint8_t*>(&pillow_working_buf[threadsIdx * buffer_size * srcDataSize]);
            }
        } else if (xPass && !yPass) {
            xpass_out_ptr_nc = out_ptr_nc;
        } else if (!xPass && yPass) {
            ypass_in_ptr_nc = in_ptr_nc;
        } else if (!xPass && !yPass) {
            cpu_memcpy(out_ptr_nc, in_ptr_nc, OH * OW * dstDataSize);
        }
        float result;
        int f, filterS, filterL;
        float* weight;
        if (xPass) {
            for (size_t ih = 0; ih < static_cast<size_t>(IH); ih++) {
                for (size_t ow = 0; ow < static_cast<size_t>(OW); ow++) {
                    filterS = indexX[ow * 2];
                    filterL = indexX[ow * 2 + 1];
                    weight = reinterpret_cast<float*>(&weightX[ow * filterLenX]);
                    result = 0.f;
                    for (f = 0; f < filterL; f++) {
                        float pixel = getValue(in_ptr_nc, (ih * IW + f + filterS) * srcDataSize, inputPrec);
                        result += pixel * weight[f];
                    }
                    if (!isFloatCompatible(outputPrec)) {
                        result = static_cast<float>(static_cast<int>(result >= 0.0 ? result + 0.5f : result - 0.5f));
                    }
                    setValue(xpass_out_ptr_nc, (ih * OW + ow) * dstDataSize, result, outputPrec);
                }
            }
        }
        if (yPass) {
            for (size_t oh = 0; oh < static_cast<size_t>(OH); oh++) {
                filterS = indexY[oh * 2];
                filterL = indexY[oh * 2 + 1];
                weight = reinterpret_cast<float*>(&weightY[oh * filterLenY]);
                for (size_t ow = 0; ow < static_cast<size_t>(OW); ow++) {
                    result = 0.f;
                    for (f = 0; f < filterL; f++) {
                        float pixel = getValue(ypass_in_ptr_nc, ((f + filterS) * OW + ow) * srcDataSize, inputPrec);
                        result += pixel * weight[f];
                    }
                    if (!isFloatCompatible(outputPrec)) {
                        result = static_cast<float>(static_cast<int>(result >= 0.0 ? result + 0.5f : result - 0.5f));
                    }
                    setValue(out_ptr_nc, (oh * OW + ow) * dstDataSize, result, outputPrec);
                }
            }
        }
    };

    parallel_nt_static(m_threads_num, [&](const int ithr, const int nthr) {
        for_2d(ithr, nthr, B, C, bc_loop);
    });
}

void Interpolate::InterpolateExecutorBase::create_pillow_working_buf(InterpolateLayoutType layout) {
    if (srcDimPad5d[3] == dstDim5d[3] || srcDimPad5d[4] == dstDim5d[4])
        return;
    size_t bufSize = srcDimPad5d[3] * dstDim5d[4] * srcDataSize; // IH * OW
    m_threads_num = parallel_get_max_threads();
    if (layout == InterpolateLayoutType::planar) {
        // B and C execute in parallel, need separate buf
        size_t parallel_num = srcDimPad5d[0] * srcDimPad5d[1];
        bufSize *= std::min(m_threads_num, parallel_num);
    } else {
        bufSize *= srcDimPad5d[1]; // *C
        // B execute in parallel, need separate buf
        size_t parallel_num = srcDimPad5d[0];
        bufSize *= std::min(m_threads_num, parallel_num);
    }
    pillow_working_buf.resize(bufSize);
}

Interpolate::InterpolateExecutorBase::InterpolateExecutorBase(const InterpolateAttrs& interpAttrs,
                                                      const VectorDims &srcDims,
                                                      const VectorDims &dstDims,
                                                      const std::vector<float> &dataScales) :
        mode(interpAttrs.mode), coordTransMode(interpAttrs.coordTransMode), configured_for_layout(interpAttrs.layout),
        inputPrec(interpAttrs.inPrc), outputPrec(interpAttrs.outPrc) {
    srcDimPad5d = to5Dim(getPaddedInputShape(srcDims, interpAttrs.padBegin, interpAttrs.padEnd));
    dstDim5d = to5Dim(dstDims);
    srcDataSize = interpAttrs.inPrc.size();
    dstDataSize = interpAttrs.outPrc.size();
    dataRank = srcDims.size();
    spatialDimSize = getSpatialDimsNum(dataRank);

    switch (mode) {
        case InterpolateMode::nearest: {
            buildTblNN(srcDimPad5d, dstDim5d, dataScales, interpAttrs.layout, interpAttrs.nearestMode);
            break;
        }
        case InterpolateMode::linear_onnx: {
            buildTblLinearOnnx(srcDimPad5d, dstDim5d, dataScales, interpAttrs.layout);
            break;
        }
        case InterpolateMode::linear: {
            static constexpr int LINEAR_KERNEL = 2;
            buildTblLinear(srcDimPad5d, dstDim5d, dataScales, LINEAR_KERNEL, interpAttrs.antialias);
            break;
        }
        case InterpolateMode::cubic: {
            buildTblCubic(srcDimPad5d, dstDim5d, dataScales, interpAttrs.cubeCoeff, interpAttrs.layout);
            break;
        }
        case InterpolateMode::bilinear_pillow:
        case InterpolateMode::bicubic_pillow: {
            buildTblPillow(srcDimPad5d, dstDim5d, dataScales, interpAttrs.cubeCoeff, interpAttrs.layout);
            if ((srcDimPad5d[4] != dstDim5d[4]) && (srcDimPad5d[3] != dstDim5d[3])) {
                create_pillow_working_buf(interpAttrs.layout);
            }
            break;
        }
        default: {
            OPENVINO_THROW("Interpolate executor does not support interpolate mode: ", mode);
            break;
        }
    }
}

Interpolate::InterpolateJitExecutor::InterpolateJitExecutor(const InterpolateAttrs& interpAttrs,
                                                                      const VectorDims &srcDims,
                                                                      const VectorDims &dstDims,
                                                                      const std::vector<float> &dataScales,
                                                                      const dnnl::primitive_attr &attr) :
        InterpolateExecutorBase(interpAttrs, srcDims, dstDims, dataScales) {
    auto jcp = jit_interpolate_config_params();
    jcp.mode = mode;
    jcp.src_prc = interpAttrs.inPrc;
    jcp.dst_prc = interpAttrs.outPrc;
    jcp.src_data_size = jcp.src_prc.size();
    jcp.dst_data_size = jcp.dst_prc.size();
    jcp.indices_size = sizeof(int);
    jcp.C = dstDim5d[1];
    jcp.OW = dstDim5d[4];
    jcp.OH = dstDim5d[3];
    jcp.OD = dstDim5d[2];
    jcp.IW = srcDimPad5d[4];
    jcp.IH = srcDimPad5d[3];
    jcp.ID = srcDimPad5d[2];
    jcp.spatial_dim_size = getSpatialDimsNum(srcDims.size());
    jcp.layout = interpAttrs.layout;
    if (mode == InterpolateMode::bilinear_pillow || mode == InterpolateMode::bicubic_pillow) {
        jcp.filterLenX = auxTable[0];
        jcp.filterLenY = auxTable[1];
        jcp.bound = static_cast<int*>(&auxTable[2 + jcp.OW * jcp.filterLenX + jcp.OH * jcp.filterLenY]);
    }
#if defined(OPENVINO_ARCH_X86_64)
    if (jcp.layout != InterpolateLayoutType::planar) {
        if (mayiuse(cpu::x64::avx512_core)) {
            interpolateKernel.reset(new jit_uni_interpolate_kernel_f32<cpu::x64::avx512_core>(jcp, *attr.get()));
        } else if (mayiuse(cpu::x64::avx2)) {
            interpolateKernel.reset(new jit_uni_interpolate_kernel_f32<cpu::x64::avx2>(jcp, *attr.get()));
        } else if (mayiuse(cpu::x64::sse41)) {
            interpolateKernel.reset(new jit_uni_interpolate_kernel_f32<cpu::x64::sse41>(jcp, *attr.get()));
        }
    } else if (mayiuse(cpu::x64::avx2) && interpAttrs.inPrc == ov::element::f32) {
        // gather ISA(for planar JIT kernel) for avx2 and fp32
        interpolateKernel.reset(new jit_uni_interpolate_kernel_f32<cpu::x64::avx2>(jcp, *attr.get()));
    } else {
        OPENVINO_THROW("Can't create InterpolateJitExecutor");
    }
#endif // OPENVINO_ARCH_X86_64
    if (interpolateKernel) {
        interpolateKernel->create_ker();
    } else {
        OPENVINO_THROW("Can't compile InterpolateJitExecutor");
    }
}

void Interpolate::InterpolateJitExecutor::exec(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_) {
    size_t N = srcDimPad5d[0], C = srcDimPad5d[1], ID = srcDimPad5d[2], IH = srcDimPad5d[3], IW = srcDimPad5d[4];
    size_t OD = dstDim5d[2], OH = dstDim5d[3], OW = dstDim5d[4];

    if (!interpolateKernel) {
        OPENVINO_THROW("Can't execute, kernel for Interpolate node is not compiled");
    }
    switch (mode) {
        case InterpolateMode::nearest: {
            if (configured_for_layout == InterpolateLayoutType::planar) {
                NNPlanar(in_ptr_, out_ptr_, post_ops_data_, N, C, ID, IH, IW, OD, OH, OW);
            } else {
                NNCGathered(in_ptr_, out_ptr_, post_ops_data_, N, C, ID, IH, IW, OD, OH, OW);
            }
            break;
        }
        case InterpolateMode::linear_onnx: {
            if (configured_for_layout == InterpolateLayoutType::planar) {
                linearOnnxPlanar(in_ptr_, out_ptr_, post_ops_data_, N, C, ID, IH, IW, OD, OH, OW);
            } else {
                linearOnnxCGathered(in_ptr_, out_ptr_, post_ops_data_, N, C, ID, IH, IW, OD, OH, OW);
            }
            break;
        }
        case InterpolateMode::cubic: {
            if (configured_for_layout == InterpolateLayoutType::planar) {
                cubicPlanar(in_ptr_, out_ptr_, post_ops_data_, N, C, IH, IW, OH, OW);
            } else {
                cubicCGathered(in_ptr_, out_ptr_, post_ops_data_, N, C, IH, IW, OH, OW);
            }
            break;
        }
        case InterpolateMode::bilinear_pillow:
        case InterpolateMode::bicubic_pillow: {
            if (configured_for_layout == InterpolateLayoutType::by_channel) {
                pillowCGathered(in_ptr_, out_ptr_, post_ops_data_, N, C, IH, IW, OH, OW);
            } else {
                OPENVINO_THROW("Only channel_first jit kernel is supported for pillow mode", mode);
            }
            break;
        }
        default: {
            OPENVINO_THROW("InterpolateJitExecutor has unsupported interpolate mode: ", mode);
        }
    }
}

void Interpolate::InterpolateRefExecutor::exec(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_) {
    size_t N = srcDimPad5d[0], C = srcDimPad5d[1], ID = srcDimPad5d[2], IH = srcDimPad5d[3], IW = srcDimPad5d[4];
    size_t OD = dstDim5d[2], OH = dstDim5d[3], OW = dstDim5d[4];

    switch (mode) {
        case InterpolateMode::nearest: {
            NNRef(in_ptr_, out_ptr_, N, C, ID, IH, IW, OD, OH, OW);
            break;
        }
        case InterpolateMode::linear_onnx: {
            linearOnnxRef(in_ptr_, out_ptr_, N, C, ID, IH, IW, OD, OH, OW);
            break;
        }
        case InterpolateMode::cubic: {
            cubicRef(in_ptr_, out_ptr_, N, C, IH, IW, OH, OW);
            break;
        }
        case InterpolateMode::linear: {
            float fz = (dataRank == 5) ? dataScales[dataRank - 3] : 1.f;
            float fy = dataScales[dataRank - 2];
            float fx = dataScales[dataRank - 1];

            bool isDownsample = (fx < 1.f) || (fy < 1.f) || (fz < 1.f);
            int kernel_width = 2;
            linearInterpolation(in_ptr_, out_ptr_, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW, kernel_width, isDownsample && antialias);
            break;
        }
        case InterpolateMode::bilinear_pillow:
        case InterpolateMode::bicubic_pillow: {
            pillowRef(in_ptr_, out_ptr_, N, C, IH, IW, OH, OW);
            break;
        }
        default: {
            OPENVINO_THROW("Interpolate layer has unsupported interpolate mode: ", mode);
        }
    }
}

size_t Interpolate::getSpatialDimsNum(const Dim rank) {
    switch (rank) {
        case 1:
        case 3:
            return 1;
        case 2:
        case 4:
            return 2;
        case 5:
            return 3;
        default:
            OPENVINO_THROW("Can't define number spatial");
    }
}

bool Interpolate::canFuse(const NodePtr& node) const {
    if (!mayiuse(cpu::x64::sse41) ||
        interpAttrs.mode == InterpolateMode::linear ||
        interpAttrs.mode == InterpolateMode::bilinear_pillow ||
        interpAttrs.mode == InterpolateMode::bicubic_pillow ||
        (!one_of(dataRank, 4u, 5u) && !mayiuse(cpu::x64::avx2))) {
        return false;
    }

    return canFuseSimpleOperation(node);
}

bool Interpolate::created() const {
    return getType() == Type::Interpolate;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
