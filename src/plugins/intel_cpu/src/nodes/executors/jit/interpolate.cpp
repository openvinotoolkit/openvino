// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "interpolate.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <common/primitive_attr.hpp>
#include <common/primitive_hashing_utils.hpp>
#include <common/utils.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <utility>
#include <vector>

#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "eltwise.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/common/blocked_desc_creator.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/interpolate_config.hpp"
#include "nodes/node_config.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/enum_names.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/interpolate.hpp"
#include "shape_inference/shape_inference.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/bfloat16.hpp"
#include "utils/general_utils.h"
#include "utils/ngraph_utils.hpp"
#include "utils/precision_support.h"
#include "interpolate.h"

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#    include <xbyak/xbyak.h>

#    include <common/c_types_map.hpp>
#    include <unordered_map>

#    include "cpu/x64/injectors/jit_uni_depthwise_injector.hpp"
#    include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#    include "cpu/x64/injectors/jit_uni_quantization_injector.hpp"
#    include "cpu/x64/jit_generator.hpp"
#    include "emitters/plugin/x64/jit_emitter.hpp"
#    include "emitters/plugin/x64/jit_load_store_emitters.hpp"
#    include "utils/cpu_utils.hpp"
#endif

using namespace dnnl;

using namespace dnnl::impl;
using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_interpolate_call_args, field)

namespace ov::intel_cpu {
static inline bool isFloatCompatible(ov::element::Type prc) {
    return any_of(prc, ov::element::f32, ov::element::bf16, ov::element::f16, ov::element::f64);
}

#if defined(OPENVINO_ARCH_X86_64)

template <cpu_isa_t isa>
struct jit_uni_interpolate_kernel_f32 : public jit_uni_interpolate_kernel, public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_interpolate_kernel_f32)

    explicit jit_uni_interpolate_kernel_f32(jit_interpolate_config_params jcp, const dnnl_primitive_attr& attr)
        : jit_uni_interpolate_kernel(jcp, attr),
          jit_generator_t(jit_name()) {}

    void create_ker() override {
        jit_generator_t::create_kernel();
        ker_ = jit_kernel_cast<decltype(ker_)>(jit_ker());
    }

    void generate() override {
        // dummy second reg_tmp_64 as no fill needed
        load_pool_gpr_idxs = {static_cast<size_t>(reg_tmp_64.getIdx()), static_cast<size_t>(reg_tmp_64.getIdx())};
        store_pool_gpr_idxs = {static_cast<size_t>(reg_tmp_64.getIdx())};
        store_pool_vec_idxs = {static_cast<size_t>(vmm_zero.getIdx())};

        const auto& p = attr_.post_ops_;
        for (int i = 0; i < p.len(); i++) {
            auto& post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors.push_back(std::make_shared<jit_uni_eltwise_injector_t<isa>>(this,
                                                                                              post_op.eltwise.alg,
                                                                                              post_op.eltwise.alpha,
                                                                                              post_op.eltwise.beta,
                                                                                              1.F,
                                                                                              data_type::f32));
            } else if (post_op.is_depthwise()) {
                depthwise_injectors.push_back(std::make_shared<jit_uni_depthwise_injector_f32<isa>>(this, post_op));
            } else if (post_op.is_quantization()) {
                quantization_injectors.push_back(std::make_shared<jit_uni_quantization_injector_f32<isa>>(this,
                                                                                                          post_op,
                                                                                                          vmm_d_weights,
                                                                                                          vmm_d_bias,
                                                                                                          reg_d_weights,
                                                                                                          reg_d_bias));
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
                assert(
                    !"unsupported memory layout for interpolate layer with bilinear_pillow and bicubic_pillow modes.");
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
        for (auto& inj : eltwise_injectors) {
            inj->prepare_table();
        }
        if ((jcp_.mode == InterpolateMode::cubic) && (jcp_.layout == InterpolateLayoutType::planar)) {
            prepare_cubic_planar_table();
        }
    }

private:
    using Vmm =
        typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;

    const int vlen = cpu_isa_traits_t<isa>::vlen;
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
    Xbyak::Reg64 reg_table = rdx;  // do not need reg_index_offset in this mode, so use rdx

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

    std::vector<std::shared_ptr<jit_uni_eltwise_injector_t<isa>>> eltwise_injectors;
    std::vector<std::shared_ptr<jit_uni_depthwise_injector_f32<isa>>> depthwise_injectors;
    std::vector<std::shared_ptr<jit_uni_quantization_injector_f32<isa>>> quantization_injectors;

    void emit_emitters_data() {
        for (const auto& emitter : emitters) {
            if (emitter.second) {
                emitter.second->emit_data();
            }
        }
    }

    void load(Xbyak::Reg64 reg_src, Vmm vmm_src, const int elt_num, const int offset = 0) {
        emit_load(reg_src, vmm_src, jcp_.src_prc, ov::element::f32, elt_num, offset);
    }

    void load_weights(Xbyak::Reg64 reg_src, Vmm vmm_src, const int elt_num, const int offset = 0) {
        emit_load(reg_src, vmm_src, ov::element::f32, ov::element::f32, elt_num, offset);
    }

    void emit_load(Xbyak::Reg64 reg_src,
                   Vmm vmm_src,
                   ov::element::Type src_prc,
                   ov::element::Type dst_prc,
                   const int elt_num,
                   const int offset = 0) {
        const auto seed = load_emitter_params(src_prc, dst_prc, elt_num).hash();
        if (!emitters[seed]) {
            emitters[seed] = std::make_unique<jit_load_emitter>(this, isa, src_prc, dst_prc, elt_num);
        }

        emitters[seed]->emit_code({static_cast<size_t>(reg_src.getIdx()), static_cast<size_t>(offset)},
                                  {static_cast<size_t>(vmm_src.getIdx())},
                                  {},
                                  {load_pool_gpr_idxs});
    }

    void store(Vmm vmm_dst, Xbyak::Reg64 reg_dst, const int elt_num, const int offset = 0) {
        const auto seed = store_emitter_params(ov::element::f32, jcp_.dst_prc, elt_num).hash();
        if (!emitters[seed]) {
            emitters[seed] = std::make_unique<jit_store_emitter>(this, isa, ov::element::f32, jcp_.dst_prc, elt_num);
        }

        // for cases when Store emitter need 2 aux vmm we can use vmm_dst as second aux vmm
        std::vector<size_t> local_store_pool_vec_idxs = {static_cast<size_t>(vmm_dst.getIdx())};
        local_store_pool_vec_idxs.insert(local_store_pool_vec_idxs.begin(),
                                         store_pool_vec_idxs.begin(),
                                         store_pool_vec_idxs.end());

        emitters[seed]->emit_code({static_cast<size_t>(vmm_dst.getIdx())},
                                  {static_cast<size_t>(reg_dst.getIdx()), static_cast<size_t>(offset)},
                                  {local_store_pool_vec_idxs},
                                  {store_pool_gpr_idxs});
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
        auto reg_params = abi_param1;

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
        int f = 0;
        int filterS = 0;
        int filterL = 0;
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
                            uni_vroundps(vmm_dst, vmm_dst, 0x0);  // Round near
                        }
                        // src_prc, dst_prc and buf ov::element::Type is the same, otherwise need another store with
                        // buf(src) precision
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
                            uni_vroundps(vmm_dst, vmm_dst, 0x0);  // Round near
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
                            uni_vroundps(vmm_dst, vmm_dst, 0x0);  // Round near
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
                            uni_vroundps(vmm_dst, vmm_dst, 0x0);  // Round near
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

            // reset work_amount to OW
            mov(reg_work_amount, jcp_.OW);

            Xbyak::Reg64 reg_src_h = rsi;
            mov(reg_src_h, reg_src);
            // index_h * IW * dataSize done when built to avoid redundent compute
            mov(reg_index_offset, dword[reg_index_h]);
            // If H is out-of-range (sentinel -1), fill zeros for the entire row
            Xbyak::Label row_zero_label, row_zero_loop, row_zero_end, row_tail_zero;
            cmp(reg_index_offset, -1);
            je(row_zero_label, T_NEAR);
            add(reg_src_h, Xbyak::Reg64(reg_index_offset.getIdx()));  // reg_src_h now point to begin of row

            // reset index_w, index_w * dataSize done when built to avoid redundent compute
            mov(reg_index, reg_index_w);

            Xbyak::Label nn_loop_label;
            Xbyak::Label nn_loop_end_label;
            Xbyak::Label nn_tail_loop_label;
            Xbyak::Label nn_tail_loop_end_label;

            L(nn_loop_label);  // inner loop
            {
                cmp(reg_work_amount, vector_step);
                jl(nn_loop_end_label, T_NEAR);

                uni_vmovdqu(vmm_index, ptr[reg_index]);
                uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
                // Gather for planar; sentinel handling omitted in vector path (planar path not used for failing cases)
                vgatherdps(vmm_val, ptr[reg_src_h + vmm_index], vmm_mask);
                if (attr_.post_ops_.len() != 0) {
                    apply_post_ops(jcp_.dst_prc, 1);
                }
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
                cmp(reg_index_offset, -1);
                je(row_tail_zero, T_NEAR);
                add(reg_src_aux, Xbyak::Reg64(reg_index_offset.getIdx()));
                load(reg_src_aux, vmm_val, scalar_step);
                if (attr_.post_ops_.len() != 0) {
                    apply_post_ops(jcp_.dst_prc, 1);
                }
                store(vmm_val, reg_dst, scalar_step);
                add(reg_dst, scalar_step * jcp_.dst_data_size);
                add(reg_index, scalar_step * jcp_.indices_size);
                sub(reg_work_amount, scalar_step);
                jmp(nn_tail_loop_label, T_NEAR);
                L(row_tail_zero);
                uni_vpxor(vmm_val, vmm_val, vmm_val);
                if (attr_.post_ops_.len() != 0) {
                    apply_post_ops(jcp_.dst_prc, 1);
                }
                store(vmm_val, reg_dst, scalar_step);
                add(reg_dst, scalar_step * jcp_.dst_data_size);
                add(reg_index, scalar_step * jcp_.indices_size);
                sub(reg_work_amount, scalar_step);
                jmp(nn_tail_loop_label, T_NEAR);
            }
            L(nn_tail_loop_end_label);  // inner loop end

            // increment index_h to next row
            add(reg_index_h, jcp_.indices_size);

            sub(reg_work_amount_oh, 1);
            jmp(out_loop_label, T_NEAR);

            // Zero entire row path (out-of-range H)
            L(row_zero_label);
            {
                mov(reg_index, reg_index_w);
                // zero main loop over OW
                L(row_zero_loop);
                {
                    cmp(reg_work_amount, vector_step);
                    jl(row_zero_end, T_NEAR);
                    uni_vpxor(vmm_val, vmm_val, vmm_val);
                    if (attr_.post_ops_.len() != 0) {
                        apply_post_ops(jcp_.dst_prc, 1);
                    }
                    store(vmm_val, reg_dst, vector_step);
                    add(reg_dst, vector_step * jcp_.dst_data_size);
                    add(reg_index, vector_step * jcp_.indices_size);
                    sub(reg_work_amount, vector_step);
                    jmp(row_zero_loop, T_NEAR);
                }
                L(row_zero_end);
                Xbyak::Label row_zero_tail_loop;
                Xbyak::Label row_zero_tail_end;
                L(row_zero_tail_loop);
                {
                    cmp(reg_work_amount, 1);
                    jl(row_zero_tail_end, T_NEAR);
                    uni_vpxor(vmm_val, vmm_val, vmm_val);
                    if (attr_.post_ops_.len() != 0) {
                        apply_post_ops(jcp_.dst_prc, 1);
                    }
                    store(vmm_val, reg_dst, scalar_step);
                    add(reg_dst, scalar_step * jcp_.dst_data_size);
                    add(reg_index, scalar_step * jcp_.indices_size);
                    sub(reg_work_amount, scalar_step);
                    jmp(row_zero_tail_loop, T_NEAR);
                }
                L(row_zero_tail_end);
                // increment index_h to next row
                add(reg_index_h, jcp_.indices_size);
                sub(reg_work_amount_oh, 1);
                jmp(out_loop_label, T_NEAR);
            }
        }
        L(out_loop_end);
    }

    void nn_blk() {
        Xbyak::Label nn_loop_label;
        Xbyak::Label nn_loop_end_label;
        Xbyak::Label nn_zero_vec_label;
        Xbyak::Label nn_zero_vec_continue;
        L(nn_loop_label);
        {
            cmp(reg_work_amount, 0);
            jle(nn_loop_end_label, T_NEAR);

            mov(reg_src_aux, reg_src);
            mov(reg_index_offset, dword[reg_index]);
            // If index is -1, it indicates out-of-range (zero padding)
            cmp(reg_index_offset, -1);
            je(nn_zero_vec_label, T_NEAR);
            add(reg_src_aux, Xbyak::Reg64(reg_index_offset.getIdx()));

            load(reg_src_aux, vmm_val, vector_step);
            if (attr_.post_ops_.len() != 0) {
                apply_post_ops(jcp_.dst_prc, 0);
            }
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
        L(nn_zero_vec_label);
        {
            uni_vpxor(vmm_val, vmm_val, vmm_val);
            if (attr_.post_ops_.len() != 0) {
                apply_post_ops(jcp_.dst_prc, 0);
            }
            store(vmm_val, reg_dst, vector_step);
            add(reg_dst, vector_step * jcp_.dst_data_size);
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

            // inner loop for C
            Xbyak::Label nn_loop_label;
            Xbyak::Label nn_loop_end_label;
            Xbyak::Label nn_tail_loop_label;
            Xbyak::Label nn_tail_loop_end_label;
            Xbyak::Label nn_zero_vec_label;
            Xbyak::Label nn_zero_tail_label;

            // inner loop for C
            // get current loop address reg_src_aux, from reg_src which is unchange, point this C * OW.
            // reset offset and work_amount.
            // dst and index address is continous, advanced each interator.
            mov(reg_src_aux, reg_src);
            // index*C*dataSize done when built to avoid redundent compute
            mov(reg_index_offset, dword[reg_index]);
            // If index is -1 for this column (out-of-range), fill zeros for the entire channel vector
            cmp(reg_index_offset, -1);
            je(nn_zero_vec_label, T_NEAR);
            // opRR need same bit length input
            add(reg_src_aux, Xbyak::Reg64(reg_index_offset.getIdx()));

            mov(reg_work_amount, reg_work_amount_bk);
            if (attr_.post_ops_.len() != 0) {
                mov(reg_oc_off, reg_oc_off_bk);
            }

            L(nn_loop_label);
            {
                cmp(reg_work_amount, vector_step);
                jl(nn_loop_end_label, T_NEAR);

                load(reg_src_aux, vmm_val, vector_step);
                if (attr_.post_ops_.len() != 0) {
                    apply_post_ops(jcp_.dst_prc, 0);
                }
                store(vmm_val, reg_dst, vector_step);

                add(reg_dst, vector_step * jcp_.dst_data_size);
                add(reg_src_aux, vector_step * jcp_.src_data_size);
                add(reg_oc_off, vector_step * sizeof(float));
                sub(reg_work_amount, vector_step);

                jmp(nn_loop_label, T_NEAR);
            }
            L(nn_loop_end_label);

            if (tail_step != 0) {
                // Tail load; if sentinel index, go zero path
                cmp(reg_index_offset, -1);
                je(nn_zero_tail_label, T_NEAR);
                load(reg_src_aux, vmm_val, tail_step);
                if (attr_.post_ops_.len() != 0) {
                    apply_post_ops(jcp_.dst_prc, 0);
                }
                store(vmm_val, reg_dst, tail_step);

                // check to remove below
                add(reg_dst, tail_step * jcp_.dst_data_size);
                add(reg_src_aux, tail_step * jcp_.src_data_size);
                add(reg_oc_off, tail_step * sizeof(float));
                sub(reg_work_amount, tail_step);
                // tail handled; proceed to next column
                L(nn_zero_tail_label);
                uni_vpxor(vmm_val, vmm_val, vmm_val);
                if (attr_.post_ops_.len() != 0) {
                    apply_post_ops(jcp_.dst_prc, 0);
                }
                store(vmm_val, reg_dst, tail_step);
                add(reg_dst, tail_step * jcp_.dst_data_size);
                add(reg_oc_off, tail_step * sizeof(float));
                sub(reg_work_amount, tail_step);
            } else {
                // Define label to satisfy assembler when tail_step == 0 (no jumps to it in this branch)
                L(nn_zero_tail_label);
            }
            add(reg_index, jcp_.indices_size);
            sub(reg_work_amount_out, 1);
            jmp(out_loop_label, T_NEAR);

            L(nn_zero_vec_label);
            {
                mov(reg_work_amount, reg_work_amount_bk);
                if (attr_.post_ops_.len() != 0) {
                    mov(reg_oc_off, reg_oc_off_bk);
                }
                // main vectorized zero loop
                Xbyak::Label zero_vec_loop, zero_vec_end;
                L(zero_vec_loop);
                {
                    cmp(reg_work_amount, vector_step);
                    jl(zero_vec_end, T_NEAR);
                    uni_vpxor(vmm_val, vmm_val, vmm_val);
                    if (attr_.post_ops_.len() != 0) {
                        apply_post_ops(jcp_.dst_prc, 0);
                    }
                    store(vmm_val, reg_dst, vector_step);
                    add(reg_dst, vector_step * jcp_.dst_data_size);
                    add(reg_oc_off, vector_step * sizeof(float));
                    sub(reg_work_amount, vector_step);
                    jmp(zero_vec_loop, T_NEAR);
                }
                L(zero_vec_end);
                if (tail_step != 0) {
                    uni_vpxor(vmm_val, vmm_val, vmm_val);
                    if (attr_.post_ops_.len() != 0) {
                        apply_post_ops(jcp_.dst_prc, 0);
                    }
                    store(vmm_val, reg_dst, tail_step);
                    add(reg_dst, tail_step * jcp_.dst_data_size);
                    add(reg_oc_off, tail_step * sizeof(float));
                    sub(reg_work_amount, tail_step);
                }
                add(reg_index, jcp_.indices_size);
                sub(reg_work_amount_out, 1);
                jmp(out_loop_label, T_NEAR);
            }
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
        int dst_stride = (jcp_.layout == InterpolateLayoutType::by_channel)
                             ? (vector_step * jcp_.dst_data_size)
                             : (blk * jcp_.OW * jcp_.OH * jcp_.OD * jcp_.dst_data_size);
        int src_stride = (jcp_.layout == InterpolateLayoutType::by_channel)
                             ? (vector_step * jcp_.src_data_size)
                             : (blk * jcp_.IW * jcp_.IH * jcp_.ID * jcp_.src_data_size);

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
                uni_vmulps(vmm_valTR, vmm_valTR, vmm_weightE);  // end_value * end_weight
                uni_vfmadd231ps(vmm_valTR,
                                vmm_d_bias,
                                vmm_weightF);  // start_value * start_weight + end_value * end_weight
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
                    uni_vmulps(vmm_valTR, vmm_valTR, vmm_weightE);  // end_value * end_weight
                    uni_vfmadd231ps(vmm_valTR,
                                    vmm_d_bias,
                                    vmm_weightF);  // start_value * start_weight + end_value * end_weight
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
                sub(reg_work_amount, vector_step);  // work_amount is c
            } else {
                sub(reg_work_amount, 1);  // work_amount = div_up(c, blk), no tails
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
                uni_vmulps(vmm_valTR, vmm_valTR, vmm_weightE);  // end_value * end_weight
                uni_vfmadd231ps(vmm_valTR,
                                vmm_d_bias,
                                vmm_weightF);  // start_value * start_weight + end_value * end_weight
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

                uni_vmulps(vmm_valTR, vmm_valTR, vmm_weightE);  // end_value * end_weight
                uni_vfmadd231ps(vmm_valTR,
                                vmm_d_bias,
                                vmm_weightF);  // start_value * start_weight + end_value * end_weight
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
            add(reg_src_aux1, Xbyak::Reg64(reg_index_offset.getIdx()));
            load(reg_src_aux1, vmm_valTL, scalar_step);

            mov(reg_src_aux1, reg_src);
            mov(reg_index_offset, dword[reg_index + index_stride]);
            add(reg_src_aux1, Xbyak::Reg64(reg_index_offset.getIdx()));
            load(reg_src_aux1, vmm_valTR, scalar_step);

            load_weights(reg_src_aux, vmm_weightL, scalar_step, 0);
            load_weights(reg_src_aux, vmm_weightR, scalar_step, weight_stride);

            if (jcp_.spatial_dim_size == 1) {
                linear_onnx_worker_1d();
            }
            if (jcp_.spatial_dim_size > 1) {
                mov(reg_src_aux1, reg_src);
                mov(reg_index_offset, dword[reg_index + 2 * index_stride]);
                add(reg_src_aux1, Xbyak::Reg64(reg_index_offset.getIdx()));
                load(reg_src_aux1, vmm_valBL, scalar_step);

                mov(reg_src_aux1, reg_src);
                mov(reg_index_offset, dword[reg_index + 3 * index_stride]);
                add(reg_src_aux1, Xbyak::Reg64(reg_index_offset.getIdx()));
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
                add(reg_src_aux1, Xbyak::Reg64(reg_index_offset.getIdx()));
                load(reg_src_aux1, vmm_valTL, scalar_step);

                mov(reg_src_aux1, reg_src);
                mov(reg_index_offset, dword[reg_index + 5 * index_stride]);
                add(reg_src_aux1, Xbyak::Reg64(reg_index_offset.getIdx()));
                load(reg_src_aux1, vmm_valTR, scalar_step);

                mov(reg_src_aux1, reg_src);
                mov(reg_index_offset, dword[reg_index + 6 * index_stride]);
                add(reg_src_aux1, Xbyak::Reg64(reg_index_offset.getIdx()));
                load(reg_src_aux1, vmm_valBL, scalar_step);

                mov(reg_src_aux1, reg_src);
                mov(reg_index_offset, dword[reg_index + 7 * index_stride]);
                add(reg_src_aux1, Xbyak::Reg64(reg_index_offset.getIdx()));
                load(reg_src_aux1, vmm_valBR, scalar_step);

                linear_onnx_worker_2d();

                load_weights(reg_src_aux, vmm_weightE, scalar_step, 5 * weight_stride);
                load_weights(reg_src_aux, vmm_weightF, scalar_step, 4 * weight_stride);

                uni_vmulps(vmm_valTR, vmm_valTR, vmm_weightE);  // end_value * end_weight
                uni_vfmadd231ps(vmm_valTR,
                                vmm_d_bias,
                                vmm_weightF);  // start_value * start_weight + end_value * end_weight
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

    void linear_onnx_worker_1d() {
        uni_vmulps(vmm_valTR, vmm_valTR, vmm_weightR);
        uni_vfmadd231ps(vmm_valTR, vmm_valTL, vmm_weightL);
    }

    // weightT * (srcTL * weightL + srcTR * weightR) +
    // weightB * (srcBL * weightL + srcBR * weightR)
    void linear_onnx_worker_2d() {
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
                apply_post_ops(jcp_.dst_prc, false);  // vmm_val is default dst value to post_ops and store
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
                sub(reg_work_amount, vector_step);  // work_amount is c
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
                apply_post_ops(jcp_.dst_prc, false);  // vmm_val is default dst value
                add(reg_oc_off, scalar_step * sizeof(float));
            }
            store(vmm_val, reg_dst, scalar_step);

            int dst_stride = scalar_step * jcp_.dst_data_size;
            int src_stride = scalar_step * jcp_.src_data_size;
            add(reg_dst, dst_stride);
            add(reg_src, src_stride);
            sub(reg_work_amount, scalar_step);  // work_amount is c

            jmp(tail_loop_label, T_NEAR);
        }
        L(tail_loop_end_label);
    }

    void cubic_c_gathered_matrix(bool is_scalar) {
        // y0:  (x0 * weightX0 + x1 * weightX1 + x2 * weightX2 + x3 * weightX3) * weightY0
        cubic_c_gathered_line(0, vmm_weightY0, is_scalar);
        // y1
        cubic_c_gathered_line(4, vmm_weightY1, is_scalar);
        // y2
        cubic_c_gathered_line(8, vmm_weightY2, is_scalar);
        // y3
        cubic_c_gathered_line(12, vmm_weightY3, is_scalar);
    }

    void cubic_c_gathered_line(int index_start, Vmm vmm_weight, bool is_scalar) {
        uni_vpxor(vmm_dstX, vmm_dstX, vmm_dstX);
        cubic_c_gathered_pixel(index_start, vmm_weightX0, is_scalar);
        cubic_c_gathered_pixel(index_start + 1, vmm_weightX1, is_scalar);
        cubic_c_gathered_pixel(index_start + 2, vmm_weightX2, is_scalar);
        cubic_c_gathered_pixel(index_start + 3, vmm_weightX3, is_scalar);
        uni_vfmadd231ps(vmm_val, vmm_dstX, vmm_weight);
    }

    void cubic_c_gathered_pixel(int i, Vmm vmm_weight, bool is_scalar) {
        mov(reg_src_aux, reg_src);
        mov(reg_index_offset, dword[reg_index + i * jcp_.indices_size]);
        add(reg_src_aux, Xbyak::Reg64(reg_index_offset.getIdx()));
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
            vgatherdps(vmm_weightX0,
                       ptr[reg_weight_x + vmm_val * grid_len],
                       vmm_mask);  // 4 in vmm_val for weight_size, another 4 for grid_len

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
            gather_i32_indices(vmm_weightX2,
                               reg_weight_x,
                               2 * sizeof(float),
                               vmm_val,
                               grid_len,
                               ov::element::f32,
                               true);
            gather_i32_indices(vmm_weightX3,
                               reg_weight_x,
                               3 * sizeof(float),
                               vmm_val,
                               grid_len,
                               ov::element::f32,
                               true);
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
            gather_i32_indices(vmm_weightY,
                               reg_weight_y,
                               2 * sizeof(float),
                               vmm_tbl_y,
                               grid_len,
                               ov::element::f32,
                               true);
            cubic_planar_line(true);

            // y3
            vpaddd(vmm_index_y_itr, vmm_index_in_y, vmm_one);
            vpaddd(vmm_index_y_itr, vmm_index_y_itr, vmm_one);
            // crop to [0, IH - 1]
            vpminsd(vmm_index_y_itr, vmm_index_y_itr, cubic_planar_table_val(1));
            vpmaxsd(vmm_index_y_itr, vmm_index_y_itr, vmm_zero);
            // weight y3
            gather_i32_indices(vmm_weightY,
                               reg_weight_y,
                               3 * sizeof(float),
                               vmm_tbl_y,
                               grid_len,
                               ov::element::f32,
                               true);
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

    void cubic_planar_line(bool is_scalar) {
        uni_vpxor(vmm_dstX, vmm_dstX, vmm_dstX);
        cubic_planar_pixel(0, is_scalar);
        cubic_planar_pixel(1, is_scalar);
        cubic_planar_pixel(2, is_scalar);
        cubic_planar_pixel(3, is_scalar);
        uni_vfmadd231ps(vmm_val, vmm_dstX, vmm_weightY);
    }

    void cubic_planar_pixel(int itr, bool is_scalar) {
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

    void prepare_cubic_planar_table() {
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

    Xbyak::Address cubic_planar_table_val(int index) {
        return ptr[reg_table + index * vlen];
    }

    // always gather to Vmm, compute with Vmm, store with Xmm if scalar_step
    void gather_i32_indices(Vmm vmm_src,
                            const Xbyak::Reg64& base,
                            int offset,
                            Vmm vmm_indices,
                            int scale,
                            ov::element::Type src_prc,
                            bool is_scalar) {
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
                mov(reg_tmp_64.cvt32(), ptr[rsp + i * sizeof(int)]);  // sizeof(int)  index_size
                table_idx = ptr[base + offset + reg_tmp_64 * scale];  // scale: sizeof(float)   value_size
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

    // is_broadcast for broadcasting param for depth_wise and quantize(channel-sensitive post-ops), for fusion with
    // plain layout.
    void apply_post_ops(ov::element::Type dst_prc, bool is_broadcast) {
        const auto& p = attr_.post_ops_;
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
                depthwise_injectors[depthwise_inj_idx]->compute_vector_range(vmm_val.getIdx(),
                                                                             vmm_val.getIdx() + 1,
                                                                             reg_d_weights,
                                                                             reg_d_weights,
                                                                             is_broadcast);

                post_ops_data_offset += depthwise_injectors[depthwise_inj_idx]->memoryStep();
                depthwise_inj_idx++;
            } else if (post_op.is_quantization()) {
                bool do_dequantization = post_op.quantization.alg == alg_kind::quantization_quantize_dequantize;
                bool do_rounding = do_dequantization || dst_prc == ov::element::f32 || i != p.len() - 1;

                int s_idx = vmm_val.getIdx();

                quantization_injectors[quantization_inj_idx]->init_crop_ptrs(reg_post_ops_data + post_ops_data_offset,
                                                                             reg_oc_off);
                quantization_injectors[quantization_inj_idx]->compute_crop(s_idx, s_idx + 1, 0, 0, is_broadcast);

                quantization_injectors[quantization_inj_idx]->init_input_scale_shift_ptrs(
                    reg_post_ops_data + post_ops_data_offset,
                    reg_oc_off);
                quantization_injectors[quantization_inj_idx]
                    ->compute_input_scale_shift(s_idx, s_idx + 1, 0, do_rounding, 0, is_broadcast);

                if (do_dequantization) {
                    quantization_injectors[quantization_inj_idx]->init_output_scale_shift_ptrs(
                        reg_post_ops_data + post_ops_data_offset,
                        reg_oc_off);
                    quantization_injectors[quantization_inj_idx]->compute_output_scale_shift(s_idx,
                                                                                             s_idx + 1,
                                                                                             0,
                                                                                             0,
                                                                                             is_broadcast);
                }

                post_ops_data_offset += quantization_injectors[quantization_inj_idx]->memoryStep();
                quantization_inj_idx++;
            }
        }
    }
};

#endif  // OPENVINO_ARCH_X86_64

namespace {
struct InterpolateKey {
    InterpolateAttrs nodeAttrs;
    VectorDims srcDims;
    VectorDims dstDims;
    std::vector<float> dataScales;
    dnnl::primitive_attr attr;

    [[nodiscard]] size_t hash() const;
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

bool InterpolateKey::operator==(const InterpolateKey& rhs) const {
    if (nodeAttrs.mode != rhs.nodeAttrs.mode) {
        return false;
    }
    if (nodeAttrs.coordTransMode != rhs.nodeAttrs.coordTransMode) {
        return false;
    }
    if (nodeAttrs.nearestMode != rhs.nodeAttrs.nearestMode) {
        return false;
    }
    if (nodeAttrs.layout != rhs.nodeAttrs.layout) {
        return false;
    }
    if (nodeAttrs.antialias != rhs.nodeAttrs.antialias) {
        return false;
    }
    if (nodeAttrs.cubeCoeff != rhs.nodeAttrs.cubeCoeff) {
        return false;
    }
    if (nodeAttrs.padBegin != rhs.nodeAttrs.padBegin) {
        return false;
    }
    if (nodeAttrs.padEnd != rhs.nodeAttrs.padEnd) {
        return false;
    }
    if (nodeAttrs.inPrc != rhs.nodeAttrs.inPrc) {
        return false;
    }
    if (nodeAttrs.outPrc != rhs.nodeAttrs.outPrc) {
        return false;
    }

    if (srcDims != rhs.srcDims) {
        return false;
    }
    if (dstDims != rhs.dstDims) {
        return false;
    }
    if (dataScales != rhs.dataScales) {
        return false;
    }
    return *attr.get() == *rhs.attr.get();
}
}

// for ndhwc and nCdhw8c[16c]
// input may be f32/bf16/int8, fused->output varies
void InterpolateJitExecutor::NNCGathered(const uint8_t* in_ptr_,
                                                      uint8_t* out_ptr_,
                                                      const void* post_ops_data_,
                                                      int B,
                                                      int C,
                                                      int ID,
                                                      int IH,
                                                      int IW,
                                                      int OD,
                                                      int OH,
                                                      int OW) {
    auto* index_d = static_cast<int*>(auxTable.data());
    auto* index_h = static_cast<int*>(&auxTable[OD]);
    auto* index_w = static_cast<int*>(&auxTable[OD + OH]);

    bool is_nhwc = (configured_for_layout == by_channel);

    // Prebuild zero rows for OOB handling; read-only and thread-safe
    const int row_size_nhwc = C * IW * srcDataSize;
    std::vector<uint8_t> zero_row_nhwc(row_size_nhwc, 0);
    int blk_size = mayiuse(cpu::x64::avx512_core) ? 16 : 8;
    const int row_size_blk = blk_size * IW * srcDataSize;
    std::vector<uint8_t> zero_row_blk(row_size_blk, 0);

    for (int b = 0; b < B; b++) {
        if (is_nhwc) {
            const uint8_t* in_ptr = in_ptr_ + (IW * IH * ID * C * b) * srcDataSize;
            uint8_t* out_ptr = out_ptr_ + (OW * OH * OD * C * b) * dstDataSize;
            std::vector<int> index_w_kernel(OW);
            parallel_for2d(OD, OH, [&](size_t d, size_t h) {
                // kernel for C * OW
                uint8_t* out_ptr_dh = out_ptr + (C * OW * OH * d + C * OW * h) * dstDataSize;
                const bool oob_row = (index_d[d] < 0) || (index_h[h] < 0);
                const uint8_t* in_ptr_dh = oob_row ? zero_row_nhwc.data()
                                                   : in_ptr + (C * IW * IH * index_d[d] + C * IW * index_h[h]) * srcDataSize;
                // Build per-width kernel indices: use 0 for oob to read zeros
                for (int ox = 0; ox < OW; ox++) {
                    if (oob_row || index_w[ox] < 0) {
                        index_w_kernel[ox] = 0;
                    } else {
                        index_w_kernel[ox] = index_w[ox] * C * srcDataSize;
                    }
                }
                // debug sanity (first row only)
                if (d == 0 && h == 0 && b == 0) {
                    float first_in = 0.f;
                    if (!oob_row) {
                        first_in = *reinterpret_cast<const float*>(in_ptr_dh);
                    }
                    fprintf(stderr,
                            "NHWC NN dbg: IH=%d IW=%d OH=%d OW=%d C=%d oob=%d id0=%d ih0=%d iw0=%d iw1=%d first_in=%f\\n",
                            IH,
                            IW,
                            OH,
                            OW,
                            C,
                            (int)oob_row,
                            index_d[0],
                            index_h[0],
                            index_w[0],
                            (OW > 1 ? index_w[1] : -999),
                            first_in);
                }
                auto arg = jit_interpolate_call_args();
                arg.dst = out_ptr_dh;
                arg.src_ptr[0] = in_ptr_dh;
                arg.index = static_cast<int*>(index_w_kernel.data());
                arg.work_amount = C;
                arg.oc_off = 0;
                arg.post_op_data = post_ops_data_;
                (*interpolateKernel)(&arg);
            });
        } else {  // for blk
            int CB = div_up(C, blk_size);
            const uint8_t* in_ptr = in_ptr_ + (IW * IH * ID * CB * blk_size * b) * srcDataSize;
            uint8_t* out_ptr = out_ptr_ + (OW * OH * OD * CB * blk_size * b) * dstDataSize;
            std::vector<int> index_w_kernel(OW);
            parallel_for2d(CB, OD, [&](size_t cb, size_t d) {
                uint8_t* out_ptr_cbd = out_ptr + (blk_size * OW * OH * OD * cb + blk_size * OW * OH * d) * dstDataSize;
                const uint8_t* in_ptr_cbd =
                    in_ptr + (blk_size * IW * IH * ID * cb + blk_size * IW * IH * index_d[d]) * srcDataSize;
                auto arg = jit_interpolate_call_args();
                for (int h = 0; h < OH; h++) {  // kernel for blk_size * OW
                    const bool oob_row = (index_h[h] < 0) || (index_d[d] < 0);
                    arg.dst = out_ptr_cbd + blk_size * OW * h * dstDataSize;
                    arg.src_ptr[0] = oob_row ? zero_row_blk.data() : in_ptr_cbd + blk_size * IW * index_h[h] * srcDataSize;
                    for (int ox = 0; ox < OW; ox++) {
                        if (oob_row || index_w[ox] < 0) {
                            index_w_kernel[ox] = 0;
                        } else {
                            index_w_kernel[ox] = index_w[ox] * blk_size * srcDataSize;
                        }
                    }
                    arg.index = static_cast<int*>(index_w_kernel.data());
                    arg.work_amount = static_cast<size_t>(OW);
                    arg.oc_off = cb * blk_size * sizeof(float);
                    arg.post_op_data = post_ops_data_;
                    (*interpolateKernel)(&arg);
                }
            });
        }
    }  // batch end
}

void InterpolateJitExecutor::NNPlanar(const uint8_t* in_ptr_,
                                                   uint8_t* out_ptr_,
                                                   const void* post_ops_data_,
                                                   int B,
                                                   int C,
                                                   int ID,
                                                   int IH,
                                                   int IW,
                                                   int OD,
                                                   int OH,
                                                   int OW) {
    auto* index_d = static_cast<int*>(auxTable.data());
    auto* index_h = static_cast<int*>(&auxTable[OD]);
    auto* index_w = static_cast<int*>(&auxTable[OD + OH]);

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
        const uint8_t* in_ptr =
            in_ptr_ + (IW * IH * ID * C * b + IW * IH * ID * c + IW * IH * index_d[od]) * srcDataSize;
        uint8_t* out_ptr = out_ptr_ + (OW * OH * OD * C * b + OW * OH * OD * c + OW * OH * od) * dstDataSize;

        auto arg = jit_interpolate_call_args();
        arg.src_ptr[0] = in_ptr;
        arg.dst = out_ptr;
        arg.index = static_cast<int*>(
            index_kernel.data());  // need index_h and index_w in kernel, it's in continous memory so one param
        arg.oc_off = static_cast<size_t>(c * sizeof(float));
        // work_amount is OH(out loop) and OW(inner loop), can get in kernel from jcp.
        arg.post_op_data = post_ops_data_;
        (*interpolateKernel)(&arg);
    });
}

void InterpolateJitExecutor::linearOnnxPlanar(const uint8_t* in_ptr_,
                                                           uint8_t* out_ptr_,
                                                           const void* post_ops_data_,
                                                           int B,
                                                           int C,
                                                           int ID,
                                                           int IH,
                                                           int IW,
                                                           int OD,
                                                           int OH,
                                                           int OW) {
    // FrontTopLeft:0, FrontTopRight:1, FrontBottomLeft:2, FrontBottomRight:3, EndTopLeft:4,   EndTopRight:5,
    // EndBottomLeft:6,   EndBottomRight:7 weight: Left:0, ritht:1, top:2, bottom:3, front:4, end:5
    auto* index = static_cast<int*>(auxTable.data());
    int eltInGrid = [&]() -> int {
        if (spatialDimSize > 2) {
            return MAX_INPUT_INTERPOLATE;
        }
        if (spatialDimSize > 1) {
            return 4;
        }
        return 2;
    }();
    int scratchLen = rnd_up(eltInGrid * OW * OH * OD, 16);
    auto* weight = reinterpret_cast<float*>(&auxTable[scratchLen]);

    parallel_for2d(B, C, [&](size_t b, size_t c) {
        uint8_t* out_ptr_nc = out_ptr_ + (OH * OW * OD * C * b + OH * OW * OD * c) * dstDataSize;
        const uint8_t* in_ptr_nc = in_ptr_ + (IH * IW * ID * C * b + IH * IW * ID * c) * srcDataSize;
        auto arg = jit_interpolate_call_args();
        arg.src_ptr[0] = in_ptr_nc;
        arg.index = (&index[0]);
        arg.weight_ptr[0] = (&weight[0]);
        arg.dst = out_ptr_nc;
        arg.work_amount = OW * OH * OD;
        arg.oc_off = static_cast<size_t>(c * sizeof(float));
        arg.post_op_data = post_ops_data_;
        (*interpolateKernel)(&arg);
    });
}

void InterpolateJitExecutor::linearOnnxCGathered(const uint8_t* in_ptr_,
                                                              uint8_t* out_ptr_,
                                                              const void* post_ops_data_,
                                                              int B,
                                                              int C,
                                                              int ID,
                                                              int IH,
                                                              int IW,
                                                              int OD,
                                                              int OH,
                                                              int OW) {
    // left:OW right:OW Top:OH Bottom:OH Front:OD End:OD
    std::vector<int*> indexPtr(MAX_INPUT_INTERPOLATE, nullptr);
    std::vector<float*> weightPtr(MAX_INPUT_INTERPOLATE, nullptr);
    size_t scratchLen = rnd_up(OW + OW + OH + OH + OD + OD, 16);
    indexPtr[0] = static_cast<int*>(auxTable.data());
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

    bool isByChannel = configured_for_layout == by_channel;

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
        uint8_t* out_ptr_ndh = out_ptr_ + (C3 * b + C1 * d + C0 * h) * dstDataSize;

        const uint8_t* in_ptr_n = in_ptr_ + (I3 * b) * srcDataSize;
        const uint8_t* in_ptr_nf = in_ptr_n + (indexPtr[4][d] * I1) * srcDataSize;
        const uint8_t* in_ptr_nft = in_ptr_nf + (indexPtr[2][h] * I0) * srcDataSize;
        const uint8_t* in_ptr_nfb = in_ptr_nf + (indexPtr[3][h] * I0) * srcDataSize;
        const uint8_t* in_ptr_ne = in_ptr_n + (indexPtr[5][d] * I1) * srcDataSize;
        const uint8_t* in_ptr_net = in_ptr_ne + (indexPtr[2][h] * I0) * srcDataSize;
        const uint8_t* in_ptr_neb = in_ptr_ne + (indexPtr[3][h] * I0) * srcDataSize;
        auto arg = jit_interpolate_call_args();
        for (int w = 0; w < OW; ++w) {
            uint8_t* out_ptr_ndhw = out_ptr_ndh + CGatherLen * w * dstDataSize;

            arg.src_ptr[0] = in_ptr_nft + (indexPtr[0][w] * CGatherLen) * srcDataSize;
            arg.src_ptr[1] = in_ptr_nft + (indexPtr[1][w] * CGatherLen) * srcDataSize;
            arg.src_ptr[2] = in_ptr_nfb + (indexPtr[0][w] * CGatherLen) * srcDataSize;
            arg.src_ptr[3] = in_ptr_nfb + (indexPtr[1][w] * CGatherLen) * srcDataSize;
            arg.src_ptr[4] = in_ptr_net + (indexPtr[0][w] * CGatherLen) * srcDataSize;
            arg.src_ptr[5] = in_ptr_net + (indexPtr[1][w] * CGatherLen) * srcDataSize;
            arg.src_ptr[6] = in_ptr_neb + (indexPtr[0][w] * CGatherLen) * srcDataSize;
            arg.src_ptr[7] = in_ptr_neb + (indexPtr[1][w] * CGatherLen) * srcDataSize;
            arg.weight_ptr[0] = (&weightPtr[0][w]);
            arg.weight_ptr[1] = (&weightPtr[1][w]);
            arg.weight_ptr[2] = (&weightPtr[2][h]);
            arg.weight_ptr[3] = (&weightPtr[3][h]);
            arg.weight_ptr[4] = (&weightPtr[4][d]);
            arg.weight_ptr[5] = (&weightPtr[5][d]);
            arg.dst = out_ptr_ndhw;
            arg.work_amount = workAmount;
            arg.oc_off = 0;
            arg.post_op_data = post_ops_data_;
            (*interpolateKernel)(&arg);
        }
    });
}

void InterpolateJitExecutor::cubicCGathered(const uint8_t* in_ptr_,
                                                         uint8_t* out_ptr_,
                                                         const void* post_ops_data_,
                                                         int B,
                                                         int C,
                                                         int IH,
                                                         int IW,
                                                         int OH,
                                                         int OW) {
    const int idxNum = 1;
    auto* xOrigin = static_cast<int*>(auxTable.data());
    auto* xFactor = reinterpret_cast<float*>(&auxTable[OW]);
    auto* yOrigin = static_cast<int*>(&auxTable[(CUBIC_GRID_LEN + idxNum) * OW]);
    auto* yFactor = reinterpret_cast<float*>(&auxTable[(CUBIC_GRID_LEN + idxNum) * OW + OH]);

    int blkSize = mayiuse(cpu::x64::avx512_core) ? 16 : 8;
    int CB = div_up(C, blkSize);
    int CSize = configured_for_layout == InterpolateLayoutType::by_channel ? C : blkSize * CB;
    int CGatherLen = configured_for_layout == InterpolateLayoutType::by_channel ? C : blkSize;
    int workAmount = configured_for_layout == InterpolateLayoutType::by_channel ? C : CB;

    parallel_for3d(B, OH, OW, [&](size_t b, size_t h, size_t w) {
        uint8_t* out_ptr_nhw = out_ptr_ + (OH * OW * CSize * b + OW * CGatherLen * h + CGatherLen * w) * dstDataSize;
        const uint8_t* in_ptr_n = in_ptr_ + (IH * IW * CSize * b) * srcDataSize;

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
        arg.index = static_cast<int*>(kernelIndex.data());
        // 0 for weight_W, 1 for weight_H
        arg.weight_ptr[0] = (&xFactor[w * CUBIC_GRID_LEN]);
        arg.weight_ptr[1] = (&yFactor[h * CUBIC_GRID_LEN]);

        // for by channel, src + step, dst + step, process next step on continuous memory
        // for blk, src + IW*IH*blkSize, dst + OW*OH*blkSize, process the blkSize on next CB
        arg.work_amount = workAmount;
        arg.oc_off = 0;
        arg.post_op_data = post_ops_data_;
        (*interpolateKernel)(&arg);
    });
}

void InterpolateJitExecutor::cubicPlanar(const uint8_t* in_ptr_,
                                                      uint8_t* out_ptr_,
                                                      const void* post_ops_data_,
                                                      int B,
                                                      int C,
                                                      int IH,
                                                      int IW,
                                                      int OH,
                                                      int OW) {
    int tblAdvance = 0;
    auto* xOrigin = static_cast<int*>(&auxTable[tblAdvance]);
    tblAdvance += OW;
    auto* xFactor = reinterpret_cast<float*>(&auxTable[tblAdvance]);
    tblAdvance += CUBIC_GRID_LEN * OW;
    auto* yOrigin = static_cast<int*>(&auxTable[tblAdvance]);
    tblAdvance += OH;
    auto* yFactor = reinterpret_cast<float*>(&auxTable[tblAdvance]);

    tblAdvance += CUBIC_GRID_LEN * OH;
    auto* sequenceOH = static_cast<int*>(&auxTable[tblAdvance]);
    tblAdvance += OW * OH;
    auto* sequenceOW = static_cast<int*>(&auxTable[tblAdvance]);

    parallel_for2d(B, C, [&](size_t n, size_t c) {
        const uint8_t* in_ptr_nc = in_ptr_ + (IW * IH * C * n + IW * IH * c) * srcDataSize;
        uint8_t* out_ptr_nc = out_ptr_ + (OW * OH * C * n + OW * OH * c) * dstDataSize;

        auto arg = jit_interpolate_call_args();
        arg.dst = out_ptr_nc;
        arg.src_ptr[0] = in_ptr_nc;
        arg.index = xOrigin;
        arg.src_ptr[1] = yOrigin;
        arg.src_ptr[2] = (&sequenceOH[0]);
        arg.src_ptr[3] = (&sequenceOW[0]);
        arg.weight_ptr[0] = xFactor;
        arg.weight_ptr[1] = yFactor;
        arg.work_amount = static_cast<size_t>(OW) * OH;
        arg.oc_off = static_cast<size_t>(c * sizeof(float));
        arg.post_op_data = post_ops_data_;
        (*interpolateKernel)(&arg);
    });
}

void InterpolateJitExecutor::pillowCGathered(const uint8_t* in_ptr_,
                                                          uint8_t* out_ptr_,
                                                          [[maybe_unused]] const void* post_ops_data_,
                                                          int B,
                                                          int C,
                                                          int IH,
                                                          int IW,
                                                          int OH,
                                                          int OW) {
    // workBuffer needed when both pass are true
    bool xPass = IW != OW;
    bool yPass = IH != OH;

    auto b_loop = [&](size_t b) {
        auto arg = jit_interpolate_call_args();
        arg.src_ptr[0] = in_ptr_ + (IW * IH * C * b) * srcDataSize;
        if (xPass && yPass) {
            size_t parallel_num = B;
            // IH * OW * C buf needed
            auto buffer_size = static_cast<size_t>(OW) * IH * C;
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

// ===== Added: InterpolateJitExecutor constructor and exec =====
InterpolateJitExecutor::InterpolateJitExecutor(const InterpolateAttrs& interpAttrs,
                                               const VectorDims& srcDims,
                                               const VectorDims& dstDims,
                                               const std::vector<float>& dataScales,
                                               const dnnl::primitive_attr& attr)
    : InterpolateExecutorBase(interpAttrs, srcDims, dstDims, dataScales) {
#if defined(OPENVINO_ARCH_X86_64)
    jit_interpolate_config_params jcp{};
    jcp.layout = configured_for_layout;
    jcp.mode = interpAttrs.mode;
    jcp.src_prc = inputPrec;
    jcp.dst_prc = outputPrec;
    jcp.src_data_size = static_cast<int>(inputPrec.size());
    jcp.dst_data_size = static_cast<int>(outputPrec.size());
    jcp.indices_size = static_cast<int>(sizeof(int));
    jcp.spatial_dim_size = spatialDimSize;
    jcp.C = static_cast<int>(srcDimPad5d[1]);
    jcp.ID = static_cast<int>(srcDimPad5d[2]);
    jcp.IH = static_cast<int>(srcDimPad5d[3]);
    jcp.IW = static_cast<int>(srcDimPad5d[4]);
    jcp.OD = static_cast<int>(dstDim5d[2]);
    jcp.OH = static_cast<int>(dstDim5d[3]);
    jcp.OW = static_cast<int>(dstDim5d[4]);

    if (interpAttrs.mode == InterpolateMode::bilinear_pillow || interpAttrs.mode == InterpolateMode::bicubic_pillow) {
        if (auxTable.size() >= 2) {
            jcp.filterLenX = auxTable[0];
            jcp.filterLenY = auxTable[1];
            size_t offset = 2 + static_cast<size_t>(jcp.filterLenX) * jcp.OW +
                            static_cast<size_t>(jcp.filterLenY) * jcp.OH;
            if (offset < auxTable.size()) {
                jcp.bound = reinterpret_cast<int*>(&auxTable[offset]);
            }
        }
    }

    if (mayiuse(cpu::x64::avx512_core)) {
        interpolateKernel = std::make_shared<jit_uni_interpolate_kernel_f32<cpu::x64::avx512_core>>(jcp, *attr.get());
    } else if (mayiuse(cpu::x64::avx2)) {
        interpolateKernel = std::make_shared<jit_uni_interpolate_kernel_f32<cpu::x64::avx2>>(jcp, *attr.get());
    } else if (mayiuse(cpu::x64::sse41)) {
        interpolateKernel = std::make_shared<jit_uni_interpolate_kernel_f32<cpu::x64::sse41>>(jcp, *attr.get());
    } else {
        OPENVINO_THROW("Can't create JIT interpolate kernel on current x64 ISA");
    }

    OPENVINO_ASSERT(interpolateKernel, "Failed to create JIT Interpolate kernel");
    interpolateKernel->create_ker();
#else
    OPENVINO_THROW("JIT Interpolate is only available on x86_64 builds");
#endif
}

void InterpolateJitExecutor::exec(const uint8_t* in_ptr_, uint8_t* out_ptr_, const void* post_ops_data_) {
#if defined(OPENVINO_ARCH_X86_64)
    const int B = static_cast<int>(srcDimPad5d[0]);
    const int C = static_cast<int>(srcDimPad5d[1]);
    const int ID = static_cast<int>(srcDimPad5d[2]);
    const int IH = static_cast<int>(srcDimPad5d[3]);
    const int IW = static_cast<int>(srcDimPad5d[4]);
    const int OD = static_cast<int>(dstDim5d[2]);
    const int OH = static_cast<int>(dstDim5d[3]);
    const int OW = static_cast<int>(dstDim5d[4]);

    switch (mode) {
    case InterpolateMode::nearest:
        if (configured_for_layout == InterpolateLayoutType::planar) {
            NNPlanar(in_ptr_, out_ptr_, post_ops_data_, B, C, ID, IH, IW, OD, OH, OW);
        } else {
            NNCGathered(in_ptr_, out_ptr_, post_ops_data_, B, C, ID, IH, IW, OD, OH, OW);
        }
        break;
    case InterpolateMode::linear_onnx:
        if (configured_for_layout == InterpolateLayoutType::planar) {
            linearOnnxPlanar(in_ptr_, out_ptr_, post_ops_data_, B, C, ID, IH, IW, OD, OH, OW);
        } else {
            linearOnnxCGathered(in_ptr_, out_ptr_, post_ops_data_, B, C, ID, IH, IW, OD, OH, OW);
        }
        break;
    case InterpolateMode::cubic:
        if (configured_for_layout == InterpolateLayoutType::planar) {
            cubicPlanar(in_ptr_, out_ptr_, post_ops_data_, B, C, IH, IW, OH, OW);
        } else {
            cubicCGathered(in_ptr_, out_ptr_, post_ops_data_, B, C, IH, IW, OH, OW);
        }
        break;
    case InterpolateMode::bilinear_pillow:
    case InterpolateMode::bicubic_pillow:
        pillowCGathered(in_ptr_, out_ptr_, post_ops_data_, B, C, IH, IW, OH, OW);
        break;
    default:
        OPENVINO_THROW("Unsupported interpolate mode in JIT executor");
    }
#else
    OPENVINO_THROW("JIT Interpolate is only available on x86_64 builds");
#endif
}

}
