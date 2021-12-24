// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_roi_align_node.h"
#include <mkldnn.hpp>
#include <string>
#include <vector>
#include <math.h>
#include <mkldnn_extension_utils.h>
#include <mkldnn_types.h>
#include <utils/bfloat16.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include "ie_parallel.hpp"
#include <mkldnn_selective_build.h>
#include <ngraph/opsets/opset3.hpp>

#include <cpu/x64/jit_generator.hpp>
#include "emitters/jit_load_store_emitters.hpp"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn;
using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu;
using namespace mkldnn::impl::cpu::x64;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

using ngPoolingMode = ngraph::op::v3::ROIAlign::PoolingMode;

#define GET_OFF(field) offsetof(jit_roi_align_call_args, field)

template <cpu_isa_t isa>
struct jit_uni_roi_align_kernel_f32 : public jit_uni_roi_align_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_roi_align_kernel_f32);

    explicit jit_uni_roi_align_kernel_f32(jit_roi_align_params jcp) : jit_uni_roi_align_kernel(jcp), jit_generator() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    };

    void generate() override {
        load_emitter.reset(new jit_load_emitter(this, isa, nullptr));
        store_emitter.reset(new jit_store_emitter(this, isa, nullptr));

        this->preamble();

        mov(reg_src, ptr[this->reg_params + GET_OFF(src)]);
        mov(reg_idx_y, ptr[this->reg_params + GET_OFF(idx_y)]);
        mov(reg_idx_x, ptr[this->reg_params + GET_OFF(idx_x)]);
        mov(reg_stride_y, ptr[this->reg_params + GET_OFF(stride_y)]);
        mov(reg_stride_x, ptr[this->reg_params + GET_OFF(stride_x)]);
        mov(reg_weights, ptr[this->reg_params + GET_OFF(weights)]);
        mov(reg_work_amount, ptr[this->reg_params + GET_OFF(work_amount)]);
        mov(reg_dst, ptr[this->reg_params + GET_OFF(dst)]);
        if (jcp_.alg == Algorithm::ROIAlignAvg) {
            mov(reg_scale, ptr[this->reg_params + GET_OFF(scale)]);
            uni_vbroadcastss(vmm_scale, ptr[reg_scale]);
        }

        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        load_pool_gpr_idxs = {static_cast<size_t>(reg_load_store_mask.getIdx()), static_cast<size_t>(reg_load_table.getIdx())};
        store_pool_gpr_idxs = {static_cast<size_t>(reg_load_store_mask.getIdx())};
        store_pool_vec_idxs = {static_cast<size_t>(vmm_zero.getIdx())};

        uni_vpbroadcastd(vmm_stride_y, ptr[reg_stride_y]);
        uni_vpbroadcastd(vmm_stride_x, ptr[reg_stride_x]);

        auto load_idx = [&](Xbyak::Reg64 reg_idx, Vmm vmm_idx, int elt_num) {
                    load_emitter->emit_code({static_cast<size_t>(reg_idx.getIdx())}, {static_cast<size_t>(vmm_idx.getIdx())},
                    std::make_shared<load_emitter_context>(Precision::I32, Precision::I32, elt_num),
                    {}, {load_pool_gpr_idxs});
        };

        Xbyak::Label main_loop_label;
        Xbyak::Label main_loop_end_label;
        Xbyak::Label tail_loop_label;
        Xbyak::Label tail_loop_end_label;

        int lane = v_len / cpu_isa_traits<sse41>::vlen;
        uni_vpxor(vmm_dst, vmm_dst, vmm_dst);
        L(main_loop_label);
        {
            cmp(reg_work_amount, lane);
            jl(main_loop_end_label, T_NEAR);

            load_idx(reg_idx_y, vmm_idx_y, v_step);
            load_idx(reg_idx_x, vmm_idx_x, v_step);

            uni_vpmulld(vmm_idx_y, vmm_idx_y, vmm_stride_y);
            uni_vpmulld(vmm_idx_x, vmm_idx_x, vmm_stride_x);
            uni_vpaddd(vmm_idx_y, vmm_idx_y, vmm_idx_x);

            if (jcp_.data_prc == Precision::FP32)
                gather_f32(vmm_src, reg_src, vmm_idx_y);
            else if (jcp_.data_prc == Precision::BF16)
                gather_bf16_to_f32_zmm(vmm_src, reg_src, vmm_idx_y);

            uni_vmovups(vmm_weights, ptr[reg_weights]);

            if (jcp_.alg == Algorithm::ROIAlignAvg) {
                uni_vfmadd231ps(vmm_dst, vmm_src, vmm_weights);
            } else {
                uni_vmulps(vmm_src, vmm_src, vmm_weights);
                // horizontal add for each lane
                // xmm_dst[0] hold the max
                if (isa == cpu::x64::avx512_common) {
                    for (int i = 0; i < lane; i++) {
                        vextractf32x4(xmm_temp1, Xbyak::Zmm(vmm_src.getIdx()), i);
                        horizontal_add_xmm(xmm_temp1, xmm_temp2);
                        uni_vmaxps(xmm_dst, xmm_dst, xmm_temp1);
                    }
                } else if (isa == cpu::x64::avx2) {
                    for (int i = 0; i < lane; i++) {
                        vextractf128(xmm_temp1, Xbyak::Ymm(vmm_src.getIdx()), i);
                        horizontal_add_xmm(xmm_temp1, xmm_temp2);
                        uni_vmaxps(xmm_dst, xmm_dst, xmm_temp1);
                    }
                } else {
                    horizontal_add_xmm(xmm_src, xmm_temp2);
                    uni_vmaxps(xmm_dst, xmm_dst, xmm_src);
                }
            }

            add(reg_idx_y, v_len);
            add(reg_idx_x, v_len);
            add(reg_weights, v_len);
            sub(reg_work_amount, lane);

            jmp(main_loop_label, T_NEAR);
        }
        L(main_loop_end_label);

        if (jcp_.alg == Algorithm::ROIAlignAvg)
            uni_vpxor(vmm_dst_tail, vmm_dst_tail, vmm_dst_tail);

        lane = 1;
        L(tail_loop_label);
        {
            cmp(reg_work_amount, 1);
            jl(tail_loop_end_label, T_NEAR);

            load_idx(reg_idx_y, vmm_idx_y, x_step);
            load_idx(reg_idx_x, vmm_idx_x, x_step);

            uni_vpmulld(xmm_idx_y, xmm_idx_y, xmm_stride_y);
            uni_vpmulld(xmm_idx_x, xmm_idx_x, xmm_stride_x);
            uni_vpaddd(xmm_idx_y, xmm_idx_y, xmm_idx_x);

            if (jcp_.data_prc == Precision::FP32)
                gather_f32_xmm(xmm_src, reg_src, xmm_idx_y);
            else if (jcp_.data_prc == Precision::BF16)
                gather_bf16_to_f32_xmm(xmm_src, reg_src, xmm_idx_y);

            uni_vmovups(xmm_weights, ptr[reg_weights]);
            if (jcp_.alg == Algorithm::ROIAlignAvg) {
                // as vex instruction will zero upper bit for xmm version, store result in seperate xmm_dst_tail
                uni_vfmadd231ps(xmm_dst_tail, xmm_src, xmm_weights);
            } else {
                uni_vmulps(xmm_src, xmm_src, xmm_weights);
                horizontal_add_xmm(xmm_src, xmm_temp2);
                uni_vmaxps(xmm_dst, xmm_dst, xmm_src);
            }

            add(reg_idx_y, x_len);
            add(reg_idx_x, x_len);
            add(reg_weights, x_len);
            sub(reg_work_amount, lane);

            jmp(tail_loop_label, T_NEAR);
        }
        L(tail_loop_end_label);

        if (jcp_.alg == Algorithm::ROIAlignAvg) {
            uni_vaddps(vmm_dst, vmm_dst, vmm_dst_tail);
            horizontal_add();  // xmm_dst[0] is the dst value
            uni_vmulps(vmm_dst, vmm_dst, vmm_scale);
        }

        // xmm_dst[0] of f32 is the dst value
        if (jcp_.data_prc == Precision::FP32)
            uni_vpextrd(ptr[reg_dst], xmm_dst, 0);
        else if (jcp_.data_prc == Precision::BF16)
            uni_vpextrw(ptr[reg_dst], xmm_dst, 1);

        this->postamble();

        load_emitter->emit_data();
        store_emitter->emit_data();
    }

private:
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;

    const int v_len = cpu_isa_traits<isa>::vlen;
    const int x_len = cpu_isa_traits<sse41>::vlen;
    const int v_step = v_len / sizeof(float);
    const int x_step = x_len / sizeof(float);

    Vmm vmm_zero = Vmm(0);

    Vmm vmm_idx_y = Vmm(2);
    Xmm xmm_idx_y = Xmm(2);
    Vmm vmm_idx_x = Vmm(3);
    Xmm xmm_idx_x = Xmm(3);
    Vmm vmm_stride_y = Vmm(4);
    Xmm xmm_stride_y = Xmm(4);
    Vmm vmm_stride_x = Vmm(5);
    Xmm xmm_stride_x = Xmm(5);

    Vmm vmm_src = Vmm(6);
    Xmm xmm_src = Xmm(6);
    Vmm vmm_dst = Vmm(7);
    Xmm xmm_dst = Xmm(7);
    Vmm vmm_dst_tail = Vmm(13);
    Xmm xmm_dst_tail = Xmm(13);

    Vmm vmm_temp1 = Vmm(8);
    Xmm xmm_temp1 = Xmm(8);
    Vmm vmm_temp2 = Vmm(9);
    Xmm xmm_temp2 = Xmm(9);

    Vmm vmm_weights = Vmm(10);
    Xmm xmm_weights = Xmm(10);
    Vmm vmm_scale = Vmm(11);

    Vmm vmm_mask = Vmm(12);

    std::unique_ptr<jit_load_emitter> load_emitter = nullptr;
    std::vector<size_t> load_pool_gpr_idxs;

    std::unique_ptr<jit_store_emitter> store_emitter = nullptr;
    std::vector<size_t> store_pool_gpr_idxs;
    std::vector<size_t> store_pool_vec_idxs;

    Opmask k_mask = Opmask(7);

    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using reg16_t = const Xbyak::Reg16;
    reg64_t reg_src         = r8;
    reg64_t reg_idx_y       = r9;
    reg64_t reg_idx_x       = r10;
    reg64_t reg_stride_y    = r11;
    reg64_t reg_stride_x    = r12;
    reg64_t reg_weights     = r13;
    reg64_t reg_work_amount = r14;
    reg64_t reg_dst         = r15;
    reg64_t reg_scale       = rdx;

    reg64_t reg_load_table = rax;
    reg64_t reg_load_store_mask = rbx;

    reg64_t reg_tmp_64 = rbp;
    reg32_t reg_tmp_32 = ebp;
    reg16_t reg_tmp_16 = bp;

    reg64_t reg_params = abi_param1;

    // gather f32 data from reg_src with vmm_idx(data_size) to vmm_src with f32 precision
    inline void gather_f32(Vmm &vmm_src, const reg64_t &reg_src, const Vmm &vmm_idx) {
        constexpr bool is_ymm = std::is_same<Vmm, Xbyak::Ymm>::value;
        constexpr bool is_zmm = std::is_same<Vmm, Xbyak::Zmm>::value;

        if (is_zmm) {
            kxnord(k_mask, k_mask, k_mask);
            vgatherdps(vmm_src | k_mask, ptr[reg_src + vmm_idx * jcp_.data_size]);
        } else if (is_ymm) {
            uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
            vgatherdps(vmm_src, ptr[reg_src + vmm_idx * jcp_.data_size], vmm_mask);
        } else {
            gather_f32_xmm(Xbyak::Xmm(vmm_src.getIdx()), reg_src, Xbyak::Xmm(vmm_idx.getIdx()));
        }
    }

    inline void gather_f32_xmm(Xbyak::Xmm xmm_src, const reg64_t reg_src, const Xbyak::Xmm xmm_idx) {
        sub(rsp, x_len);
        uni_vmovdqu(ptr[rsp], xmm_idx);
        for (int i = 0; i < x_step; i++) {
            mov(reg_tmp_32, ptr[rsp + i * sizeof(int)]);       // sizeof(int)  index_size
            mov(reg_tmp_32, ptr[reg_src + reg_tmp_64 * jcp_.data_size]);  // scale: sizeof(float)   value_size
            mov(ptr[rsp + i * sizeof(int)], reg_tmp_32);
        }
        uni_vmovups(xmm_src, ptr[rsp]);
        add(rsp, x_len);
    }

    // gather bf16 data from reg_src with vmm_idx(data_size) to vmm_src with f32 precision
    // bf16 is needed from avx512_core
    inline void gather_bf16_to_f32_zmm(Vmm vmm_src, const reg64_t reg_src, const Vmm vmm_idx) {
        if (!std::is_same<Vmm, Xbyak::Zmm>::value)
            IE_THROW() << "bf16 is only supported from avx512_core platform for ROIAlign node.";
        sub(rsp, v_len);
        uni_vmovdqu(ptr[rsp], vmm_idx);
        for (int i = 0; i < v_step; i++) {
            mov(reg_tmp_32, ptr[rsp + i * sizeof(int)]);       // sizeof(int)  index_size
            mov(reg_tmp_16, word[reg_src + reg_tmp_64 * jcp_.data_size]);  // scale: sizeof(bf16)   value_size
            mov(ptr[rsp + i * sizeof(int)], reg_tmp_16);
        }
        uni_vmovups(vmm_src, ptr[rsp]);    // |_ x|_ x|_ x|_ x|
        uni_vpslld(vmm_src, vmm_src, 16);  // |x 0|x 0|x 0|x 0|

        add(rsp, v_len);
    }

    inline void gather_bf16_to_f32_xmm(Xbyak::Xmm xmm_src, const reg64_t reg_src, const Xbyak::Xmm xmm_idx) {
        sub(rsp, x_len);
        uni_vmovdqu(ptr[rsp], xmm_idx);
        for (int i = 0; i < x_step; i++) {
            mov(reg_tmp_32, ptr[rsp + i * sizeof(int)]);
            mov(reg_tmp_16, ptr[reg_src + reg_tmp_64 * jcp_.data_size]);
            mov(ptr[rsp + i * sizeof(int)], reg_tmp_16);
        }
        uni_vmovups(xmm_src, ptr[rsp]);    // |_ x|_ x|_ x|_ x|
        uni_vpslld(xmm_src, xmm_src, 16);  // |x 0|x 0|x 0|x 0|

        add(rsp, x_len);
    }

    inline void horizontal_add_xmm(const Xbyak::Xmm &xmm_dst, const Xbyak::Xmm &xmm_aux) {
        uni_vmovshdup(xmm_aux, xmm_dst);              //  dst:1,2,3,4; aux:2,2,4,4
        uni_vaddps(xmm_dst, xmm_dst, xmm_aux);        //  dst:1+2,2+2,3+4,4+4
        uni_vmovhlps(xmm_aux, xmm_aux, xmm_dst);      //  aux:3+4,4+4,4,4
        uni_vaddps(xmm_dst, xmm_dst, xmm_aux);        //  dst:1+2+3+4,...
    }

    // horizontal add for vmm_dst, temp1 and temp2 as aux
    inline void horizontal_add() {
        Xbyak::Xmm xmm_dst = Xbyak::Xmm(vmm_dst.getIdx());
        Xbyak::Xmm xmm_temp1 = Xbyak::Xmm(vmm_temp1.getIdx());
        Xbyak::Xmm xmm_temp2 = Xbyak::Xmm(vmm_temp2.getIdx());
        if (isa == cpu::x64::sse41) {
            horizontal_add_xmm(xmm_dst, xmm_temp1);
        } else if (isa == cpu::x64::avx2) {
            Xbyak::Ymm ymm_dst = Xbyak::Ymm(vmm_dst.getIdx());
            vextractf128(xmm_temp1, ymm_dst, 0);
            vextractf128(xmm_temp2, ymm_dst, 1);
            uni_vaddps(xmm_dst, xmm_temp1, xmm_temp2);
            horizontal_add_xmm(xmm_dst, xmm_temp1);
        } else {
            Xbyak::Zmm zmm_dst = Xbyak::Zmm(vmm_dst.getIdx());
            vextractf32x4(xmm_temp1, zmm_dst, 0);
            vextractf32x4(xmm_temp2, zmm_dst, 1);
            uni_vaddps(xmm_temp1, xmm_temp1, xmm_temp2);
            vextractf32x4(xmm_temp2, zmm_dst, 2);
            vextractf32x4(xmm_dst, zmm_dst, 3);
            uni_vaddps(xmm_dst, xmm_dst, xmm_temp2);
            uni_vaddps(xmm_dst, xmm_dst, xmm_temp1);
            horizontal_add_xmm(xmm_dst, xmm_temp1);
        }
    }
};

bool MKLDNNROIAlignNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        auto roiAlign = ngraph::as_type_ptr<const ngraph::opset3::ROIAlign>(op);
        if (!roiAlign) {
            errorMessage = "Only opset3 ROIAlign operation is supported";
            return false;
        }

        const ngPoolingMode mode = roiAlign->get_mode();
        if (mode != ngPoolingMode::AVG && mode != ngPoolingMode::MAX) {
            errorMessage = "Doesn't support mode: " + ngraph::as_string(mode);
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNROIAlignNode::MKLDNNROIAlignNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
                                       MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "ROIPooling layer with name '" + getName() + "' ";

        auto roiAlign = ngraph::as_type_ptr<const ngraph::opset3::ROIAlign>(op);
        pooledH = roiAlign->get_pooled_h();
        pooledW = roiAlign->get_pooled_w();
        spatialScale = roiAlign->get_spatial_scale();
        samplingRatio = roiAlign->get_sampling_ratio();
        const ngPoolingMode m = roiAlign->get_mode();
        if (m == ngPoolingMode::MAX) {
            algorithm = Algorithm::ROIAlignMax;
        } else if (m == ngPoolingMode::AVG) {
            algorithm = Algorithm::ROIAlignAvg;
        }
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

void MKLDNNROIAlignNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    if (getParentEdges().size() != 3)
        IE_THROW() << errorPrefix << "has incorrect number of input edges: " << getParentEdges().size();
    if (getChildEdges().empty())
        IE_THROW() << errorPrefix << "has incorrect number of output edges: " << getChildEdges().size();

    if (getInputShapeAtPort(0).getRank() != 4) {
        IE_THROW() << errorPrefix << "doesn't support 0th input with rank: " << getInputShapeAtPort(0).getRank();
    }

    if (getInputShapeAtPort(1).getRank() != 2) {
        IE_THROW() << errorPrefix << "doesn't support 1st input with rank: " << getInputShapeAtPort(1).getRank();
    }

    if (getInputShapeAtPort(2).getRank() != 1) {
        IE_THROW() << errorPrefix << "doesn't support 2nd input with rank: " << getInputShapeAtPort(2).getRank();
    }

    if (getOutputShapeAtPort(0).getRank() != 4) {
        IE_THROW() << errorPrefix << "doesn't support output with rank: " << getOutputShapeAtPort(0).getRank();
    }

    const auto& proposalsDims = getInputShapeAtPort(1).getDims();
    if (proposalsDims[1] != 4) {
        IE_THROW() << errorPrefix << "has invalid shape on 1st input: [" << proposalsDims[0] << "," << proposalsDims[1] << "]";
    }

    const auto& indexesDims = getInputShapeAtPort(2).getDims();
    if (!dimsEqualWeak(proposalsDims[0], indexesDims[0])) {
        IE_THROW() << errorPrefix << "has different sizes of inputs for proposals ("
                   << proposalsDims[0] << ") and indexes (" << indexesDims[0] << ")";
    }
}

void MKLDNNROIAlignNode::createJitKernel(const InferenceEngine::Precision& dataPrec) {
    auto jcp = jit_roi_align_params();
    jcp.alg = algorithm;
    jcp.data_prc = dataPrec;
    jcp.data_size = dataPrec.size();

    if (mayiuse(cpu::x64::avx512_common)) {
        roi_align_kernel.reset(new jit_uni_roi_align_kernel_f32<cpu::x64::avx512_common>(jcp));
    } else if (mayiuse(cpu::x64::avx2)) {
        roi_align_kernel.reset(new jit_uni_roi_align_kernel_f32<cpu::x64::avx2>(jcp));
    } else if (mayiuse(cpu::x64::sse41)) {
        roi_align_kernel.reset(new jit_uni_roi_align_kernel_f32<cpu::x64::sse41>(jcp));
    }

    if (roi_align_kernel)
        roi_align_kernel->create_ker();
}

void MKLDNNROIAlignNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    Precision inputPrec0 = getOriginalInputPrecisionAtPort(0);
    Precision outputPrec = getOriginalOutputPrecisionAtPort(0);

    if (inputPrec0 != Precision::FP32 || outputPrec != Precision::FP32) {
        if ((outputPrec == Precision::BF16 || inputPrec0 == Precision::BF16) && mayiuse(avx512_core)) {
            outputPrec = inputPrec0 = Precision::BF16;
        } else {
            outputPrec = inputPrec0 = Precision::FP32;
        }
    }

    NodeConfig config;
    config.dynBatchSupport = false;
    config.inConfs.resize(3);
    config.outConfs.resize(1);

    std::vector<std::pair<LayoutType, LayoutType>> supportedFormats {
            {LayoutType::ncsp, LayoutType::ncsp},
            {LayoutType::nspc, LayoutType::nspc},
            {LayoutType::nCsp16c, LayoutType::nCsp16c},
            {LayoutType::nCsp8c, LayoutType::nCsp8c}
    };

    impl_desc_type impl_type;
    if (mayiuse(cpu::x64::avx512_common)) {
        impl_type = impl_desc_type::jit_avx512;
    } else if (mayiuse(cpu::x64::avx2)) {
        impl_type = impl_desc_type::jit_avx2;
    } else if (mayiuse(cpu::x64::sse41)) {
        impl_type = impl_desc_type::jit_sse42;
    } else {
        impl_type = impl_desc_type::ref;
    }

    for (auto fmts : supportedFormats) {
        addSupportedPrimDesc({{fmts.first, inputPrec0},
                              {LayoutType::ncsp, Precision::FP32},
                              {LayoutType::ncsp, Precision::I32}},
                             {{fmts.second, outputPrec}},
                              impl_type);
    }

    // input and output precision is always the same.
    createJitKernel(outputPrec);
}

namespace {
struct ROIAlignContext {
    MKLDNNROIAlignNode &node;
};
}

template<typename T>
struct MKLDNNROIAlignNode::ROIAlignExecute {
    using srcT = typename std::tuple_element<0, T>::type;
    using dstT = typename std::tuple_element<1, T>::type;

    void operator()(ROIAlignContext & ctx) {
        ctx.node.executeSpecified<srcT, dstT>();
    }
};
void MKLDNNROIAlignNode::execute(mkldnn::stream strm) {
    auto inputPrec = getParentEdgeAt(0)->getMemory().GetDataType();
    auto outputPrec = getChildEdgeAt(0)->getMemory().GetDataType();
    if (!((inputPrec == mkldnn_bf16 && outputPrec == mkldnn_bf16) ||
          (inputPrec == mkldnn_f32 && outputPrec == mkldnn_f32)))
        IE_THROW() <<"ROIAlign doesn't support demanded precisions";

    ROIAlignContext ctx = {
            *this
    };

    OV_SWITCH(MKLDNNPlugin, ROIAlignExecute, ctx, std::tie(inputPrec, outputPrec),
              OV_CASE2(mkldnn_f32, mkldnn_f32, float, float),
              OV_CASE2(mkldnn_bf16, mkldnn_bf16, bfloat16_t, bfloat16_t))
}

template <typename inputType, typename outputType>
void MKLDNNROIAlignNode::executeSpecified() {
    auto &srcMemory0 = getParentEdgeAt(0)->getMemory();
    auto &srcMemory1 = getParentEdgeAt(1)->getMemory();
    auto &dstMemory = getChildEdgeAt(0)->getMemory();

    auto srcBlockDesc = srcMemory0.GetDescWithType<BlockedMemoryDesc>();
    auto dstBlockDesc = dstMemory.GetDescWithType<BlockedMemoryDesc>();

    auto isPlainFmt = srcBlockDesc->hasLayoutType(LayoutType::ncsp);
    auto isNhwcFmt =  srcBlockDesc->hasLayoutType(LayoutType::nspc);
    auto isBlkFmt =   srcBlockDesc->hasLayoutType(LayoutType::nCsp16c) || srcBlockDesc->hasLayoutType(LayoutType::nCsp8c);

    int blockSize = isBlkFmt ? srcBlockDesc->getBlockDims().back() : 1;

    const auto *srcData = reinterpret_cast<const inputType *>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    const auto *srcRoi = reinterpret_cast<const float *>(getParentEdgeAt(1)->getMemoryPtr()->GetPtr());
    const auto *srcRoiIdx = reinterpret_cast<const int *>(getParentEdgeAt(2)->getMemoryPtr()->GetPtr());
    auto *dst = reinterpret_cast<outputType *>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    auto nominalRoiCount = static_cast<int>(srcMemory1.getStaticDims()[0]);
    int realRois = 0;
    auto inputDimVector = srcMemory0.getStaticDims();
    const int C = static_cast<int>(inputDimVector[1]);
    const int H = static_cast<int>(inputDimVector[2]);
    const int W = static_cast<int>(inputDimVector[3]);

    const int binCount = pooledH * pooledW;

    const size_t tailDimsOffset = (isNhwcFmt ? -1 : 0);
    const auto &srcStrides = srcBlockDesc->getStrides();
    const auto &dstStrides = dstBlockDesc->getStrides();
    const int hInputStride = srcStrides[2 + tailDimsOffset];
    const int wInputStride = srcStrides[3 + tailDimsOffset];
    const int hOutputStride = dstStrides[2 + tailDimsOffset];
    const int wOutputStride = dstStrides[3 + tailDimsOffset];
    const int chPadding = blockSize * srcBlockDesc->getBlockDims()[1];
    const int blockCount = chPadding / blockSize;

    for (; realRois < nominalRoiCount; realRois++) {
        auto roiBatchInd = srcRoiIdx[realRois];
        if (roiBatchInd == -1) {
            break;
        }
    }

    for (int n = 0; n < realRois; ++n) {
        int roiOff = n * 4;
        const float* srcRoiPtr = &srcRoi[roiOff];
        int roiBatchInd = srcRoiIdx[n];
        if (roiBatchInd < -1) {  // -1 means switched off region
            IE_THROW() << "Batch index cannot be less, than -1";
        } else if (roiBatchInd >= inputDimVector[0]) {
            IE_THROW() << "Demanded batch (id = " << roiBatchInd << ") doesn't exist";
        }

        float x1 = srcRoiPtr[0] * spatialScale;
        float y1 = srcRoiPtr[1] * spatialScale;
        float x2 = srcRoiPtr[2] * spatialScale;
        float y2 = srcRoiPtr[3] * spatialScale;

        float roiHeight = std::max(y2 - y1, 1.0f);
        float roiWidth = std::max(x2 - x1, 1.0f);
        float binHeight = roiHeight / pooledH;
        float binWidth = roiWidth / pooledW;

        auto samplingRatioX = samplingRatio == 0 ? static_cast<int>(ceil(binWidth)) : samplingRatio;
        auto samplingRatioY = samplingRatio == 0 ? static_cast<int>(ceil(binHeight)) : samplingRatio;

        uint64_t numSamplesInBin = static_cast<uint64_t>(samplingRatioX) * samplingRatioY;

        float sampleDistanceX = binWidth / samplingRatioX;
        float sampleDistanceY = binHeight / samplingRatioY;
        // prepare arrays for sampling points and weights
        std::vector<int> pointVectorY;
        std::vector<int> pointVectorX;
        std::vector<float> weightVector;
        pointVectorY.reserve(4 * numSamplesInBin * binCount);
        pointVectorX.reserve(4 * numSamplesInBin * binCount);
        weightVector.reserve(4 * numSamplesInBin * binCount);

        for (int yBinInd = 0; yBinInd < pooledH; ++yBinInd) {
            for (int xBinInd = 0; xBinInd < pooledW; ++xBinInd) {
                // run into bin
                for (int ySampleInd = 0; ySampleInd < samplingRatioY; ySampleInd++) {
                    float sampleY = y1 + yBinInd * binHeight + sampleDistanceY * (0.5f + ySampleInd);
                    for (int xSampleInd = 0; xSampleInd < samplingRatioX; xSampleInd++) {
                        float sampleX = x1 + xBinInd * binWidth + sampleDistanceX * (0.5f + xSampleInd);
                        if (sampleX < -1.0 || sampleX > W ||
                            sampleY < -1.0 || sampleY > H) {
                            // For this sample we save 4x point (0,0) with weight 0
                            pointVectorY.insert(pointVectorY.end(), 4, 0);
                            pointVectorX.insert(pointVectorX.end(), 4, 0);
                            weightVector.insert(weightVector.end(), 4, float{0});
                            continue;
                        }
                        sampleX = std::max(sampleX, float{0});
                        sampleY = std::max(sampleY, float{0});

                        auto sampleYLow = static_cast<unsigned int>(sampleY);
                        auto sampleXLow = static_cast<unsigned int>(sampleX);
                        unsigned int sampleYHigh;
                        unsigned int sampleXHigh;
                        if (sampleYLow >= H - 1) {
                            sampleYHigh = sampleYLow = H - 1;
                            sampleY = static_cast<float>(sampleYLow);
                        } else {
                            sampleYHigh = sampleYLow + 1;
                        }
                        if (sampleXLow >= W - 1) {
                            sampleXHigh = sampleXLow = W - 1;
                            sampleX = static_cast<float>(sampleXLow);
                        } else {
                            sampleXHigh = sampleXLow + 1;
                        }
                        pointVectorY.push_back(sampleYLow);
                        pointVectorY.push_back(sampleYLow);
                        pointVectorY.push_back(sampleYHigh);
                        pointVectorY.push_back(sampleYHigh);
                        pointVectorX.push_back(sampleXLow);
                        pointVectorX.push_back(sampleXHigh);
                        pointVectorX.push_back(sampleXLow);
                        pointVectorX.push_back(sampleXHigh);

                        // weight calculation for bilinear interpolation
                        auto ly = sampleY - sampleYLow;
                        auto lx = sampleX - sampleXLow;
                        auto hy = 1.0f - ly;
                        auto hx = 1.0f - lx;

                        weightVector.push_back(hy * hx);
                        weightVector.push_back(hy * lx);
                        weightVector.push_back(ly * hx);
                        weightVector.push_back(ly * lx);
                    }
                }
            }
        }

        if (roi_align_kernel) {
            float numSamplesInBinInvert = 1.f / numSamplesInBin;
            if (isNhwcFmt) {
                parallel_for2d(pooledH, pooledW, [&](int yBinInd, int xBinInd) {
                    unsigned int sampleIndex = 4 * (yBinInd * pooledW + xBinInd) * numSamplesInBin;
                    auto arg = jit_roi_align_call_args(nullptr,
                        static_cast<const int*>(&pointVectorY[sampleIndex]),
                        static_cast<const int*>(&pointVectorX[sampleIndex]),
                        static_cast<const int*>(&hInputStride),
                        static_cast<const int*>(&wInputStride),
                        static_cast<const float*>(&weightVector[sampleIndex]),
                        static_cast<const float*>(&numSamplesInBinInvert),
                        nullptr,
                        static_cast<size_t>(numSamplesInBin));
                    size_t dstOffset = yBinInd * hOutputStride + xBinInd * wOutputStride;

                    for (int c = 0; c < C; c++) {
                        size_t binOffsetInput = roiBatchInd * C * H * W + c;
                        size_t binOffsetOutput = n * C * binCount + c;

                        size_t dstIndex = dstOffset + binOffsetOutput;
                        arg.src = static_cast<const void*>(&srcData[binOffsetInput]);
                        arg.dst = static_cast<void*>(&dst[dstIndex]);
                        (*roi_align_kernel)(&arg);
                    }
                });
            } else {  // nchw, nChw16c, nChw8c
                parallel_for3d(blockCount, pooledH, pooledW, [&](int blkIdx, int yBinInd, int xBinInd) {
                    unsigned int sampleIndex = 4 * (yBinInd * pooledW + xBinInd) * numSamplesInBin;
                    auto arg = jit_roi_align_call_args(nullptr,
                        static_cast<const int*>(&pointVectorY[sampleIndex]),
                        static_cast<const int*>(&pointVectorX[sampleIndex]),
                        static_cast<const int*>(&hInputStride),
                        static_cast<const int*>(&wInputStride),
                        static_cast<const float*>(&weightVector[sampleIndex]),
                        static_cast<const float*>(&numSamplesInBinInvert),
                        nullptr,
                        static_cast<size_t>(numSamplesInBin));
                    int cStart = blkIdx * blockSize;
                    int cEnd = (blkIdx == blockCount - 1 ? C : cStart + blockSize);
                    size_t dstOffset = yBinInd * hOutputStride + xBinInd * wOutputStride;
                    for (int c = cStart; c < cEnd; c++) {
                        const int blockResidual = (isPlainFmt ? 0 : c % blockSize);
                        const int blockIdx = (c / blockSize) * blockSize;
                        size_t binOffsetInput = (roiBatchInd * chPadding + blockIdx) * H * W;
                        size_t binOffsetOutput = (n * chPadding + blockIdx) * binCount;

                        size_t srcIndex = binOffsetInput + blockResidual;
                        size_t dstIndex = dstOffset + binOffsetOutput + blockResidual;
                        arg.src = static_cast<const void*>(&srcData[srcIndex]);
                        arg.dst = static_cast<void*>(&dst[dstIndex]);
                        (*roi_align_kernel)(&arg);
                    }
                });
            }
        } else {
            auto pool = [&] (int xBinInd_, int yBinInd_, int binOffsetInput_, int binOffsetOutput_, int blockResidual_) {
                float pooledValue = 0;
                unsigned int sampleIndex = 4 * (yBinInd_ * pooledW + xBinInd_) * numSamplesInBin;
                for (unsigned int binSampleInd = 0; binSampleInd < numSamplesInBin; binSampleInd++) {
                    size_t part1Index = binOffsetInput_ + pointVectorY[sampleIndex] * hInputStride +
                                        pointVectorX[sampleIndex] * wInputStride + blockResidual_;
                    float part1 = srcData[part1Index];
                    size_t part2Index = binOffsetInput_ + pointVectorY[sampleIndex + 1] * hInputStride +
                                        pointVectorX[sampleIndex + 1] * wInputStride + blockResidual_;
                    float part2 = srcData[part2Index];
                    size_t part3Index = binOffsetInput_ + pointVectorY[sampleIndex + 2] * hInputStride +
                                        pointVectorX[sampleIndex + 2] * wInputStride + blockResidual_;
                    float part3 = srcData[part3Index];
                    size_t part4Index = binOffsetInput_ + pointVectorY[sampleIndex + 3] * hInputStride +
                                        pointVectorX[sampleIndex + 3] * wInputStride + blockResidual_;
                    float part4 = srcData[part4Index];

                    float sampleValue =
                            weightVector[sampleIndex] * part1 +
                            weightVector[sampleIndex + 1] * part2 +
                            weightVector[sampleIndex + 2] * part3 +
                            weightVector[sampleIndex + 3] * part4;

                    switch (getAlgorithm()) {
                        case Algorithm::ROIAlignMax:
                        {
                            pooledValue = sampleValue > pooledValue ? sampleValue : pooledValue;
                            break;
                        }
                        case Algorithm::ROIAlignAvg:
                        default:
                        {
                            pooledValue += sampleValue / numSamplesInBin;
                        }
                    }
                    sampleIndex += 4;
                }
                size_t dstIndex = binOffsetOutput_ + yBinInd_ * hOutputStride +
                                   xBinInd_ * wOutputStride + blockResidual_;
                dst[dstIndex] = pooledValue;
            };
            if (isNhwcFmt) {
                parallel_for2d(pooledH, pooledW, [&](int yBinInd, int xBinInd) {
                    for (int c = 0; c < C; c++) {
                        size_t binOffsetInput = roiBatchInd * C * H * W + c;
                        size_t binOffsetOutput = n * C * binCount + c;
                        pool(xBinInd, yBinInd, binOffsetInput, binOffsetOutput, 0);
                    }
                });
            } else {  // nchw, nChw16c, nChw8c
                parallel_for3d(blockCount, pooledH, pooledW, [&](int blkIdx, int yBinInd, int xBinInd) {
                    int cStart = blkIdx * blockSize;
                    int cEnd = (blkIdx == blockCount - 1 ? C : cStart + blockSize);
                    for (int c = cStart; c < cEnd; c++) {
                        const int blockResidual = (isPlainFmt ? 0 : c % blockSize);
                        const int blockIdx = (c / blockSize) * blockSize;
                        size_t binOffsetInput = (roiBatchInd * chPadding + blockIdx) * H * W;
                        size_t binOffsetOutput = (n * chPadding + blockIdx) * binCount;
                        pool(xBinInd, yBinInd, binOffsetInput, binOffsetOutput, blockResidual);
                    }
                });
            }
        }
    }
}

bool MKLDNNROIAlignNode::created() const {
    return getType() == ROIAlign;
}

bool MKLDNNROIAlignNode::needPrepareParams() const {
    return false;
}

void MKLDNNROIAlignNode::executeDynamicImpl(mkldnn::stream strm) {
    execute(strm);
}

REG_MKLDNN_PRIM_FOR(MKLDNNROIAlignNode, ROIAlign)
