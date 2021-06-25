// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <vector>
#include <string>
#include <mkldnn_types.h>
#include "ie_parallel.hpp"
#include "mkldnn_region_yolo_node.h"
#include <nodes/common/blocked_desc_creator.h>
#include <ngraph/opsets/opset1.hpp>
#include "common/cpu_convert.h"
#include <cpu/x64/jit_generator.hpp>
#include <emitters/jit_bf16_emitters.hpp>
#include <cpu/x64/jit_uni_eltwise_injector.hpp>
#include "utils/bfloat16.hpp"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn::impl::cpu;
using namespace mkldnn::impl::cpu::x64;
using namespace mkldnn::impl::utils;

#define GET_OFF(field) offsetof(jit_args_logistic, field)

template <cpu_isa_t isa>
struct jit_uni_logistic_kernel_f32 : public jit_uni_logistic_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_logistic_kernel_f32)

    jit_uni_logistic_kernel_f32(jit_logistic_config_params jcp) : jcp_(jcp), jit_uni_logistic_kernel(), jit_generator() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        exp_injector.reset(new jit_uni_eltwise_injector_f32<isa>(this, mkldnn::impl::alg_kind::eltwise_exp, 0.f, 0.f, 1.f));

        if (!mayiuse(avx512_core_bf16) && mayiuse(avx512_core))
            emu_vcvtneps2bf16.reset(new jit_emu_vcvtneps2bf16(this, isa, nullptr));

        this->preamble();

        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);
        mov(reg_table, l_table);

        Xbyak::Label main_loop_label;
        Xbyak::Label tail_loop_label;
        Xbyak::Label exit_label;

        int step = vlen / sizeof(float);
        L(main_loop_label); {
            cmp(reg_work_amount, step);
            jl(tail_loop_label, T_NEAR);

            load_vector(vmm_src, ptr[reg_src], jcp_.src_dt);
            compute_kernel();
            store_vector(ptr[reg_dst], vmm_src, jcp_.dst_dt);

            add(reg_src, step * jcp_.src_data_size);
            add(reg_dst, step * jcp_.dst_data_size);
            sub(reg_work_amount, step);

            jmp(main_loop_label, T_NEAR);
        }

        step = 1;
        L(tail_loop_label); {
            cmp(reg_work_amount, step);
            jl(exit_label, T_NEAR);

            load_scalar(xmm_src, ptr[reg_src], jcp_.src_dt);
            compute_kernel();
            store_scalar(ptr[reg_dst], xmm_src, jcp_.dst_dt);

            add(reg_src, step * jcp_.src_data_size);
            add(reg_dst, step * jcp_.dst_data_size);
            sub(reg_work_amount, step);

            jmp(tail_loop_label, T_NEAR);
        }

        L(exit_label);

        this->postamble();

        if (!mayiuse(avx512_core_bf16) && mayiuse(avx512_core))
            emu_vcvtneps2bf16->emit_data();

        exp_injector->prepare_table();

        prepare_table();
    }

private:
    using Vmm = typename conditional3<isa == x64::sse41, Xbyak::Xmm, isa == x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    size_t vlen = cpu_isa_traits<isa>::vlen;

    Xbyak::Address table_val(int index) { return ptr[reg_table + index * vlen]; }

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_dst = r9;
    Xbyak::Reg64 reg_table = r10;
    Xbyak::Reg64 reg_work_amount = r11;
    Xbyak::Reg64 reg_params = abi_param1;

    Vmm vmm_aux0 = Vmm(0);
    Vmm vmm_src = Vmm(1);
    Xbyak::Xmm xmm_src = Xbyak::Xmm(1);
    Vmm vmm_aux1 = Vmm(2);
    Vmm vmm_aux2 = Vmm(3);

    const Xbyak::Opmask k_mask = Xbyak::Opmask(1);

    std::unique_ptr<jit_emu_vcvtneps2bf16> emu_vcvtneps2bf16;

    Xbyak::Label l_table;

    std::shared_ptr<jit_uni_eltwise_injector_f32<isa>> exp_injector;

    jit_logistic_config_params jcp_;

    void compute_kernel() {
        uni_vmovups(vmm_aux0, vmm_src);
        uni_vandps(vmm_aux0, vmm_aux0, table_val(0));
        uni_vorps(vmm_src, vmm_src, table_val(0));

        exp_injector->compute_vector_range(vmm_src.getIdx(), vmm_src.getIdx() + 1);

        uni_vmovups(vmm_aux1, vmm_src);
        uni_vaddps(vmm_aux1, vmm_aux1, table_val(1));
        uni_vdivps(vmm_src, vmm_src, vmm_aux1);

        uni_vmovups(vmm_aux2, table_val(1));
        uni_vsubps(vmm_aux2, vmm_aux2, vmm_src);

        if (isa == x64::sse41) {
            uni_vblendvps(vmm_aux2, vmm_aux2, vmm_src, vmm_aux0);
            uni_vmovups(vmm_src, vmm_aux2);
        } else if (isa == x64::avx2) {
            uni_vblendvps(vmm_src, vmm_aux2, vmm_src, vmm_aux0);
        } else {
            vptestmd(k_mask, vmm_aux0, vmm_aux0);
            vblendmps(vmm_src | k_mask, vmm_aux2, vmm_src);
        }
    }

    void prepare_table() {
        auto broadcast_int = [&](int val) {
            for (size_t d = 0; d < vlen / sizeof(float); ++d) {
                dd(val);
            }
        };

        align(64);
        L(l_table);

        broadcast_int(vals_for_logistic_activate.mask_sign);
        broadcast_int(vals_for_logistic_activate.float_1);
    }

    const struct vals_for_logistic_activate_type {
        int mask_sign = 0x80000000;  // 0 //  mask to extract sign
        int float_1   = 0x3f800000;  // 1 //  1.0f
    } vals_for_logistic_activate;

    inline void load_vector(Vmm vmm_src, const Xbyak::Address &op, InferenceEngine::Precision src_dt) {
        switch (src_dt) {
            case InferenceEngine::Precision::FP32:
                uni_vmovups(vmm_src, op);
                break;
            case InferenceEngine::Precision::BF16:
                vpmovzxwd(vmm_src, op);
                uni_vpslld(vmm_src, vmm_src, 16);
                break;
            default:
                assert(!"unknown src_dt");
        }
    }
    inline void store_vector(const Xbyak::Address &op, Vmm vmm_dst, InferenceEngine::Precision dst_dt) {
        Xbyak::Ymm ymm_dst = Xbyak::Ymm(vmm_dst.getIdx());

        switch (dst_dt) {
            case InferenceEngine::Precision::FP32:
                uni_vmovups(op, vmm_dst);
                break;
            case InferenceEngine::Precision::BF16:
                if (mayiuse(avx512_core_bf16))
                    vcvtneps2bf16(ymm_dst, vmm_dst);
                else
                    emu_vcvtneps2bf16->emit_code({static_cast<size_t>(vmm_dst.getIdx())}, {static_cast<size_t>(ymm_dst.getIdx())});
                vmovdqu16(op, ymm_dst);
                break;
            default:
                assert(!"unknown dst_dt");
        }
    }
    inline void load_scalar(Xbyak::Xmm xmm_src, const Xbyak::Address &op, InferenceEngine::Precision src_dt) {
        switch (src_dt) {
            case InferenceEngine::Precision::FP32:
                movss(xmm_src, op);
                break;
            case InferenceEngine::Precision::BF16:
                pinsrw(xmm_src, op, 0x0);
                uni_vpslld(xmm_src, xmm_src, 16);
                break;
            default:
                assert(!"unknown src_dt");
        }
    }
    inline void store_scalar(const Xbyak::Address &op, Xbyak::Xmm xmm_dst, InferenceEngine::Precision dst_dt) {
        switch (dst_dt) {
            case InferenceEngine::Precision::FP32:
                movss(op, xmm_dst);
                break;
            case InferenceEngine::Precision::BF16:
                uni_vpsrld(xmm_dst, xmm_dst, 16);
                pextrw(op, xmm_dst, 0x0);
                break;
           default:
                assert(!"unknown dst_dt");
        }
    }
};

bool MKLDNNRegionYoloNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto regionYolo = std::dynamic_pointer_cast<const ngraph::opset1::RegionYolo>(op);
        if (!regionYolo) {
            errorMessage = "Only opset1 RegionYolo operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNRegionYoloNode::MKLDNNRegionYoloNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = std::string(op->get_type_name()) + " node with name '" + op->get_friendly_name() + "'";
    if (op->get_input_size() != 1 || op->get_output_size() != 1)
        IE_THROW() << errorPrefix << " has incorrect number of input/output edges!";

    const auto regionYolo = std::dynamic_pointer_cast<const ngraph::opset1::RegionYolo>(op);
    classes = regionYolo->get_num_classes();
    coords = regionYolo->get_num_coords();
    num = regionYolo->get_num_regions();
    do_softmax = regionYolo->get_do_softmax();
    mask = regionYolo->get_mask();
}

void MKLDNNRegionYoloNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    input_prec = getOriginalInputPrecisionAtPort(0);
    output_prec = getOriginalOutputPrecisionAtPort(0);

    if (input_prec != Precision::FP32 && input_prec != Precision::BF16) {
        input_prec = Precision::FP32;
    }

    if (output_prec != Precision::FP32 && output_prec != Precision::BF16) {
        output_prec = Precision::FP32;
    }

    if (Precision::BF16 == output_prec) {
        if (!mayiuse(avx512_core)) {
            output_prec = Precision::FP32;
        }
    }

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

    addSupportedPrimDesc({{TensorDescCreatorTypes::ncsp, input_prec}},
                         {{TensorDescCreatorTypes::ncsp, output_prec}},
                         impl_type);
}

void MKLDNNRegionYoloNode::createPrimitive() {
    jit_logistic_config_params jcp;
    jcp.src_dt = jcp.dst_dt = output_prec;
    jcp.src_data_size = jcp.dst_data_size = output_prec.size();

    block_size = 1;
    if (mayiuse(x64::avx512_common)) {
        logistic_kernel.reset(new jit_uni_logistic_kernel_f32<x64::avx512_common>(jcp));
        block_size = 16;
    } else if (mayiuse(x64::avx2)) {
        logistic_kernel.reset(new jit_uni_logistic_kernel_f32<x64::avx2>(jcp));
        block_size = 8;
    } else if (mayiuse(x64::sse41)) {
        logistic_kernel.reset(new jit_uni_logistic_kernel_f32<x64::sse41>(jcp));
        block_size = 4;
    }

    softmax_kernel = std::make_shared<SoftmaxGeneric>(input_prec, output_prec);

    if (logistic_kernel)
        logistic_kernel->create_ker();
}

inline float MKLDNNRegionYoloNode::logistic_scalar(float src) {
    U aux2;
    aux2.as_float_value = src;
    int sign = aux2.as_int_value >> 31;
    if (sign == 0)
        src *= -1;

    src = std::exp(src);

    src = src / (src + 1);
    if (sign == 0)
        src = 1 - src;

    return src;
}

inline void MKLDNNRegionYoloNode::calculate_logistic(size_t start_index, int count, uint8_t * dst_data) {
    auto dst_data_size = output_prec.size();
    if (logistic_kernel) {
        int blocks_num = div_up(count, block_size);
        parallel_for(blocks_num, [&](int ib) {
            int idx = ib * block_size;
            int work_amount = std::min(count - idx, block_size);

            auto arg = jit_args_logistic();
            arg.src = arg.dst = dst_data + dst_data_size * (start_index + idx);
            arg.work_amount = static_cast<size_t>(work_amount);

            (*logistic_kernel)(&arg);
        });
    } else {
        if (Precision::FP32 == output_prec) {
            auto float_dst_data = reinterpret_cast<float*>(dst_data);
            for (int i = 0; i < count; i++) {
                float_dst_data[i + start_index] = logistic_scalar(float_dst_data[i + start_index]);
            }
        } else if (Precision::BF16 == output_prec) {
            auto bf16_dst_data = reinterpret_cast<MKLDNNPlugin::bfloat16_t*>(dst_data);
            for (int i = 0; i < count; i++) {
                bf16_dst_data[i + start_index] = logistic_scalar(bf16_dst_data[i + start_index]);
            }
        } else {
            IE_THROW() << "Unsupported precision configuration outPrc=" << output_prec.name();
        }
    }
}

void MKLDNNRegionYoloNode::execute(mkldnn::stream strm) {
    auto inputDesc = getParentEdgeAt(0)->getDesc();
    auto outputDesc = getChildEdgeAt(0)->getDesc();

    size_t B = (inputDesc.getDims().size() > 0) ? inputDesc.getDims()[0] : 1;
    size_t IC = (inputDesc.getDims().size() > 1) ? inputDesc.getDims()[1] : 1;
    size_t IH = (inputDesc.getDims().size() > 2) ? inputDesc.getDims()[2] : 1;
    size_t IW = (inputDesc.getDims().size() > 3) ? inputDesc.getDims()[3] : 1;

    size_t mask_size = mask.size();
    int end_index = 0;
    int num_ = 0;
    int output_size = 0;
    if (do_softmax) {
        // Region layer (Yolo v2)
        end_index = IW * IH;
        num_ = num;
        output_size = B * IH * IW * IC; // different shape combinations with the same overall size;
    } else {
        // Yolo layer (Yolo v3)
        end_index = IW * IH * (classes + 1);
        num_ = mask_size;
        output_size = B * IH * IW * mask_size * (classes + coords + 1);
    }

    if (output_size != getChildEdgeAt(0)->getMemoryPtr()->GetElementsCount())
        IE_THROW() << "Incorrect layer configuration or output dimensions. " << output_size << " != " << getChildEdgeAt(0)->getMemoryPtr()->GetElementsCount();

    size_t inputs_size = IH * IW * num_ * (classes + coords + 1);
    size_t total_size = 2 * IH * IW;

    const auto *src_data = reinterpret_cast<const uint8_t *>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    auto *dst_data = reinterpret_cast<uint8_t *>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    cpu_convert(src_data, dst_data, inputDesc.getPrecision(), outputDesc.getPrecision(), output_size);

    for (int b = 0; b < B; b++) {
        for (int n = 0; n < num_; n++) {
            size_t index = b * inputs_size + n * IW * IH * (classes + coords + 1);
            calculate_logistic(index, total_size, dst_data);

            index = b * inputs_size + IW * IH * (n * (classes + coords + 1) + coords);
            calculate_logistic(index, end_index, dst_data);
        }
    }

    if (do_softmax) {
        int index = IW * IH * (coords + 1);
        int batch_offset = inputs_size / num;
        for (int b = 0; b < B * num; b++) {
            softmax_kernel->execute(src_data + input_prec.size() * (index + b * batch_offset),
                                    dst_data + output_prec.size() * (index + b * batch_offset), 1, classes, IH, IW);
        }
    }
}

bool MKLDNNRegionYoloNode::created() const {
    return getType() == RegionYolo;
}

REG_MKLDNN_PRIM_FOR(MKLDNNRegionYoloNode, RegionYolo)
