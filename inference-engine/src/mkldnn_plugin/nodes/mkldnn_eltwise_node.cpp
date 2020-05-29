// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_eltwise_node.h"
#include <ie_layers.h>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include "ie_parallel.hpp"
#include "mkldnn_quantize_node.h"
#include "mkldnn_activation_node.h"
#include <map>
#include "jit_uni_eltwise.hpp"
#include "jit_uni_quantization.hpp"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_eltwise_fq_call_args, field)

template <cpu_isa_t isa>
struct jit_uni_eltwise_fq_generic : public jit_uni_eltwise_fq_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_eltwise_fq_generic)

    explicit jit_uni_eltwise_fq_generic(jit_eltwise_fq_params jep, const mkldnn_primitive_attr &attr) : jit_uni_eltwise_fq_kernel(jep, attr), jit_generator() {
        const auto &p = attr_.post_ops_;
        for (int i = 0; i < p.len_; i++) {
            auto &post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors.push_back(std::make_shared<jit_uni_eltwise_injector_f32<isa>>(
                        this, post_op.eltwise.alg, post_op.eltwise.alpha, post_op.eltwise.beta));
            } else if (post_op.is_quantization()) {
                quantization_injectors.push_back(std::make_shared<jit_uni_quantization_injector_f32<isa>>(
                        this, post_op, vmm_d_weights, vmm_d_bias, reg_d_weights, reg_d_bias));
            }
        }

        this->preamble();

        mov(reg_src0, ptr[reg_params + GET_OFF(src0)]);
        mov(reg_src1, ptr[reg_params + GET_OFF(src1)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);
        xor_(reg_oc_off, reg_oc_off);

        Xbyak::Label main_loop_label;
        Xbyak::Label main_loop_end_label;
        Xbyak::Label tail_loop_label;
        Xbyak::Label tail_loop_end_label;

        if (isa == avx512_common)
            vpxord(vmm_zero, vmm_zero, vmm_zero);

        if (jep.src0_step == 0)
            uni_vbroadcastss(vmm_src0, ptr[reg_src0]);
        if (jep.src1_step == 0)
            uni_vbroadcastss(vmm_src1, ptr[reg_src1]);

        L(main_loop_label);
        {
            cmp(reg_work_amount, simd_w);
            jl(main_loop_end_label, T_NEAR);

            if (jep.src0_step != 0)
                load_vector(vmm_src0, ptr[reg_src0], jep.src0_dt);
            if (jep.src1_step != 0)
                load_vector(vmm_src1, ptr[reg_src1], jep.src1_dt);

            switch (jep.eltwise_op) {
                case EltwiseLayer::eOperation::Sum:
                    if (isa == cpu::sse42) {
                        uni_vmovups(vmm_dst, vmm_src0);
                        uni_vaddps(vmm_dst, vmm_dst, vmm_src1);
                    } else {
                        uni_vaddps(vmm_dst, vmm_src0, vmm_src1);
                    }
                    break;
                case EltwiseLayer::eOperation::Prod:
                    if (isa == cpu::sse42) {
                        uni_vmovups(vmm_dst, vmm_src0);
                        uni_vmulps(vmm_dst, vmm_dst, vmm_src1);
                    } else {
                        uni_vmulps(vmm_dst, vmm_src0, vmm_src1);
                    }
                    break;
                default: THROW_IE_EXCEPTION << "Unsupported operation type for Eltwise node";
            }

            int eltwise_inj_idx = 0;
            int quantization_inj_idx = 0;
            for (int i = 0; i < p.len_; i++) {
                auto &post_op = p.entry_[i];
                if (post_op.is_eltwise()) {
                    eltwise_injectors[eltwise_inj_idx]->compute_vector_range(vmm_dst.getIdx(), vmm_dst.getIdx() + 1);
                    eltwise_inj_idx++;
                } else if (post_op.is_quantization()) {
                    bool do_dequantization = post_op.quantization.alg == alg_kind::quantization_quantize_dequantize;
                    bool do_rounding = do_dequantization || jep_.dst_dt == data_type::f32 || i != p.len_ - 1;
                    int s_idx = vmm_dst.getIdx();

                    quantization_injectors[quantization_inj_idx]->init_crop_ptrs(reg_oc_off);
                    quantization_injectors[quantization_inj_idx]->compute_crop(s_idx, s_idx + 1, 0);

                    quantization_injectors[quantization_inj_idx]->init_input_scale_shift_ptrs(reg_oc_off);
                    quantization_injectors[quantization_inj_idx]->compute_input_scale_shift(s_idx, s_idx + 1, 0, do_rounding);

                    quantization_injectors[quantization_inj_idx]->init_output_scale_shift_ptrs(reg_oc_off);
                    quantization_injectors[quantization_inj_idx]->compute_output_scale_shift(s_idx, s_idx + 1, 0);

                    quantization_inj_idx++;
                }
            }

            store_vector(ptr[reg_dst], vmm_dst, jep.dst_dt);

            if (jep.src0_step != 0)
                add(reg_src0, jep.src0_step * jep.src0_data_size * simd_w);
            if (jep.src1_step != 0)
                add(reg_src1, jep.src1_step * jep.src1_data_size * simd_w);
            add(reg_dst, jep.dst_step * jep.dst_data_size * simd_w);
            sub(reg_work_amount, simd_w);
            add(reg_oc_off, simd_w * sizeof(float));

            jmp(main_loop_label, T_NEAR);
        }

        L(main_loop_end_label);

        L(tail_loop_label);
        {
            cmp(reg_work_amount, 1);
            jl(tail_loop_end_label, T_NEAR);

            if (jep.src0_step != 0)
                load_scalar(xmm_src0, ptr[reg_src0], jep.src0_dt);
            if (jep.src1_step != 0)
                load_scalar(xmm_src1, ptr[reg_src1], jep.src1_dt);

            switch (jep.eltwise_op) {
                case EltwiseLayer::eOperation::Sum: uni_vaddps(vmm_dst, vmm_src0, vmm_src1); break;
                case EltwiseLayer::eOperation::Prod: uni_vmulps(vmm_dst, vmm_src0, vmm_src1); break;
                default: THROW_IE_EXCEPTION << "Unsupported operation type for Eltwise node";
            }

            int eltwise_inj_idx = 0;
            int quantization_inj_idx = 0;
            for (int i = 0; i < p.len_; i++) {
                auto &post_op = p.entry_[i];
                if (post_op.is_eltwise()) {
                    eltwise_injectors[eltwise_inj_idx]->compute_vector_range(vmm_dst.getIdx(), vmm_dst.getIdx() + 1);
                    eltwise_inj_idx++;
                } else if (post_op.is_quantization()) {
                    bool do_dequantization = post_op.quantization.alg == alg_kind::quantization_quantize_dequantize;
                    bool do_rounding = do_dequantization || jep_.dst_dt == data_type::f32 || i != p.len_ - 1;
                    int s_idx = vmm_dst.getIdx();

                    quantization_injectors[quantization_inj_idx]->init_crop_ptrs(reg_oc_off);
                    quantization_injectors[quantization_inj_idx]->compute_crop(s_idx, s_idx + 1, 0, true);

                    quantization_injectors[quantization_inj_idx]->init_input_scale_shift_ptrs(reg_oc_off);
                    quantization_injectors[quantization_inj_idx]->compute_input_scale_shift(s_idx, s_idx + 1, 0, do_rounding, true);

                    quantization_injectors[quantization_inj_idx]->init_output_scale_shift_ptrs(reg_oc_off);
                    quantization_injectors[quantization_inj_idx]->compute_output_scale_shift(s_idx, s_idx + 1, 0, true);

                    quantization_inj_idx++;
                }
            }

            store_scalar(ptr[reg_dst], xmm_dst, jep.dst_dt);

            if (jep.src0_step != 0)
                add(reg_src0, jep.src0_step * jep.src0_data_size);
            if (jep.src1_step != 0)
                add(reg_src1, jep.src1_step * jep.src1_data_size);
            add(reg_dst, jep.dst_step * jep.dst_data_size);
            sub(reg_work_amount, 1);
            add(reg_oc_off, 1 * sizeof(float));

            jmp(tail_loop_label, T_NEAR);
        }

        L(tail_loop_end_label);

        this->postamble();

        for (auto& inj : eltwise_injectors)
            inj->prepare_table();

        ker_ = (decltype(ker_)) this->getCode();
    }

private:
    using Vmm = typename conditional3<isa == cpu::sse42, Xmm, isa == cpu::avx2, Ymm, Zmm>::type;

    const int simd_w = cpu_isa_traits<isa>::vlen / sizeof(float);

    Reg64 reg_src0 = r8;
    Reg64 reg_src1 = r9;
    Reg64 reg_dst = r10;
    Reg64 reg_work_amount = r11;
    Reg64 reg_oc_off = r13;
    Reg64 reg_params = abi_param1;

    Reg8 reg_tmp_8 = r12b;
    Reg32 reg_tmp_32 = r12d;
    Reg64 reg_tmp_64 = r12;

    Reg64 reg_d_weights = r14;
    Reg64 reg_d_bias = r15;

    Vmm vmm_src0 = Vmm(0);
    Vmm vmm_src1 = Vmm(1);
    Vmm vmm_dst = Vmm(2);
    Xmm xmm_src0 = Xmm(0);
    Xmm xmm_src1 = Xmm(1);
    Xmm xmm_dst = Xmm(2);

    Vmm vmm_d_weights = Vmm(3);
    Vmm vmm_d_bias = Vmm(4);

    Vmm vmm_zero = Vmm(5);

    std::vector<std::shared_ptr<jit_uni_eltwise_injector_f32<isa>>> eltwise_injectors;
    std::vector<std::shared_ptr<jit_uni_quantization_injector_f32<isa>>> quantization_injectors;

    inline void load_vector(Vmm vmm_src, const Xbyak::Address &op, memory::data_type src_dt) {
        switch (src_dt) {
            case memory::f32:
            case memory::s32:
                uni_vmovups(vmm_src, op);
                break;
            case memory::s8:
                uni_vpmovsxbd(vmm_src, op);
                break;
            case memory::u8:
                uni_vpmovzxbd(vmm_src, op);
                break;
            default:
                assert(!"unknown dst_dt");
        }

        if (src_dt != data_type::f32) {
            uni_vcvtdq2ps(vmm_src, vmm_src);
        }
    }

    inline void load_scalar(Xmm xmm_src, const Xbyak::Address &op, memory::data_type src_dt) {
        switch (src_dt) {
            case memory::f32:
            case memory::s32:
                movss(xmm_src, op);
                break;
            case memory::s8:
                movsx(reg_tmp_32, op);
                movq(xmm_src, reg_tmp_64);
                break;
            case memory::u8:
                movzx(reg_tmp_32, op);
                movq(xmm_src, reg_tmp_64);
                break;
            default:
                assert(!"unknown dst_dt");
        }

        if (src_dt != data_type::f32) {
            uni_vcvtdq2ps(xmm_src, xmm_src);
        }
    }

    inline void store_vector(const Xbyak::Address &op, Vmm vmm_dst, memory::data_type dst_dt) {
        Xmm xmm_dst = Xmm(vmm_dst.getIdx());
        Ymm ymm_dst = Ymm(vmm_dst.getIdx());

        if (dst_dt != data_type::f32) {
            uni_vcvtps2dq(vmm_dst, vmm_dst);
        }

        switch (dst_dt) {
            case memory::f32:
            case memory::s32:
                uni_vmovups(op, vmm_dst);
                break;
            case memory::s8:
                if (isa == avx512_common) {
                    vmaxps(vmm_dst, vmm_zero, vmm_dst);
                    vpmovsdb(op, vmm_dst);
                } else {
                    uni_vpackssdw(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != cpu::sse42)
                        vpermq(ymm_dst, ymm_dst, 0x08);
                    uni_vpacksswb(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != cpu::sse42)
                        vmovq(op, xmm_dst);
                    else
                        movd(op, xmm_dst);
                }
                break;
            case memory::u8:
                if (isa == avx512_common) {
                    vpmovusdb(op, vmm_dst);
                } else {
                    uni_vpackusdw(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != cpu::sse42)
                        vpermq(ymm_dst, ymm_dst, 0x08);
                    uni_vpackuswb(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != cpu::sse42)
                        vmovq(op, xmm_dst);
                    else
                        movd(op, xmm_dst);
                }
                break;
            default:
                assert(!"unknown dst_dt");
        }
    }

    inline void store_scalar(const Xbyak::Address &op, Xmm xmm_dst, memory::data_type dst_dt) {
        if (dst_dt != data_type::f32) {
            uni_vcvtps2dq(xmm_dst, xmm_dst);
        }

        switch (dst_dt) {
            case memory::f32:
            case memory::s32:
                movss(op, xmm_dst);
                break;
            case memory::s8:
                uni_vpackssdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpacksswb(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
                break;
            case memory::u8:
                uni_vpackusdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpackuswb(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
                break;
            default:
                assert(!"unknown dst_dt");
        }
    }
};

MKLDNNEltwiseNode::MKLDNNEltwiseNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(layer, eng, cache), eltiwse_fq_kernel(nullptr) {
    op = EltwiseLayer::Sum;
}

bool MKLDNNEltwiseNode::isSum() {
    auto * eltwiseLayer = dynamic_cast<EltwiseLayer*>(getCnnLayer().get());
    if (eltwiseLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get eltwise layer " << getName();
    return eltwiseLayer->_operation == EltwiseLayer::Sum;
}

bool MKLDNNEltwiseNode::isUnitScales() {
    auto * eltwiseLayer = dynamic_cast<EltwiseLayer*>(getCnnLayer().get());
    if (eltwiseLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get eltwise layer " << getName();

    if (eltwiseLayer->coeff.empty())
        return true;

    for (auto scale : eltwiseLayer->coeff) {
        if (scale != 1.0f)
            return false;
    }

    return true;
}

bool MKLDNNEltwiseNode::isWithBroadcast() {
    bool withBroadcast = false;
    auto oDims = outDims[0].ToSizeVector();
    for (size_t i = 0; i < inDims.size(); i++) {
        auto iDims = inDims[i].ToSizeVector();
        for (size_t j = 1; j <= iDims.size(); j++) {
            if (oDims[oDims.size() - j] != iDims[iDims.size() - j]) {
                if (iDims[iDims.size() - j] == 1) {
                    withBroadcast = true;
                } else {
                    THROW_IE_EXCEPTION << "Incorrect dimensions for broadcasting for " << getName();
                }
            }
            if (iDims.size() < oDims.size())
                withBroadcast = true;
        }
        if (iDims.size() == 0 && oDims.size())
            withBroadcast = true;
    }

    return withBroadcast;
}

void MKLDNNEltwiseNode::getSupportedDescriptors() {
    auto * eltwiseLayer = dynamic_cast<EltwiseLayer*>(getCnnLayer().get());

    if (eltwiseLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert eltwise layer.";
    op = eltwiseLayer->_operation;

    if (getParentEdges().size() < 2)
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();
    if (op == EltwiseLayer::Squared_diff)
        if (getParentEdges().size() != 2)
            THROW_IE_EXCEPTION  << "Incorrect number of input edges for layer " << getName() << " for operation squared_diff.\n"
                << "Expected: 2\n" << "Actual: " << getParentEdges().size();

    auto outDims = getChildEdgeAt(0)->getDims();
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto inDims = getParentEdgeAt(i)->getDims();
        batch_dim = std::min(batch_dim, 5 - inDims.ndims());
    }

    broadcast = isWithBroadcast();
    if (broadcast) {
        auto outDims = getChildEdgeAt(0)->getDims();
        for (size_t i = 0; i < getParentEdges().size(); i++) {
            auto inDims = getParentEdgeAt(i)->getDims();
            if (inDims.ndims() > 5 || outDims.ndims() > 5)
                THROW_IE_EXCEPTION << "Eltwise node in broadcasting mode doesn't support more than 5 dims for blobs";
        }
    }

    bool with_coeffs = !eltwiseLayer->coeff.empty();
    if (op != EltwiseLayer::Sum && with_coeffs)
        THROW_IE_EXCEPTION << "Only sum operation supports operands coefficients";

    if (with_coeffs && eltwiseLayer->coeff.size() != getParentEdges().size())
        THROW_IE_EXCEPTION << "Number of provided coefficients is not equal to number of operands";

    if (with_coeffs && eltwiseLayer->precision != Precision::FP32)
        THROW_IE_EXCEPTION << "Sum with coefficients supports only FP32 precision";

    sum_scales.clear();
    for (int i = 0; i < getParentEdges().size(); i++)
        sum_scales.push_back(with_coeffs ? eltwiseLayer->coeff[i] : 1.0f);
}

void MKLDNNEltwiseNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    setPostOps(attr, true);

    auto initDesc = [&] (mkldnn::memory::data_type inputDT, mkldnn::memory::data_type outputDT, memory::format format) -> PrimitiveDescInfo {
        InferenceEngine::LayerConfig config;
        impl_desc_type impl_type = impl_desc_type::ref;
        config.dynBatchSupport = true;
        for (size_t i = 0; i < getParentEdges().size(); i++) {
            InferenceEngine::DataConfig dataConfig;
            dataConfig.inPlace = (!i && canBeInPlace()) ? 0 : -1;
            dataConfig.constant = false;

            if (!broadcast) {
                dataConfig.desc = MKLDNNMemoryDesc(getParentEdgeAt(i)->getDims(), inputDT, format);
                config.inConfs.push_back(dataConfig);
            } else {
                // Broadcasting support
                if (MKLDNNMemory::IsPlainFormat(format)) {
                    dataConfig.desc = MKLDNNMemoryDesc(getParentEdgeAt(i)->getDims(), inputDT,
                            MKLDNNMemory::GetPlainFormat(getParentEdgeAt(i)->getDims()));
                    config.inConfs.push_back(dataConfig);
                } else {
                    // Unsupported format for broadcast mode. Should be skipped.
                    // Will mark it as undef and outer code should filter it.
                    impl_type = impl_desc_type::undef;
                }
            }
        }

        InferenceEngine::DataConfig dataConfig;
            dataConfig.inPlace = -1;
            dataConfig.constant = false;
            dataConfig.desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDT, format);
            config.outConfs.push_back(dataConfig);
        return {config, impl_type, format};
    };

    if (fusedWith.empty()) {
        for (const auto& format : getAvailableFormatsForDims(getChildEdgeAt(0)->getDims())) {
            // Precision of implementation is defined by precision of output tensor
            auto prec = getCnnLayer()->outData[0]->getPrecision();
            mkldnn::memory::data_type inputDT = MKLDNNExtensionUtils::IEPrecisionToDataType(prec);
            mkldnn::memory::data_type outputDT = MKLDNNExtensionUtils::IEPrecisionToDataType(prec);

            // Eltwise compare operation can have the input type different from the output type
            auto node_op = this->op;
            bool is_eltwise_compare_node = ((node_op == EltwiseLayer::eOperation::Equal) ||
                                            (node_op == EltwiseLayer::eOperation::Not_equal) ||
                                            (node_op == EltwiseLayer::eOperation::Greater) ||
                                            (node_op == EltwiseLayer::eOperation::Greater_equal) ||
                                            (node_op == EltwiseLayer::eOperation::Less) ||
                                            (node_op == EltwiseLayer::eOperation::Less_equal));
            if (is_eltwise_compare_node) {
                auto in_prec = getCnnLayer()->insData[0].lock()->getPrecision();
                inputDT = MKLDNNExtensionUtils::IEPrecisionToDataType(in_prec);
            }

            if (inputDT == memory::bf16 || outputDT == memory::bf16) {
                inputDT = memory::f32;
                outputDT = memory::f32;
            }

            auto impl_desc = initDesc(inputDT, outputDT, format);

            if (impl_desc.getImplementationType() != impl_desc_type::undef) {
                supportedPrimitiveDescriptors.push_back(impl_desc);
            }
        }
    } else {
        auto ndims = getCnnLayer()->outData[0]->getDims().size();
        auto format = ndims == 2 ? memory::format::nc :
                      ndims == 4 ? memory::format::nhwc :
                      memory::format::ndhwc;

        InferenceEngine::LayerConfig config;
        impl_desc_type impl_type = impl_desc_type::ref;
        config.dynBatchSupport = true;
        for (size_t i = 0; i < getParentEdges().size(); i++) {
            InferenceEngine::DataConfig dataConfig;
            dataConfig.inPlace = -1;
            dataConfig.constant = false;
            auto inputDT = MKLDNNExtensionUtils::IEPrecisionToDataType(
                    getCnnLayer()->insData[i].lock()->getPrecision());
            dataConfig.desc = MKLDNNMemoryDesc(getParentEdgeAt(i)->getDims(), inputDT, format);
            config.inConfs.push_back(dataConfig);
        }

        auto outputDT = memory::f32;
        auto lastFusedLayer = fusedWith[fusedWith.size() - 1].get()->getCnnLayer();
        if (lastFusedLayer) {
            outputDT = MKLDNNExtensionUtils::IEPrecisionToDataType(lastFusedLayer->outData[0]->getPrecision());
        }

        InferenceEngine::DataConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;
        dataConfig.desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDT, format);
        config.outConfs.push_back(dataConfig);

        supportedPrimitiveDescriptors.push_back({config, impl_type, format});

        jep.src0_step = config.inConfs[0].desc.getDims()[1] == 1 ? 0 : 1;
        jep.src1_step = config.inConfs[1].desc.getDims()[1] == 1 ? 0 : 1;
        jep.dst_step = 1;
        jep.src0_dt = MKLDNNExtensionUtils::IEPrecisionToDataType(config.inConfs[0].desc.getPrecision());
        jep.src1_dt = MKLDNNExtensionUtils::IEPrecisionToDataType(config.inConfs[1].desc.getPrecision());
        jep.dst_dt = MKLDNNExtensionUtils::IEPrecisionToDataType(config.outConfs[0].desc.getPrecision());
        jep.src0_data_size = MKLDNNExtensionUtils::sizeOfDataType(jep.src0_dt);
        jep.src1_data_size = MKLDNNExtensionUtils::sizeOfDataType(jep.src1_dt);
        jep.dst_data_size = MKLDNNExtensionUtils::sizeOfDataType(jep.dst_dt);
        jep.eltwise_op = op;

        if (mayiuse(cpu::avx512_common)) {
            eltiwse_fq_kernel.reset(new jit_uni_eltwise_fq_generic<cpu::avx512_common>(jep, *attr.get()));
        } else if (mayiuse(cpu::avx2)) {
            eltiwse_fq_kernel.reset(new jit_uni_eltwise_fq_generic<cpu::avx2>(jep, *attr.get()));
        } else if (mayiuse(cpu::sse42)) {
            eltiwse_fq_kernel.reset(new jit_uni_eltwise_fq_generic<cpu::sse42>(jep, *attr.get()));
        }
    }
}

void MKLDNNEltwiseNode::createPrimitive() {
    if (prim)
        return;

    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";

    std::vector<memory::primitive_desc> srcs_pd;
    std::vector<primitive::at> srcs_p;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto& srcMemPtr = getParentEdgeAt(i)->getMemoryPtr();
        if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr()) {
            auto parent = getParentEdgeAt(i)->getParent();
            THROW_IE_EXCEPTION << "Source memory from " << parent->getName() << " didn't allocate.";
        }

        if (op == EltwiseLayer::Sum) {
            srcs_pd.push_back(srcMemPtr->GetPrimitiveDescriptor());
            srcs_p.emplace_back(srcMemPtr->GetPrimitive());
        }
    }
    if (op == EltwiseLayer::Sum && !broadcast && fusedWith.empty()) {
        try {
            auto primitive_desc = mkldnn::sum::primitive_desc(dstMemPtr->GetDescriptor(), sum_scales, srcs_pd);
            prim = std::shared_ptr<mkldnn::sum>(new mkldnn::sum(primitive_desc, srcs_p, dstMemPtr->GetPrimitive()));
        } catch (...) {
            std::cerr << "Handle this problem correctly!" << std::endl;
            prim = nullptr;
        }
    }
}

void MKLDNNEltwiseNode::initOptimalPrimitiveDescriptor() {
    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";
    auto config = selected_pd->getConfig();
    if (isInitConfig(config))
        return;

    MKLDNNNode::initOptimalPrimitiveDescriptor();

    auto* selectedPD = getSelectedPrimitiveDescriptor();
    if (!selectedPD) {
        return;
    }

    auto& selectedConfig = getSelectedPrimitiveDescriptor()->getConfig();
    for (size_t i = 1; i < selectedConfig.inConfs.size(); i++) {
        if (selectedConfig.inConfs[0].desc.getPrecision() != selectedConfig.inConfs[i].desc.getPrecision()) {
            selectedConfig.inConfs[i].desc.setPrecision(selectedConfig.inConfs[0].desc.getPrecision());
        }
    }
}

void MKLDNNEltwiseNode::setPostOps(mkldnn::primitive_attr &attr, bool initWeights) {
    mkldnn::post_ops ops;

    for (auto &node : fusedWith) {
        auto* activationNode = dynamic_cast<MKLDNNActivationNode *>(node.get());
        if (activationNode) {
            ops.append_eltwise(1.0, activationNode->getAlgorithm(), activationNode->getAlpha(), activationNode->getBeta());

            continue;
        }

        auto* quantizeNode = dynamic_cast<MKLDNNQuantizeNode *>(node.get());
        if (quantizeNode) {
            quantizeNode->appendPostOps(ops);
            continue;
        }

        THROW_IE_EXCEPTION << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType()) << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

void MKLDNNEltwiseNode::dims_calc(int *dims, const MKLDNNDims &edge_dims, bool channels_first = false) {
    for (int i = 0; i < 5; i++)
        dims[i] = 1;
    int ndims = edge_dims.ndims();
    if (ndims > 5) {
        THROW_IE_EXCEPTION << "ndims should be less then 5";
    }
    for (int i = 0; i < ndims; i++) {
        dims[4 - i] = edge_dims[ndims - 1 - i];
    }
    if (edge_dims.ndims() && !(broadcast && edge_dims[0] == getChildEdgeAt(0)->getDims()[0]))
        dims[batch_dim] = std::min(dims[batch_dim], batchToProcess());

    if (channels_first) {
        auto ch_idx = 5 - ndims + 1;
        auto ch = dims[ch_idx];
        for (int i = ch_idx; i < 4; i++) {
            dims[i] = dims[i + 1];
        }
        dims[4] = ch;
    }
}

void MKLDNNEltwiseNode::offset_out_calc(int *offset, int *dims) {
    int k = 1;
    for (int i = 4; i >= 0; i--) {
        offset[i] = k;
        k *= dims[i];
    }
}

void MKLDNNEltwiseNode::offset_in_calc(int *offset, int *dims_in, int *dims_out) {
    int k = 1;
    for (int i = 4; i >= 0; i--) {
        offset[i] = (dims_in[i] == dims_out[i]) ? k : 0;
        k *= dims_in[i];
    }
}

// Intel C++ Compiler 18.0 for Windows contains bug that doesn't allow to use templates to generate eltwise implementations
// and to avoid all copypaste below
template <typename T0, typename T1> void MKLDNNEltwiseNode::eltwise_add(
        const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = src0_ptr[i] + src1_ptr[i];
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = src0_ptr[i] + src1_ptr[i];
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = dst_ptr[i] + src_ptr[i];
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = dst_ptr[i] + src_ptr[i];
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = src0_ptr[index_in0] + src1_ptr[index_in1];
                    }
                }
            }
        }
    }
#else
        parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
            for (int i4 = 0; i4 < dims_out[4]; i4++) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = src0_ptr[index_in0] + src1_ptr[index_in1];
            }
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = dst_ptr[index_out] + src_ptr[index_in];
                        }
                    }
                }
            }
        }
#else
            parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
                for (int i4 = 0; i4 < dims_out[4]; i4++) {
                    size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                    size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                    dst_ptr[index_out] = dst_ptr[index_out] + src_ptr[index_in];
                }
            });
#endif
        }
    }
}

template <typename T0, typename T1> void MKLDNNEltwiseNode::eltwise_prod(
        const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = src0_ptr[i] * src1_ptr[i];
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = src0_ptr[i] * src1_ptr[i];
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = dst_ptr[i] * src_ptr[i];
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = dst_ptr[i] * src_ptr[i];
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = src0_ptr[index_in0] * src1_ptr[index_in1];
                    }
                }
            }
        }
    }
#else
        parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
            for (int i4 = 0; i4 < dims_out[4]; i4++) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = src0_ptr[index_in0] * src1_ptr[index_in1];
            }
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = dst_ptr[index_out] * src_ptr[index_in];
                        }
                    }
                }
            }
        }
#else
            parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = dst_ptr[index_out] * src_ptr[index_in];
            });
#endif
        }
    }
}

template <typename T0, typename T1> void MKLDNNEltwiseNode::eltwise_max(
        const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = std::max(src0_ptr[i], (T0)src1_ptr[i]);
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = std::max(src0_ptr[i], (T0)src1_ptr[i]);
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = std::max(dst_ptr[i], (T0)src_ptr[i]);
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = std::max(dst_ptr[i], (T0)src_ptr[i]);
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = std::max(src0_ptr[index_in0], (T0)src1_ptr[index_in1]);
                    }
                }
            }
        }
    }
#else
        parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
            for (int i4 = 0; i4 < dims_out[4]; i4++) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = std::max(src0_ptr[index_in0], (T0)src1_ptr[index_in1]);
            }
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = std::max(dst_ptr[index_out], (T0)src_ptr[index_in]);
                        }
                    }
                }
            }
        }
#else
            parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
                for (int i4 = 0; i4 < dims_out[4]; i4++) {
                    size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                    size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                    dst_ptr[index_out] = std::max(dst_ptr[index_out], (T0)src_ptr[index_in]);
                }
            });
#endif
        }
    }
}

template <typename T0, typename T1> void MKLDNNEltwiseNode::eltwise_sub(
        const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = src0_ptr[i] - src1_ptr[i];
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = src0_ptr[i] - src1_ptr[i];
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = dst_ptr[i] - src_ptr[i];
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = dst_ptr[i] - src_ptr[i];
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = src0_ptr[index_in0] - src1_ptr[index_in1];
                    }
                }
            }
        }
    }
#else
        parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
            for (int i4 = 0; i4 < dims_out[4]; i4++) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = src0_ptr[index_in0] - src1_ptr[index_in1];
            }
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = dst_ptr[index_out] - src_ptr[index_in];
                        }
                    }
                }
            }
        }
#else
            parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
                for (int i4 = 0; i4 < dims_out[4]; i4++) {
                    size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                    size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                    dst_ptr[index_out] = dst_ptr[index_out] - src_ptr[index_in];
                }
            });
#endif
        }
    }
}

template <typename T0, typename T1> void MKLDNNEltwiseNode::eltwise_min(
        const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = std::min(src0_ptr[i], (T0)src1_ptr[i]);
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = std::min(src0_ptr[i], (T0)src1_ptr[i]);
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = std::min(dst_ptr[i], (T0)src_ptr[i]);
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = std::min(dst_ptr[i], (T0)src_ptr[i]);
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = std::min(src0_ptr[index_in0], (T0)src1_ptr[index_in1]);
                    }
                }
            }
        }
    }
#else
        parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
            for (int i4 = 0; i4 < dims_out[4]; i4++) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = std::min(src0_ptr[index_in0], (T0)src1_ptr[index_in1]);
            }
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = std::min(dst_ptr[index_out], (T0)src_ptr[index_in]);
                        }
                    }
                }
            }
        }
#else
            parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
                for (int i4 = 0; i4 < dims_out[4]; i4++) {
                    size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                    size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                    dst_ptr[index_out] = std::min(dst_ptr[index_out], (T0)src_ptr[index_in]);
                }
            });
#endif
        }
    }
}

template <typename T0, typename T1> void MKLDNNEltwiseNode::eltwise_div(
        const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = src0_ptr[i] / src1_ptr[i];
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = src0_ptr[i] / src1_ptr[i];
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = dst_ptr[i] / src_ptr[i];
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = dst_ptr[i] / src_ptr[i];
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = src0_ptr[index_in0] / src1_ptr[index_in1];
                    }
                }
            }
        }
    }
#else
        parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
            for (int i4 = 0; i4 < dims_out[4]; i4++) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = src0_ptr[index_in0] / src1_ptr[index_in1];
            }
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = dst_ptr[index_out] / src_ptr[index_in];
                        }
                    }
                }
            }
        }
#else
            parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
                for (int i4 = 0; i4 < dims_out[4]; i4++) {
                    size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                    size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                    dst_ptr[index_out] = dst_ptr[index_out] / src_ptr[index_in];
                }
            });
#endif
        }
    }
}

template <typename T0, typename T1> void MKLDNNEltwiseNode::eltwise_squared_diff(
        const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = (src0_ptr[i] - src1_ptr[i]) * (src0_ptr[i] - src1_ptr[i]);
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = (src0_ptr[i] - src1_ptr[i]) * (src0_ptr[i] - src1_ptr[i]);
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = (dst_ptr[i] - src_ptr[i]) * (dst_ptr[i] - src_ptr[i]);
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = (dst_ptr[i] - src_ptr[i]) * (dst_ptr[i] - src_ptr[i]);
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = (src0_ptr[index_in0] - src1_ptr[index_in1]) * (src0_ptr[index_in0] - src1_ptr[index_in1]);
                    }
                }
            }
        }
    }
#else
        parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
            for (int i4 = 0; i4 < dims_out[4]; i4++) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = (src0_ptr[index_in0] - src1_ptr[index_in1]) * (src0_ptr[index_in0] - src1_ptr[index_in1]);
            }
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = (dst_ptr[index_out] - src_ptr[index_in]) * (dst_ptr[index_out] - src_ptr[index_in]);
                        }
                    }
                }
            }
        }
#else
            parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
                for (int i4 = 0; i4 < dims_out[4]; i4++) {
                    size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                    size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                    dst_ptr[index_out] = (dst_ptr[index_out] - src_ptr[index_in]) * (dst_ptr[index_out] - src_ptr[index_in]);
                }
            });
#endif
        }
    }
}

template <typename T0, typename T1> void MKLDNNEltwiseNode::eltwise_floor_mod(
        const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = src0_ptr[i] - src0_ptr[i] / src1_ptr[i] * src1_ptr[i];
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = src0_ptr[i] - src0_ptr[i] / src1_ptr[i] * src1_ptr[i];
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = dst_ptr[i] - dst_ptr[i] / src_ptr[i] * src_ptr[i];
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = dst_ptr[i] - dst_ptr[i] / src_ptr[i] * src_ptr[i];
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = src0_ptr[index_in0] - src0_ptr[index_in0] / src1_ptr[index_in1] * src1_ptr[index_in1];
                    }
                }
            }
        }
    }
#else
        parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
            for (int i4 = 0; i4 < dims_out[4]; i4++) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = src0_ptr[index_in0] - src0_ptr[index_in0] / src1_ptr[index_in1] * src1_ptr[index_in1];
            }
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = dst_ptr[index_out] - dst_ptr[index_out] / src_ptr[index_in] * src_ptr[index_in];
                        }
                    }
                }
            }
        }
#else
            parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
                for (int i4 = 0; i4 < dims_out[4]; i4++) {
                    size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                    size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                    dst_ptr[index_out] = dst_ptr[index_out] - dst_ptr[index_out] / src_ptr[index_in] * src_ptr[index_in];
                }
            });
#endif
        }
    }
}

template <typename T0, typename T1> void MKLDNNEltwiseNode::eltwise_pow(
        const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = std::pow(src0_ptr[i], src1_ptr[i]);
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = std::pow(src0_ptr[i], src1_ptr[i]);
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = std::pow(dst_ptr[i], src_ptr[i]);
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = std::pow(dst_ptr[i], src_ptr[i]);
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = std::pow(src0_ptr[index_in0], src1_ptr[index_in1]);
                    }
                }
            }
        }
    }
#else
        parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
            for (int i4 = 0; i4 < dims_out[4]; i4++) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = std::pow(src0_ptr[index_in0], src1_ptr[index_in1]);
            }
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = std::pow(dst_ptr[index_out], src_ptr[index_in]);
                        }
                    }
                }
            }
        }
#else
            parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
                for (int i4 = 0; i4 < dims_out[4]; i4++) {
                    size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                    size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                    dst_ptr[index_out] = std::pow(dst_ptr[index_out], src_ptr[index_in]);
                }
            });
#endif
        }
    }
}

template <typename T0, typename T1, typename T2> void MKLDNNEltwiseNode::eltwise_equal(
        const T0 *src0_ptr, const T1 *src1_ptr, T2 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = src0_ptr[i] == src1_ptr[i];
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = src0_ptr[i] == src1_ptr[i];
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = dst_ptr[i] == src_ptr[i];
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = dst_ptr[i] == src_ptr[i];
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = src0_ptr[index_in0] == src1_ptr[index_in1];
                    }
                }
            }
        }
    }
#else
        parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
            for (int i4 = 0; i4 < dims_out[4]; i4++) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = src0_ptr[index_in0] == src1_ptr[index_in1];
            }
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = dst_ptr[index_out] == src_ptr[index_in];
                        }
                    }
                }
            }
        }
#else
            parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
                for (int i4 = 0; i4 < dims_out[4]; i4++) {
                    size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                    size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                    dst_ptr[index_out] = dst_ptr[index_out] == src_ptr[index_in];
                }
            });
#endif
        }
    }
}

template <typename T0, typename T1, typename T2> void MKLDNNEltwiseNode::eltwise_not_equal(
        const T0 *src0_ptr, const T1 *src1_ptr, T2 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = src0_ptr[i] != src1_ptr[i];
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = src0_ptr[i] != src1_ptr[i];
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = dst_ptr[i] != src_ptr[i];
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = dst_ptr[i] != src_ptr[i];
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = src0_ptr[index_in0] != src1_ptr[index_in1];
                    }
                }
            }
        }
    }
#else
        parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
            for (int i4 = 0; i4 < dims_out[4]; i4++) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = src0_ptr[index_in0] != src1_ptr[index_in1];
            }
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = dst_ptr[index_out] != src_ptr[index_in];
                        }
                    }
                }
            }
        }
#else
            parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
                for (int i4 = 0; i4 < dims_out[4]; i4++) {
                    size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                    size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                    dst_ptr[index_out] = dst_ptr[index_out] != src_ptr[index_in];
                }
            });
#endif
        }
    }
}

template <typename T0, typename T1, typename T2> void MKLDNNEltwiseNode::eltwise_less(
        const T0 *src0_ptr, const T1 *src1_ptr, T2 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = src0_ptr[i] < src1_ptr[i];
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = src0_ptr[i] < src1_ptr[i];
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = dst_ptr[i] < src_ptr[i];
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = dst_ptr[i] < src_ptr[i];
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = src0_ptr[index_in0] < src1_ptr[index_in1];
                    }
                }
            }
        }
    }
#else
        parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
            for (int i4 = 0; i4 < dims_out[4]; i4++) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = src0_ptr[index_in0] < src1_ptr[index_in1];
            }
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = dst_ptr[index_out] < src_ptr[index_in];
                        }
                    }
                }
            }
        }
#else
            parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
                for (int i4 = 0; i4 < dims_out[4]; i4++) {
                    size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                    size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                    dst_ptr[index_out] = dst_ptr[index_out] < src_ptr[index_in];
                }
            });
#endif
        }
    }
}

template <typename T0, typename T1, typename T2> void MKLDNNEltwiseNode::eltwise_less_equal(
        const T0 *src0_ptr, const T1 *src1_ptr, T2 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = src0_ptr[i] <= src1_ptr[i];
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = src0_ptr[i] <= src1_ptr[i];
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = dst_ptr[i] <= src_ptr[i];
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = dst_ptr[i] <= src_ptr[i];
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = src0_ptr[index_in0] <= src1_ptr[index_in1];
                    }
                }
            }
        }
    }
#else
        parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
            for (int i4 = 0; i4 < dims_out[4]; i4++) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = src0_ptr[index_in0] <= src1_ptr[index_in1];
            }
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = dst_ptr[index_out] <= src_ptr[index_in];
                        }
                    }
                }
            }
        }
#else
            parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
                for (int i4 = 0; i4 < dims_out[4]; i4++) {
                    size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                    size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                    dst_ptr[index_out] = dst_ptr[index_out] <= src_ptr[index_in];
                }
            });
#endif
        }
    }
}

template <typename T0, typename T1, typename T2> void MKLDNNEltwiseNode::eltwise_greater(
        const T0 *src0_ptr, const T1 *src1_ptr, T2 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = src0_ptr[i] > src1_ptr[i];
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = src0_ptr[i] > src1_ptr[i];
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = dst_ptr[i] > src_ptr[i];
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = dst_ptr[i] > src_ptr[i];
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = src0_ptr[index_in0] > src1_ptr[index_in1];
                    }
                }
            }
        }
    }
#else
        parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
            for (int i4 = 0; i4 < dims_out[4]; i4++) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = src0_ptr[index_in0] > src1_ptr[index_in1];
            }
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = dst_ptr[index_out] > src_ptr[index_in];
                        }
                    }
                }
            }
        }
#else
            parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
                for (int i4 = 0; i4 < dims_out[4]; i4++) {
                    size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                    size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                    dst_ptr[index_out] = dst_ptr[index_out] > src_ptr[index_in];
                }
            });
#endif
        }
    }
}

template <typename T0, typename T1, typename T2> void MKLDNNEltwiseNode::eltwise_greater_equal(
        const T0 *src0_ptr, const T1 *src1_ptr, T2 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = src0_ptr[i] >= src1_ptr[i];
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = src0_ptr[i] >= src1_ptr[i];
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = dst_ptr[i] >= src_ptr[i];
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = dst_ptr[i] >= src_ptr[i];
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = src0_ptr[index_in0] >= src1_ptr[index_in1];
                    }
                }
            }
        }
    }
#else
        parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
            for (int i4 = 0; i4 < dims_out[4]; i4++) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = src0_ptr[index_in0] >= src1_ptr[index_in1];
            }
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = dst_ptr[index_out] >= src_ptr[index_in];
                        }
                    }
                }
            }
        }
#else
            parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
                for (int i4 = 0; i4 < dims_out[4]; i4++) {
                    size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                    size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                    dst_ptr[index_out] = dst_ptr[index_out] >= src_ptr[index_in];
                }
            });
#endif
        }
    }
}

template <typename T0, typename T1> void MKLDNNEltwiseNode::eltwise_logical_and(
        const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = src0_ptr[i] && src1_ptr[i];
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = src0_ptr[i] && src1_ptr[i];
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = dst_ptr[i] && src_ptr[i];
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = dst_ptr[i] && src_ptr[i];
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = src0_ptr[index_in0] && src1_ptr[index_in1];
                    }
                }
            }
        }
    }
#else
        parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
            for (int i4 = 0; i4 < dims_out[4]; i4++) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = src0_ptr[index_in0] && src1_ptr[index_in1];
            }
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = dst_ptr[index_out] && src_ptr[index_in];
                        }
                    }
                }
            }
        }
#else
            parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
                for (int i4 = 0; i4 < dims_out[4]; i4++) {
                    size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                    size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                    dst_ptr[index_out] = dst_ptr[index_out] && src_ptr[index_in];
                }
            });
#endif
        }
    }
}

template <typename T0, typename T1> void MKLDNNEltwiseNode::eltwise_logical_or(
        const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = src0_ptr[i] || src1_ptr[i];
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = src0_ptr[i] || src1_ptr[i];
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = dst_ptr[i] || src_ptr[i];
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = dst_ptr[i] || src_ptr[i];
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = src0_ptr[index_in0] || src1_ptr[index_in1];
                    }
                }
            }
        }
    }
#else
        parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
            for (int i4 = 0; i4 < dims_out[4]; i4++) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = src0_ptr[index_in0] || src1_ptr[index_in1];
            }
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = dst_ptr[index_out] || src_ptr[index_in];
                        }
                    }
                }
            }
        }
#else
            parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
                for (int i4 = 0; i4 < dims_out[4]; i4++) {
                    size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                    size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                    dst_ptr[index_out] = dst_ptr[index_out] || src_ptr[index_in];
                }
            });
#endif
        }
    }
}

template <typename T0, typename T1> void MKLDNNEltwiseNode::eltwise_logical_xor(
        const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, const size_t dst_data_size) {
    if (!broadcast) {
#ifdef _WIN32
        for (size_t i = 0; i < dst_data_size; i++) {
            dst_ptr[i] = (src0_ptr[i] || src1_ptr[i]) - (src0_ptr[i] && src1_ptr[i]);
    }
#else
        parallel_for(dst_data_size, [&](size_t i) {
            dst_ptr[i] = (src0_ptr[i] || src1_ptr[i]) - (src0_ptr[i] && src1_ptr[i]);
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                                getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (size_t i = 0; i < dst_data_size; i++) {
                dst_ptr[i] = (dst_ptr[i] || src_ptr[i]) - (dst_ptr[i] && src_ptr[i]);
            }
#else
            parallel_for(dst_data_size, [&](size_t i) {
                dst_ptr[i] = (dst_ptr[i] || src_ptr[i]) - (dst_ptr[i] && src_ptr[i]);
            });
#endif
        }
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims);
        dims_calc(dims_in0, parent0_edge_dims);
        dims_calc(dims_in1, parent1_edge_dims);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
        for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
        for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
            for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                    for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                        size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                        size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                        size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                        dst_ptr[index_out] = (src0_ptr[index_in0] || src1_ptr[index_in1]) - (src0_ptr[index_in0] && src1_ptr[index_in1]);
                    }
                }
            }
        }
    }
#else
        parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
            for (int i4 = 0; i4 < dims_out[4]; i4++) {
                size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3] + i4 * offset_in0[4];
                size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                dst_ptr[index_out] = (src0_ptr[index_in0] || src1_ptr[index_in1]) - (src0_ptr[index_in0] && src1_ptr[index_in1]);
            }
        });
#endif
        for (size_t n = 2; n < getParentEdges().size(); n++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(n)->getMemory().GetData()) +
                                getParentEdgeAt(n)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

            auto& parent_edge_dims = getParentEdgeAt(n)->getDims();
            dims_calc(dims_in1, parent_edge_dims);
            offset_in_calc(offset_in1, dims_in1, dims_out);

#ifdef _WIN32
            for (size_t i0 = 0; i0 < dims_out[0]; i0++) {
            for (size_t i1 = 0; i1 < dims_out[1]; i1++) {
                for (size_t i2 = 0; i2 < dims_out[2]; i2++) {
                    for (size_t i3 = 0; i3 < dims_out[3]; i3++) {
                        for (size_t i4 = 0; i4 < dims_out[4]; i4++) {
                            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                            size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                            dst_ptr[index_out] = (dst_ptr[index_out] || src_ptr[index_in]) - (dst_ptr[index_out] && src_ptr[index_in]);
                        }
                    }
                }
            }
        }
#else
            parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
                for (int i4 = 0; i4 < dims_out[4]; i4++) {
                    size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3] + i4 * offset_out[4];
                    size_t index_in = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3] + i4 * offset_in1[4];
                    dst_ptr[index_out] = (dst_ptr[index_out] || src_ptr[index_in]) - (dst_ptr[index_out] && src_ptr[index_in]);
                }
            });
#endif
        }
    }
}

template <typename T0, typename T1, typename T2> void MKLDNNEltwiseNode::ref_eltwise2(int in0, int in1) {
    IE_ASSERT(getParentEdges().size() > 1);

    auto& srcMemory0 = getParentEdgeAt(in0)->getMemory();
    auto& srcMemory1 = getParentEdgeAt(in1)->getMemory();
    const T0 *src0_ptr = reinterpret_cast<const T0*>(srcMemory0.GetData()) +
        srcMemory0.GetDescriptor().data.layout_desc.blocking.offset_padding;
    const T1 *src1_ptr = reinterpret_cast<const T1*>(srcMemory1.GetData()) +
        srcMemory1.GetDescriptor().data.layout_desc.blocking.offset_padding;
    T2 *dst_ptr = reinterpret_cast<T2*>(getChildEdgeAt(0)->getMemory().GetData()) +
        getChildEdgeAt(0)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

    const size_t dst_data_size = srcMemory0.GetSize() / sizeof(T0) / srcMemory0.GetDims()[0] * batchToProcess();

    switch (op) {
        case EltwiseLayer::eOperation::Equal: eltwise_equal(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Not_equal: eltwise_not_equal(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Less: eltwise_less(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Less_equal: eltwise_less_equal(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Greater: eltwise_greater(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Greater_equal: eltwise_greater_equal(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        default: THROW_IE_EXCEPTION << "Unsupported operation type for Eltwise node";
    }
}

template <typename T0, typename T1> void MKLDNNEltwiseNode::ref_eltwise(int in0, int in1) {
    IE_ASSERT(getParentEdges().size() > 1);

    auto& srcMemory0 = getParentEdgeAt(in0)->getMemory();
    auto& srcMemory1 = getParentEdgeAt(in1)->getMemory();
    const T0 *src0_ptr = reinterpret_cast<const T0*>(srcMemory0.GetData()) +
            srcMemory0.GetDescriptor().data.layout_desc.blocking.offset_padding;
    const T1 *src1_ptr = reinterpret_cast<const T1*>(srcMemory1.GetData()) +
            srcMemory1.GetDescriptor().data.layout_desc.blocking.offset_padding;
    T0 *dst_ptr = reinterpret_cast<T0*>(getChildEdgeAt(0)->getMemory().GetData()) +
            getChildEdgeAt(0)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

    const size_t dst_data_size = srcMemory0.GetSize() / sizeof(T0) / srcMemory0.GetDims()[0] * batchToProcess();

    switch (op) {
        case EltwiseLayer::eOperation::Sum: eltwise_add(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Prod: eltwise_prod(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Max: eltwise_max(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Sub: eltwise_sub(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Min: eltwise_min(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Div: eltwise_div(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Squared_diff: eltwise_squared_diff(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Floor_mod: eltwise_floor_mod(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Pow: eltwise_pow(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Equal: eltwise_equal(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Not_equal: eltwise_not_equal(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Less: eltwise_less(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Less_equal: eltwise_less_equal(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Greater: eltwise_greater(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Greater_equal: eltwise_greater_equal(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Logical_AND: eltwise_logical_and(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Logical_OR: eltwise_logical_or(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        case EltwiseLayer::eOperation::Logical_XOR: eltwise_logical_xor(src0_ptr, src1_ptr, dst_ptr, dst_data_size); break;
        default: THROW_IE_EXCEPTION << "Unsupported operation type for Eltwise node";
    }
}

void MKLDNNEltwiseNode::jit_eltwise_fq() {
    auto& srcMemory0 = getParentEdgeAt(0)->getMemory();
    auto& srcMemory1 = getParentEdgeAt(1)->getMemory();
    auto& dstMemory = getChildEdgeAt(0)->getMemory();

    const uint8_t *src0_ptr = reinterpret_cast<const uint8_t*>(srcMemory0.GetData()) +
        srcMemory0.GetDescriptor().data.layout_desc.blocking.offset_padding *
        MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(srcMemory0.GetDescriptor().data.data_type));
    const uint8_t *src1_ptr = reinterpret_cast<const uint8_t*>(srcMemory1.GetData()) +
        srcMemory1.GetDescriptor().data.layout_desc.blocking.offset_padding *
        MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(srcMemory1.GetDescriptor().data.data_type));
    uint8_t *dst_ptr = reinterpret_cast<uint8_t*>(dstMemory.GetData()) +
        dstMemory.GetDescriptor().data.layout_desc.blocking.offset_padding *
        MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(dstMemory.GetDescriptor().data.data_type));

    if (!broadcast) {
        auto& dims = getParentEdgeAt(0)->getDims();

        int N = batchToProcess();
        int C = dims[1];
        int D = dims.ndims() > 4 ? dims[2] : 1;
        int H = dims.ndims() > 2 ? dims[dims.ndims() - 2] : 1;
        int W = dims.ndims() > 3 ? dims[dims.ndims() - 1] : 1;

        parallel_for4d(N, D, H, W, [&](int n, int d, int h, int w) {
            size_t off = n * D * H * W * C + d * H * W * C + h * W * C + w * C;

            auto arg = jit_eltwise_fq_call_args();
            arg.src0 = src0_ptr + off * jep.src0_data_size;
            arg.src1 = src1_ptr + off * jep.src1_data_size;
            arg.dst = dst_ptr + off * jep.dst_data_size;
            arg.work_amount = static_cast<size_t>(C);

            (*eltiwse_fq_kernel)(&arg);
        });
    } else {
        int dims_out[5], dims_in0[5], dims_in1[5];
        int offset_out[5], offset_in0[5], offset_in1[5];
        auto& child_edge_dims = getChildEdgeAt(0)->getDims();
        auto& parent0_edge_dims = getParentEdgeAt(0)->getDims();
        auto& parent1_edge_dims = getParentEdgeAt(1)->getDims();
        dims_calc(dims_out, child_edge_dims, true);
        dims_calc(dims_in0, parent0_edge_dims, true);
        dims_calc(dims_in1, parent1_edge_dims, true);
        offset_out_calc(offset_out, dims_out);
        offset_in_calc(offset_in0, dims_in0, dims_out);
        offset_in_calc(offset_in1, dims_in1, dims_out);

        parallel_for4d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], [&](size_t i0, size_t i1, size_t i2, size_t i3) {
            size_t index_out = i0 * offset_out[0] + i1 * offset_out[1] + i2 * offset_out[2] + i3 * offset_out[3];
            size_t index_in0 = i0 * offset_in0[0] + i1 * offset_in0[1] + i2 * offset_in0[2] + i3 * offset_in0[3];
            size_t index_in1 = i0 * offset_in1[0] + i1 * offset_in1[1] + i2 * offset_in1[2] + i3 * offset_in1[3];

            auto arg = jit_eltwise_fq_call_args();
            arg.src0 = src0_ptr + index_in0 * jep.src0_data_size;
            arg.src1 = src1_ptr + index_in1 * jep.src1_data_size;
            arg.dst = dst_ptr + index_out * jep.dst_data_size;
            arg.work_amount = static_cast<size_t>(dims_out[4]);

            (*eltiwse_fq_kernel)(&arg);
        });
    }
}

void MKLDNNEltwiseNode::execute(mkldnn::stream strm) {
    if (prim) {
        MKLDNNNode::execute(strm);
    } else {
        if (op == EltwiseLayer::Floor_mod) {
            for (size_t i = 0; i < getParentEdges().size(); i++)
                if (getParentEdgeAt(i)->getDesc().getPrecision() != Precision::I32)
                    THROW_IE_EXCEPTION << "Floor_mod supports only I32 precision of inputs";
            if (getChildEdgeAt(0)->getDesc().getPrecision() != Precision::I32)
                THROW_IE_EXCEPTION << "Floor_mod supports only I32 precision of output";
        }

        if (getParentEdges().size() > 2) {
            Precision pi = getParentEdgeAt(0)->getDesc().getPrecision();
            Precision po = getChildEdgeAt(0)->getDesc().getPrecision();
            for (int i = 1; i < getParentEdges().size(); i++) {
                if (getParentEdgeAt(i)->getDesc().getPrecision() != pi)
                    THROW_IE_EXCEPTION << "If Eltwise node has more than 2 inputs, all inputs must have same precision";
            }
            if (pi != po) {
                THROW_IE_EXCEPTION << "If Eltwise node has more than 2 inputs, all inputs and output must have same precision";
            }
            if (pi == Precision::FP32)
                ref_eltwise<float, float>(0, 1);
            else if (pi == Precision::I32)
                ref_eltwise<int32_t, int32_t>(0, 1);
            else if (pi == Precision::I8)
                ref_eltwise<int8_t, int8_t>(0, 1);
            else if (pi == Precision::U8)
                ref_eltwise<uint8_t, uint8_t>(0, 1);
            else
                THROW_IE_EXCEPTION << "If Eltwise node has more than 2 inputs, only FP32, I32, I8, U8 are supported";
            return;
        }

        Precision pi0 = getParentEdgeAt(0)->getDesc().getPrecision();
        Precision pi1 = getParentEdgeAt(1)->getDesc().getPrecision();
        Precision po = getChildEdgeAt(0)->getDesc().getPrecision();

        IE_ASSERT(getParentEdges().size() > 1);

        if (!fusedWith.empty()) {
            jit_eltwise_fq();
        } else {
            // Input and output types for eltwise compare operations can be different
            bool is_eltwise_compare_node = (op == EltwiseLayer::Equal || op == EltwiseLayer::Not_equal ||
                                            op == EltwiseLayer::Greater || op == EltwiseLayer::Greater_equal ||
                                            op == EltwiseLayer::Less || op == EltwiseLayer::Less_equal);

            if (po == Precision::FP32 && pi0 == po && pi1 == po) {
                ref_eltwise<float, float>(0, 1);
            } else if (po == Precision::FP32 && pi0 == po && pi1 == Precision::I8) {
                ref_eltwise<float, int8_t>(0, 1);
            } else if (po == Precision::FP32 && pi1 == po && pi0 == Precision::I8) {
                ref_eltwise<float, int8_t>(1, 0);
            } else if (po == Precision::FP32 && pi0 == po && pi1 == Precision::U8) {
                ref_eltwise<float, uint8_t>(0, 1);
            } else if (po == Precision::FP32 && pi1 == po && pi0 == Precision::U8) {
                ref_eltwise<float, uint8_t>(1, 0);
            } else if (po == Precision::I8 && pi0 == po && pi1 == po) {
                ref_eltwise<int8_t, int8_t>(0, 1);
            } else if (po == Precision::I8 && pi0 == po && pi1 == Precision::U8) {
                ref_eltwise<int8_t, uint8_t>(0, 1);
            } else if (po == Precision::I8 && pi1 == po && pi0 == Precision::U8) {
                ref_eltwise<int8_t, uint8_t>(1, 0);
            } else if (po == Precision::I32 && pi0 == po && pi1 == po) {
                ref_eltwise<int32_t, int32_t>(0, 1);
            } else if (po == Precision::U8 && pi0 == Precision::I32 && pi0 == pi1 && is_eltwise_compare_node) {
                ref_eltwise2<int32_t, int32_t, uint8_t>(0, 1);
            } else if (po == Precision::U8 && pi0 == Precision::FP32 && pi0 == pi1 && is_eltwise_compare_node) {
                ref_eltwise2<float, float, uint8_t>(0, 1);
            } else {
                THROW_IE_EXCEPTION << "Eltwise node with unsupported combination of input and output types";
            }
        }
    }
}

bool MKLDNNEltwiseNode::created() const {
    return getType() == Eltwise;
}

bool MKLDNNEltwiseNode::canBeInPlace() const {
    size_t inPlaceWithParent = getParentEdges().size();
    for (size_t i = 0; i < inPlaceWithParent; i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (!parentEdge->getParent()->isConstant() &&
                parentEdge->getParent()->getChildEdges().size() == 1) {
            inPlaceWithParent = i;
            break;
        }
    }
    // This is WA for MKLDNN implementation
    if (inPlaceWithParent != 0)
        return false;
    MKLDNNDims dims = getParentEdgeAt(0)->getDims();
    for (size_t cIdx = 0; cIdx < getChildEdges().size(); cIdx++) {
        if (getChildEdgeAt(cIdx)->getDims() != dims) {
            return false;
        }
    }

    // Broadcast mode is complex for inplace usage
    // So will disable it
    if (broadcast) return false;

    return true;
}

#if GraphGen(Gen_Eltwise) || \
    GraphGen(Gen_Add)
REG_MKLDNN_PRIM_FOR(MKLDNNEltwiseNode, Eltwise);
#endif
