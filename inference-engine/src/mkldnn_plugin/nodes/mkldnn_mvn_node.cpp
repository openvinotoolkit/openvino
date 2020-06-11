// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_mvn_node.h"
#include "desc_iterator.hpp"
#include "mkldnn_quantize_node.h"
#include "mkldnn_depthwise_node.h"
#include "mkldnn_activation_node.h"
#include <ie_layers.h>
#include <mkldnn.hpp>
#include <string>
#include <vector>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <ie_layers_internal.hpp>
#include "ie_parallel.hpp"
#include <algorithm>

#include "jit_generator.hpp"
#include "jit_uni_eltwise.hpp"
#include "jit_uni_depthwise.hpp"
#include "jit_uni_quantization.hpp"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_mvn_call_args, field)

// normalize_variance = false : src->mean
// normalize_variance = true : src+mean->variance:sqr(x-mean)
template <cpu_isa_t isa>
struct jit_uni_mvn_mean_variance_kernel_f32 : public jit_uni_mvn_mean_variance_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_mvn_mean_kernel_f32)

    explicit jit_uni_mvn_mean_variance_kernel_f32(jit_mvn_config_params jcp) : jit_uni_mvn_mean_variance_kernel(jcp), jit_generator() {
        this->preamble();
        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        if (jcp_.normalize_variance) {
            mov(reg_mean, ptr[reg_params + GET_OFF(mean)]);
            mov(reg_variance, ptr[reg_params + GET_OFF(variance)]);
        } else {
            mov(reg_sum, ptr[reg_params + GET_OFF(sum)]);
        }
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);
        mov(reg_stride, ptr[reg_params + GET_OFF(src_stride)]);

        int repeats = (!jcp_.planar_layout && !jcp_.across_channels && isa == cpu::sse42) ? 2 : 1;  // block size is also 8 on cpu::sse42
        for (int i = 0; i < repeats; i++) {
            int offset_sse42 = i * 4;
            if (i > 0) {
                mov(reg_src, ptr[reg_params + GET_OFF(src)]);
                mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);

                add(reg_src, offset_sse42 * jcp_.src_data_size);
                if (jcp_.normalize_variance) {
                    add(reg_mean, offset_sse42 * sizeof(float));
                    add(reg_variance, offset_sse42 * sizeof(float));
                } else {
                    add(reg_sum, offset_sse42 * sizeof(float));
                }
            }

            Xbyak::Label loop_label;
            Xbyak::Label loop_end_label;

            if (jcp_.normalize_variance) {
                uni_vpxor(vmm_variance, vmm_variance, vmm_variance);

                if (jcp_.planar_layout || jcp_.across_channels) {
                    uni_vbroadcastss(vmm_mean, ptr[reg_mean]);
                } else {
                    uni_vmovups(vmm_mean, ptr[reg_mean]);
                }
            } else {
                uni_vpxor(vmm_sum, vmm_sum, vmm_sum);
            }

            L(loop_label);
            {
                cmp(reg_work_amount, 0);
                jle(loop_end_label, T_NEAR);

                load_vector(vmm_val, ptr[reg_src], jcp_.src_dt);

                if (jcp_.normalize_variance) {
                    if (jcp_.src_dt != memory::f32)
                        uni_vcvtdq2ps(vmm_val, vmm_val);

                    uni_vsubps(vmm_val, vmm_val, vmm_mean);
                    uni_vfmadd231ps(vmm_variance, vmm_val, vmm_val);
                } else {
                    if (jcp_.src_dt != memory::f32)
                        uni_vpaddd(vmm_sum, vmm_sum, vmm_val);
                    else
                        uni_vaddps(vmm_sum, vmm_sum, vmm_val);
                }

                add(reg_src, reg_stride);
                sub(reg_work_amount, 1);

                jmp(loop_label, T_NEAR);
            }
            L(loop_end_label);

            if (jcp_.planar_layout) {
                Vmm vmm_dst = jcp_.normalize_variance ? vmm_variance : vmm_sum;
                // hsum+store
                if (isa == cpu::sse42) {
                    hsum_store(vmm_dst);
                } else if (isa == cpu::avx2) {
                    Xbyak::Ymm ymm_sum = Xbyak::Ymm(vmm_dst.getIdx());
                    vextractf128(xmm_aux1, ymm_sum, 0);
                    vextractf128(xmm_aux2, ymm_sum, 1);
                    addps(xmm_aux1, xmm_aux2);
                    hsum_store(xmm_aux1);
                } else {
                    Xbyak::Zmm zmm_sum = Xbyak::Zmm(vmm_dst.getIdx());
                    vextractf32x4(xmm_aux1, zmm_sum, 0);
                    vextractf32x4(xmm_aux2, zmm_sum, 1);
                    addps(xmm_aux1, xmm_aux2);
                    vextractf32x4(xmm_aux2, zmm_sum, 2);
                    vextractf32x4(xmm_aux3, zmm_sum, 3);
                    addps(xmm_aux2, xmm_aux3);
                    addps(xmm_aux1, xmm_aux2);
                    hsum_store(xmm_aux1);
                }
            } else {
                if (jcp_.normalize_variance) {
                    if (!jcp_.planar_layout && !jcp_.across_channels) {
                        uni_vmovups(vmm_val, ptr[reg_variance]);
                        uni_vaddps(vmm_variance, vmm_variance, vmm_val);
                    }

                    uni_vmovups(ptr[reg_variance], vmm_variance);
                } else {
                    if (jcp_.src_dt != memory::f32)
                        uni_vcvtdq2ps(vmm_sum, vmm_sum);

                    if (!jcp_.planar_layout && !jcp_.across_channels) {
                        uni_vmovups(vmm_val, ptr[reg_sum]);
                        uni_vaddps(vmm_sum, vmm_sum, vmm_val);
                    }

                    uni_vmovups(ptr[reg_sum], vmm_sum);
                }
            }
        }

        this->postamble();
        ker_ = (decltype(ker_)) this->getCode();
    }

private:
    using Vmm = typename conditional3<isa == cpu::sse42, Xbyak::Xmm, isa == cpu::avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_mean = r9;
    Xbyak::Reg64 reg_variance = r10;
    Xbyak::Reg64 reg_work_amount = r11;
    Xbyak::Reg64 reg_stride = r12;
    Xbyak::Reg64 reg_sum = reg_mean;
    Xbyak::Reg64 reg_params = abi_param1;

    Vmm vmm_val = Vmm(0);
    Vmm vmm_mean = Vmm(1);
    Vmm vmm_variance = Vmm(2);
    Vmm vmm_sum = vmm_mean;
    Xbyak::Xmm xmm_aux1 = Xbyak::Xmm(3);
    Xbyak::Xmm xmm_aux2 = Xbyak::Xmm(4);
    Xbyak::Xmm xmm_aux3 = Xbyak::Xmm(5);

    inline void hsum_store(Xbyak::Xmm xmm_sum) {
        movshdup(xmm_aux3, xmm_sum);  //  sum:1,2,3,4; aux3:2,2,4,4
        addps(xmm_sum, xmm_aux3);     //  sum:1+2,2+2,3+4,4+4
        movhlps(xmm_aux3, xmm_sum);   //  aux3:3+4,4+4,4,4
        addps(xmm_sum, xmm_aux3);     //  sum:1+2+3+4,...
        if (jcp_.normalize_variance) {
            movss(ptr[reg_variance], xmm_sum);
        } else {
            movss(ptr[reg_sum], xmm_sum);
        }
    }

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
    }
};

// mean,variance->mvn
template <cpu_isa_t isa>
struct jit_uni_mvn_kernel_f32 : public jit_uni_mvn_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_mvn_kernel_f32)

    explicit jit_uni_mvn_kernel_f32(jit_mvn_config_params jcp, const mkldnn_primitive_attr &attr) : jit_uni_mvn_kernel(jcp, attr), jit_generator() {
        const auto &p = attr_.post_ops_;
        for (int i = 0; i < p.len_; i++) {
            auto &post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors.push_back(std::make_shared<jit_uni_eltwise_injector_f32<isa>>(
                        this, post_op.eltwise.alg, post_op.eltwise.alpha, post_op.eltwise.beta));
            } else if (post_op.is_depthwise()) {
                depthwise_injectors.push_back(std::make_shared<jit_uni_depthwise_injector_f32<isa>>(
                        this, post_op.depthwise.alg));
            } else if (post_op.is_quantization()) {
                quantization_injectors.push_back(std::make_shared<jit_uni_quantization_injector_f32<isa>>(
                        this, post_op, vmm_d_weights, vmm_d_bias, reg_d_weights, reg_d_bias));
            }
        }

        this->preamble();

        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_mean, ptr[reg_params + GET_OFF(mean)]);
        mov(reg_variance_inv, ptr[reg_params + GET_OFF(variance)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);
        mov(reg_src_stride, ptr[reg_params + GET_OFF(src_stride)]);
        mov(reg_dst_stride, ptr[reg_params + GET_OFF(dst_stride)]);
        if (attr_.post_ops_.len_ != 0)
            mov(reg_oc_off, ptr[reg_params + GET_OFF(oc_off)]);

        if (isa == avx512_common)
            uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        int repeats = (!jcp_.planar_layout && !jcp_.across_channels && isa == cpu::sse42) ? 2 : 1;  // block size is also 8 on cpu::sse42
        for (int i = 0; i < repeats; i++) {
            int offset_sse42 = i * 4;
            if (i > 0) {
                mov(reg_src, ptr[reg_params + GET_OFF(src)]);
                mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
                mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);

                add(reg_src, offset_sse42 * jcp_.src_data_size);
                add(reg_dst, offset_sse42 * jcp_.dst_data_size);
                add(reg_mean, offset_sse42 * sizeof(float));
                add(reg_variance_inv, offset_sse42 * sizeof(float));
                if (attr_.post_ops_.len_ != 0)
                    add(reg_oc_off, offset_sse42 * sizeof(float));
            }

            if (jcp_.planar_layout || jcp_.across_channels) {
                uni_vbroadcastss(vmm_mean, ptr[reg_mean]);
                if (jcp_.normalize_variance)
                    uni_vbroadcastss(vmm_variance_inv, ptr[reg_variance_inv]);

            } else {
                uni_vmovups(vmm_mean, ptr[reg_mean]);
                if (jcp_.normalize_variance)
                    uni_vmovups(vmm_variance_inv, ptr[reg_variance_inv]);
            }

            Xbyak::Label mvn_loop_label;
            Xbyak::Label mvn_loop_end_label;

            L(mvn_loop_label);
            {
                cmp(reg_work_amount, 0);
                jle(mvn_loop_end_label, T_NEAR);

                load_vector(vmm_val, ptr[reg_src], jcp_.src_dt);

                uni_vsubps(vmm_val, vmm_val, vmm_mean);
                if (jcp_.normalize_variance)
                    uni_vmulps(vmm_val, vmm_val, vmm_variance_inv);

                apply_post_ops(jcp_.dst_dt);

                store_vector(ptr[reg_dst], vmm_val, jcp_.dst_dt);

                add(reg_src, reg_src_stride);
                add(reg_dst, reg_dst_stride);
                sub(reg_work_amount, 1);

                jmp(mvn_loop_label, T_NEAR);
            }
            L(mvn_loop_end_label);
        }

        this->postamble();

        for (auto& inj : eltwise_injectors)
            inj->prepare_table();

        ker_ = (decltype(ker_)) this->getCode();
    }

private:
    using Vmm = typename conditional3<isa == cpu::sse42, Xbyak::Xmm, isa == cpu::avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;

    const int vlen = cpu_isa_traits<isa>::vlen;

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_mean = r9;
    Xbyak::Reg64 reg_variance_inv = r10;
    Xbyak::Reg64 reg_dst = r11;
    Xbyak::Reg64 reg_work_amount = r12;
    Xbyak::Reg64 reg_src_stride = r13;
    Xbyak::Reg64 reg_dst_stride = r14;
    Xbyak::Reg64 reg_params = abi_param1;

    Xbyak::Reg64 reg_oc_off = rax;
    Xbyak::Reg64 reg_d_weights = rbx;
    Xbyak::Reg64 reg_d_bias = rdx;

    Vmm vmm_val = Vmm(0);
    Vmm vmm_mean = Vmm(1);
    Vmm vmm_variance_inv = Vmm(2);
    Vmm vmm_zero = Vmm(3);

    Vmm vmm_d_weights = Vmm(5);
    Vmm vmm_d_bias = Vmm(6);

    Xbyak::Label l_table;

    std::vector<std::shared_ptr<jit_uni_eltwise_injector_f32<isa>>> eltwise_injectors;
    std::vector<std::shared_ptr<jit_uni_depthwise_injector_f32<isa>>> depthwise_injectors;
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

        if (src_dt != memory::f32)
            uni_vcvtdq2ps(vmm_src, vmm_src);
    }

    inline void store_vector(const Xbyak::Address &op, Vmm vmm_dst, memory::data_type dst_dt) {
        Ymm ymm_dst = Ymm(vmm_dst.getIdx());
        Xmm xmm_dst = Xmm(vmm_dst.getIdx());

        if (dst_dt == memory::f32) {
            uni_vmovups(op, vmm_dst);
        } else if (dst_dt == memory::u8) {
            uni_vcvtps2dq(vmm_dst, vmm_dst);
            if (isa == cpu::avx512_common) {
                vpmaxsd(vmm_dst, vmm_dst, vmm_zero);
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
        } else if (dst_dt == memory::s8) {
            uni_vcvtps2dq(vmm_dst, vmm_dst);
            if (isa == cpu::avx512_common) {
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
        }
    }

    void apply_post_ops(memory::data_type dst_dt) {
        const auto &p = attr_.post_ops_;
        int eltwise_inj_idx = 0;
        int depthwise_inj_idx = 0;
        int quantization_inj_idx = 0;
        for (int i = 0; i < p.len_; i++) {
            auto& post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors[eltwise_inj_idx]->compute_vector_range(vmm_val.getIdx(), vmm_val.getIdx() + 1);
                eltwise_inj_idx++;
            } else if (post_op.is_depthwise()) {
                mov(reg_d_weights, reinterpret_cast<size_t>(post_op.depthwise.weights_data));
                mov(reg_d_bias, reinterpret_cast<size_t>(post_op.depthwise.biases_data));
                add(reg_d_weights, reg_oc_off);
                add(reg_d_bias, reg_oc_off);
                depthwise_injectors[depthwise_inj_idx]->compute_vector_range(vmm_val.getIdx(), vmm_val.getIdx() + 1, reg_d_weights, reg_d_bias);
                depthwise_inj_idx++;
            } else if (post_op.is_quantization()) {
                bool do_dequantization = post_op.quantization.alg == alg_kind::quantization_quantize_dequantize;
                bool do_rounding = do_dequantization || dst_dt == memory::f32 || i != p.len_ - 1;
                int s_idx = vmm_val.getIdx();

                quantization_injectors[quantization_inj_idx]->init_crop_ptrs(reg_oc_off);
                quantization_injectors[quantization_inj_idx]->compute_crop(s_idx, s_idx + 1, 0);

                quantization_injectors[quantization_inj_idx]->init_input_scale_shift_ptrs(reg_oc_off);
                quantization_injectors[quantization_inj_idx]->compute_input_scale_shift(s_idx, s_idx + 1, 0, do_rounding);

                quantization_injectors[quantization_inj_idx]->init_output_scale_shift_ptrs(reg_oc_off);
                quantization_injectors[quantization_inj_idx]->compute_output_scale_shift(s_idx, s_idx + 1, 0);

                quantization_inj_idx++;
            }
        }
    }
};
//////////////////////////////////////////////////////////////////////////////////

MKLDNNMVNNode::MKLDNNMVNNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache) {}

void MKLDNNMVNNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    const auto& numOfDims = getParentEdgeAt(0)->getDims().ndims();
    if (numOfDims < 1 || numOfDims > 5)
        THROW_IE_EXCEPTION << "MVN layer with name '" << getCnnLayer()->name << "' doesn't support input with size of dimensions: " << numOfDims;

    auto * mvnLayer = dynamic_cast<MVNLayer*>(getCnnLayer().get());
    if (mvnLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert MVN layer.";

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();

    across_channels = mvnLayer->across_channels;
    normalize_variance = mvnLayer->normalize;
    eps = mvnLayer->GetParamAsFloat("eps");
}

void MKLDNNMVNNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    setPostOps(attr, true);

    Precision inputPrecision = getCnnLayer()->insData[0].lock()->getPrecision();
    Precision outputPrecision = getCnnLayer()->outData[0]->getPrecision();

    if (!fusedWith.empty()) {
        auto lastFusedLayer = fusedWith[fusedWith.size() - 1].get()->getCnnLayer();
        if (lastFusedLayer) {
            outputPrecision = lastFusedLayer->outData[0]->getPrecision();
        }
    }

    if (getParentEdgeAt(0)->getDims().ndims() < 4 || getParentEdgeAt(0)->getDims().ndims() > 5
        || across_channels != 0 || normalize_variance != 1) {
        inputPrecision = Precision::FP32;
        outputPrecision = Precision::FP32;
    }

    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(inputPrecision);
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(outputPrecision);

    input_prec = inputPrecision;
    output_prec = outputPrecision;
    src_data_size = MKLDNNExtensionUtils::sizeOfDataType(inputDataType);
    dst_data_size = MKLDNNExtensionUtils::sizeOfDataType(outputDataType);

    bool canBeInplace = src_data_size == dst_data_size && getParentEdgeAt(0)->getParent()->getChildEdges().size() == 1;

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = false;
    config.inConfs.resize(1);
    config.outConfs.resize(1);
    config.inConfs[0].constant = false;
    config.outConfs[0].constant = false;
    config.inConfs[0].inPlace = -1;
    config.outConfs[0].inPlace = canBeInplace ? 0 : -1;

    auto pushDesc = [&](memory::format format) {
        config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, format);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), outputDataType, format);
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown});
    };

    if (across_channels == 0 && normalize_variance == 1) {
        if (getParentEdgeAt(0)->getDims().ndims() == 4) {
            pushDesc(memory::nhwc);
        } else if (getParentEdgeAt(0)->getDims().ndims() == 5) {
            pushDesc(memory::ndhwc);
        }
    }

    if (inputPrecision == Precision::FP32 && outputPrecision == Precision::FP32) {
        if (getParentEdgeAt(0)->getDims().ndims() == 4) {
            if (mayiuse(cpu::avx512_common)) {
                pushDesc(memory::nChw16c);
            } else if (mayiuse(cpu::avx2) || mayiuse(cpu::sse42)) {
                pushDesc(memory::nChw8c);
            }
        } else if (getParentEdgeAt(0)->getDims().ndims() == 5) {
            if (mayiuse(cpu::avx512_common)) {
                pushDesc(memory::nCdhw16c);
            } else if (mayiuse(cpu::avx2) || mayiuse(cpu::sse42)) {
                pushDesc(memory::nCdhw8c);
            }
        }

        if (fusedWith.empty()) {
            if (canBeInplace)
                config.inConfs[0].inPlace = 0;
            pushDesc(MKLDNNMemory::GetPlainFormat(getChildEdgeAt(0)->getDims()));
        }
    }
}

void MKLDNNMVNNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";

    auto selectedPD = getSelectedPrimitiveDescriptor();
    auto jcp = jit_mvn_config_params();
    jcp.src_dt = MKLDNNExtensionUtils::IEPrecisionToDataType(selectedPD->getConfig().inConfs[0].desc.getPrecision());
    jcp.dst_dt = MKLDNNExtensionUtils::IEPrecisionToDataType(selectedPD->getConfig().outConfs[0].desc.getPrecision());
    jcp.src_data_size = MKLDNNExtensionUtils::sizeOfDataType(jcp.src_dt);
    jcp.dst_data_size = MKLDNNExtensionUtils::sizeOfDataType(jcp.dst_dt);
    jcp.planar_layout = MKLDNNMemory::GetPlainLayout(getChildEdgeAt(0)->getDims()) == selectedPD->getConfig().inConfs[0].desc.getLayout();
    jcp.normalize_variance = normalize_variance;
    jcp.across_channels = across_channels;

    if (mayiuse(cpu::avx512_common)) {
        mvn_kernel.reset(new jit_uni_mvn_kernel_f32<cpu::avx512_common>(jcp, *attr.get()));

        jcp.normalize_variance = false;
        mvn_mean_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<cpu::avx512_common>(jcp));
        if (normalize_variance) {
            jcp.normalize_variance = true;
            mvn_variance_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<cpu::avx512_common>(jcp));
        }
    } else if (mayiuse(cpu::avx2)) {
        mvn_kernel.reset(new jit_uni_mvn_kernel_f32<cpu::avx2>(jcp, *attr.get()));

        jcp.normalize_variance = false;
        mvn_mean_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<cpu::avx2>(jcp));
        if (normalize_variance) {
            jcp.normalize_variance = true;
            mvn_variance_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<cpu::avx2>(jcp));
        }
    } else if (mayiuse(cpu::sse42)) {
        mvn_kernel.reset(new jit_uni_mvn_kernel_f32<cpu::sse42>(jcp, *attr.get()));

        jcp.normalize_variance = false;
        mvn_mean_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<cpu::sse42>(jcp));
        if (normalize_variance) {
            jcp.normalize_variance = true;
            mvn_variance_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<cpu::sse42>(jcp));
        }
    }
}

void MKLDNNMVNNode::setPostOps(mkldnn::primitive_attr &attr, bool initWeights) {
    int blob_idx = 0;
    mkldnn::post_ops ops;

    for (auto &node : fusedWith) {
        auto* quantizeNode = dynamic_cast<MKLDNNQuantizeNode *>(node.get());
        if (quantizeNode) {
            quantizeNode->appendPostOps(ops);
            continue;
        }

        auto* depthwiseNode = dynamic_cast<MKLDNNDepthwiseNode *>(node.get());
        if (depthwiseNode) {
            if (initWeights) {
                auto* depthwiseLayer = reinterpret_cast<WeightableLayer*>(depthwiseNode->getCnnLayer().get());
                MKLDNNDims depthwiseDims({static_cast<ptrdiff_t>(rnd_up(getChildEdgeAt(0)->getDims()[1], 16))});

                PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
                PostOpsIntBlobMemory[blob_idx]->Create(depthwiseDims, memory::data_type::f32, memory::format::x);

                PostOpsIntBlobMemory[blob_idx]->SetData(memory::data_type::f32, memory::x,
                                                        depthwiseLayer->_weights->buffer(),
                                                        depthwiseLayer->_weights->size() *
                                                        MKLDNNExtensionUtils::sizeOfDataType(memory::data_type::f32));

                if (depthwiseNode->isBroadcast()) {
                    float broadcastValue = static_cast<float *>(PostOpsIntBlobMemory[blob_idx]->GetData())[0];
                    for (int i = 1; i < PostOpsIntBlobMemory[blob_idx]->GetPrimitiveDescriptor().desc().data.dims[0]; i++) {
                        static_cast<float *>(PostOpsIntBlobMemory[blob_idx]->GetData())[i] = broadcastValue;
                    }
                }

                if (depthwiseNode->getAlgorithm() == depthwise_scale_shift) {
                    PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
                    PostOpsIntBlobMemory[blob_idx + 1]->Create(depthwiseDims, memory::data_type::f32,
                                                               memory::format::x);
                    PostOpsIntBlobMemory[blob_idx + 1]->SetData(memory::data_type::f32, memory::x,
                                                                depthwiseLayer->_biases->buffer(),
                                                                depthwiseLayer->_biases->size() *
                                                                MKLDNNExtensionUtils::sizeOfDataType(memory::data_type::f32));

                    if (depthwiseNode->isBroadcast()) {
                        float broadcastValue = static_cast<float *>(PostOpsIntBlobMemory[blob_idx + 1]->GetData())[0];
                        for (int i = 1; i < PostOpsIntBlobMemory[blob_idx + 1]->GetPrimitiveDescriptor().desc().data.dims[0]; i++) {
                            static_cast<float *>(PostOpsIntBlobMemory[blob_idx + 1]->GetData())[i] = broadcastValue;
                        }
                    }

                    ops.append_depthwise(depthwiseNode->getAlgorithm(),
                                         (const float *) PostOpsIntBlobMemory[blob_idx]->GetData(),
                                         (const float *) PostOpsIntBlobMemory[blob_idx + 1]->GetData());

                    blob_idx += 2;
                }
            } else {
                ops.append_depthwise(depthwiseNode->getAlgorithm(),
                                     nullptr,
                                     nullptr);
            }

            continue;
        }

        auto* activationNode = dynamic_cast<MKLDNNActivationNode *>(node.get());
        if (activationNode) {
            ops.append_eltwise(1.0, activationNode->getAlgorithm(), activationNode->getAlpha(), activationNode->getBeta());

            continue;
        }

        THROW_IE_EXCEPTION << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType()) << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

void MKLDNNMVNNode::execute(mkldnn::stream strm) {
    auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto &srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();

    Layout layout = getParentEdgeAt(0)->getDesc().getLayout();

    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetData());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetData());

    if (layout == C || layout == NC || layout == CHW || layout == NCHW || layout == NCDHW) {
        mvn_pln(src_data, dst_data, getParentEdgeAt(0)->getDesc().getDims());
    } else {
        if (output_prec == Precision::U8) {
            auto dst_data = reinterpret_cast<uint8_t *>(dstMemPtr->GetData());
            if (input_prec == Precision::U8) {
                auto src_data = reinterpret_cast<const uint8_t *>(srcMemPtr->GetData());
                mvn_blk<uint8_t, uint8_t>(src_data, dst_data, getParentEdgeAt(0)->getDesc().getDims());
            } else if (input_prec == Precision::I8) {
                auto src_data = reinterpret_cast<const int8_t *>(srcMemPtr->GetData());
                mvn_blk<int8_t, uint8_t>(src_data, dst_data, getParentEdgeAt(0)->getDesc().getDims());
            } else if (input_prec == Precision::FP32) {
                auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetData());
                mvn_blk<float, uint8_t>(src_data, dst_data, getParentEdgeAt(0)->getDesc().getDims());
            }
        } else if (output_prec == Precision::I8) {
            auto dst_data = reinterpret_cast<int8_t *>(dstMemPtr->GetData());
            if (input_prec == Precision::U8) {
                auto src_data = reinterpret_cast<const uint8_t *>(srcMemPtr->GetData());
                mvn_blk<uint8_t, int8_t>(src_data, dst_data, getParentEdgeAt(0)->getDesc().getDims());
            } else if (input_prec == Precision::I8) {
                auto src_data = reinterpret_cast<const int8_t *>(srcMemPtr->GetData());
                mvn_blk<int8_t, int8_t>(src_data, dst_data, getParentEdgeAt(0)->getDesc().getDims());
            } else if (input_prec == Precision::FP32) {
                auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetData());
                mvn_blk<float, int8_t>(src_data, dst_data, getParentEdgeAt(0)->getDesc().getDims());
            }
        } else if (output_prec == Precision::FP32) {
            auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetData());
            if (input_prec == Precision::U8) {
                auto src_data = reinterpret_cast<const uint8_t *>(srcMemPtr->GetData());
                mvn_blk<uint8_t, float>(src_data, dst_data, getParentEdgeAt(0)->getDesc().getDims());
            } else if (input_prec == Precision::I8) {
                auto src_data = reinterpret_cast<const int8_t *>(srcMemPtr->GetData());
                mvn_blk<int8_t, float>(src_data, dst_data, getParentEdgeAt(0)->getDesc().getDims());
            } else if (input_prec == Precision::FP32) {
                auto src_data = reinterpret_cast<float *>(srcMemPtr->GetData());
                mvn_blk<float, float>(src_data, dst_data, getParentEdgeAt(0)->getDesc().getDims());
            }
        }
    }
}

std::tuple<size_t, size_t, size_t, size_t, size_t> MKLDNNMVNNode::get5dShapes(const SizeVector& dims) {
    std::tuple<size_t, size_t, size_t, size_t, size_t> shapes;
    switch (dims.size()) {
        case 1 : { shapes = std::make_tuple(1, dims[0], 1, 1, 1); break; }
        case 2 : { shapes = std::make_tuple(dims[0], dims[1], 1, 1, 1); break; }
        case 3 : { shapes = std::make_tuple(dims[0], dims[1], 1, dims[2], 1); break; }
        case 4 : { shapes = std::make_tuple(dims[0], dims[1], 1, dims[2], dims[3]); break; }
        case 5 : { shapes = std::make_tuple(dims[0], dims[1], dims[2], dims[3], dims[4]); break; }
        default : { THROW_IE_EXCEPTION << "MVN layer with name '" << getCnnLayer()->name << "' doesn't support planar layout with rank: " << dims.size(); }
    }
    return shapes;
}

void MKLDNNMVNNode::mvn_pln(const float* src_data, float* dst_data, const SizeVector& dims) {
    size_t blk_size = 1;  // blk size in vmm
    if (mayiuse(cpu::avx512_common)) {
        blk_size = 16;
    } else if (mayiuse(cpu::avx2)) {
        blk_size = 8;
    } else if (mayiuse(cpu::sse42)) {
        blk_size = 4;
    }

    size_t N = 0; size_t C = 0; size_t D = 0; size_t H = 0; size_t W = 0;
    std::tie(N, C, D, H, W) = get5dShapes(dims);

    size_t C1 = H * W;
    size_t C2 = C1 * D;
    size_t C3 = C2 * C;

    for (size_t b = 0lu; b < N; b++) {
        size_t cb = b * C3;
        if (across_channels) {
            // Calculate mean value for one instance in batch
            // Parallel sum for each channel
            float C3inv = 1.f / static_cast<float>(C3);
            float mean_temp = 0.0f;
            size_t tail_across_channels = (C2 / blk_size) * blk_size;
            if (mvn_mean_kernel) {
                mean_temp = parallel_sum(C, mean_temp, [&](size_t c)->float {
                    float mean_internal = 0.0f;
                    size_t cc = cb + c * C2;
                    auto arg = jit_mvn_call_args();
                    arg.src = src_data + cc;
                    arg.sum = static_cast<float*>(&mean_internal);
                    arg.src_stride = static_cast<size_t>(blk_size * sizeof(float));
                    arg.work_amount = static_cast<size_t>(C2 / blk_size);
                    (*mvn_mean_kernel)(&arg);
                    for (size_t tail = tail_across_channels; tail < C2; tail++) {
                        mean_internal += src_data[cc + tail];
                    }
                    return mean_internal;
                });
            } else {
                mean_temp = parallel_sum(C, mean_temp, [&](size_t c)->float {
                    float mean_internal = 0.0f;
                    size_t cc = cb + c * C2;
                    for (size_t tail = 0lu; tail < C2; tail++) {
                        mean_internal += src_data[cc + tail];
                    }
                    return mean_internal;
                });
            }
            float mean = mean_temp * C3inv;

            // calculate variance value for one instance in batch
            // parallel sum for each channel
            if (normalize_variance) {
                float variance_temp = 0.0f;
                if (mvn_variance_kernel) {
                    variance_temp = parallel_sum(C, variance_temp, [&](size_t c)->float {
                        float variance_internal = 0.0f;
                        size_t cc = cb + c * C2;
                        auto arg = jit_mvn_call_args();
                        arg.src = src_data + cc;
                        arg.mean = static_cast<float*>(&mean);
                        arg.variance = static_cast<float*>(&variance_internal);
                        arg.src_stride = static_cast<size_t>(blk_size * sizeof(float));
                        arg.work_amount = static_cast<size_t>(C2 / blk_size);
                        (*mvn_variance_kernel)(&arg);

                        for (size_t tail = tail_across_channels; tail < C2; tail++) {
                            variance_internal += (src_data[cc + tail] - mean) * (src_data[cc + tail] - mean);
                        }
                        return variance_internal;
                    });
                } else {
                    variance_temp = parallel_sum(C, variance_temp, [&](size_t c)->float {
                        float variance_internal = 0.0f;
                        size_t cc = cb + c * C2;
                        for (size_t tail = 0lu; tail < C2; tail++) {
                            variance_internal += (src_data[cc + tail] - mean) * (src_data[cc + tail] - mean);
                        }
                        return variance_internal;
                    });
                }
                float variance = 1.f / sqrtf(variance_temp * C3inv + eps);
                // mvn for one instance in batch
                if (mvn_kernel) {
                    parallel_for(C, [&](int c) {
                        size_t cc = cb + c * C2;
                        auto arg = jit_mvn_call_args();
                        arg.src = src_data + cc;
                        arg.dst = dst_data + cc;
                        arg.mean = static_cast<float*>(&mean);
                        arg.variance = static_cast<float*>(&variance);
                        arg.src_stride = static_cast<size_t>(blk_size * sizeof(float));
                        arg.dst_stride = static_cast<size_t>(blk_size * sizeof(float));
                        arg.work_amount = static_cast<size_t>(C2 / blk_size);
                        (*mvn_kernel)(&arg);

                        for (size_t tail = tail_across_channels; tail < C2; tail++) {
                            dst_data[cc + tail] = (src_data[cc + tail] - mean) * variance;
                        }
                    });
                } else {
                    parallel_for(C, [&](int c) {
                        size_t cc = cb + c * C2;
                        for (size_t tail = 0lu; tail < C2; tail++) {
                            dst_data[cc + tail] = (src_data[cc + tail] - mean) * variance;
                        }
                    });
                }
            } else {
                // mvn for one instance in batch
                if (mvn_kernel) {
                    parallel_for(C, [&](int c) {
                        size_t cc = cb + c * C2;
                        auto arg = jit_mvn_call_args();
                        arg.src = src_data + cc;
                        arg.dst = dst_data + cc;
                        arg.mean = static_cast<float*>(&mean);
                        arg.src_stride = static_cast<size_t>(blk_size * sizeof(float));
                        arg.dst_stride = static_cast<size_t>(blk_size * sizeof(float));
                        arg.work_amount = static_cast<size_t>(C2 / blk_size);
                        (*mvn_kernel)(&arg);

                        for (size_t tail = tail_across_channels; tail < C2; tail++) {
                            dst_data[cc + tail] = src_data[cc + tail] - mean;
                        }
                    });
                } else {
                    parallel_for(C, [&](int c) {
                        size_t cc = cb + c * C2;
                        for (size_t tail = 0lu; tail < C2; tail++) {
                            dst_data[cc + tail] = src_data[cc + tail] - mean;
                        }
                    });
                }
            }
        } else {  // per channel
            float C2inv = 1.f / static_cast<float>(C2);
            if (mvn_mean_kernel && mvn_variance_kernel && mvn_kernel) {
                parallel_for(C, [&](size_t c) {
                    // mean for this channel
                    size_t tail_per_channel = (C2 / blk_size) * blk_size;
                    float mean = 0.f;
                    size_t cc = cb + c * C2;
                    // the same arg for three kernels
                    auto arg = jit_mvn_call_args();
                    arg.src = src_data + cc;
                    arg.dst = dst_data + cc;
                    arg.sum = static_cast<float*>(&mean);
                    arg.src_stride = static_cast<size_t>(blk_size * sizeof(float));
                    arg.dst_stride = static_cast<size_t>(blk_size * sizeof(float));
                    arg.work_amount = static_cast<size_t>(C2 / blk_size);
                    (*mvn_mean_kernel)(&arg);

                    for (size_t tail = tail_per_channel; tail < C2; tail++) {
                        mean += src_data[cc + tail];
                    }
                    mean *= C2inv;

                    // variance for this channel
                    if (normalize_variance) {
                        float variance = 0.f;
                        arg.mean = static_cast<float*>(&mean);
                        arg.variance = static_cast<float*>(&variance);
                        (*mvn_variance_kernel)(&arg);

                        for (size_t tail = tail_per_channel; tail < C2; tail++) {
                            variance += (src_data[cc + tail] - mean) * (src_data[cc + tail] - mean);
                        }
                        variance = 1.f / sqrtf(variance * C2inv + eps);

                        // mvn for this channel
                        (*mvn_kernel)(&arg);
                        for (size_t tail = tail_per_channel; tail < C2; tail++) {
                            dst_data[cc + tail] = (src_data[cc + tail] - mean) * variance;
                        }
                    } else {
                        // mvn for this channel
                        arg.mean = static_cast<float*>(&mean);
                        (*mvn_kernel)(&arg);

                        for (size_t tail = tail_per_channel; tail < C2; tail++) {
                            dst_data[cc + tail] = src_data[cc + tail] - mean;
                        }
                    }
                });
            } else {
                parallel_for(C, [&](size_t c) {
                    // mean for this channel
                    float mean = 0.f;
                    size_t cc = cb + c * C2;
                    for (size_t tail = 0lu; tail < C2; tail++) {
                        mean += src_data[cc + tail];
                    }
                    mean *= C2inv;

                    // variance for this channel
                    if (normalize_variance) {
                        float variance = 0.f;
                        for (size_t tail = 0lu; tail < C2; tail++) {
                            variance += (src_data[cc + tail] - mean) * (src_data[cc + tail] - mean);
                        }
                        variance = 1.f / sqrtf(variance * C2inv + eps);

                        // mvn for this channel
                        for (size_t tail = 0lu; tail < C2; tail++) {
                            dst_data[cc + tail] = (src_data[cc + tail] - mean) * variance;
                        }
                    } else {
                        // mvn for this channel
                        for (size_t tail = 0lu; tail < C2; tail++) {
                            dst_data[cc + tail] = src_data[cc + tail] - mean;
                        }
                    }
                });
            }
        }
    }
}

template <typename in_data_t, typename out_data_t>
void MKLDNNMVNNode::mvn_blk(const in_data_t* src_data, out_data_t* dst_data, const SizeVector& dims) {
    size_t blk_size = 1;  // channel blk for memory layout
    size_t ele_in_vmm = 4;
    if (mayiuse(cpu::avx512_common)) {
        blk_size = 16;
        ele_in_vmm = 16;
    } else if (mayiuse(cpu::avx2)) {
        blk_size = 8;
        ele_in_vmm = 8;
    } else {
        blk_size = 8;
        ele_in_vmm = 4;
    }

    size_t dims_size = dims.size();
    size_t N = (dims_size > 0) ? dims[0] : 1lu;
    size_t C = (dims_size > 1) ? dims[1] : 1lu;
    size_t D = (dims_size > 4) ? dims[dims_size - 3] : 1lu;
    size_t H = (dims_size > 3) ? dims[dims_size - 2] : 1lu;
    size_t W = (dims_size > 2) ? dims[dims_size - 1] : 1lu;

    bool is_nhwc = false;
    Layout layout = getParentEdgeAt(0)->getDesc().getLayout();
    if (layout == NHWC || layout == NDHWC)
        is_nhwc = true;

    size_t CB = is_nhwc ? C / blk_size : div_up(C, blk_size);

    size_t C0 = is_nhwc ? W * C : W * blk_size;
    size_t C1 = C0 * H;
    size_t C2 = C1 * D;
    size_t C3 = C2 * CB;
    size_t C5 = C * D * H * W;

    size_t threads_num =  mkldnn_get_max_threads();
    size_t aux_buffer_size = across_channels ? blk_size : rnd_up(C, blk_size);
    std::vector<float> mean_buffer(aux_buffer_size * threads_num);
    std::vector<float> variance_buffer(aux_buffer_size * threads_num);

    for (size_t b = 0lu; b < N; b++) {
        size_t ccb = is_nhwc ? b * C2 : b * C3;
        if (across_channels) {
            // mean for this instance in batch
            float C5inv = 1.f / static_cast<float>(C5);
            float mean_temp = 0.0f;
            mean_temp = parallel_sum3d(CB, D, H, mean_temp, [&](size_t cb, size_t d, size_t h)->float {
                size_t ccbd = ccb + cb * C2 + d * C1 + h * C0;
                size_t min_cb = (std::min)(blk_size, C - cb * blk_size);

                float mean_internal = 0.0f;
                if ((min_cb == blk_size) && mvn_mean_kernel) {
                    auto mean_buffer_ptr = &mean_buffer[blk_size * mkldnn_get_thread_num()];
                    for (int i = 0; i < blk_size; i++)
                        mean_buffer_ptr[i] = 0.f;

                    auto arg = jit_mvn_call_args();
                    arg.src = src_data + ccbd;
                    arg.sum = mean_buffer_ptr;
                    arg.src_stride = static_cast<size_t>(ele_in_vmm * src_data_size);
                    arg.work_amount = static_cast<size_t>((W * blk_size)/ele_in_vmm);
                    (*mvn_mean_kernel)(&arg);

                    for (int i = 0; i < blk_size; i++)
                        mean_internal += mean_buffer_ptr[i];

                    // no tail here due blk/ele_in_vmm is 1 or 2.
                } else {
                    for (size_t w = 0lu; w < W; w++) {
                        size_t cw = ccbd + w * blk_size;
                        for (size_t c = 0lu; c < min_cb; c++) {
                            mean_internal += src_data[cw + c];
                        }
                    }
                }
                return mean_internal;
            });
            float mean = mean_temp * C5inv;

            if (normalize_variance) {
                // variance for one instance in batch
                float variance_temp = 0.0f;
                variance_temp = parallel_sum3d(CB, D, H, variance_temp, [&](size_t cb, size_t d, size_t h)->float {
                    size_t ccbd = ccb + cb * C2 + d * C1 + h * C0;
                    size_t min_cb = (std::min)(blk_size, C - cb * blk_size);

                    float variance_internal = 0.0f;
                    if ((blk_size == min_cb) && mvn_variance_kernel) {
                        auto variance_buffer_ptr = &variance_buffer[blk_size * mkldnn_get_thread_num()];
                        for (int i = 0; i < blk_size; i++)
                            variance_buffer_ptr[i] = 0.f;

                        auto arg = jit_mvn_call_args();
                        arg.src = src_data + ccbd;
                        arg.mean = static_cast<float*>(&mean);
                        arg.variance = variance_buffer_ptr;
                        arg.src_stride = static_cast<size_t>(ele_in_vmm * src_data_size);
                        arg.work_amount = static_cast<size_t>((W * blk_size)/ele_in_vmm);
                        (*mvn_variance_kernel)(&arg);

                        for (int i = 0; i < blk_size; i++)
                            variance_internal += variance_buffer_ptr[i];
                    } else {
                        for (size_t w = 0lu; w < W; w++) {
                            size_t cw = ccbd + w * blk_size;
                            for (size_t c = 0lu; c < min_cb; c++) {
                                variance_internal += (src_data[cw + c] - mean) * (src_data[cw + c] - mean);
                            }
                        }
                    }
                    return variance_internal;
                });

                float variance = 1.f / sqrtf(variance_temp * C5inv + eps);
                // mvn for one instance in batch
                parallel_for3d(CB, D, H, [&](size_t cb, size_t d, size_t h) {
                    size_t ccbd = ccb + cb * C2 + d * C1 + h * C0;
                    size_t min_cb = (std::min)(blk_size, C - cb * blk_size);
                    if ((blk_size == min_cb) && mvn_kernel) {
                        auto arg = jit_mvn_call_args();
                        arg.src = src_data + ccbd;
                        arg.dst = dst_data + ccbd;
                        arg.mean = static_cast<float*>(&mean);
                        arg.variance = static_cast<float*>(&variance);
                        arg.src_stride = static_cast<size_t>(ele_in_vmm * src_data_size);
                        arg.dst_stride = static_cast<size_t>(ele_in_vmm * dst_data_size);
                        arg.work_amount = static_cast<size_t>((W * blk_size)/ele_in_vmm);
                        (*mvn_kernel)(&arg);
                    } else {
                        for (size_t w = 0lu; w < W; w++) {
                            size_t cw = ccbd + w * blk_size;
                            for (size_t c = 0lu; c < min_cb; c++) {
                                size_t src_offset = cw + c;
                                dst_data[src_offset] = (src_data[src_offset] - mean) * variance;
                            }
                        }
                    }
                });
            } else {
                // mvn for one instance in batch
                parallel_for3d(CB, D, H, [&](size_t cb, size_t d, size_t h) {
                    size_t ccbd = ccb + cb * C2 + d * C1 + h * C0;
                    size_t min_cb = (std::min)(blk_size, C - cb * blk_size);
                    if ((blk_size == min_cb) && mvn_kernel) {
                        auto arg = jit_mvn_call_args();
                        arg.src = src_data + ccbd;
                        arg.dst = dst_data + ccbd;
                        arg.mean = static_cast<float*>(&mean);
                        arg.src_stride = static_cast<size_t>(ele_in_vmm * src_data_size);
                        arg.dst_stride = static_cast<size_t>(ele_in_vmm * dst_data_size);
                        arg.work_amount = static_cast<size_t>((W * blk_size)/ele_in_vmm);
                        (*mvn_kernel)(&arg);
                    } else {
                        for (size_t w = 0lu; w < W; w++) {
                            size_t cw = ccbd + w * blk_size;
                            for (size_t c = 0lu; c < min_cb; c++) {
                                size_t src_offset = cw + c;
                                dst_data[src_offset] = src_data[src_offset] - mean;
                            }
                        }
                    }
                });
            }
        } else {  // for per_channel
            size_t tail_cb_end = div_up(static_cast<size_t>(C), static_cast<size_t>(blk_size));
            size_t src_stride = is_nhwc ? C : blk_size;

            size_t tail_cb_start = 0;
            float size_inv = 1.f / static_cast<float>(D * H * W);
            if (mvn_mean_kernel) {
                tail_cb_start = CB;

                for (int i = 0; i < mean_buffer.size(); i++)
                    mean_buffer[i] = 0.f;

                parallel_for2d(D, H, [&](size_t d, size_t h) {
                    for (size_t cb = 0; cb < CB; cb++) {
                        size_t src_off = is_nhwc ? ccb + d * H * W * C + h * W * C + cb * blk_size
                                                 : ccb + d * H * W * blk_size + h * W * blk_size + cb * D * H * W * blk_size;
                        auto thr_idx = mkldnn_get_thread_num();
                        auto mean_buffer_ptr = &mean_buffer[blk_size * cb + aux_buffer_size * thr_idx];

                        auto arg = jit_mvn_call_args();
                        arg.src = src_data + src_off;
                        arg.sum = mean_buffer_ptr;
                        arg.src_stride = src_stride * src_data_size;
                        arg.work_amount = static_cast<size_t>(W);
                        (*mvn_mean_kernel)(&arg);
                    }
                });

                for (size_t i = 1; i < threads_num; i++) {
                    for (size_t c = 0; c < C; c++)
                        mean_buffer[c] += mean_buffer[c + aux_buffer_size * i];
                }
                for (size_t c = 0; c < C; c++)
                    mean_buffer[c] *= size_inv;
            }

            for (size_t cb = tail_cb_start; cb < tail_cb_end; cb++) {
                size_t src_off = is_nhwc ? ccb + cb * blk_size : ccb + cb * C2;
                size_t min_cb = (std::min)(blk_size, C - cb * blk_size);
                auto mean_buffer_ptr = &mean_buffer[blk_size * cb];

                for (size_t c = 0lu; c < min_cb; c++) {
                    size_t cc = src_off + c;

                    mean_buffer_ptr[c] = 0.0f;
                    for (size_t d = 0; d < D; d++) {
                        size_t cd = cc + d * C1;
                        for (size_t h = 0; h < H; h++) {
                            size_t ch = cd + h * C0;
                            for (size_t w = 0; w < W; w++) {
                                mean_buffer_ptr[c] += src_data[ch + w * src_stride];
                            }
                        }
                    }
                    mean_buffer_ptr[c] *= size_inv;
                }
            }

            if (normalize_variance) {
                tail_cb_start = 0;
                if (mvn_variance_kernel) {
                    tail_cb_start = CB;

                    for (int i = 0; i < variance_buffer.size(); i++)
                        variance_buffer[i] = 0.f;

                    parallel_for2d(D, H, [&](size_t d, size_t h) {
                        for (size_t cb = 0; cb < CB; cb++) {
                            size_t src_off = is_nhwc ? ccb + d * H * W * C + h * W * C + cb * blk_size
                                                     : ccb + d * H * W * blk_size + h * W * blk_size + cb * D * H * W * blk_size;
                            auto thr_idx = mkldnn_get_thread_num();
                            auto mean_buffer_ptr = &mean_buffer[blk_size * cb];
                            auto variance_buffer_ptr = &variance_buffer[blk_size * cb + aux_buffer_size * thr_idx];

                            auto arg = jit_mvn_call_args();
                            arg.src = src_data + src_off;
                            arg.mean = mean_buffer_ptr;
                            arg.variance = variance_buffer_ptr;
                            arg.src_stride = src_stride * src_data_size;
                            arg.work_amount = static_cast<size_t>(W);
                            (*mvn_variance_kernel)(&arg);
                        }
                    });

                    for (size_t i = 1; i < threads_num; i++) {
                        for (size_t c = 0; c < C; c++)
                            variance_buffer[c] += variance_buffer[c + aux_buffer_size * i];
                    }
                    for (size_t c = 0; c < C; c++)
                        variance_buffer[c] = 1.f / sqrtf(variance_buffer[c] * size_inv + eps);
                }

                for (size_t cb = tail_cb_start; cb < tail_cb_end; cb++) {
                    size_t src_off = is_nhwc ? ccb + cb * blk_size : ccb + cb * C2;
                    size_t min_cb = (std::min)(blk_size, C - cb * blk_size);
                    auto mean_buffer_ptr = &mean_buffer[blk_size * cb];
                    auto variance_buffer_ptr = &variance_buffer[blk_size * cb];

                    for (size_t c = 0lu; c < min_cb; c++) {
                        size_t cc = src_off + c;

                        variance_buffer_ptr[c] = 0.0f;
                        for (size_t d = 0lu; d < D; d++) {
                            size_t cd = cc + d * C1;
                            for (size_t h = 0lu; h < H; h++) {
                                size_t ch = cd + h * C0;
                                for (size_t w = 0lu; w < W; w++) {
                                    variance_buffer_ptr[c] +=
                                            (src_data[ch + w * src_stride] - mean_buffer_ptr[c]) *
                                            (src_data[ch + w * src_stride] - mean_buffer_ptr[c]);
                                }
                            }
                        }
                        variance_buffer_ptr[c] = 1.f / sqrtf(variance_buffer_ptr[c] * size_inv + eps);
                    }
                }

                tail_cb_start = 0;
                if (mvn_kernel) {
                    tail_cb_start = CB;

                    parallel_for2d(D, H, [&](size_t d, size_t h) {
                        for (size_t cb = 0; cb < CB; cb++) {
                            size_t src_off = is_nhwc ? ccb + d * H * W * C + h * W * C + cb * blk_size
                                                     : ccb + d * H * W * blk_size + h * W * blk_size + cb * D * H * W * blk_size;
                            auto mean_buffer_ptr = &mean_buffer[blk_size * cb];
                            auto variance_buffer_ptr = &variance_buffer[blk_size * cb];

                            auto arg = jit_mvn_call_args();
                            arg.src = src_data + src_off;
                            arg.dst = dst_data + src_off;
                            arg.mean = mean_buffer_ptr;
                            arg.variance = variance_buffer_ptr;
                            arg.src_stride = src_stride * src_data_size;
                            arg.dst_stride = src_stride * dst_data_size;
                            arg.work_amount = static_cast<size_t>(W);
                            arg.oc_off = cb * blk_size * sizeof(float);
                            (*mvn_kernel)(&arg);
                        }
                    });
                }

                for (size_t cb = tail_cb_start; cb < tail_cb_end; cb++) {
                    size_t src_off = is_nhwc ? ccb + cb * blk_size : ccb + cb * C2;
                    size_t min_cb = (std::min)(blk_size, C - cb * blk_size);
                    auto mean_buffer_ptr = &mean_buffer[blk_size * cb];
                    auto variance_buffer_ptr = &variance_buffer[blk_size * cb];

                    for (size_t c = 0lu; c < min_cb; c++) {
                        size_t cc = src_off + c;

                        for (size_t d = 0lu; d < D; d++) {
                            size_t cd = cc + d * C1;
                            for (size_t h = 0lu; h < H; h++) {
                                size_t ch = cd + h * C0;
                                for (size_t w = 0lu; w < W; w++) {
                                    float dst_value = (src_data[ch + w * src_stride] - mean_buffer_ptr[c]) * variance_buffer_ptr[c];
                                    if (!fusedWith.empty()) {
                                        const auto &p = (*attr.get()).post_ops_;
                                        for (int i = 0; i < p.len_; i++) {
                                            auto &post_op = p.entry_[i];
                                            if (post_op.is_eltwise()) {
                                                //  only eltwise_relu supported
                                                if (dst_value < 0) dst_value = 0;
                                            } else if (post_op.is_depthwise()) {
                                                //  only ScaleShift supported
                                                float scale = post_op.depthwise.weights_data[cb * blk_size + c];
                                                float shift = post_op.depthwise.biases_data[cb * blk_size + c];
                                                dst_value = dst_value * scale + shift;
                                            } else if (post_op.is_quantization()) {
                                                bool do_dequantization = post_op.quantization.alg ==
                                                                         alg_kind::quantization_quantize_dequantize;
                                                bool do_rounding = do_dequantization || output_prec == Precision::FP32 ||
                                                                   i != p.len_ - 1;

                                                auto quant = post_op.quantization;
                                                float crl = quant.crop_low_data->shifts_[quant.crop_low_data->count_ == 1 ? 0 : cb * blk_size + c];
                                                float crh = quant.crop_high_data->shifts_[quant.crop_high_data->count_ == 1 ? 0 : cb * blk_size + c];
                                                float isc = quant.input_scale_data->scales_[quant.input_scale_data->count_ == 1 ? 0 : cb * blk_size + c];
                                                float ish = quant.input_shift_data->shifts_[quant.input_shift_data->count_ == 1 ? 0 : cb * blk_size + c];
                                                float osc = quant.output_scale_data->scales_[quant.output_scale_data->count_ == 1 ? 0 : cb * blk_size + c];
                                                float osh = quant.output_shift_data->shifts_[quant.output_shift_data->count_ == 1 ? 0 : cb * blk_size + c];

                                                dst_value = nstl::min(crh, nstl::max(crl, dst_value));
                                                dst_value = dst_value * isc + ish;

                                                if (do_rounding) {
                                                    dst_value = roundf(dst_value);
                                                }

                                                if (do_dequantization) {
                                                    dst_value = dst_value * osc + osh;
                                                }
                                            }
                                        }
                                    }
                                    if (output_prec == Precision::FP32) {
                                        dst_data[ch + w * src_stride] = dst_value;
                                    } else if (output_prec == Precision::U8) {
                                        dst_data[ch + w * src_stride] = (dst_value >= 0) ? lroundf(dst_value) : 0;
                                    } else if (output_prec == Precision::I8) {
                                        dst_data[ch + w * src_stride] = lroundf(dst_value);
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                tail_cb_start = 0;
                if (mvn_kernel) {
                    tail_cb_start = CB;

                    parallel_for2d(D, H, [&](size_t d, size_t h) {
                        for (size_t cb = 0; cb < CB; cb++) {
                            size_t src_off = is_nhwc ? ccb + d * H * W * C + h * W * C + cb * blk_size
                                                     : ccb + d * H * W * blk_size + h * W * blk_size + cb * D * H * W * blk_size;
                            auto mean_buffer_ptr = &mean_buffer[blk_size * cb];

                            auto arg = jit_mvn_call_args();
                            arg.src = src_data + src_off;
                            arg.dst = dst_data + src_off;
                            arg.mean = mean_buffer_ptr;
                            arg.src_stride = src_stride * src_data_size;
                            arg.dst_stride = src_stride * dst_data_size;
                            arg.work_amount = static_cast<size_t>(W);
                            (*mvn_kernel)(&arg);
                        }
                    });
                }

                for (size_t cb = tail_cb_start; cb < tail_cb_end; cb++) {
                    size_t src_off = is_nhwc ? ccb + cb * blk_size : ccb + cb * C2;
                    size_t min_cb = (std::min)(blk_size, C - cb * blk_size);
                    auto mean_buffer_ptr = &mean_buffer[blk_size * cb];

                    for (size_t c = 0lu; c < min_cb; c++) {
                        size_t cc = src_off + c;

                        for (size_t d = 0lu; d < D; d++) {
                            size_t cd = cc + d * C1;
                            for (size_t h = 0lu; h < H; h++) {
                                size_t ch = cd + h * C0;
                                for (size_t w = 0lu; w < W; w++) {
                                    float dst_value = src_data[ch + w * src_stride] - mean_buffer_ptr[c];
                                    if (output_prec == Precision::FP32) {
                                        dst_data[ch + w * src_stride] = dst_value;
                                    } else if (output_prec == Precision::U8) {
                                        dst_data[ch + w * src_stride] = (dst_value >= 0) ? lroundf(dst_value) : 0;
                                    } else if (output_prec == Precision::I8) {
                                        dst_data[ch + w * src_stride] = lroundf(dst_value);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

bool MKLDNNMVNNode::created() const {
    return getType() == MVN;
}

REG_MKLDNN_PRIM_FOR(MKLDNNMVNNode, MVN);
