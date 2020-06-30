// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_resample_node.h"
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
#include "common/simple_copy.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu;
using namespace mkldnn::impl::utils;
using namespace Xbyak;


#define GET_OFF(field) offsetof(jit_resample_call_args, field)

template <cpu_isa_t isa>
struct jit_uni_resample_nearest_kernel_f32 : public jit_uni_resample_nearest_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_resample_nearest_kernel_f32)

    explicit jit_uni_resample_nearest_kernel_f32(jit_resample_config_params jcp, const mkldnn_primitive_attr &attr)
            : jit_uni_resample_nearest_kernel(jcp, attr), jit_generator() {
        const auto &p = attr_.post_ops_;
        for (int i = 0; i < p.len_; i++) {
            auto &post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors.push_back(std::make_shared<jit_uni_eltwise_injector_f32<isa>>(
                        this,
                        post_op.eltwise.alg,
                        post_op.eltwise.alpha,
                        post_op.eltwise.beta));
            } else if (post_op.is_depthwise()) {
                depthwise_injectors.push_back(std::make_shared<jit_uni_depthwise_injector_f32<isa>>(
                        this,
                        post_op.depthwise.alg));
            } else if (post_op.is_quantization()) {
                quantization_injectors.push_back(std::make_shared<jit_uni_quantization_injector_f32<isa>>(
                        this, post_op, vmm_d_weights, vmm_d_bias, reg_d_weights, reg_d_bias));
            }
        }

        this->preamble();

        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_index, ptr[reg_params + GET_OFF(index)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);
        mov(reg_src_stride, ptr[reg_params + GET_OFF(src_stride)]);
        mov(reg_index_stride, ptr[reg_params + GET_OFF(index_stride)]);
        mov(reg_dst_stride, ptr[reg_params + GET_OFF(dst_stride)]);
        if (attr_.post_ops_.len_ != 0)
            mov(reg_oc_off, ptr[reg_params + GET_OFF(oc_off)]);

        if (isa == cpu::avx512_common)
            uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        int blk_size = vlen / sizeof(float);
        if (isa == cpu::sse42)
            blk_size *= 2;

        Xbyak::Label resample_nearest_loop_label;
        Xbyak::Label resample_nearest_loop_end_label;
        L(resample_nearest_loop_label);
        {
            cmp(reg_work_amount, 0);
            jle(resample_nearest_loop_end_label, T_NEAR);

            if (jcp_.planar_layout) {
                uni_vmovdqu(vmm_index, ptr[reg_index]);
                uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
                vgatherdps(vmm_val, ptr[reg_src + vmm_index * jcp.src_data_size], vmm_mask);
                store_vector(ptr[reg_dst], vmm_val, jcp_.dst_dt);

                add(reg_dst, reg_dst_stride);
                add(reg_index, reg_index_stride);
                sub(reg_work_amount, 1);
            } else if (jcp_.nhwc_format) {  // support int8 and fusion for this format
                load_vector(vmm_val, ptr[reg_src], jcp_.src_dt);
                if (attr_.post_ops_.len_ != 0)
                    apply_post_ops(jcp_.dst_dt);
                store_vector(ptr[reg_dst], vmm_val, jcp_.dst_dt);

                if (isa == cpu::sse42) {
                    int sse42_offset = 4;
                    load_vector(vmm_val, ptr[reg_src + sse42_offset * jcp_.src_data_size], jcp_.src_dt);
                    if (attr_.post_ops_.len_ != 0) {
                        add(reg_oc_off, sse42_offset * sizeof(float));
                        apply_post_ops(jcp_.dst_dt);
                        sub(reg_oc_off, sse42_offset * sizeof(float));
                    }
                    store_vector(ptr[reg_dst + sse42_offset * jcp_.dst_data_size], vmm_val, jcp_.dst_dt);
                }

                add(reg_dst, reg_dst_stride);
                add(reg_src, reg_src_stride);
                add(reg_oc_off, blk_size * sizeof(float));
                sub(reg_work_amount, 1);
            } else {  // for blk
                mov(reg_src_aux, reg_src);
                mov(reg_index_oc, dword[reg_index]);
                add(reg_src_aux, reg_index_oc);

                load_vector(vmm_val, ptr[reg_src_aux], jcp_.src_dt);
                if (attr_.post_ops_.len_ != 0)
                    apply_post_ops(jcp_.dst_dt);
                store_vector(ptr[reg_dst], vmm_val, jcp_.dst_dt);

                if (isa == cpu::sse42) {
                    int sse42_offset = 4;
                    add(reg_src_aux, sse42_offset * jcp_.src_data_size);
                    load_vector(vmm_val, ptr[reg_src_aux], jcp_.src_dt);
                    if (attr_.post_ops_.len_ != 0) {
                        add(reg_oc_off, sse42_offset * sizeof(float));
                        apply_post_ops(jcp_.dst_dt);
                        sub(reg_oc_off, sse42_offset * sizeof(float));
                    }
                    store_vector(ptr[reg_dst + sse42_offset * jcp_.dst_data_size], vmm_val, jcp_.dst_dt);
                }

                add(reg_dst, reg_dst_stride);
                add(reg_index, reg_index_stride);
                sub(reg_work_amount, 1);
            }

            jmp(resample_nearest_loop_label, T_NEAR);
        }
        L(resample_nearest_loop_end_label);

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
    Xbyak::Reg64 reg_dst = r9;
    Xbyak::Reg64 reg_src_stride = r10;
    Xbyak::Reg64 reg_dst_stride = r11;
    Xbyak::Reg64 reg_index_stride = r12;
    Xbyak::Reg64 reg_work_amount = r13;
    Xbyak::Reg64 reg_index = r14;
    Xbyak::Reg64 reg_src_aux = r15;
    Xbyak::Reg64 reg_params = abi_param1;

    Xbyak::Reg64 reg_oc_off = rax;
    Xbyak::Reg64 reg_d_weights = rbx;
    Xbyak::Reg64 reg_d_bias = rcx;
    Xbyak::Reg32 reg_index_oc = edx;

    Vmm vmm_val = Vmm(0);
    Vmm vmm_index = Vmm(1);
    Vmm vmm_zero = Vmm(2);
    Vmm vmm_mask = Vmm(3);
    Vmm vmm_d_weights = Vmm(4);
    Vmm vmm_d_bias = Vmm(5);

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


MKLDNNResampleNode::MKLDNNResampleNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache) {}

void MKLDNNResampleNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();

    auto *layer = getCnnLayer().get();
    type = layer->GetParamAsString("type");
    antialias = layer->GetParamAsBool("antialias", false);
    factor = layer->GetParamAsFloat("factor");
}

void MKLDNNResampleNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    if (getParentEdgeAt(0)->getDims().ndims() < 4 || getParentEdgeAt(0)->getDims().ndims() > 5) {
        return;
    }

    setPostOps(attr, true);

    Precision inputPrecision = getCnnLayer()->insData[0].lock()->getPrecision();
    Precision outputPrecision = getCnnLayer()->outData[0]->getPrecision();

    if (!fusedWith.empty()) {
        auto lastFusedLayer = fusedWith[fusedWith.size() - 1].get()->getCnnLayer();
        if (lastFusedLayer) {
            outputPrecision = lastFusedLayer->outData[0]->getPrecision();
        }
    }

    if (inputPrecision == Precision::BF16) {
        inputPrecision = Precision::FP32;
    }

    if (outputPrecision == Precision::BF16) {
        outputPrecision = Precision::FP32;
    }

    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(inputPrecision);
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(outputPrecision);

    input_prec = inputPrecision;
    output_prec = outputPrecision;
    src_data_size = MKLDNNExtensionUtils::sizeOfDataType(inputDataType);
    dst_data_size = MKLDNNExtensionUtils::sizeOfDataType(outputDataType);

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = false;
    config.inConfs.resize(1);
    config.outConfs.resize(1);
    config.inConfs[0].constant = false;
    config.outConfs[0].constant = false;
    config.inConfs[0].inPlace = -1;
    config.outConfs[0].inPlace = -1;

    auto pushDesc = [&](memory::format format) {
        config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, format);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, format);
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, format});
    };

    if (type == "caffe.ResampleParameter.NEAREST") {
        if (getParentEdgeAt(0)->getDims().ndims() == 4) {
            pushDesc(memory::nhwc);
        } else if (getParentEdgeAt(0)->getDims().ndims() == 5) {
            pushDesc(memory::ndhwc);
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
                pushDesc(MKLDNNMemory::GetPlainFormat(getChildEdgeAt(0)->getDims()));
            }
        }
    }
    if (type == "caffe.ResampleParameter.LINEAR") {
        if (getParentEdgeAt(0)->getDims().ndims() == 4) {
            pushDesc(memory::nchw);
        } else if (getParentEdgeAt(0)->getDims().ndims() == 5) {
            pushDesc(memory::ncdhw);
        }
    }
}

void MKLDNNResampleNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";

    auto selectedPD = getSelectedPrimitiveDescriptor();
    Layout selected_layout = selectedPD->getConfig().inConfs[0].desc.getLayout();
    auto jcp = jit_resample_config_params();
    jcp.src_dt = MKLDNNExtensionUtils::IEPrecisionToDataType(selectedPD->getConfig().inConfs[0].desc.getPrecision());
    jcp.dst_dt = MKLDNNExtensionUtils::IEPrecisionToDataType(selectedPD->getConfig().outConfs[0].desc.getPrecision());
    jcp.src_data_size = MKLDNNExtensionUtils::sizeOfDataType(jcp.src_dt);
    jcp.dst_data_size = MKLDNNExtensionUtils::sizeOfDataType(jcp.dst_dt);
    jcp.planar_layout = MKLDNNMemory::GetPlainLayout(getChildEdgeAt(0)->getDims()) == selected_layout;
    jcp.nhwc_format = (selected_layout == NHWC) || (selected_layout == NDHWC);

    if (type == "caffe.ResampleParameter.NEAREST") {
        if (mayiuse(cpu::avx512_common)) {
            if (jcp.planar_layout) {
                resample_nearest_kernel.reset(new jit_uni_resample_nearest_kernel_f32<cpu::avx2>(jcp, *attr.get()));
                blk_size = 8;
            } else {
                resample_nearest_kernel.reset(new jit_uni_resample_nearest_kernel_f32<cpu::avx512_common>(jcp, *attr.get()));
                blk_size = 16;
            }
        } else if (mayiuse(cpu::avx2)) {
            resample_nearest_kernel.reset(new jit_uni_resample_nearest_kernel_f32<cpu::avx2>(jcp, *attr.get()));
            blk_size = 8;
        } else if (mayiuse(cpu::sse42) && !jcp.planar_layout) {
            resample_nearest_kernel.reset(new jit_uni_resample_nearest_kernel_f32<cpu::sse42>(jcp, *attr.get()));
            blk_size = 8;
        }
    }
}

void MKLDNNResampleNode::setPostOps(mkldnn::primitive_attr &attr, bool initWeights) {
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
                PostOpsIntBlobMemory[blob_idx]->FillZero();

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
                    PostOpsIntBlobMemory[blob_idx + 1]->FillZero();
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


void MKLDNNResampleNode::execute(mkldnn::stream strm) {
    auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto &srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();

    Layout layout = getParentEdgeAt(0)->getDesc().getLayout();

    const auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetData());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetData());

    SizeVector src_dim = getParentEdgeAt(0)->getDesc().getDims();
    SizeVector dst_dim = getChildEdgeAt(0)->getDesc().getDims();

    size_t dims_size = src_dim.size();
    size_t N = src_dim[0];
    size_t C = src_dim[1];
    size_t ID = (dims_size == 5) ? src_dim[dims_size - 3] : 1lu;
    size_t IH = src_dim[dims_size - 2];
    size_t IW = src_dim[dims_size - 1];

    size_t OD = (dims_size == 5) ? dst_dim[dims_size - 3] : 1lu;
    size_t OH = dst_dim[dims_size - 2];
    size_t OW = dst_dim[dims_size - 1];

    float fx = static_cast<float>(IW) / static_cast<float>(OW);
    float fy = static_cast<float>(IH) / static_cast<float>(OH);
    float fz = static_cast<float>(ID) / static_cast<float>(OD);

    if (type == "caffe.ResampleParameter.NEAREST") {
        if (layout == NCHW || layout == NCDHW) {
            NearestNeighbor_PLN(src_data, dst_data, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW);
        } else {
            if (output_prec == Precision::U8) {
                auto dst_data = reinterpret_cast<uint8_t *>(dstMemPtr->GetData());
                if (input_prec == Precision::U8) {
                    auto src_data = reinterpret_cast<const uint8_t *>(srcMemPtr->GetData());
                    NearestNeighbor_BLK<uint8_t, uint8_t>(src_data, dst_data, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW);
                } else if (input_prec == Precision::I8) {
                    auto src_data = reinterpret_cast<const int8_t *>(srcMemPtr->GetData());
                    NearestNeighbor_BLK<int8_t, uint8_t>(src_data, dst_data, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW);
                } else if (input_prec == Precision::FP32) {
                    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetData());
                    NearestNeighbor_BLK<float, uint8_t>(src_data, dst_data, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW);
                }
            } else if (output_prec == Precision::I8) {
                auto dst_data = reinterpret_cast<int8_t *>(dstMemPtr->GetData());
                if (input_prec == Precision::U8) {
                    auto src_data = reinterpret_cast<const uint8_t *>(srcMemPtr->GetData());
                    NearestNeighbor_BLK<uint8_t, int8_t>(src_data, dst_data, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW);
                } else if (input_prec == Precision::I8) {
                    auto src_data = reinterpret_cast<const int8_t *>(srcMemPtr->GetData());
                    NearestNeighbor_BLK<int8_t, int8_t>(src_data, dst_data, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW);
                } else if (input_prec == Precision::FP32) {
                    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetData());
                    NearestNeighbor_BLK<float, int8_t>(src_data, dst_data, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW);
                }
            } else if (output_prec == Precision::FP32) {
                auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetData());
                if (input_prec == Precision::U8) {
                    auto src_data = reinterpret_cast<const uint8_t *>(srcMemPtr->GetData());
                    NearestNeighbor_BLK<uint8_t, float>(src_data, dst_data, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW);
                } else if (input_prec == Precision::I8) {
                    auto src_data = reinterpret_cast<const int8_t *>(srcMemPtr->GetData());
                    NearestNeighbor_BLK<int8_t, float>(src_data, dst_data, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW);
                } else if (input_prec == Precision::FP32) {
                    auto src_data = reinterpret_cast<float *>(srcMemPtr->GetData());
                    NearestNeighbor_BLK<float, float>(src_data, dst_data, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW);
                }
            }
        }
    } else if (type == "caffe.ResampleParameter.LINEAR") {
        // currently no fusion, the input and output precision is the same
        bool isDownsample = (fx > 1) || (fy > 1) || (fz > 1);
        int kernel_width = 2;
        if (input_prec == Precision::U8) {
            auto src_data = reinterpret_cast<const uint8_t *>(srcMemPtr->GetData());
            auto dst_data = reinterpret_cast<uint8_t *>(dstMemPtr->GetData());
            LinearInterpolation<uint8_t, uint8_t>(src_data, dst_data, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW, kernel_width, isDownsample && antialias);
        } else if (input_prec == Precision::I8) {
            auto src_data = reinterpret_cast<const int8_t *>(srcMemPtr->GetData());
            auto dst_data = reinterpret_cast<int8_t *>(dstMemPtr->GetData());
            LinearInterpolation<int8_t, int8_t>(src_data, dst_data, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW, kernel_width, isDownsample && antialias);
        } else if (input_prec == Precision::FP32) {
            auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetData());
            auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetData());
            LinearInterpolation<float, float>(src_data, dst_data, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW, kernel_width, isDownsample && antialias);
        }
    }
}

// f32 and no fused, f32->input is f32, no fuse->output is f32
void MKLDNNResampleNode::NearestNeighbor_PLN(const float *in_ptr_, float *out_ptr_, int B, int C, int ID, int IH, int IW,
                                             float fx, float fy, float fz, int OD, int OH, int OW) {
    std::vector<int> index_buffer(OD * OH * OW);
    for (int oz = 0; oz < OD; oz++) {
        float iz = oz * fz;
        int iz_offset = static_cast<int>(std::floor(iz)) * IH * IW;
        int oz_offset = oz * OH * OW;
        for (int oy = 0; oy < OH; oy++) {
            float iy = oy * fy;
            int iy_offset = static_cast<int>(std::floor(iy)) * IW + iz_offset;
            int oy_offset = oy * OW + oz_offset;
            for (int ox = 0; ox < OW; ox++) {
                float ix = ox * fx;
                int ix_index = static_cast<int>(std::floor(ix)) + iy_offset;
                index_buffer[oy_offset + ox] = ix_index;
            }
        }
    }
    if (resample_nearest_kernel) {
        parallel_for2d(B, C, [&](size_t b, size_t c) {
            const float *in_ptr = in_ptr_ + IW * IH * ID * C * b + IW * IH * ID * c;
            float *out_ptr = out_ptr_ + OW * OH * OD * C * b + OW * OH * OD * c;

            // for OW*OH*OD
            auto arg = jit_resample_call_args();
            arg.src = in_ptr;
            arg.dst = out_ptr;
            arg.index = static_cast<int*>(&index_buffer[0]);
            arg.index_stride = blk_size * sizeof(int);
            arg.dst_stride = blk_size * dst_data_size;
            arg.work_amount = OW * OH * OD / blk_size;
            (*resample_nearest_kernel)(&arg);

            int tail_start = (OW * OH * OD / blk_size) * blk_size;
            for (int tail = tail_start; tail < OW * OH * OD; tail++) {
                out_ptr[tail] = in_ptr[index_buffer[tail]];
            }
        });
    } else {
        parallel_for2d(B, C, [&](size_t b, size_t c) {
            const float *in_ptr = in_ptr_ + IW * IH * ID * C * b + IW * IH * ID * c;
            float *out_ptr = out_ptr_ + OW * OH * OD * C * b + OW * OH * OD * c;

            for (int i_dst = 0; i_dst < OW * OH * OD; i_dst++) {
                out_ptr[i_dst] = in_ptr[index_buffer[i_dst]];
            }
        });
    }
}

// for ndhwc and nCdhw8/16d
// int8->input may be int8, fused->output may be int8
template <typename in_data_t, typename out_data_t>
void MKLDNNResampleNode::NearestNeighbor_BLK(const in_data_t *in_ptr_, out_data_t *out_ptr_, int B, int C, int ID, int IH, int IW,
                                             float fx, float fy, float fz, int OD, int OH, int OW) {
    std::vector<int> index_d(OD);
    std::vector<int> index_h(OH);
    std::vector<int> index_w(OW);
    for (int oz = 0; oz < OD; oz++) {
        float iz = oz * fz;
        index_d[oz] = static_cast<int>(std::floor(iz));
    }
    for (int oy = 0; oy < OH; oy++) {
        float iy = oy * fy;
        index_h[oy] = static_cast<int>(std::floor(iy));
    }
    for (int ox = 0; ox < OW; ox++) {
        float ix = ox * fx;
        index_w[ox] = static_cast<int>(std::floor(ix));
    }

    Layout layout = getParentEdgeAt(0)->getDesc().getLayout();
    bool is_nhwc = (layout == NHWC || layout == NDHWC) ? true : false;

    for (int b = 0; b < B; b++) {
        if (is_nhwc) {
            const in_data_t *in_ptr = in_ptr_ + IW * IH * ID * C * b;
            out_data_t *out_ptr = out_ptr_ + OW * OH * OD * C * b;
            if (resample_nearest_kernel) {
                int tail = (C / blk_size) * blk_size;
                parallel_for2d(OD, OH, [&](size_t d, size_t h) {
                    // better that same core process continuous memory
                    out_data_t *out_ptr_dh = out_ptr + C * OW * OH * d + C * OW * h;
                    const in_data_t *in_ptr_dh = in_ptr + C * IW * IH * index_d[d] + C * IW * index_h[h];
                    auto arg = jit_resample_call_args();
                    for (int ox = 0; ox < OW; ox++) {
                        // kernel for OC
                        arg.dst = out_ptr_dh + C * ox;
                        arg.src = in_ptr_dh + C * index_w[ox];
                        arg.dst_stride = blk_size * sizeof(out_data_t);
                        arg.src_stride = blk_size * sizeof(in_data_t);
                        arg.work_amount = C / blk_size;
                        arg.oc_off = 0;
                        (*resample_nearest_kernel)(&arg);
                    }
                    // tail
                    if (tail != C) {
                        for (int ox = 0; ox < OW; ox++) {
                            out_data_t *out_ptr_dhw = out_ptr_dh + C * ox;
                            const in_data_t *in_ptr_dhw = in_ptr_dh + C * index_w[ox];
                            if (fusedWith.empty() && output_prec == input_prec) {
                                memcpy(out_ptr_dhw + tail, in_ptr_dhw + tail, (C - tail) * sizeof(in_data_t));
                            } else {
                                for (int c = tail; c < C; c++) {
                                    float dst_value = static_cast<float>(in_ptr_dhw[c]);
                                    apply_post_ops_scalar(dst_value, c);
                                    if (output_prec == Precision::FP32) {
                                        out_ptr_dhw[c] = dst_value;
                                    } else if (output_prec == Precision::U8) {
                                        out_ptr_dhw[c] = (dst_value >= 0) ? lroundf(dst_value) : 0;
                                    } else if (output_prec == Precision::I8) {
                                        out_ptr_dhw[c] = lroundf(dst_value);
                                    }
                                }
                            }
                        }
                    }
                });
            } else {  // without kernel
                parallel_for2d(OD, OH, [&](size_t d, size_t h) {
                    out_data_t *out_ptr_dh = out_ptr + C * OW * OH * d + C * OW * h;
                    const in_data_t *in_ptr_dh = in_ptr + C * IW * IH * index_d[d] + C * IW * index_h[h];
                    for (int ox = 0; ox < OW; ox++) {
                        out_data_t *out_ptr_dhw = out_ptr_dh + C * ox;
                        const in_data_t *in_ptr_dhw = in_ptr_dh + C * index_w[ox];
                        if (fusedWith.empty() && output_prec == input_prec) {
                            memcpy(out_ptr_dhw, in_ptr_dhw, C * sizeof(in_data_t));
                        } else {
                            for (int c = 0; c < C; c++) {
                                float dst_value = static_cast<float>(in_ptr_dhw[c]);
                                apply_post_ops_scalar(dst_value, c);
                                if (output_prec == Precision::FP32) {
                                    out_ptr_dhw[c] = dst_value;
                                } else if (output_prec == Precision::U8) {
                                    out_ptr_dhw[c] = (dst_value >= 0) ? lroundf(dst_value) : 0;
                                } else if (output_prec == Precision::I8) {
                                    out_ptr_dhw[c] = lroundf(dst_value);
                                }
                            }
                        }
                    }
                });
            }
        } else {  // for nC(d)hw8/16c
            int CB = div_up(C, blk_size);
            const in_data_t *in_ptr = in_ptr_ + IW * IH * ID * CB * blk_size * b;
            out_data_t *out_ptr = out_ptr_ + OW * OH * OD * CB * blk_size * b;
            if (resample_nearest_kernel) {
                std::vector<int> index_w_kernel(OW);
                for (int ox = 0; ox < OW; ox++) {
                    index_w_kernel[ox] = index_w[ox] * blk_size * sizeof(in_data_t);
                }
                parallel_for2d(CB, OD, [&](size_t cb, size_t d) {
                    out_data_t *out_ptr_cbd = out_ptr + blk_size * OW * OH * OD * cb + blk_size * OW * OH * d;
                    const in_data_t *in_ptr_cbd = in_ptr +  blk_size * IW * IH * ID * cb + blk_size * IW * IH * index_d[d];
                    auto arg = jit_resample_call_args();
                    for (int h = 0; h < OH; h++) {  // kernel for blk_size * OW
                        arg.dst = out_ptr_cbd + blk_size * OW * h;
                        arg.src = in_ptr_cbd + blk_size * IW * index_h[h];
                        arg.index = static_cast<int*>(&(index_w_kernel[0]));
                        arg.dst_stride = static_cast<size_t>(blk_size * sizeof(out_data_t));
                        arg.index_stride = static_cast<size_t>(1 * sizeof(int));
                        arg.work_amount = static_cast<size_t>(OW);
                        arg.oc_off = cb * blk_size;
                        (*resample_nearest_kernel)(&arg);
                    }
                });
            } else {
                parallel_for2d(CB, OD, [&](int cb, int d) {
                    out_data_t *out_ptr_cbd = out_ptr + blk_size * OW * OH * OD * cb + blk_size * OW * OH * d;
                    const in_data_t *in_ptr_cbd = in_ptr +  blk_size * IW * IH * ID * cb + blk_size * IW * IH * index_d[d];
                    for (int h = 0; h < OH; h++) {
                        out_data_t *out_ptr_cbdh = out_ptr_cbd + blk_size * OW * h;
                        const in_data_t *in_ptr_cbdh = in_ptr_cbd + blk_size * IW * index_h[h];
                        for (int w = 0; w < OW; w++) {
                            out_data_t *out_ptr_cbdhw = out_ptr_cbdh + blk_size * w;
                            const in_data_t *in_ptr_cbdhw = in_ptr_cbdh + blk_size * index_w[w];
                            if (fusedWith.empty()) {
                                memcpy(out_ptr_cbdhw, in_ptr_cbdhw, blk_size * sizeof(in_data_t));
                            } else {
                                for (int blk = 0; blk < blk_size; blk++) {
                                    float dst_value = static_cast<float>(in_ptr_cbdhw[blk]);
                                    apply_post_ops_scalar(dst_value, cb * blk_size + blk);
                                    if (output_prec == Precision::FP32) {
                                        out_ptr_cbdhw[blk] = dst_value;
                                    } else if (output_prec == Precision::U8) {
                                        out_ptr_cbdhw[blk] = (dst_value >= 0) ? lroundf(dst_value) : 0;
                                    } else if (output_prec == Precision::I8) {
                                        out_ptr_cbdhw[blk] = lroundf(dst_value);
                                    }
                                }
                            }
                        }
                    }
                });
            }
        }
    }  // batch end
}

static inline float triangleCoeff(float x) {
    return (std::max)(0.0f, 1 - std::abs(x));
}

template <typename in_data_t, typename out_data_t>
void MKLDNNResampleNode::LinearInterpolation(const in_data_t *in_ptr_, out_data_t *out_ptr_, int B, int C, int ID, int IH, int IW,
                                             float fx, float fy, float fz, int OD, int OH, int OW, int kernel_width, bool antialias) {
    if (IW == OW && IH == OH && ID == OD) {
        size_t size = B * C * ID * IH * IW;
        if (input_prec == Precision::FP32) {
            size *= sizeof(float);
        }
        simple_copy(out_ptr_, size, in_ptr_, size);
        return;
    }

    for (size_t b = 0; b < B; b++) {
        const in_data_t *in_ptr_n = in_ptr_ + IW * IH * ID * C * b;
        out_data_t *out_ptr_n = out_ptr_ + OW * OH * OD * C * b;
        for (size_t c = 0; c < C; c++) {
            const in_data_t *in_ptr_nc = in_ptr_n + IW * IH * ID * c;
            out_data_t *out_ptr_nc = out_ptr_n + OW * OH * OD * c;

            for (size_t oz = 0; oz < OD; oz++) {
                out_data_t *out_ptr_ncd = out_ptr_nc + OW * OH * oz;
                for (size_t oy = 0; oy < OH; oy++) {
                    out_data_t *out_ptr_ncdh = out_ptr_ncd + OW * oy;
                    for (size_t ox = 0; ox < OW; ox++) {
                        float ix = ox * fx + fx / 2.0f - 0.5f;
                        float iy = oy * fy + fy / 2.0f - 0.5f;
                        float iz = oz * fz + fz / 2.0f - 0.5f;

                        int ix_r = static_cast<int>(round(ix));
                        int iy_r = static_cast<int>(round(iy));
                        int iz_r = static_cast<int>(round(iz));

                        float sum = 0;
                        float wsum = 0;

                        float ax = 1.0f / (antialias ? fx : 1.0f);
                        float ay = 1.0f / (antialias ? fy : 1.0f);
                        float az = 1.0f / (antialias ? fz : 1.0f);

                        int rx = (fx < 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / ax));
                        int ry = (fy < 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / ay));
                        int rz = (fz < 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / az));

                        for (int z = iz_r - rz; z <= iz_r + rz; z++) {
                            for (int y = iy_r - ry; y <= iy_r + ry; y++) {
                                for (int x = ix_r - rx; x <= ix_r + rx; x++) {
                                    bool is_continue =  z < 0                     ||
                                                        y < 0                     ||
                                                        x < 0                     ||
                                                        z >= static_cast<int>(ID) ||
                                                        y >= static_cast<int>(IH) ||
                                                        x >= static_cast<int>(IW);
                                    if (is_continue)
                                        continue;

                                    float dx = ix - x;
                                    float dy = iy - y;
                                    float dz = iz - z;

                                    float w = ax * triangleCoeff(ax * dx) *
                                              ay * triangleCoeff(ay * dy) *
                                              az * triangleCoeff(az * dz);

                                    sum += w * static_cast<float>(in_ptr_nc[z * IH * IW + y * IW + x]);
                                    wsum += w;
                                }
                            }
                        }
                        if (!wsum) {
                            out_ptr_ncdh[ox] = 0;
                        } else {
                            float dst_value = sum / wsum;
                            if (output_prec == Precision::FP32) {
                                out_ptr_ncdh[ox] = dst_value;
                            } else if (output_prec == Precision::U8) {
                                out_ptr_ncdh[ox] = (dst_value >= 0) ? lroundf(dst_value) : 0;
                            } else if (output_prec == Precision::I8) {
                                out_ptr_ncdh[ox] = lroundf(dst_value);
                            }
                        }
                    }
                }
            }
        }
    }
}

inline void MKLDNNResampleNode::apply_post_ops_scalar(float &dst_value, int index_c) {
    const auto &p = (*attr.get()).post_ops_;
    for (int i = 0; i < p.len_; i++) {
        auto &post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            //  only eltwise_relu supported
            if (dst_value < 0) dst_value = 0;
        } else if (post_op.is_depthwise()) {
            //  only ScaleShift supported
            float scale = post_op.depthwise.weights_data[index_c];
            float shift = post_op.depthwise.biases_data[index_c];
            dst_value = dst_value * scale + shift;
        } else if (post_op.is_quantization()) {
            bool do_dequantization = post_op.quantization.alg ==
                                     alg_kind::quantization_quantize_dequantize;
            bool do_rounding = do_dequantization || output_prec == Precision::FP32 ||
                               i != p.len_ - 1;

            auto quant = post_op.quantization;

            float crop_low = quant.crop_low_data->shifts_[quant.crop_low_data->count_ == 1 ? 0 : index_c];
            float crop_high = quant.crop_high_data->shifts_[quant.crop_high_data->count_ == 1 ? 0 : index_c];
            float input_scale = quant.input_scale_data->scales_[quant.input_scale_data->count_ == 1 ? 0 : index_c];
            float input_shift = quant.input_shift_data->shifts_[quant.input_shift_data->count_ == 1 ? 0 : index_c];

            dst_value = nstl::min(crop_high, nstl::max(crop_low, dst_value));
            dst_value = dst_value * input_scale + input_shift;

            if (do_rounding) {
                dst_value = roundf(dst_value);
            }

            if (do_dequantization) {
                float output_scale = quant.output_scale_data->scales_[quant.output_scale_data->count_ == 1 ? 0 : index_c];
                float output_shift = quant.output_shift_data->shifts_[quant.output_shift_data->count_ == 1 ? 0 : index_c];
                dst_value = dst_value * output_scale + output_shift;
            }
        }
    }
}

bool MKLDNNResampleNode::created() const {
    return getType() == Resample;
}

REG_MKLDNN_PRIM_FOR(MKLDNNResampleNode, Resample);