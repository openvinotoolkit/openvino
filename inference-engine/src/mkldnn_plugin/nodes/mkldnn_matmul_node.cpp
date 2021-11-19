// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_matmul_node.h"

#include "ie_precision.hpp"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "cpu_types.h"
#include "mkldnn_eltwise_node.h"

#include <numeric>
#include <string>
#include <vector>
#include <memory>
#include <mkldnn_types.h>
#include "common/cpu_memcpy.h"
#include <ngraph/opsets/opset1.hpp>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "utils/general_utils.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "mkldnn_extension_utils.h"
#include "utils/cpu_utils.hpp"

#include "emitters/jit_emitter.hpp"
#include "emitters/jit_eltwise_emitters.hpp"
#include "emitters/jit_mkldnn_emitters.hpp"
#include "emitters/jit_load_store_emitters.hpp"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn;
using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::cpu;
using namespace mkldnn::impl::cpu::x64;
using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_matmul_args, field)


template <cpu_isa_t isa>
struct jit_uni_matmul_kernel_f32 : public jit_uni_matmul_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_matmul_kernel_f32)

    explicit jit_uni_matmul_kernel_f32(jit_matmul_config_params jcp_, const mkldnn_primitive_attr &attr) : jit_uni_matmul_kernel(jcp_, attr), jit_generator() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        const auto &p = attr_.post_ops_;
        for (int i = 0; i < p.len(); i++) {
            auto &post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors.push_back(std::make_shared<jit_uni_eltwise_injector_f32<isa>>(
                        this,
                        post_op.eltwise.alg,
                        post_op.eltwise.alpha,
                        post_op.eltwise.beta,
                        1));
            } else {
                IE_THROW() << "MatMul supports only eltwise post ops!";
            }
        }

        load_emitter.reset(new jit_load_emitter(this, isa, nullptr));
        store_emitter.reset(new jit_store_emitter(this, isa, nullptr));

        this->preamble();

        mov(reg_src_0, ptr[reg_params + GET_OFF(src0)]);
        mov(reg_src_1, ptr[reg_params + GET_OFF(src1)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);

        load_pool_gpr_idxs = {static_cast<size_t>(reg_load_store_mask.getIdx()), static_cast<size_t>(reg_load_table.getIdx())};
        store_pool_gpr_idxs = {static_cast<size_t>(reg_load_store_mask.getIdx())};
        store_pool_vec_idxs = {static_cast<size_t>(vmm_zero.getIdx())};

        amount_full = (jcp_.scalar_product ? jcp_.k : jcp_.n) / vec_step;
        amount_tail = (jcp_.scalar_product ? jcp_.k : jcp_.n) % vec_step;

        Xbyak::Label label_batch;
        Xbyak::Label label_batch_end;

        mov(batch, 0);
        L(label_batch); {
            cmp(batch, jcp_.b);
            je(label_batch_end, T_NEAR);

            body();

            // strides for batch
            if (jcp_.stride0 == 0)  // ptr for the first matrix is automatically incremented after each iteration by m,
                sub(reg_src_0, jcp_.stride0 * sizeof(float)); // but if there should be no shift, we return ptr to the previous position
            if (jcp_.stride1 != 0)
                add(reg_src_1, jcp_.stride1 * sizeof(float));

            add(batch, 1);
            jmp(label_batch, T_NEAR);
        }
        L(label_batch_end);

        this->postamble();

        load_emitter->emit_data();
        store_emitter->emit_data();

        for (auto& inj : eltwise_injectors)
            inj->prepare_table();
    }

private:
    using Vmm = typename conditional3<isa == x64::sse41, Xbyak::Xmm, isa == x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    const int vlen = cpu_isa_traits<isa>::vlen;
    const int vec_step = vlen / sizeof(float);
    int amount_full;
    int amount_tail;
    int unroll_dst_count = 1;

    Vmm get_src_vmm(const int idx) {
        return Vmm(1 + unroll_dst_count + idx);
    }

    Vmm get_dst_vmm(const int idx = 0) {
        return Vmm(1 + idx);
    }

    Xmm get_aux_xmm(const int idx) {
        return Xmm(idx);
    }

    Xbyak::Reg64 reg_src_0 = r8;
    Xbyak::Reg64 reg_src_1 = r9;
    Xbyak::Reg64 reg_dst = r10;
    Xbyak::Reg64 reg_src_aux_0 = r11;
    Xbyak::Reg64 reg_src_aux_1 = r12;
    Xbyak::Reg64 reg_dst_aux = r13; // only for 2nd algorithm

    // indexes
    Xbyak::Reg64 batch = r14;
    Xbyak::Reg64 m = r15;
    Xbyak::Reg64 k = rax;
    Xbyak::Reg64 n = rbx;
    Xbyak::Reg64 n_inside = rdx;

    Xbyak::Reg64 reg_params = abi_param1; // RDI | RCX

    // loaders and stores
    Xbyak::Reg64 reg_load_store_mask = rsi;
    Xbyak::Reg64 reg_load_table = rbp;

    Vmm vmm_zero = Vmm(0);

    std::unique_ptr<jit_load_emitter> load_emitter = nullptr;
    std::unique_ptr<jit_store_emitter> store_emitter = nullptr;

    std::vector<size_t> load_pool_gpr_idxs;
    std::vector<size_t> store_pool_gpr_idxs;
    std::vector<size_t> store_pool_vec_idxs;

    std::vector<std::shared_ptr<jit_uni_eltwise_injector_f32<isa>>> eltwise_injectors;

    inline void body() {
        Xbyak::Label label_m;
        Xbyak::Label label_m_end;

        mov(m, 0);
        L(label_m); {
            cmp(m, jcp_.m);
            je(label_m_end, T_NEAR);

            if (jcp_.scalar_product) {
                optimized_body_loop();
            } else {
                body_loop();
            }

            add(reg_src_0, jcp_.k * sizeof(float));
            add(m, 1);
            jmp(label_m, T_NEAR);
        }
        L(label_m_end);
    }

    // common execution : broadcast(a) * vmm_b
    inline void body_loop() {
        Xbyak::Label label_k;
        Xbyak::Label label_k_end;

        unroll_dst_count = amount_full + static_cast<int>(amount_tail != 0);

        mov(reg_src_aux_0, reg_src_0);
        mov(reg_src_aux_1, reg_src_1);
        for (int i = 0; i < unroll_dst_count; ++i)
            uni_vpxor(get_dst_vmm(i), get_dst_vmm(i), get_dst_vmm(i));

        mov(k, 0);
        L(label_k); {
            cmp(k, jcp_.k);
            je(label_k_end, T_NEAR);

            uni_vbroadcastss(get_src_vmm(0), ptr[reg_src_aux_0]);  // src0

            for (int i = 0; i < amount_full; ++i) {
                load_ptr(get_src_vmm(1), ptr[reg_src_aux_1 + i * vlen]);  // src1
                uni_vfmadd231ps(get_dst_vmm(i), get_src_vmm(0), get_src_vmm(1));  // broadcast(a) * vmm_1
            }
            add(reg_src_aux_1, amount_full * vlen);

            if (amount_tail != 0) {
                load(reg_src_aux_1, get_src_vmm(1), amount_tail);
                uni_vfmadd231ps(get_dst_vmm(amount_full), get_src_vmm(0), get_src_vmm(1));
                add(reg_src_aux_1, amount_tail * sizeof(float));
            }

            add(reg_src_aux_0, sizeof(float));
            add(k, 1);
            jmp(label_k, T_NEAR);
        }
        L(label_k_end);

        for (int i = 0; i < amount_full; ++i) {
            apply_post_ops(get_dst_vmm(i).getIdx());
            store_ptr(ptr[reg_dst + i * vlen], get_dst_vmm(i));
        }
        add(reg_dst, amount_full * vlen);

        if (amount_tail != 0) {
            apply_post_ops(get_dst_vmm(amount_full).getIdx());
            store(get_dst_vmm(amount_full), reg_dst, amount_tail);
            add(reg_dst, amount_tail * sizeof(float));
        }
    }

    // optimized execution for cases with transposed matrix b or k = 1
    inline void optimized_body_loop() {
        Xbyak::Label label_n;
        Xbyak::Label label_n_end;

        const int full_n_amount = jcp_.n / vec_step;
        const int tail_n_amount = jcp_.n % vec_step;

        mov(reg_dst_aux, reg_dst);
        mov(reg_src_aux_1, reg_src_1);

        mov(n, 0);
        L(label_n); {
            cmp(n, full_n_amount);
            je(label_n_end, T_NEAR);

            optimized_vectorization_body_loop(vec_step);

            if (attr_.post_ops_.len() != 0) {
                load(reg_dst_aux, get_dst_vmm(0), vec_step);
                apply_post_ops();
                store(get_dst_vmm(0), reg_dst_aux, vec_step);
                add(reg_dst_aux, vlen);
            }

            add(n, 1);
            jmp(label_n, T_NEAR);
        }
        L(label_n_end);

        if (tail_n_amount != 0) {
            optimized_vectorization_body_loop(tail_n_amount);

            if (attr_.post_ops_.len() != 0) {
                load(reg_dst_aux, get_dst_vmm(0), tail_n_amount);
                apply_post_ops();
                store(get_dst_vmm(0), reg_dst_aux, tail_n_amount);
                add(reg_dst_aux, tail_n_amount * sizeof(float));
            }
        }
    }

    inline void optimized_vectorization_body_loop(const int amount) {
        Xbyak::Label label_o;
        Xbyak::Label label_o_end;

        mov(n_inside, 0);
        L(label_o); {
            cmp(n_inside, amount);
            je(label_o_end, T_NEAR);

            mov(reg_src_aux_0, reg_src_0);
            uni_vpxor(get_dst_vmm(), get_dst_vmm(), get_dst_vmm());

            for (int i = 0; i < amount_full; ++i) {
                load_ptr(get_src_vmm(0), ptr[reg_src_aux_0 + i * vlen]);  // src0
                load_ptr(get_src_vmm(1), ptr[reg_src_aux_1 + i * vlen]);  // src1

                uni_vfmadd231ps(get_dst_vmm(), get_src_vmm(0), get_src_vmm(1));
            }
            add(reg_src_aux_1, amount_full * vlen);

            if (amount_tail != 0) {
                add(reg_src_aux_0, amount_full * vlen);
                load(reg_src_aux_0, get_src_vmm(0), amount_tail, true);
                load(reg_src_aux_1, get_src_vmm(1), amount_tail, true);
                uni_vfmadd231ps(get_dst_vmm(), get_src_vmm(0), get_src_vmm(1));

                add(reg_src_aux_1, amount_tail * sizeof(float));
            }

            // hsum
            if (isa == x64::avx512_common) {
                Xbyak::Zmm zmm_dst = Xbyak::Zmm(get_dst_vmm().getIdx());
                vextractf32x4(get_aux_xmm(2), zmm_dst, 0);
                vextractf32x4(get_aux_xmm(3), zmm_dst, 1);
                addps(get_aux_xmm(2), get_aux_xmm(3));
                vextractf32x4(get_aux_xmm(3), zmm_dst, 2);
                vextractf32x4(get_aux_xmm(4), zmm_dst, 3);
                addps(get_aux_xmm(3), get_aux_xmm(4));
                vaddps(get_aux_xmm(1), get_aux_xmm(2), get_aux_xmm(3));
                hsum(get_aux_xmm(1));
            } else if (isa == x64::avx2) {
                Xbyak::Ymm ymm_dst = Xbyak::Ymm(get_dst_vmm().getIdx());
                vextractf128(get_aux_xmm(2), ymm_dst, 0);
                vextractf128(get_aux_xmm(3), ymm_dst, 1);
                vaddps(get_aux_xmm(1), get_aux_xmm(2), get_aux_xmm(3));
                hsum(get_aux_xmm(1));
            } else {
                hsum(get_dst_vmm());
            }

            store(get_dst_vmm(), reg_dst, 1);
            add(reg_dst, sizeof(float));

            add(n_inside, 1);
            jmp(label_o, T_NEAR);
        }
        L(label_o_end);
    }

    inline void hsum(Xbyak::Xmm xmm) {
        movshdup(get_aux_xmm(2), xmm);
        addps(xmm, get_aux_xmm(2));
        movhlps(get_aux_xmm(2), xmm);
        addps(xmm, get_aux_xmm(2));
    }

    inline void apply_post_ops(const int idx = 1) {
        const auto &p = attr_.post_ops_;
        int eltwise_inj_idx = 0;
        for (int i = 0; i < p.len(); i++) {
            auto& post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors[eltwise_inj_idx++]->compute_vector_range(idx, idx + 1);
            }
        }
    }

    inline void load(Xbyak::Reg64 reg, Vmm vmm, int load_num, bool is_fill = false) {
        load_emitter->emit_code({static_cast<size_t>(reg.getIdx())}, {static_cast<size_t>(vmm.getIdx())},
                                std::make_shared<load_emitter_context>(Precision::FP32, Precision::FP32, load_num, 0, is_fill),
                                {}, load_pool_gpr_idxs);
    }

    inline void store(Vmm vmm, Xbyak::Reg64 reg, int load_num) {
        store_emitter->emit_code({static_cast<size_t>(vmm.getIdx())}, {static_cast<size_t>(reg.getIdx())},
                                 std::make_shared<store_emitter_context>(Precision::FP32, Precision::FP32, load_num),
                                 store_pool_vec_idxs, store_pool_gpr_idxs);
    }

    inline void load_ptr(Vmm vmm_src, const Xbyak::Address &op) {
        uni_vmovups(vmm_src, op);
    }

    inline void store_ptr(const Xbyak::Address &op, Vmm vmm_dst) {
        uni_vmovups(op, vmm_dst);
    }
};

bool MKLDNNMatMulNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto matMul = std::dynamic_pointer_cast<const ngraph::opset1::MatMul>(op);
        if (!matMul) {
            errorMessage = "Only opset1 MatMul operation is supported";
            return false;
        }

        for (size_t i = 0; i < matMul->get_input_size(); i++) {
            const auto inShapeRank = matMul->get_input_partial_shape(i).rank().get_length();
            if (inShapeRank < 2) {
                errorMessage = "Unsupported rank: " + std::to_string(inShapeRank) + " on " + std::to_string(i) + " input";
                return false;
            }
        }

        const auto outShapeRank = matMul->get_output_partial_shape(0).rank().get_length();
        if (outShapeRank < 2) {
            errorMessage = "Unsupported rank: " + std::to_string(outShapeRank) + " on output";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNMatMulNode::MKLDNNMatMulNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
    MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage))
        IE_THROW(NotImplemented) << errorMessage;

    errorPrefix = "MatMul node with name '" + getName() + "'";

    const auto matMul = std::dynamic_pointer_cast<const ngraph::opset1::MatMul>(op);

    transposeIn[0] = matMul->get_transpose_a();
    transposeIn[1] = matMul->get_transpose_b();
}

bool MKLDNNMatMulNode::canFuse(const MKLDNNNodePtr& node) const {
    return one_of(node->getAlgorithm(), EltwiseRelu, EltwiseGelu, EltwiseElu, EltwiseSigmoid, EltwiseClamp, EltwiseTanh,
                  EltwiseSwish, EltwiseHswish, EltwiseMish, EltwiseHsigmoid, EltwiseRoundHalfToEven,
                  EltwiseRoundHalfAwayFromZero, EltwiseAbs, EltwiseSqrt, EltwiseSoftRelu);
}

void MKLDNNMatMulNode::setPostOps(mkldnn::primitive_attr &attr, const VectorDims& dims, bool initWeights = false) const {
    mkldnn::post_ops ops;

    for (auto &node : fusedWith) {
        if (auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(node.get())) {
            eltwiseNode->appendPostOps(ops, dims);
            continue;
        }

        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType()) << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

MKLDNNNode::AttrPtr MKLDNNMatMulNode::initPrimitiveAttr(const VectorDims &dims) const {
    auto attr = std::make_shared<mkldnn::primitive_attr>(mkldnn::primitive_attr());

    setPostOps(*attr, dims, true);

    return attr;
}

MKLDNNNode::AttrPtr MKLDNNMatMulNode::initPrimitiveAttr() const {
    auto dummyShape = MemoryDescUtils::makeDummyShape(getOutputShapeAtPort(0));
    return initPrimitiveAttr(dummyShape.getStaticDims());
}

/* Example MatMul:
 * 2x128x512(T) * 2x128x512 = 2x512x512
 * First input 2x128x512(T) should be transposed
 * oneDNN requires memory::desc for this input to:
 * - change shapes configuration as if input already transposed (2x128x512) -> (2x512x128)
 * - provide transposed strides (66536, 128, 1) -> (66536, 1, 512)
 */
static VectorDims getStridesAndModifyShape(Shape& shape, const bool transpose) {
    const auto getRank = shape.getRank();

    VectorDims strides(getRank, 1);
    const auto& staticDims = shape.getStaticDims();
    for (size_t i = 1; i < getRank; i++) {
        strides[getRank - i - 1 ] = strides[getRank - i] * staticDims[getRank - i];
    }

    if (transpose && getRank > 1) {
        // form new shape
        auto dims = staticDims;
        std::swap(dims[getRank - 2], dims[getRank - 1]);
        shape = Shape{dims};
        // update strides
        strides[getRank - 1] = staticDims[getRank - 2];
        strides[getRank - 2] = 1;
    }

    return strides;
}

void MKLDNNMatMulNode::getSupportedDescriptors() {
    if (getParentEdges().size() != 2)
        IE_THROW()  << errorPrefix << " has incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        IE_THROW()  << errorPrefix << " has incorrect number of output edges for layer " << getName();

    auto firstInPortPrec = getOriginalInputPrecisionAtPort(0);
    auto secondInPortPrec = getOriginalInputPrecisionAtPort(1);
    auto outPortPrec = getOriginalOutputPrecisionAtPort(0);

    if (firstInPortPrec.size() != secondInPortPrec.size())
        firstInPortPrec = secondInPortPrec = getMaxPrecision(getOriginalInputPrecisions());

    // fallback to fp32 for any precision that cannot be handled natively
    if ((!one_of(firstInPortPrec , Precision::U8, Precision::I8, Precision::BF16, Precision::FP32) ||
         !one_of(secondInPortPrec , Precision::I8, Precision::BF16, Precision::FP32))) {
        outPortPrec = firstInPortPrec = secondInPortPrec = Precision::FP32;
    }

    if (!fusedWith.empty()) {
        outPortPrec = fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0);
    }

    const auto& inputShape0 = getInputShapeAtPort(0);
    const auto& inputShape1 = getInputShapeAtPort(1);
    const auto& outputShape = getOutputShapeAtPort(0);

    if (inputShape0.getRank() != inputShape1.getRank() || inputShape0.getRank() != outputShape.getRank())
        IE_THROW()  << errorPrefix << " has invalid dims count";

    const int nDims = inputShape0.getRank();
    const auto xAxis = nDims - 1;
    const auto yAxis = nDims - 2;
    const auto xAxis0 = transposeIn[0] ? yAxis : xAxis;
    const auto yAxis0 = transposeIn[0] ? xAxis : yAxis;
    const auto xAxis1 = transposeIn[1] ? yAxis : xAxis;
    const auto yAxis1 = transposeIn[1] ? xAxis : yAxis;

    const auto& inDims0 = getInputShapeAtPort(0).getDims();
    const auto& inDims1 = getInputShapeAtPort(1).getDims();
    const auto& outDims = getOutputShapeAtPort(0).getDims();

    // coverity[copy_paste_error]
    if (!dimsEqualWeak(inDims0[xAxis0], inDims1[yAxis1]) ||
        !dimsEqualWeak(inDims0[yAxis0], outDims[yAxis]) ||
        !dimsEqualWeak(inDims1[xAxis1], outDims[xAxis]))
        IE_THROW()  << errorPrefix << " has incorrect spatial input and output dimensions";

    for (int dim_idx = nDims - 3; dim_idx >= 0; dim_idx--) {
        if ((!dimsEqualWeak(inDims0[dim_idx], outDims[dim_idx]) &&
             !dimsEqualWeak(inDims0[dim_idx], 1)) ||
            (!dimsEqualWeak(inDims1[dim_idx], outDims[dim_idx]) &&
             !dimsEqualWeak(inDims1[dim_idx], 1))) {
            IE_THROW()  << errorPrefix << " has incorrect input batch dimensions";
        }
    }

    std::vector<Shape> staticInputShapes(2);
    staticInputShapes[0] = inputShape0.isStatic() ? inputShape0 : MemoryDescUtils::makeDummyShape(inputShape0);
    staticInputShapes[1] = inputShape1.isStatic() ? inputShape1 : MemoryDescUtils::makeDummyShape(inputShape1);

    auto staticOutputShape = outputShape.isStatic() ? outputShape : Shape(shapeInferGeneric(staticInputShapes).front());

    const VectorDims inStrides0 = getStridesAndModifyShape(staticInputShapes[0], transposeIn[0]);
    const VectorDims inStrides1 = getStridesAndModifyShape(staticInputShapes[1], transposeIn[1]);

    inDataDesc[0] = std::make_shared<DnnlBlockedMemoryDesc>(firstInPortPrec, staticInputShapes[0], inStrides0);
    inDataDesc[1] = std::make_shared<DnnlBlockedMemoryDesc>(secondInPortPrec, staticInputShapes[1], inStrides1);
    outDataDesc   = std::make_shared<DnnlBlockedMemoryDesc>(outPortPrec, staticOutputShape);

    createDescriptor({inDataDesc[0], inDataDesc[1]}, {outDataDesc});
}

void MKLDNNMatMulNode::createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                                        const std::vector<MemoryDescPtr>& outputDesc) {
    MKLDNNDescriptor desc{
        std::make_shared<matmul::desc>(inDataDesc[0]->getDnnlDesc(),
                                       inDataDesc[1]->getDnnlDesc(),
                                       outDataDesc->getDnnlDesc())};

    descs.push_back(desc);
}

void MKLDNNMatMulNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto attr = initPrimitiveAttr();

    for (auto& desc : descs) {
        auto itpd = desc.createPrimitiveDescriptorIterator(getEngine(), *attr);
        while (static_cast<bool>(itpd)) {
            NodeConfig config;
            config.dynBatchSupport = true;
            for (size_t i = 0; i < descInputNumbers(desc); i++) {
                PortConfig portConfig;
                portConfig.inPlace = -1;
                portConfig.constant = false;
                portConfig.desc = getSrcMemDesc(itpd, i);

                config.inConfs.push_back(portConfig);
            }

            for (size_t i = 0; i < descOutputNumbers(desc); i++) {
                PortConfig portConfig;
                portConfig.inPlace = canBeInPlace() ? 0 : -1;
                portConfig.constant = false;
                portConfig.desc = getDstMemDesc(itpd, i);

                config.outConfs.push_back(portConfig);
            }

            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());

            supportedPrimitiveDescriptors.emplace_back(config, impl_type);
            if (!itpd.next_impl())
                break;
        }
    }
}

void MKLDNNMatMulNode::createPrimitive() {
    if (inputShapesDefined()) {
        if (needPrepareParams())
            prepareParams();
        updateLastInputDims();
    }
}

void MKLDNNMatMulNode::prepareCustomKernel(const mkldnn::primitive_attr& attrs) {
    if (transposeIn[0])
        return;

    // custom matmul supports only fp32
    const auto precisionIn0 = getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc->getPrecision();
    const auto precisionIn1 = getSelectedPrimitiveDescriptor()->getConfig().inConfs[1].desc->getPrecision();
    const auto precisionOut = getSelectedPrimitiveDescriptor()->getConfig().outConfs[0].desc->getPrecision();
    if (!everyone_is(Precision::FP32, precisionIn0, precisionIn1, precisionOut))
        return;

    const size_t ndims = inputShapes[0].getRank();
    const size_t m = inputShapes[0].getStaticDims()[ndims - 2];
    const size_t k = inputShapes[0].getStaticDims()[ndims - 1];
    const size_t n = outputShapes[0].getStaticDims()[ndims - 1];

    // custom matmul doesn't support cases with batch broadcasting
    size_t inputBatch0 = 1, inputBatch1 = 1;
    const int lastBatchIdx = ndims - 2;
    if (lastBatchIdx >= 0) {
        inputBatch0 = std::accumulate(inputShapes[0].getStaticDims().begin(), inputShapes[0].getStaticDims().begin() + lastBatchIdx,
                                      1, std::multiplies<size_t>());
        inputBatch1 = std::accumulate(inputShapes[1].getStaticDims().begin(), inputShapes[1].getStaticDims().begin() + lastBatchIdx,
                                      1, std::multiplies<size_t>());

        if (inputBatch0 != inputBatch1 && (inputBatch0 != 1 || inputBatch1 != 1))
            return;

        if (inputBatch0 == inputBatch1) {
            for (auto i = 0; i < lastBatchIdx; ++i) {
                if (inputShapes[0].getStaticDims()[i] != inputShapes[1].getStaticDims()[i]) {
                    return;
                }
            }
        }
    }
    const size_t batch = std::max(inputBatch0, inputBatch1);
    if (batch > 10)
        return;

    // check that data is cached
    const int cacheFloatSize = dnnl::utils::get_cache_size(1, true) / sizeof(float);
    if ((inputShapes[0].getElementsCount() + inputShapes[1].getElementsCount() + outputShapes[0].getElementsCount()) > cacheFloatSize)
        return;

    // custom matmul supports only eltwise post-ops
    for (const auto& node : fusedWith) {
        if (node->getType() != Eltwise)
            return;
    }

    // to have enough vmm for unrolling by n for first algorithm with broadcast(a)
    const bool isAlgorithmWithScalarProduct = transposeIn[1] || n == 1;
    if (!isAlgorithmWithScalarProduct) {
        const int nofree_registers = 3;  // vmm_zero, vmm_src_0, vmm_src_1
        int size = 1;
        int vmm_count = 16;

        if (mayiuse(impl::cpu::x64::avx512_common)) {
            size = 16;
            vmm_count = 32;
        } else if (mayiuse(cpu::x64::avx2)) {
            size = 8;
        } else if (mayiuse(cpu::x64::sse41)) {
            size = 4;
        }

        if (n > ((vmm_count - nofree_registers) * size))
            return;
    }

    jit_matmul_config_params jep;
    jep.b = batch;
    jep.m = m;
    jep.k = k;
    jep.n = n;
    jep.stride0 = inputBatch0 > 1 ? m * k : 0;
    jep.stride1 = inputBatch1 > 1 ? k * n : 0;
    jep.scalar_product = isAlgorithmWithScalarProduct;

    arg = jit_matmul_args();
    memSrc0 = getParentEdgeAt(0)->getMemoryPtr();
    memSrc1 = getParentEdgeAt(1)->getMemoryPtr();
    memDst = getChildEdgeAt(0)->getMemoryPtr();

    if (mayiuse(x64::avx512_common)) {
        matmul_kernel.reset(new jit_uni_matmul_kernel_f32<x64::avx512_common>(jep, *attrs.get()));
    } else if (mayiuse(x64::avx2)) {
        matmul_kernel.reset(new jit_uni_matmul_kernel_f32<x64::avx2>(jep, *attrs.get()));
    } else if (mayiuse(x64::sse41)) {
        matmul_kernel.reset(new jit_uni_matmul_kernel_f32<x64::sse41>(jep, *attrs.get()));
    }

    if (matmul_kernel)
        matmul_kernel->create_ker();
}

void MKLDNNMatMulNode::execute(mkldnn::stream strm) {
    if (matmul_kernel) {
        arg.src0 = memSrc0->GetPtr();
        arg.src1 = memSrc1->GetPtr();
        arg.dst   = memDst->GetPtr();

        (*matmul_kernel)(&arg);
        return;
    }

    MKLDNNNode::execute(strm);
}

MemoryDescPtr MKLDNNMatMulNode::getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    auto desc = idx > 0 ? primitive_desc_it.weights_desc(idx - 1): primitive_desc_it.src_desc(idx);
    return std::make_shared<CpuBlockedMemoryDesc>(
        MKLDNNExtensionUtils::DataTypeToIEPrecision(static_cast<mkldnn::memory::data_type>(desc.data.data_type)),
        getInputShapeAtPort(idx)); /* provide initial shapes, so hide transpose effect */
}

bool MKLDNNMatMulNode::created() const {
    return getType() == MatMul;
}

size_t MKLDNNMatMulNode::getMaxBatch() const {
    if (!outputShapes.empty())
        return outputShapes[0].getStaticDims()[0];
    return 0;
}

InferenceEngine::Precision MKLDNNMatMulNode::getRuntimePrecision() const {
    return getMaxPrecision(getInputPrecisions());
}

void MKLDNNMatMulNode::prepareParams() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& src0MemPtr = getParentEdgeAt(0)->getMemoryPtr();
    auto& src1MemPtr = getParentEdgeAt(1)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        IE_THROW()  << errorPrefix << " did not allocate destination memory";
    if (!src0MemPtr || !src0MemPtr->GetPrimitivePtr() || !src1MemPtr || !src1MemPtr->GetPrimitivePtr())
        IE_THROW()  << errorPrefix << " did not allocate input memory";

    const NodeDesc *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW()  << errorPrefix << " did not set preferable primitive descriptor";

    DnnlMemoryDescPtr src0TransposedDesc;
    DnnlMemoryDescPtr src1TransposedDesc;

    AttrPtr attr;

    if (isDynamicNode()) {
        if (!pAttr) {
            pAttr = initPrimitiveAttr(src0MemPtr->getStaticDims());
        }
        attr = pAttr;

        const auto& src0Desc = src0MemPtr->getDesc();
        const auto& src1Desc = src1MemPtr->getDesc();

        auto src0Shape = src0Desc.getShape();
        auto src0Strides = getStridesAndModifyShape(src0Shape, transposeIn[0]);
        src0TransposedDesc = std::make_shared<DnnlBlockedMemoryDesc>(src0Desc.getPrecision(), src0Shape, src0Strides);

        auto src1Shape = src1Desc.getShape();
        auto src1Strides = getStridesAndModifyShape(src1Shape, transposeIn[1]);
        src1TransposedDesc = std::make_shared<DnnlBlockedMemoryDesc>(src1Desc.getPrecision(), src1Shape, src1Strides);
    } else {
        attr = initPrimitiveAttr();
        src0TransposedDesc = inDataDesc[0];
        src1TransposedDesc = inDataDesc[1];

        prepareCustomKernel(*attr);
        if (matmul_kernel)
            return;
    }

    auto dstDnnlDesc = dstMemPtr->GetDescWithType<DnnlMemoryDesc>();

    MKLDNNDescriptor desc{
            std::make_shared<matmul::desc>(src0TransposedDesc->getDnnlDesc(),
                                           src1TransposedDesc->getDnnlDesc(),
                                           dstDnnlDesc->getDnnlDesc())};

    matmul::primitive_desc prim_desc;
    primitive_desc_iterator itpd = desc.createPrimitiveDescriptorIterator(getEngine(), *attr);

    while (static_cast<bool>(itpd))  {
        impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());

        if (impl_type == selected_pd->getImplementationType()) {
            prim_desc = itpd.get();
            break;
        }
        if (!itpd.next_impl())
            IE_THROW() << "Primitive descriptor was not found for node " << getName() << ".";
    }

    prim.reset(new matmul(prim_desc));

    primArgs[DNNL_ARG_SRC_0] = src0MemPtr->GetPrimitive();
    primArgs[DNNL_ARG_WEIGHTS_0] = src1MemPtr->GetPrimitive();
    primArgs[DNNL_ARG_DST] = dstMemPtr->GetPrimitive();
}

void MKLDNNMatMulNode::executeDynamicImpl(dnnl::stream strm) {
    MKLDNNNode::execute(strm);
}

REG_MKLDNN_PRIM_FOR(MKLDNNMatMulNode, MatMul);
