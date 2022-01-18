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
#include "common/cpu_memcpy.h"
#include <ngraph/opsets/opset1.hpp>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "nodes/mkldnn_fake_quantize_node.h"
#include "utils/general_utils.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "mkldnn_extension_utils.h"
#include <common/primitive_hashing_utils.hpp>

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
struct jit_uni_small_matmul_kernel_f32 : public jit_uni_matmul_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_small_matmul_kernel_f32)

    explicit jit_uni_small_matmul_kernel_f32(jit_matmul_config_params jcp_, const mkldnn_primitive_attr &attr) :
        jit_uni_matmul_kernel(jcp_, attr), jit_generator() {}

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

        load_emitter.reset(new jit_load_emitter(this, isa));
        store_emitter.reset(new jit_store_emitter(this, isa));

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

            body_m();

            // strides for batch
            if (jcp_.stride0 != 0)  // ptr for the first matrix is automatically incremented after each iteration by m,
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

    inline void body_m() {
        Xbyak::Label label_m;
        Xbyak::Label label_m_end;

        mov(m, 0);
        L(label_m); {
            cmp(m, jcp_.m);
            je(label_m_end, T_NEAR);

            if (jcp_.scalar_product) {
                loop_n();
            } else {
                loop_k();
            }

            add(reg_src_0, jcp_.k * sizeof(float));
            add(m, 1);
            jmp(label_m, T_NEAR);
        }
        L(label_m_end);
    }

    // common execution : broadcast(a) * vmm_b
    inline void loop_k() {
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
                load(reg_src_aux_1, get_src_vmm(1), vec_step, i * vlen);  // src1
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
            store(get_dst_vmm(i), reg_dst, vec_step, i * vlen);
        }
        add(reg_dst, amount_full * vlen);

        if (amount_tail != 0) {
            apply_post_ops(get_dst_vmm(amount_full).getIdx());
            store(get_dst_vmm(amount_full), reg_dst, amount_tail);
            add(reg_dst, amount_tail * sizeof(float));
        }
    }

    // optimized execution for cases with transposed matrix b or k = 1
    inline void loop_n() {
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

            loop_n_body(vec_step);

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
            loop_n_body(tail_n_amount);

            if (attr_.post_ops_.len() != 0) {
                load(reg_dst_aux, get_dst_vmm(0), tail_n_amount);
                apply_post_ops();
                store(get_dst_vmm(0), reg_dst_aux, tail_n_amount);
                add(reg_dst_aux, tail_n_amount * sizeof(float));
            }
        }
    }

    inline void loop_n_body(const int amount) {
        Xbyak::Label label_o;
        Xbyak::Label label_o_end;

        mov(n_inside, 0);
        L(label_o); {
            cmp(n_inside, amount);
            je(label_o_end, T_NEAR);

            mov(reg_src_aux_0, reg_src_0);
            uni_vpxor(get_dst_vmm(), get_dst_vmm(), get_dst_vmm());

            for (int i = 0; i < amount_full; ++i) {
                load(reg_src_aux_0, get_src_vmm(0), vec_step, i * vlen);
                load(reg_src_aux_1, get_src_vmm(1), vec_step, i * vlen);

                uni_vfmadd231ps(get_dst_vmm(), get_src_vmm(0), get_src_vmm(1));
            }
            add(reg_src_aux_1, amount_full * vlen);

            if (amount_tail != 0) {
                add(reg_src_aux_0, amount_full * vlen);
                load(reg_src_aux_0, get_src_vmm(0), amount_tail, 0, true);
                load(reg_src_aux_1, get_src_vmm(1), amount_tail, 0, true);
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

    inline void load(Xbyak::Reg64 reg, Vmm vmm, int load_num, int offset_byte = 0, bool is_fill = false) {
        load_emitter->emit_code({static_cast<size_t>(reg.getIdx())}, {static_cast<size_t>(vmm.getIdx())},
                                std::make_shared<load_emitter_context>(Precision::FP32, Precision::FP32, load_num, offset_byte, is_fill),
                                {}, load_pool_gpr_idxs);
    }

    inline void store(Vmm vmm, Xbyak::Reg64 reg, int store_num, int offset_byte = 0) {
        store_emitter->emit_code({static_cast<size_t>(vmm.getIdx())}, {static_cast<size_t>(reg.getIdx())},
                                 std::make_shared<store_emitter_context>(Precision::FP32, Precision::FP32, store_num, offset_byte),
                                 store_pool_vec_idxs, store_pool_gpr_idxs);
    }
};

namespace {
struct MatMulKey {
    DnnlMemoryDescCPtr inp0;
    DnnlMemoryDescCPtr inp1;
    DnnlMemoryDescCPtr bias;
    DnnlMemoryDescCPtr out;
    mkldnn::primitive_attr attr;
    impl_desc_type implType;

    size_t hash() const;
    bool operator==(const MatMulKey& rhs) const;
};

size_t MatMulKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;

    for (const auto& ptr : {inp0, inp1, bias, out}) {
        if (ptr) {
            seed = hash_combine(seed, get_md_hash(ptr->getDnnlDesc().data));
        }
    }

    seed = hash_combine(seed, get_attr_hash(*attr.get()));
    seed = hash_combine(seed, implType);
    return seed;
}

bool MatMulKey::operator==(const MatMulKey &rhs) const {
    bool retVal = true;
    if (inp0 != rhs.inp0) {
        retVal = retVal && inp0 && rhs.inp0 && inp0->getDnnlDesc() == rhs.inp0->getDnnlDesc();
    }
    if (inp1 != rhs.inp1) {
        retVal = retVal && inp1 && rhs.inp1 && inp1->getDnnlDesc() == rhs.inp1->getDnnlDesc();
    }
    if (bias != rhs.bias) {
        retVal = retVal && bias && rhs.bias && bias->getDnnlDesc() == rhs.bias->getDnnlDesc();
    }
    if (out != rhs.out) {
        retVal = retVal && out && rhs.out && out->getDnnlDesc() == rhs.out->getDnnlDesc();
    }
    retVal = retVal && *attr.get() == *rhs.attr.get() &&
             implType == rhs.implType;
    return retVal;
}

} // namespace

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
    MKLDNNNode(op, eng, cache), withBiases(false) {
    std::string errorMessage;
    errorPrefix = "MatMul node with name '" + getName() + "'";

    if (!isSupportedOperation(op, errorMessage))
        IE_THROW(NotImplemented) << errorMessage;

    const auto matMul = std::dynamic_pointer_cast<const ngraph::opset1::MatMul>(op);

    if (!matMul) {
        IE_THROW(NotImplemented) << "Operation with name " << op->get_friendly_name() << ":" << op->get_type_name() <<
            " is not an instance of MatMul from opset1";
    }

    transposeIn[0] = matMul->get_transpose_a();
    transposeIn[1] = matMul->get_transpose_b();
}

bool MKLDNNMatMulNode::canFuse(const MKLDNNNodePtr& node) const {
    // per channel binary post op for rank > 2D is supported only by oneDNN reference implementation because of unusual MatMul channel axis (issue 6669)
    if (getOutputShapeAtPort(0).getRank() > 2) {
        if (const auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(node.get())) {
            if (one_of(eltwiseNode->getAlgorithm(),
                       EltwiseAdd, EltwiseMultiply, EltwiseSubtract, EltwiseDivide, EltwisePrelu, EltwiseMulAdd, EltwisePowerStatic) &&
                eltwiseNode->getBroadcastingPolicy() != MKLDNNEltwiseNode::PerTensor) {
                return false;
            }
        } else if (const auto* fakeQuantizeNode = dynamic_cast<MKLDNNFakeQuantizeNode *>(node.get())) {
            if (fakeQuantizeNode->getBroadcastingPolicy() != MKLDNNFakeQuantizeNode::PerTensor) {
                return false;
            }
        }
    }

    return canFuseSimpleOperation(node);
}

void MKLDNNMatMulNode::setPostOps(mkldnn::primitive_attr &attr, const VectorDims& dims, bool initWeights = false) {
    mkldnn::post_ops ops;

    auto getBinPostOpShape = [&](){
        const auto outShapeRank = dims.size();
        const auto chIdx = getFusingAxis();
        std::vector<size_t> binaryShape(outShapeRank, 1);
        binaryShape[chIdx] = dims[chIdx];
        return binaryShape;
    };

    for (const auto &node : fusedWith) {
        if (auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(node.get())) {
            if (eltwiseNode->getMKLDNNAlgorithm() != mkldnn::algorithm::undef) {
                eltwiseNode->appendPostOps(ops, dims);
            } else {
                eltwiseNode->appendBinPostOps(ops, getBinPostOpShape(), binaryPostOpsArgs);
            }
            continue;
        } else if (auto* fakeQuantizeNode = dynamic_cast<MKLDNNFakeQuantizeNode *>(node.get())) {
            fakeQuantizeNode->appendBinPostOps(ops, getBinPostOpShape(), binaryPostOpsArgs);
            continue;
        }

        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType()) << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

MKLDNNNode::AttrPtr MKLDNNMatMulNode::initPrimitiveAttr(const VectorDims &dims) {
    auto attr = std::make_shared<mkldnn::primitive_attr>(mkldnn::primitive_attr());

    setPostOps(*attr, dims, true);

    return attr;
}

MKLDNNNode::AttrPtr MKLDNNMatMulNode::initPrimitiveAttr() {
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

mkldnn::memory::desc MKLDNNMatMulNode::getBiasDescFrom(const DnnlMemoryDescCPtr outMemDesc) {
    // oneDNN matmul requires shape for bias desc to be the same rank
    VectorDims biasDims(outMemDesc->getShape().getRank(), 1);
    const auto outDims = outMemDesc->getShape().getStaticDims();
    const auto chIdx = getFusingAxis();
    biasDims[chIdx] = outDims[chIdx];
    const auto bdt = MKLDNNExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(2));

    return mkldnn::memory::desc(MKLDNNExtensionUtils::convertToDnnlDims(biasDims), bdt, memory::format_tag::any);
}

void MKLDNNMatMulNode::getSupportedDescriptors() {
    if (getParentEdges().size() != getOriginalInputsNumber())
        IE_THROW()  << errorPrefix << " has incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        IE_THROW()  << errorPrefix << " has incorrect number of output edges for layer " << getName();

    withBiases = getOriginalInputsNumber() == 3;

    auto canBeExecutedInInt8 = [](const Precision firstInput, const Precision secondInput) {
        return one_of(firstInput, Precision::U8, Precision::I8) && secondInput == Precision::I8;
    };

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

    if (!canBeExecutedInInt8(firstInPortPrec, secondInPortPrec) && one_of(outPortPrec, Precision::U8, Precision::I8))
        outPortPrec = Precision::FP32; // INT output is not supported for non-INT inputs

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
    std::shared_ptr<mkldnn::matmul::desc> matmul_desc;
    if (withBiases) {
        matmul_desc.reset(new matmul::desc(inDataDesc[0]->getDnnlDesc(),
                                           inDataDesc[1]->getDnnlDesc(),
                                           getBiasDescFrom(outDataDesc),
                                           outDataDesc->getDnnlDesc()));
    } else {
        matmul_desc.reset(new matmul::desc(inDataDesc[0]->getDnnlDesc(),
                                           inDataDesc[1]->getDnnlDesc(),
                                           outDataDesc->getDnnlDesc()));
    }

    descs.emplace_back(matmul_desc);
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

MemoryDescPtr MKLDNNMatMulNode::getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    auto desc = idx > 0 ? primitive_desc_it.weights_desc(idx - 1): primitive_desc_it.src_desc(idx);

    if (idx < 2) // inputs
        return std::make_shared<CpuBlockedMemoryDesc>(
            MKLDNNExtensionUtils::DataTypeToIEPrecision(static_cast<mkldnn::memory::data_type>(desc.data.data_type)),
            getInputShapeAtPort(idx)); /* provide initial shapes, so hide transpose effect */
    else // bias
        return MKLDNNExtensionUtils::makeDescriptor(desc);
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
        attr = initPrimitiveAttr(dstMemPtr->getStaticDims());

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

        prepareCustomKernel(src0TransposedDesc, src1TransposedDesc, *attr);
        if (matmul_kernel)
            return;
    }

    auto dstDnnlDesc = dstMemPtr->GetDescWithType<DnnlMemoryDesc>();

    DnnlMemoryDescPtr dnnlBiasMemDesc = nullptr;
    if (withBiases) {
        auto& biasMemory = getParentEdgeAt(2)->getMemoryPtr();
        if (!biasMemory || !biasMemory->GetPrimitivePtr())
            IE_THROW()  << errorPrefix << " did not allocate bias memory";
        dnnlBiasMemDesc = biasMemory->GetDescWithType<DnnlMemoryDesc>();
    }

    MatMulKey key = {src0TransposedDesc, src1TransposedDesc, dnnlBiasMemDesc,
                     dstDnnlDesc, *attr, selected_pd->getImplementationType()};

    auto engine = getEngine();

    auto builder = [&engine](const MatMulKey& key) -> std::shared_ptr<mkldnn::primitive> {
        std::shared_ptr<mkldnn::matmul::desc> matmul_desc;

        if (key.bias) {
            matmul_desc.reset(new mkldnn::matmul::desc{key.inp0->getDnnlDesc(),
                                                       key.inp1->getDnnlDesc(),
                                                       key.bias->getDnnlDesc(),
                                                       key.out->getDnnlDesc()});
        } else {
            matmul_desc.reset(new mkldnn::matmul::desc(key.inp0->getDnnlDesc(),
                                                       key.inp1->getDnnlDesc(),
                                                       key.out->getDnnlDesc()));
        }

        MKLDNNDescriptor desc(matmul_desc);
        primitive_desc_iterator itpd = desc.createPrimitiveDescriptorIterator(engine, key.attr);
        matmul::primitive_desc prim_desc;

        while (static_cast<bool>(itpd))  {
            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());

            if (impl_type == key.implType) {
                prim_desc = itpd.get();
                break;
            }
            if (!itpd.next_impl())
                return nullptr;
        }
        return std::make_shared<matmul>(prim_desc);
    };

    auto cache = getRuntimeCache();
    auto result = cache->getOrCreate(key, builder);

    if (!result.first) {
        IE_THROW() << "Primitive descriptor was not found for node " << getName() << ".";
    }

    prim = result.first;

    primArgs[DNNL_ARG_SRC_0] = src0MemPtr->GetPrimitive();
    primArgs[DNNL_ARG_WEIGHTS_0] = src1MemPtr->GetPrimitive();
    primArgs[DNNL_ARG_DST] = dstMemPtr->GetPrimitive();
    if (withBiases)
        primArgs[DNNL_ARG_BIAS] = getParentEdgeAt(2)->getMemoryPtr()->GetPrimitive();

    appendPostOpArgs(*attr, primArgs, binaryPostOpsArgs);
}


void MKLDNNMatMulNode::prepareCustomKernel(const MemoryDescPtr& srcTransposedDesc0, const MemoryDescPtr& srcTransposedDesc1,
                                           const mkldnn::primitive_attr& attrs) {
    if (transposeIn[0] || withBiases)
        return;

    // custom matmul supports only fp32
    const auto precisionIn0 = srcTransposedDesc0->getPrecision();
    const auto precisionIn1 = srcTransposedDesc1->getPrecision();
    const auto precisionOut = getChildEdgeAt(0)->getMemoryPtr()->getDesc().getPrecision();
    if (!everyone_is(Precision::FP32, precisionIn0, precisionIn1, precisionOut))
        return;

    const auto& srcTransposedDims0 = srcTransposedDesc0->getShape().getStaticDims();
    const auto& srcTransposedDims1 = srcTransposedDesc1->getShape().getStaticDims();
    const auto& dstShape = getChildEdgeAt(0)->getMemoryPtr()->getDesc().getShape();

    const size_t ndims = srcTransposedDims0.size();
    const size_t m = srcTransposedDims0[ndims - 2];
    const size_t k = srcTransposedDims0[ndims - 1];
    const size_t n = srcTransposedDims1.back();

    // custom matmul doesn't support cases with batch broadcasting
    size_t inputBatch0 = 1, inputBatch1 = 1;
    const int lastBatchIdx0 = ndims - 2;
    const int lastBatchIdx1 = srcTransposedDims1.size() - 2;
    inputBatch0 = std::accumulate(srcTransposedDims0.begin(), srcTransposedDims0.begin() + lastBatchIdx0,
                                  1, std::multiplies<size_t>());
    inputBatch1 = std::accumulate(srcTransposedDims1.begin(), srcTransposedDims1.begin() + lastBatchIdx1,
                                  1, std::multiplies<size_t>());

    if (inputBatch0 != inputBatch1 && (inputBatch0 != 1 && inputBatch1 != 1))
        return;

    if (inputBatch0 != 1 && inputBatch0 == inputBatch1) {
        if (lastBatchIdx0 != lastBatchIdx1)
            return;

        for (auto i = 0; i < lastBatchIdx0; ++i) {
            if (srcTransposedDims0[i] != srcTransposedDims1[i]) {
                return;
            }
        }
    }
    const size_t batch = std::max(inputBatch0, inputBatch1);
    if (batch > 10)
        return;

    // check that data is cached
    const int cacheFloatSize = dnnl::utils::get_cache_size(1, true) / sizeof(float);
    if ((srcTransposedDesc0->getShape().getElementsCount() + srcTransposedDesc1->getShape().getElementsCount() + dstShape.getElementsCount()) > cacheFloatSize)
        return;

    // custom matmul supports only eltwise post-ops
    auto post_ops = attrs.get_post_ops();
    for (int i = 0; i < post_ops.len(); i++) {
        if (post_ops.kind(i) != mkldnn::primitive::kind::eltwise)
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
    jep.stride0 = inputBatch0 > 1 ? 0 : m * k;
    jep.stride1 = inputBatch1 > 1 ? k * n : 0;
    jep.scalar_product = isAlgorithmWithScalarProduct;

    arg = jit_matmul_args();
    memSrc0 = getParentEdgeAt(0)->getMemoryPtr();
    memSrc1 = getParentEdgeAt(1)->getMemoryPtr();
    memDst = getChildEdgeAt(0)->getMemoryPtr();

    if (mayiuse(x64::avx512_common)) {
        matmul_kernel.reset(new jit_uni_small_matmul_kernel_f32<x64::avx512_common>(jep, *attrs.get()));
    } else if (mayiuse(x64::avx2)) {
        matmul_kernel.reset(new jit_uni_small_matmul_kernel_f32<x64::avx2>(jep, *attrs.get()));
    } else if (mayiuse(x64::sse41)) {
        matmul_kernel.reset(new jit_uni_small_matmul_kernel_f32<x64::sse41>(jep, *attrs.get()));
    }

    if (matmul_kernel)
        matmul_kernel->create_ker();
}

void MKLDNNMatMulNode::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

void MKLDNNMatMulNode::execute(mkldnn::stream strm) {
    if (matmul_kernel) {
        arg.src0 = memSrc0->GetPtr();
        arg.src1 = memSrc1->GetPtr();
        arg.dst  = memDst->GetPtr();

        (*matmul_kernel)(&arg);
        return;
    }

    MKLDNNNode::execute(strm);
}

const std::vector<impl_desc_type>& MKLDNNMatMulNode::getPrimitivesPriority() {
    std::vector<impl_desc_type> priorities = {
            impl_desc_type::unknown,
            impl_desc_type::brgemm_avx512_amx,
            impl_desc_type::brgemm_avx512,
            impl_desc_type::gemm_blas,
            impl_desc_type::gemm_avx512,
            impl_desc_type::gemm_avx2,
            impl_desc_type::gemm_avx,
            impl_desc_type::gemm_sse42,
            impl_desc_type::gemm_any,
            impl_desc_type::gemm,
            impl_desc_type::jit_gemm,
            impl_desc_type::jit_uni_dw,
            impl_desc_type::jit_uni_1x1,
            impl_desc_type::jit_uni,
            impl_desc_type::jit_avx512_dw,
            impl_desc_type::jit_avx512_1x1,
            impl_desc_type::jit_avx512,
            impl_desc_type::jit_avx2_dw,
            impl_desc_type::jit_avx2_1x1,
            impl_desc_type::jit_avx2,
            impl_desc_type::jit_avx_dw,
            impl_desc_type::jit_avx_1x1,
            impl_desc_type::jit_avx,
            impl_desc_type::jit_sse42_dw,
            impl_desc_type::jit_sse42_1x1,
            impl_desc_type::jit_sse42,
            impl_desc_type::ref,
    };
    for (const auto& impl : priorities) {
        if (std::find(implPriorities.begin(), implPriorities.end(), impl) == implPriorities.end())
            implPriorities.push_back(impl);
    }
    return implPriorities;
}

REG_MKLDNN_PRIM_FOR(MKLDNNMatMulNode, MatMul);
