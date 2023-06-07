// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matmul.h"

#include "ie_precision.hpp"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "cpu_types.h"
#include "eltwise.h"

#include <cstddef>
#include <numeric>
#include <string>
#include <vector>
#include <memory>
#include "common/cpu_memcpy.h"
#include <ngraph/opsets/opset1.hpp>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "fake_quantize.h"
#include "utils/general_utils.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include <dnnl_extension_utils.h>
#include <common/primitive_hashing_utils.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>

using namespace dnnl;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {
namespace {

bool canBeExecutedInInt8(const Precision& firstInput, const Precision& secondInput) {
    return one_of(firstInput, Precision::U8, Precision::I8) && secondInput == Precision::I8;
}
} // namespace

bool MatMul::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
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

namespace {
class MMShapeInfer : public ShapeInferEmptyPads {
public:
    MMShapeInfer(const size_t& out_rank, const bool& transpose_a, const bool& transpose_b) :
        m_out_rank(out_rank), m_transpose_a(transpose_a), m_transpose_b(transpose_b) {
        m_shapeY = VectorDims(m_out_rank, 1); // for output and cache
    }
    Result infer(
        const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
        const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        const VectorDims& shapeA = input_shapes[0].get();
        const VectorDims& shapeB = input_shapes[1].get();
        const size_t rankA = shapeA.size();
        const size_t rankB = shapeB.size();

        // getSupportedDescriptors has done some shape check.
        // 1. Needn't assert the scalar type since the matmul_shape_inference has checked.
        // 2. Needn't check the compatibility of the last two dims
        // 3. 1-D x 1-D is needed
        // 4. transpose is necessary
        // 5. Just support the same rank of matmul
        // 6. simplify the broadcast check
        if (rankA == 1 && rankB == 1 && shapeA[0] == shapeB[0]) {
            return {{m_shapeY}, ShapeInferStatus::success};
        }

        m_shapeY[m_out_rank-2] = m_transpose_a ? shapeA[rankA-1] : shapeA[rankA-2];
        m_shapeY[m_out_rank-1] = m_transpose_b ? shapeB[rankB-2] : shapeB[rankB-1];

        for (size_t i=0; i < m_out_rank-2; ++i) {
            if (shapeA[i] != shapeB[i]) {
                if (shapeB[i] == 1) {
                    m_shapeY[i] = shapeA[i];
                    continue;
                } else if (shapeA[i] != 1) {
                    IE_THROW() << "Incompatible MatMul batch dimension. Cant merge the first input dimension=" <<
                                  shapeA[i] << " with second input dimension=" << shapeB[i] << " at index=" << i;
                }
            }
            m_shapeY[i] = shapeB[i];
        }

        return {{m_shapeY}, ShapeInferStatus::success};
    }

    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }

private:
    VectorDims m_shapeY;
    const size_t m_out_rank;
    const bool m_transpose_a;
    const bool m_transpose_b;
};

class MMShapeInferFactory : public ShapeInferFactory {
public:
    MMShapeInferFactory(const std::shared_ptr<ngraph::Node>& op) : m_op(op) {}
    ShapeInferPtr makeShapeInfer() const override {
        if (const auto matmul = ov::as_type_ptr<const ngraph::opset1::MatMul>(m_op)) {
            const auto output_rank = matmul->get_output_partial_shape(0).rank().get_length();
            const bool transpose_a = matmul->get_transpose_a();
            const bool transpose_b = matmul->get_transpose_b();
            return std::make_shared<MMShapeInfer>(output_rank, transpose_a, transpose_b);
       } else {
             IE_THROW() << "Unexpected operation type in the MatMul shape inference factory";
       }
    }
private:
    std::shared_ptr<ngraph::Node> m_op;
};
} // namespace

MatMul::MatMul(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context) :
    Node(op, context, MMShapeInferFactory(op)) {
    std::string errorMessage;
    errorPrefix = "MatMul node with name '" + getName() + "'";

    if (!isSupportedOperation(op, errorMessage))
        IE_THROW(NotImplemented) << errorMessage;

    const auto matMul = std::dynamic_pointer_cast<const ngraph::opset1::MatMul>(op);

    if (!matMul) {
        IE_THROW(NotImplemented) << "Operation with name " << op->get_friendly_name() << ":" << op->get_type_name() <<
            " is not an instance of MatMul from opset1";
    }

    matmulAttrs.transposeA = matMul->get_transpose_a();
    matmulAttrs.transposeB = matMul->get_transpose_b();
}

bool MatMul::canFuse(const NodePtr& node) const {
    // WA for CVS-84056: oneDNN brgemm impl has problem with per-OC binary-postOps for MatMul with 6D inputs
    if (impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_core)) {
        if (auto* eltwiseNode = dynamic_cast<Eltwise*>(node.get())) {
            if (eltwiseNode->getBroadcastingPolicy() == Eltwise::BroadcastingPolicy::PerChannel) {
                auto rank = getInputShapeAtPort(0).getRank();
                if (rank > 4) {
                    DEBUG_LOG("skip fusing non-perTensor Eltwise:", eltwiseNode->getName(), " into 6D MatMul:", getName());
                    return false;
                }
            }
        }
    }

    //  Consider the case when Matmul doesn't support execution in int8, but is getting fused with FQ with int8 output.
    //  Then the Matmul will change its output precision to fp32. If fusing FQ into matmul, there would be reorder inserted
    //  after matmul. In some bert model, this reorder causes great perf degradation.
    //  Todo: Remove this if onednn primitive support U8 output with floating input.
    if (node->getType() == Type::FakeQuantize && one_of(node->getOriginalOutputPrecisionAtPort(0), Precision::I8, Precision::U8) &&
        !canBeExecutedInInt8(getOriginalInputPrecisionAtPort(0), getOriginalInputPrecisionAtPort(1)) &&
        getOriginalInputPrecisionAtPort(0) == InferenceEngine::Precision::FP32 )
        return false;
    return canFuseSimpleOperation(node);
}

void MatMul::setPostOps(dnnl::primitive_attr& attr, const VectorDims& dims, bool initWeights = false) {
    dnnl::post_ops ops;

    dnnl::memory::data_type outputDataType = DnnlExtensionUtils::IEPrecisionToDataType(outputPrecisions[0]);

    bool isINT8 = canBeExecutedInInt8(getOriginalInputPrecisionAtPort(0), getOriginalInputPrecisionAtPort(1));

    DnnlPostOpsComposer dnnlpoc(
        getEngine(), attr, ops, postOpsArgs, dims, dims.size() - 1, isINT8, 1 << (dims.size() - 1), getDQScales(), matmulAttrs.withBias);

    for (size_t i = 0; i < fusedWith.size(); ++i) {
        auto& node = fusedWith[i];
        bool isLastPostOp = (i == (fusedWith.size() - 1));

        if (auto* eltwiseNode = dynamic_cast<Eltwise*>(node.get())) {
            eltwiseNode->appendAttrPostOps(dnnlpoc, isLastPostOp, outputDataType);
            continue;
        }

        if (auto* fakeQuantizeNode = dynamic_cast<FakeQuantize*>(node.get())) {
            fakeQuantizeNode->appendAttrPostOps(dnnlpoc, isLastPostOp, outputDataType);
            continue;
        }

        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType())
                   << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

Node::AttrPtr MatMul::initPrimitiveAttr(const VectorDims &dims) {
    auto attr = std::make_shared<dnnl::primitive_attr>(dnnl::primitive_attr());

    setPostOps(*attr, dims, true);

    (*attr).set_scratchpad_mode(dnnl::scratchpad_mode::user);

    return attr;
}

Node::AttrPtr MatMul::initPrimitiveAttr() {
    auto dummyShape = MemoryDescUtils::makeDummyShape(getOutputShapeAtPort(0));
    return initPrimitiveAttr(dummyShape.getStaticDims());
}

void MatMul::initSupportedPrimitiveDescriptors() {
    matmulAttrs.withBias = getOriginalInputsNumber() == 3;

    inputPrecisions = getOriginalInputPrecisions();
    outputPrecisions = getOriginalOutputPrecisions();
    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::sse41)) {
        if (inputPrecisions[0].size() != inputPrecisions[1].size())
            inputPrecisions[0] = inputPrecisions[1] = getMaxPrecision(getOriginalInputPrecisions());

        // fallback to fp32 for any precision that cannot be handled natively
        if ((!one_of(inputPrecisions[0] , Precision::U8, Precision::I8, Precision::BF16, Precision::FP32) ||
            !one_of(inputPrecisions[1] , Precision::I8, Precision::BF16, Precision::FP32))) {
            outputPrecisions[0] = inputPrecisions[0] = inputPrecisions[1] = Precision::FP32;
        }

        if (!fusedWith.empty()) {
            outputPrecisions[0] = fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0);
        }

        if (!canBeExecutedInInt8( inputPrecisions[0], inputPrecisions[1]) && one_of(outputPrecisions[0], Precision::U8, Precision::I8))
            outputPrecisions[0] = Precision::FP32; // INT output is not supported for non-INT inputs
    } else {
        inputPrecisions[0] = inputPrecisions[1] = Precision::FP32;
        outputPrecisions[0] = Precision::FP32;
    }

    auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    NodeConfig config;

    for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
        PortConfig portConfig;
        portConfig.inPlace(-1);
        portConfig.constant(false);
        portConfig.setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(inputPrecisions[i], getInputShapeAtPort(i)));

        config.inConfs.push_back(portConfig);
    }

    for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
        PortConfig portConfig;
        portConfig.inPlace(canBeInPlace() ? 0 : -1);
        portConfig.constant(false);
        portConfig.setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(outputPrecisions[i], getOutputShapeAtPort(i)));

        config.outConfs.push_back(portConfig);
    }

    std::vector<MemoryDescPtr> srcMemoryDescs;
    for (size_t i = 0; i < config.inConfs.size(); i++) {
        srcMemoryDescs.push_back(config.inConfs[i].getMemDesc());
    }
    std::vector<MemoryDescPtr> dstMemoryDescs;
    for (size_t i = 0; i < config.outConfs.size(); i++) {
        dstMemoryDescs.push_back(config.outConfs[i].getMemDesc());
    }

    auto attr = initPrimitiveAttr();
    auto factory = std::make_shared<MatMulExecutorFactory>(matmulAttrs, srcMemoryDescs, dstMemoryDescs, *attr.get(),
                                                           std::make_shared<ExecutorContext>(context, getPrimitivesPriority()));

    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::undef, factory);
}

bool MatMul::created() const {
    return getType() == Type::MatMul;
}

InferenceEngine::Precision MatMul::getRuntimePrecision() const {
    return getMaxPrecision(getInputPrecisions());
}

void MatMul::prepareParams() {
    std::vector<MemoryDescPtr> srcMemoryDescs;
    for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
        srcMemoryDescs.push_back(getParentEdgeAt(i)->getMemoryPtr()->getDescPtr());
    }
    std::vector<MemoryDescPtr> dstMemoryDescs;
    for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
        dstMemoryDescs.push_back(getChildEdgeAt(i)->getMemoryPtr()->getDescPtr());
    }

    AttrPtr attr = initPrimitiveAttr(dstMemoryDescs[0]->getShape().getStaticDims() );

    auto selectedPD = getSelectedPrimitiveDescriptor();
    execPtr = selectedPD->getExecutorFactoryAs<MatMulExecutorFactory>()->makeExecutor(
        matmulAttrs, srcMemoryDescs, dstMemoryDescs, *attr.get());
    selectedPD->setImplementationType(execPtr->getImplType());
}

void MatMul::execute(dnnl::stream strm) {
    if (!execPtr) {
        IE_THROW() << "Can't execute MatMul node. Executor is not created";
    }

    std::vector<MemoryCPtr> srcMemory;
    for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
        srcMemory.push_back(getParentEdgeAt(i)->getMemoryPtr());
    }
    std::vector<MemoryPtr> dstMemory;
    for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
        dstMemory.push_back(getChildEdgeAt(i)->getMemoryPtr());
    }

    execPtr->exec(srcMemory, dstMemory, postOpsArgs);
}

void MatMul::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

const std::vector<impl_desc_type>& MatMul::getPrimitivesPriority() {
    std::vector<impl_desc_type> priorities = {
            impl_desc_type::unknown,
            impl_desc_type::brgemm_avx512_amx,
            impl_desc_type::brgemm_avx512,
            impl_desc_type::gemm_acl,
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
}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
