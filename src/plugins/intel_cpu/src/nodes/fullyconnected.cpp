// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fullyconnected.h"

#include <cpu/x64/cpu_isa_traits.hpp>
#include <memory>
#include <openvino/op/constant.hpp>

#include "common/cpu_convert.h"
#include "dnnl_extension_utils.h"
#include "executors/memory_arguments.hpp"
#include "graph_context.h"
#include "input.h"
#include "memory_desc/blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "openvino/core/type/element_type.hpp"
#include "post_ops.hpp"
#include "shape_inference/custom/fullyconnected.hpp"
#include "transformations/cpu_opset/common/op/fully_connected.hpp"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"

using namespace dnnl;
using namespace ov::element;

namespace ov {
namespace intel_cpu {
namespace node {

bool FullyConnected::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                          std::string& errorMessage) noexcept {
    try {
        const auto fc = std::dynamic_pointer_cast<const FullyConnectedNode>(op);
        if (!fc) {
            errorMessage = "Only legacy FullyConnected operation is supported";
            return false;
        }
        if (fc->get_input_size() == 3 &&
            std::dynamic_pointer_cast<const ov::op::v0::Constant>(fc->get_input_node_shared_ptr(BIAS_ID)) == nullptr) {
            errorMessage = "Only Constant operation on 'bias' input is supported";
            return false;
        }
        const auto weightRank = fc->get_input_partial_shape(WEIGHTS_ID).size();
        if (weightRank != 2) {
            errorMessage = "Doesn't support 'weight' input with rank: " + std::to_string(weightRank);
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

FullyConnected::FullyConnected(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, FCShapeInferFactory(op)),
      errorPrefix("FullyConnected node with name '" + getName() + "'") {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage))
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);

    auto createEmptyMemoryDesc = [](const ov::element::Type type) {
        return std::make_shared<CpuBlockedMemoryDesc>(type, Shape{0});
    };

    auto createEmptyMemory = [&createEmptyMemoryDesc](const GraphContext::CPtr context, const ov::element::Type type) {
        return std::make_shared<Memory>(context->getEngine(), createEmptyMemoryDesc(type), nullptr);
    };

    emptyMemory = createEmptyMemory(context, ov::element::undefined);
}

bool FullyConnected::canBeExecutedInInt8() const {
    auto srcType = getOriginalInputPrecisionAtPort(0);
    auto weiType = getOriginalInputPrecisionAtPort(1);

    return one_of(srcType, ov::element::u8, ov::element::i8) && weiType == ov::element::i8;
}

ExecutorPtr FullyConnected::createExecutor() {
    const auto& executor = factory->make(memory);
    getSelectedPrimitiveDescriptor()->setImplementationType(executor->implType());

    return executor;
}

void FullyConnected::prepareParams() {
    executor = createExecutor();
}

void FullyConnected::execute(dnnl::stream strm) {
    executor->execute(memory);
}

void FullyConnected::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool FullyConnected::canFuse(const NodePtr& node) const {
    return canFuseSimpleOperation(node);
}

bool FullyConnected::created() const {
    return getType() == Type::FullyConnected;
}

void FullyConnected::toNumaNodeImpl(int numaID) {
    executor->moveMemToNumaNode(numaID);
}

const std::vector<impl_desc_type>& FullyConnected::getDefaultImplPriority() {
    static const std::vector<impl_desc_type> priorities = {
        impl_desc_type::unknown,
        impl_desc_type::acl,
        impl_desc_type::brgemm_sparse_avx512_amx,
        impl_desc_type::brgemm_avx512_amx,
        impl_desc_type::brgemm_avx512,
        impl_desc_type::brgemm_avx2,
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

    return priorities;
}

// @todo Should be moved to the transformations / optimization stages?
static bool useSparseWeightsDecompression(const NodePtr& weightsInput,
                                          const ov::element::Type inputType,
                                          const float sparseWeiDecompressionRate) {
    const auto minSparseRate = sparseWeiDecompressionRate;

    if (minSparseRate == 1.f) {
        return false;
    }

    if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx))
        return false;

    const auto constNode = std::dynamic_pointer_cast<Input>(weightsInput);
    if (!constNode)
        return false;

    const auto weiMemory = constNode->getMemoryPtr();
    OPENVINO_ASSERT(weiMemory, "Cannot get const blob");

    const auto weiDims = weiMemory->getShape().getStaticDims();
    if (weiDims.size() != 2 || weiDims[0] % 64 != 0 || weiDims[1] % 64 != 0) {
        return false;
    }

    const auto weightsType = weiMemory->getPrecision();
    if (!one_of(inputType, u8, i8) || weightsType != i8) {
        return false;
    }

    const auto weightsData = weiMemory->getDataAs<const int8_t>();
    auto elementsCount = weiMemory->getDescWithType<BlockedMemoryDesc>()->getPaddedElementsCount();
    size_t zerosCount = 0;
    for (size_t i = 0; i < elementsCount; i++) {
        if (weightsData[i] == 0) {
            zerosCount++;
        }
    }

    DEBUG_LOG("elementsCount = ",
              elementsCount,
              ", zerosCount = ",
              zerosCount,
              ", nnzCount = ",
              elementsCount - zerosCount);

    auto sparseRate = static_cast<float>(zerosCount) / static_cast<float>(elementsCount);

    DEBUG_LOG("Sparse rate = ",
              sparseRate * 100,
              "%, min sparse rate = ",
              minSparseRate * 100,
              "%, use sparse weights = ",
              sparseRate >= minSparseRate);

    return sparseRate >= minSparseRate;
}

void FullyConnected::initSupportedPrimitiveDescriptors() {
    attrs.withBias = getOriginalInputsNumber() == 3;
    attrs.dequantizationScales = getDQScales();
    attrs.sparseWeights = useSparseWeightsDecompression(getParentEdgeAt(WEIGHTS_ID)->getParent(),
                                                        getOriginalInputPrecisionAtPort(DATA_ID),
                                                        context->getConfig().fcSparseWeiDecompressionRate);
    attrs.dynamicQuantizationGroupSize = context->getConfig().fcDynamicQuantizationGroupSize;
    postOps = getPostOps(fusedWith);

    const auto& srcTypes = getOriginalInputPrecisions();
    auto dstTypes = getOriginalOutputPrecisions();
    // @todo graph optimizer should update original output precisions instead
    if (!fusedWith.empty())
        dstTypes = fusedWith.back()->getOriginalOutputPrecisions();

    VecMemoryDescs srcDescs;
    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    for (size_t i = 0; i < srcTypes.size(); i++) {
        const auto srcDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(srcTypes[i], getInputShapeAtPort(i));
        srcDescs.push_back(srcDesc);
    }

    VecMemoryDescs dstDescs;
    for (size_t i = 0; i < dstTypes.size(); i++) {
        const auto dstDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(dstTypes[i], getOutputShapeAtPort(i));
        dstDescs.push_back(dstDesc);
    }

    auto createEmptyMemoryDesc = [](const ov::element::Type type) {
        return std::make_shared<CpuBlockedMemoryDesc>(type, Shape{0});
    };

    MemoryDescArgs descs{
        {ARG_SRC, srcDescs[0]},
        {ARG_WEI, srcDescs[1]},
        {ARG_BIAS, attrs.withBias ? srcDescs[2] : createEmptyMemoryDesc(ov::element::undefined)},
        {ARG_DST, dstDescs[0]},
    };

    auto executionContext = std::make_shared<ExecutorContext>(context, getImplPriority(), privateWeightCache);
    factory = std::make_shared<ExecutorFactory<FCAttrs, node::FullyConnected>>(attrs, postOps, executionContext, descs);
    const auto nodeDescriptors = factory->getProperMemoryDescriptors(descs);

    NodeConfig nodeConfig;
    nodeConfig.inConfs.emplace_back(nodeDescriptors.at(ARG_SRC));
    nodeConfig.inConfs.emplace_back(nodeDescriptors.at(ARG_WEI));
    if (attrs.withBias) nodeConfig.inConfs.emplace_back(nodeDescriptors.at(ARG_BIAS));

    const int inPlace = canBeInPlace() ? 0 : -1;
    nodeConfig.outConfs.emplace_back(nodeDescriptors.at(ARG_DST), BlockedMemoryDesc::FULL_MASK, inPlace);

    supportedPrimitiveDescriptors.emplace_back(nodeConfig, impl_desc_type::undef);
}

void FullyConnected::createPrimitive() {
    memory[ARG_SRC] = getSrcMemoryAtPort(DATA_ID);
    memory[ARG_WEI] = getSrcMemoryAtPort(WEIGHTS_ID);
    memory[ARG_BIAS] = attrs.withBias ? getSrcMemoryAtPort(BIAS_ID) : emptyMemory;
    memory[ARG_DST] = getDstMemoryAtPort(0);
    // @todo should we preconfigure only for dynamic shapes?
    // Since for static shapes primitive is created in scope of compile_model() anyway
    factory->preconfigure(memory);

    Node::createPrimitive();
}

ov::element::Type FullyConnected::getRuntimePrecision() const {
    std::vector<ov::element::Type> srcTypes;
    // Don't take bias precision into account
    const size_t inputsNumLimit = 2;
    const auto inputSize = std::min(getParentEdges().size(), inputsNumLimit);

    for (size_t i = 0; i < inputSize; i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (parentEdge && parentEdge->getStatus() == Edge::Status::Validated) {
            srcTypes.emplace_back(parentEdge->getMemoryPtr()->getPrecision());
        }
    }

    return getMaxPrecision(srcTypes);
}

void FullyConnected::fuseDecompressionMultiply(const MemoryCPtr& memory) {
    attrs.decompressionMultiplyPtr = memory;
}

void FullyConnected::fuseDecompressionSubtract(const MemoryCPtr& memory) {
    attrs.decompressionSubtractPtr = memory;
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
