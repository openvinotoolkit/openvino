// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fullyconnected.h"

#include <cpu/x64/cpu_isa_traits.hpp>
#include <memory>
#include <openvino/op/constant.hpp>

#include "common/cpu_convert.h"
#include "common/cpu_memcpy.h"
#include "dnnl_extension_utils.h"
#include "executors/memory_arguments.hpp"
#include "graph_context.h"
#include "input.h"
#include "memory_desc/blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/threading/cpu_message.hpp"
#include "post_ops.hpp"
#include "shape_inference/custom/fullyconnected.hpp"
#include "transformations/cpu_opset/common/op/fully_connected.hpp"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"

#include "fake_quantize.h"

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

void FullyConnected::initTensorParallelConfig(const GraphContext::CPtr context) {
    if (context->getCPUStreamExecutor()) {
        if (!context->getCPUStreamExecutor()->get_rank().empty()) {
            // init tp_cfg.w_rank and tp_cfg.w_size
            tp_cfg.w_rank = context->getCPUStreamExecutor()->get_rank()[0];
            tp_cfg.w_size = ov::threading::message_manager()->get_num_sub_streams();
            tp_cfg.enable_tensor_parallel = tp_cfg.w_size > 1 ? true : false;
            tp_cfg.sub_memory = context->getSubMemory();
        }
    }
}

FullyConnected::FullyConnected(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, FCShapeInferFactory(op)),
      errorPrefix("FullyConnected node with name '" + getName() + "'") {
    std::string errorMessage;
    initTensorParallelConfig(context);
    if (!isSupportedOperation(op, errorMessage))
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
}

bool FullyConnected::canBeExecutedInInt8() const {
    auto srcType = getOriginalInputPrecisionAtPort(0);
    auto weiType = getOriginalInputPrecisionAtPort(1);

    return one_of(srcType, ov::element::u8, ov::element::i8) && weiType == ov::element::i8;
}

void FullyConnected::needPrepareParamsForTensorParallel() {
    if (tp_cfg.enable_tensor_parallel) {
        // must call in dynamic
        const auto dstMemoryBuffer = getDstMemoryAtPort(0);

        auto split_parts = [](int len, int n) {
            int average = len / n;
            std::vector<int> parts(n, average);
            parts.back() = len - average * (n - 1);
            return parts;
        };

        int dim = -1;
        auto dst_shape = dstMemoryBuffer->getShape();
        auto dst_desc = dstMemoryBuffer->getDescPtr();
        auto dims = dst_shape.getDims();
        if (dim < 0) {
            dim += dims.size();
        }
        OPENVINO_ASSERT(static_cast<int>(dims[dim]) >= tp_cfg.w_size,
            getName() + " dim[" + std::to_string(dim) + "] is " + std::to_string(dims[dim]) + ", which is larger than w_size " + std::to_string(tp_cfg.w_size));
        auto splited_dim_vec = split_parts(dims[dim], tp_cfg.w_size);

        VectorDims new_dims = std::move(dims);
        new_dims[dim] = splited_dim_vec[tp_cfg.w_rank];
        auto memory_desc = dst_desc->cloneWithNewDims(new_dims, true);
        tp_cfg.cached_dst->redefineDesc(std::move(memory_desc));
        memory[ARG_DST] = tp_cfg.cached_dst;
    }
}

ExecutorPtr FullyConnected::createExecutor() {
    const auto& executor = factory->make(memory);
    getSelectedPrimitiveDescriptor()->setImplementationType(executor->implType());

    return executor;
}

void FullyConnected::prepareParams() {
    needPrepareParamsForTensorParallel();
    executor = createExecutor();
}

void FullyConnected::initTensorParallelSync() {
    if (tp_cfg.enable_tensor_parallel) {
        tp_cfg.id = tp_cfg.sub_memory->get_memory_id(tp_cfg.w_rank);
        OPENVINO_ASSERT(tp_cfg.id > 0, "Tensor Parallel Config ID cannot be negative.");
        tp_cfg.sub_memory->set_memory_used(tp_cfg.id, tp_cfg.w_rank);
        while (true) {
            std::lock_guard<std::mutex> lock(tp_cfg.sub_memory->_flagMutex);
            if (tp_cfg.sub_memory->_use_count[tp_cfg.id] == tp_cfg.w_size) {
                tp_cfg.sub_memory->_use_count[tp_cfg.id] = 0;
                for (int i = 0; i < tp_cfg.w_size; i++) {
                    tp_cfg.sub_memory->_memorys_table[tp_cfg.id][i].flag = false;
                }
            }
            if (tp_cfg.sub_memory->_use_count[tp_cfg.id] == 0) {
                break;
            }
        }
    }
}

void FullyConnected::execTensorParallelSync() {
    if (tp_cfg.enable_tensor_parallel) {
        // dst
        auto dst = getDstMemoryAtPort(0);
        auto dst_ptr = static_cast<uint8_t*>(dst->getData());

        auto& shape = dst->getShape();
        auto dims = shape.getDims();
        auto prec = dst->getPrecision();

        // cur dst
        auto cur_dst = memory[ARG_DST];

        auto split_parts = [](int len, int n) {
            int average = len / n;
            std::vector<int> parts(n, average);
            parts.back() = len - average * (n - 1);
            return parts;
        };

        const int dim = dims.size() - 1;
        // selected dim bytes
        auto channel_size = dims[dim] * prec.size();
        // total bytes
        auto mem_size = dst->getSize();
        // the steps need to copy.
        const size_t count = (mem_size / channel_size);

        auto splited_dim_vec = split_parts(dims[dim], tp_cfg.w_size);
        const auto strideSize = splited_dim_vec[0] * prec.size();

        tp_cfg.sub_memory->_memorys_table[tp_cfg.id][tp_cfg.w_rank].send_buf = cur_dst->getData();
        tp_cfg.sub_memory->_memorys_table[tp_cfg.id][tp_cfg.w_rank].flag = true;

        std::vector<int> wait_list(tp_cfg.w_size, 1);
        while (true) {
            int wait_size = 0;
            for (int idx = 0; idx < tp_cfg.w_size; idx++) {
                if (wait_list[idx] > 0 && tp_cfg.sub_memory->_memorys_table[tp_cfg.id][idx].flag) {
                    auto new_ptr = static_cast<uint8_t*>(tp_cfg.sub_memory->_memorys_table[tp_cfg.id][idx].send_buf);
                    const auto copySize = splited_dim_vec[idx] * prec.size();    // bytes of half selected dim.
                    const size_t unloop = 8;
                    size_t step = count / unloop;
                    parallel_for(step, [&](size_t i){
                        cpu_memcpy(dst_ptr + idx * strideSize + (i * unloop) * channel_size, new_ptr + (i * unloop) * copySize, copySize);
                        cpu_memcpy(dst_ptr + idx * strideSize + (i * unloop + 1) * channel_size, new_ptr + (i * unloop + 1) * copySize, copySize);
                        cpu_memcpy(dst_ptr + idx * strideSize + (i * unloop + 2) * channel_size, new_ptr + (i * unloop + 2) * copySize, copySize);
                        cpu_memcpy(dst_ptr + idx * strideSize + (i * unloop + 3) * channel_size, new_ptr + (i * unloop + 3) * copySize, copySize);
                        cpu_memcpy(dst_ptr + idx * strideSize + (i * unloop + 4) * channel_size, new_ptr + (i * unloop + 4) * copySize, copySize);
                        cpu_memcpy(dst_ptr + idx * strideSize + (i * unloop + 5) * channel_size, new_ptr + (i * unloop + 5) * copySize, copySize);
                        cpu_memcpy(dst_ptr + idx * strideSize + (i * unloop + 6) * channel_size, new_ptr + (i * unloop + 6) * copySize, copySize);
                        cpu_memcpy(dst_ptr + idx * strideSize + (i * unloop + 7) * channel_size, new_ptr + (i * unloop + 7) * copySize, copySize);
                    });
                    size_t tail = count & ~(unloop - 1);
                    for (size_t i = tail; i < count; ++i) {
                        size_t dst_offset = i * channel_size + idx * strideSize;
                        size_t src_offset = i * copySize;
                        cpu_parallel_memcpy(dst_ptr + dst_offset, new_ptr + src_offset, copySize);
                    }
                    wait_list[idx] = 0;
                }
                wait_size += wait_list[idx];
            }
            if (wait_size == 0) {
                break;
            }
        }
        {
            std::lock_guard<std::mutex> lock(tp_cfg.sub_memory->_flagMutex);
            tp_cfg.sub_memory->_use_count[tp_cfg.id]++;
        }
    }
}
void FullyConnected::execute(dnnl::stream strm) {
    initTensorParallelSync();

    executor->execute(memory);

    execTensorParallelSync();
}

void FullyConnected::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool FullyConnected::canFuse(const NodePtr& node) const {
#if defined(OV_CPU_WITH_SHL)
    return false;
#endif
    if (node->getType() == Type::FakeQuantize) {
        auto* fq = dynamic_cast<FakeQuantize*>(node.get());
        if (!fq) {
            DEBUG_LOG("Invalid dynamic_cast FakeQuantize pointer");
            return false;
        }
        if (fq->getBroadcastingPolicy() != FakeQuantize::BroadcastingPolicy::PerTensor) {
            const auto& dstShape = getOutputShapeAtPort(0);
            auto dataRanks = dstShape.getRank();
            // only per-OC or per-Tensor fakequantize can be postOps
            if (fq->getAxis() != dataRanks - 1) {
                DEBUG_LOG("reject FakeQuantize ",
                          fq->getName(),
                          "(axis=",
                          fq->getAxis(),
                          ") from fusing into ",
                          getName(),
                          " with dst shape ",
                          dstShape);
                return false;
            }
        }
    }
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
        impl_desc_type::shl,
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

void FullyConnected::needUpdateDQScaleForTensorParallel(std::vector<float>& dequantizationScales) {
    if (tp_cfg.enable_tensor_parallel) {
        auto split_parts = [](int len, int n) {
            int average = len / n;
            std::vector<int> parts(n, average);
            parts.back() = len - average * (n - 1);
            return parts;
        };
        auto DQScales = getDQScales();
        auto split_lens = split_parts(DQScales.size(), tp_cfg.w_size);
        auto split_offset = tp_cfg.w_rank * split_lens[0];
        std::vector<float> newDQScales(split_lens[tp_cfg.w_rank]);
        std::copy(DQScales.begin() + split_offset, DQScales.begin() + split_offset + split_lens[tp_cfg.w_rank], newDQScales.begin());
        dequantizationScales = std::move(newDQScales);
    }
}

void FullyConnected::initSupportedPrimitiveDescriptors() {
    attrs.withBias = getOriginalInputsNumber() == 3;

    attrs.dequantizationScales = getDQScales();
    needUpdateDQScaleForTensorParallel(attrs.dequantizationScales);

    attrs.sparseWeights = useSparseWeightsDecompression(getParentEdgeAt(WEIGHTS_ID)->getParent(),
                                                        getOriginalInputPrecisionAtPort(DATA_ID),
                                                        context->getConfig().fcSparseWeiDecompressionRate);
    attrs.dynamicQuantizationGroupSize = context->getConfig().fcDynamicQuantizationGroupSize;
    attrs.modelType = context->getConfig().modelType;

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

    MemoryDescArgs descs{
        {ARG_SRC, srcDescs[0]},
        {ARG_WEI, srcDescs[1]},
        {ARG_BIAS, attrs.withBias ? srcDescs[2] : MemoryDescUtils::makeEmptyDesc()},
        {ARG_DST, dstDescs[0]},
    };

    needUpdateScaleForTensorParallel();
    needUpdateZeroPointForTensorParallel();

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

void FullyConnected::needSplitMemoryForTensorParallel() {
    if (tp_cfg.enable_tensor_parallel) {
        auto src = getSrcMemoryAtPort(DATA_ID);
        auto wgt = getSrcMemoryAtPort(WEIGHTS_ID);
        auto dst = getDstMemoryAtPort(0);
        // src
        memory[ARG_SRC] = getSrcMemoryAtPort(DATA_ID);
        // wgt
        // split N direction
        tp_cfg.cached_splited_weight = attrs.weightsNonTransposed ? split_vertical(context->getEngine(), std::move(wgt), 0, tp_cfg.w_rank, tp_cfg.w_size)
                    : split_horizontal(context->getEngine(), std::move(wgt), 0, tp_cfg.w_rank, tp_cfg.w_size);
        memory[ARG_WEI] = tp_cfg.cached_splited_weight;
        // bias
        if (attrs.withBias) {
            auto bias = getSrcMemoryAtPort(BIAS_ID);
            auto select_bias = split_horizontal(context->getEngine(), std::move(bias), 0, tp_cfg.w_rank, tp_cfg.w_size);
            tp_cfg.cached_splited_bias = std::move(select_bias);
        } else {
            tp_cfg.cached_splited_bias = MemoryDescUtils::makeEmptyMemory(context);
        }
        memory[ARG_BIAS] = tp_cfg.cached_splited_bias;
        // dst
        memory[ARG_DST] = getDstMemoryAtPort(0);
        tp_cfg.cached_dst = split_horizontal(context->getEngine(), std::move(dst), -1, tp_cfg.w_rank, tp_cfg.w_size, false);
    }
}

void FullyConnected::needUpdateTensorParalelConfig() {
    // tensor parallel should be disabled in two conditions.
    // 1. weight shape is dynamic
    // 2. last dim can be splited.
    if (tp_cfg.enable_tensor_parallel) {
        auto& shape = getSrcMemoryAtPort(WEIGHTS_ID)->getShape();
        if (shape.isDynamic()) {
            tp_cfg.enable_tensor_parallel = false;
        } else if (shape.getDims()[0] < static_cast<size_t>(tp_cfg.w_size)) {
            tp_cfg.enable_tensor_parallel = false;
        }
    }
}
void FullyConnected::createPrimitive() {
    needUpdateTensorParalelConfig();

    memory[ARG_SRC] = getSrcMemoryAtPort(DATA_ID);
    memory[ARG_WEI] = getSrcMemoryAtPort(WEIGHTS_ID);
    memory[ARG_BIAS] = attrs.withBias ? getSrcMemoryAtPort(BIAS_ID) : MemoryDescUtils::makeEmptyMemory(context);
    memory[ARG_DST] = getDstMemoryAtPort(0);

    needSplitMemoryForTensorParallel();
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

void FullyConnected::needUpdateScaleForTensorParallel() {
    if (tp_cfg.enable_tensor_parallel && tp_cfg.cached_scale) {
        attrs.decompressionMultiplyPtr = tp_cfg.cached_scale;
    }
}

void FullyConnected::needSplitScaleForTensorParallel(const MemoryCPtr& memory) {
    if (tp_cfg.enable_tensor_parallel && !tp_cfg.cached_scale) {
        auto scale_mem = std::const_pointer_cast<IMemory>(memory);
        tp_cfg.cached_scale = attrs.weightsNonTransposed ? split_vertical(context->getEngine(), std::move(scale_mem), 0, tp_cfg.w_rank, tp_cfg.w_size)
                       : split_horizontal(context->getEngine(), std::move(scale_mem), 0, tp_cfg.w_rank, tp_cfg.w_size);
    }
}

void FullyConnected::needUpdateZeroPointForTensorParallel() {
    if (tp_cfg.enable_tensor_parallel && tp_cfg.cached_zeropoint) {
        attrs.decompressionSubtractPtr = tp_cfg.cached_zeropoint;
    }
}

void FullyConnected::needSplitZeroPointForTensorParallel(const MemoryCPtr& memory) {
    if (tp_cfg.enable_tensor_parallel && !tp_cfg.cached_zeropoint) {
        auto zeropoint_mem = std::const_pointer_cast<IMemory>(memory);
        auto element_num = memory->getSize() / memory->getPrecision().size();
        if (element_num == 1) {
            tp_cfg.cached_zeropoint = std::move(zeropoint_mem);
        } else {
            tp_cfg.cached_zeropoint = attrs.weightsNonTransposed ? split_vertical(context->getEngine(), zeropoint_mem, 0, tp_cfg.w_rank, tp_cfg.w_size)
                                : split_horizontal(context->getEngine(), zeropoint_mem, 0, tp_cfg.w_rank, tp_cfg.w_size);
        }
    }
}

void FullyConnected::fuseDecompressionMultiply(const MemoryCPtr& memory) {
    attrs.decompressionMultiplyPtr = memory;
    needSplitScaleForTensorParallel(memory);
}

void FullyConnected::fuseDecompressionSubtract(const MemoryCPtr& memory) {
    attrs.decompressionSubtractPtr = memory;
    needSplitZeroPointForTensorParallel(memory);
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
