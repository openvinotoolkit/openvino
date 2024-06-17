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
#include "utils/profiler.hpp"

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
    // init w_rank and w_size
    if (!context->getCPUStreamExecutor()->get_rank().empty() && std::getenv("ENABLE_TP")) {
        enable_tensor_parallel = std::getenv("ENABLE_TP") == nullptr ? false : true;
        w_rank = context->getCPUStreamExecutor()->get_rank()[0];
        w_size = 2;
        // std::cout << "[dbg] w_rank: " << w_rank << ", w_size: " << w_size << "\n";
        sub_memory = context->getSubMemory();
    }
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage))
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
}

bool FullyConnected::canBeExecutedInInt8() const {
    auto srcType = getOriginalInputPrecisionAtPort(0);
    auto weiType = getOriginalInputPrecisionAtPort(1);

    return one_of(srcType, ov::element::u8, ov::element::i8) && weiType == ov::element::i8;
}

ExecutorPtr FullyConnected::createExecutor() {
    if (enable_tensor_parallel) {
        // must call in dynamic
        auto dstMemoryBuffer = getDstMemoryAtPort(0);
        auto select_dst = split_vertical(context->getEngine(), dstMemoryBuffer, -1, w_rank, w_size, false);
        memory[ARG_DST] = select_dst;
    }
    const auto& executor = factory->make(memory);
    getSelectedPrimitiveDescriptor()->setImplementationType(executor->implType());

    return executor;
}

void FullyConnected::prepareParams() {
    executor = createExecutor();
}

void FullyConnected::execute(dnnl::stream strm) {
    if (enable_tensor_parallel) {
        PROFILE(_prof1, "fc_sync");
        id = sub_memory->get_memory_id(w_rank);
        sub_memory->set_memory_used(id, w_rank);
        while (true) {
            std::lock_guard<std::mutex> lock(sub_memory->_flagMutex);
            if (sub_memory->_use_count[id] == w_size) {
                sub_memory->_use_count[id] = 0;
                for (int i = 0; i < w_size; i++) {
                    sub_memory->_memorys_table[id][i].flag = false;
                }
            }
            if (sub_memory->_use_count[id] == 0) {
                break;
            }
        }
    }

    {
        PROFILE(_prof1, "fc_exec");
        executor->execute(memory);
    }

    if (enable_tensor_parallel) {
        PROFILE(_prof1, "fc_post");
        // dst
        auto dst = getDstMemoryAtPort(0);
        auto dst_ptr = static_cast<uint8_t*>(dst->getData());

        // cur dst
        auto cur_dst = memory[ARG_DST];
        auto shape = dst->getShape();
        auto dims = shape.getDims();
        auto prec = dst->getPrecision();
        auto mem_size = dst->getSize(); // total bytes

        const int dim = dims.size() - 1;
        auto channel_size = dims[dim] * prec.size();    // selected dim bytes
        // const int step = (mem_size / channel_size);     // the steps need to copy.
        const size_t count = (mem_size / channel_size);     // the steps need to copy.

        const auto copySize = (dims[dim] / w_size) * prec.size();    // bytes of half selected dim.
        sub_memory->_memorys_table[id][w_rank].send_buf = cur_dst->getData();
        sub_memory->_memorys_table[id][w_rank].flag = true;

        std::vector<int> wait_list(w_size, 1);
        while (true) {
            int wait_size = 0;
            for (int idx = 0; idx < w_size; idx++) {
                if (wait_list[idx] > 0 && sub_memory->_memorys_table[id][idx].flag) {
                    auto new_ptr = static_cast<uint8_t*>(sub_memory->_memorys_table[id][idx].send_buf);
                    // parallel_for(step, [&](int i) {
                    //     int src_offset = i * copySize;
                    //     int dst_offset = i * copySize * 2 + idx * copySize;
                    //     cpu_parallel_memcpy(dst_ptr + dst_offset, new_ptr + src_offset, copySize);
                    // });
                    const size_t unloop = 8;
                    size_t step = count / unloop;
                    parallel_for(step, [&](size_t i){
                        // int src_offset = i * copySize;
                        // int dst_offset = i * copySize * 2 + idx * copySize;
                        cpu_memcpy(dst_ptr + copySize * (idx + 2 * (i * unloop)),     new_ptr + copySize * (i * unloop), copySize);
                        cpu_memcpy(dst_ptr + copySize * (idx + 2 * (i * unloop + 1)), new_ptr + copySize * (i * unloop + 1), copySize);
                        cpu_memcpy(dst_ptr + copySize * (idx + 2 * (i * unloop + 2)), new_ptr + copySize * (i * unloop + 2), copySize);
                        cpu_memcpy(dst_ptr + copySize * (idx + 2 * (i * unloop + 3)), new_ptr + copySize * (i * unloop + 3), copySize);
                        cpu_memcpy(dst_ptr + copySize * (idx + 2 * (i * unloop + 4)), new_ptr + copySize * (i * unloop + 4), copySize);
                        cpu_memcpy(dst_ptr + copySize * (idx + 2 * (i * unloop + 5)), new_ptr + copySize * (i * unloop + 5), copySize);
                        cpu_memcpy(dst_ptr + copySize * (idx + 2 * (i * unloop + 6)), new_ptr + copySize * (i * unloop + 6), copySize);
                        cpu_memcpy(dst_ptr + copySize * (idx + 2 * (i * unloop + 7)), new_ptr + copySize * (i * unloop + 7), copySize);
                    });
                    size_t tail = count & ~(unloop - 1);
                    for (size_t i = tail; i < count; ++i) {
                        int src_offset = i * copySize;
                        int dst_offset = i * copySize * 2 + idx * copySize;
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
            std::lock_guard<std::mutex> lock(sub_memory->_flagMutex);
            sub_memory->_use_count[id]++;
        }
    }
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

    if (enable_tensor_parallel && cached_scale && cached_zeropoint) {
        attrs.decompressionMultiplyPtr = cached_scale;
        attrs.decompressionSubtractPtr = cached_zeropoint;
    }

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
    auto src = getSrcMemoryAtPort(DATA_ID);
    auto wgt = getSrcMemoryAtPort(WEIGHTS_ID);
    auto dst = getDstMemoryAtPort(0);
    if (enable_tensor_parallel) {
        // src
        memory[ARG_SRC] = getSrcMemoryAtPort(DATA_ID);
        // wgt
        // split N direction
        cached_splited_weight = attrs.weightsNonTransposed ? split_vertical(context->getEngine(), wgt, 0, w_rank, w_size)
                                : split_horizontal(context->getEngine(), wgt, 0, w_rank, w_size);
        memory[ARG_WEI] = cached_splited_weight;

        // bias
        if (attrs.withBias) {
            auto bias = getSrcMemoryAtPort(BIAS_ID);
            auto select_bias = split_horizontal(context->getEngine(), bias, 0, w_rank, w_size);
            cached_splited_bias = select_bias;
        } else {
            cached_splited_bias = MemoryDescUtils::makeEmptyMemory(context);
        }

        memory[ARG_BIAS] = cached_splited_bias;

        // dst
        memory[ARG_DST] = getDstMemoryAtPort(0);
    } else {
        memory[ARG_SRC] = getSrcMemoryAtPort(DATA_ID);
        memory[ARG_WEI] = getSrcMemoryAtPort(WEIGHTS_ID);
        memory[ARG_BIAS] = attrs.withBias ? getSrcMemoryAtPort(BIAS_ID) : MemoryDescUtils::makeEmptyMemory(context);
        memory[ARG_DST] = getDstMemoryAtPort(0);
    }
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
    if (enable_tensor_parallel && !cached_scale) {
        auto scale_mem = std::const_pointer_cast<IMemory>(memory);
        cached_scale = attrs.weightsNonTransposed ? split_vertical(context->getEngine(), scale_mem, 0, w_rank, w_size)
                       : split_horizontal(context->getEngine(), scale_mem, 0, w_rank, w_size);
    }
}

void FullyConnected::fuseDecompressionSubtract(const MemoryCPtr& memory) {
    attrs.decompressionSubtractPtr = memory;
    if (enable_tensor_parallel && !cached_zeropoint) {
        auto zeropoint_mem = std::const_pointer_cast<IMemory>(memory);
        auto element_num = memory->getSize() / memory->getPrecision().size();
        if (element_num == 1) {
            cached_zeropoint = zeropoint_mem;
        } else {
            cached_zeropoint = attrs.weightsNonTransposed ? split_vertical(context->getEngine(), zeropoint_mem, 0, w_rank, w_size)
                                : split_horizontal(context->getEngine(), zeropoint_mem, 0, w_rank, w_size);
        }
    }
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
