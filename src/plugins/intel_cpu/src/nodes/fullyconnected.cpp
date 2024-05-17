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
#include "kernels/ccl/allreduce.hpp"

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
        const char* str_enable = std::getenv("ENABLE_TP");
        tp_mode = std::atoi(str_enable);
        if (!(tp_mode == 1 || tp_mode == 2 || tp_mode == 3)) {
            printf("current tp mode just supports 1(allreduce), 2(allgather_h), 3(allgather_v), %d is unexpeced!\n", tp_mode);
            exit(-1);
        } else {
            // printf("[dbg] %s is in %d mode.\n", getName().c_str(), tp_mode);
        }
        w_rank = context->getCPUStreamExecutor()->get_rank()[0];
        // TODO@Xiaoxia: get correct stream num
        // context->getConfig().streamExecutorConfig.get_streams();
        w_size = 2;
        // std::cout << "[dbg] w_rank: " << w_rank << ", w_size: " << w_size << "\n";
        message = ov::threading::message_manager();
    }
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage))
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
}

void FullyConnected::allreduce(void *send_buf, void *recv_buf, size_t count, ov::element::Type dtype) {
    ov::threading::MessageInfo send_message;
    send_message.msg_type = ov::threading::MsgType::TP;
    send_message.rank = {w_rank};
    send_message.buf = send_buf;
    message->send_message(send_message);
    auto vec_message = message->wait_message(/*cur_rank*/w_rank, /*streams_num*/w_size);
    if (dtype == ov::element::f32) {
        float* recv_ptr = static_cast<float*>(recv_buf);
        for (int idx=0; idx < w_size; ++idx) {
            // if (idx == w_rank) {
            //     continue;
            // }
            float* send_ptr = static_cast<float*>(vec_message[idx].buf);
            ov::Extensions::Cpu::XARCH::allreduce_float32(send_ptr, recv_ptr, count);
        }
    } else if (dtype == ov::element::bf16) {
        ov::bfloat16* recv_ptr = static_cast<ov::bfloat16*>(recv_buf);
        for (int idx=0; idx < w_size; ++idx) {
            // if (idx == w_rank) {
            //     continue;
            // }
            ov::bfloat16* send_ptr = static_cast<ov::bfloat16*>(vec_message[idx].buf);
            ov::Extensions::Cpu::XARCH::allreduce_bfloat16(send_ptr, recv_ptr, count);
        }
    } else {
        printf("Unsupported data type for reduceAdd.\n");
        exit(-1);
    }
    // sync before return
    ov::threading::MessageInfo msg_info;
    msg_info.msg_type = ov::threading::MsgType::REDUCE;
    message->send_message(msg_info);
    message->reduce_wait(w_rank, w_size);
}

bool FullyConnected::canBeExecutedInInt8() const {
    auto srcType = getOriginalInputPrecisionAtPort(0);
    auto weiType = getOriginalInputPrecisionAtPort(1);

    return one_of(srcType, ov::element::u8, ov::element::i8) && weiType == ov::element::i8;
}

ExecutorPtr FullyConnected::createExecutor() {
    // std::cout << "[dbg] create executor\n";
    if (tp_mode == 1) {
        auto srcMemoryBuffer = getSrcMemoryAtPort(DATA_ID);
        auto select_src = split_v(srcMemoryBuffer, -1, w_rank, w_size);
        memory[ARG_SRC] = select_src;

        // TODO: update dst shape once output shape is changed.
        memory[ARG_DST] = getDstMemoryAtPort(0);
    }
    if (tp_mode == 2) {
        // src keep
        // memory[ARG_SRC] = getSrcMemoryAtPort(DATA_ID);
        // memory[ARG_BIAS] = cached_splited_bias;

        // weight
        // memory[ARG_WEI] = cached_splited_weight;
        // {
        //     printf("[dbg] w_rank: %d, wgt address=%p\n", w_rank, cached_splited_weight->getData());
        //     auto f32_wei = cached_splited_weight->getDataAs<float>();
        //     printf("[dbg] w_rank: %d, %f - %f - %f - %f - %f - %f\n", w_rank, f32_wei[0], f32_wei[1], f32_wei[2], f32_wei[3], f32_wei[4], f32_wei[5]);
        // }
        // memory[ARG_BIAS] = cached_splited_bias;

        // must call in dynamic
        auto dstMemoryBuffer = getDstMemoryAtPort(0);
        auto select_dst = split_v(dstMemoryBuffer, -1, w_rank, w_size);
        memory[ARG_DST] = select_dst;
    }
    if (tp_mode == 3) {
        auto srcMemoryBuffer = getSrcMemoryAtPort(DATA_ID);
        auto select_src = split_h(srcMemoryBuffer, 0, w_rank, w_size);
        memory[ARG_SRC] = select_src;

        if (attrs.withBias && w_rank == 0) {
            auto bias = getSrcMemoryAtPort(BIAS_ID);
            auto select_bias = split_v(bias, 0, w_rank, w_size);
            memory[ARG_BIAS] = select_bias;
        } else {
            memory[ARG_BIAS] = MemoryDescUtils::makeEmptyMemory(context);
        }

        auto dstMemoryBuffer = getDstMemoryAtPort(0);
        auto select_dst = split_h(dstMemoryBuffer, 0, w_rank, w_size);
        memory[ARG_DST] = select_dst;
    }
    // std::cout << "[dbg] w_rank: " << w_rank
    //         << ", src_shape: " << memory[ARG_SRC]->getShape().toString()
    //         << ", wgt_shape: " << memory[ARG_WEI]->getShape().toString()
    //         << ", dst_shape: " << memory[ARG_DST]->getShape().toString()
    //         << "\n";
    const auto& executor = factory->make(memory);
    getSelectedPrimitiveDescriptor()->setImplementationType(executor->implType());
    // std::cout << "[dbg] finish executor\n";

    return executor;
}

void FullyConnected::prepareParams() {
    executor = createExecutor();
}

void FullyConnected::execute(dnnl::stream strm) {
    if (tp_mode == 1) {
        auto srcMemoryBuffer = getSrcMemoryAtPort(DATA_ID);
        auto select_src = split_v(srcMemoryBuffer, -1, w_rank, w_size);
        memory[ARG_SRC] = select_src;
    }
    if (tp_mode == 2) {
        // memory[ARG_SRC] = getSrcMemoryAtPort(DATA_ID);
        // memory[ARG_BIAS] = cached_splited_bias;
        // memory[ARG_WEI] = cached_splited_weight;
        //
        // auto dstMemoryBuffer = getDstMemoryAtPort(0);
        // auto select_dst = split_v(dstMemoryBuffer, -1, w_rank, w_size);
        // memory[ARG_DST] = select_dst;
        // printf("[dbg ###] w_rank:%d, src_address:%p, wei_address:%p, dst_address:%p\n",
        //     w_rank, memory[ARG_SRC]->getData(), memory[ARG_WEI]->getData(), memory[ARG_DST]->getData());
    }
    if (tp_mode == 3) {
        auto srcMemoryBuffer = getSrcMemoryAtPort(DATA_ID);
        auto select_src = split_h(srcMemoryBuffer, 0, w_rank, w_size);
        memory[ARG_SRC] = select_src;

        // if (attrs.withBias && w_rank == 0) {
        //     auto bias = getSrcMemoryAtPort(BIAS_ID);
        //     auto select_bias = split_v(bias, 0, w_rank, w_size);
        //     memory[ARG_BIAS] = select_bias;
        // } else {
        //     memory[ARG_BIAS] = MemoryDescUtils::makeEmptyMemory(context);
        // }

        // need not update the output buffer here!
        // auto dstMemoryBuffer = getDstMemoryAtPort(0);
        // auto select_dst = split_h(dstMemoryBuffer, 0, w_rank, w_size);
        // memory[ARG_DST] = select_dst;
    }

    {
        executor->execute(memory);
    }

    if (tp_mode == 1) {
        // post process output
        auto send_mem = memory[ARG_DST];
        auto send_ptr = send_mem->getData();
        auto prec = send_mem->getPrecision();
        auto ele_num = send_mem->getSize() / prec.size();
        // MemoryPtr recv_mem = std::make_shared<Memory>(context->getEngine(), send_mem->getDescPtr(), send_ptr);
        MemoryPtr recv_mem = std::make_shared<Memory>(context->getEngine(), send_mem->getDescPtr(), nullptr);
        auto recv_ptr = recv_mem->getData();
        memset(recv_ptr, 0, recv_mem->getSize());
        // TODO
        allreduce(send_ptr, recv_ptr, ele_num, prec);

        cpu_parallel_memcpy(send_ptr, recv_ptr, send_mem->getSize());
        memory[ARG_DST] = send_mem;
    }

    if (tp_mode == 2) {
        // dst
        auto dst = getDstMemoryAtPort(0);
        auto dst_ptr = static_cast<uint8_t*>(dst->getData());

        // cur dst
        auto cur_dst = memory[ARG_DST];
        auto cur_dst_ptr = static_cast<uint8_t*>(cur_dst->getData());

        auto shape = dst->getShape();
        // {
        //     printf("[dbg] dst shape: %s\n", shape.toString().c_str());
        //     auto fp32_ptr = cur_dst->getDataAs<float>();
        //     printf("[dbg] cur buffer shape: %s\n", cur_dst->getShape().toString().c_str());
        //     printf("[dbg] cur buffer, w_rank=%d, %f - %f - %f - %f - %f - %f -%f\n",
        //         w_rank, fp32_ptr[0], fp32_ptr[1], fp32_ptr[2], fp32_ptr[3], fp32_ptr[4], fp32_ptr[5], fp32_ptr[6]);
        // }
        auto dims = shape.getDims();
        auto prec = dst->getPrecision();
        auto mem_size = dst->getSize(); // total bytes

        const int dim = dims.size() - 1;
        auto channel_size = dims[dim] * prec.size();    // selected dim bytes
        const int step = (mem_size / channel_size);     // the steps need to copy.

        const auto copySize = (dims[dim] / w_size) * prec.size();    // bytes of half selected dim.
        parallel_for(step, [&](int i){
            int src_offset = i * copySize;
            int dst_offset = i * copySize* 2 + w_rank * copySize;
            cpu_parallel_memcpy(dst_ptr + dst_offset, cur_dst_ptr + src_offset, copySize);
        });

        ov::threading::MessageInfo send_message;
        send_message.msg_type = ov::threading::MsgType::TP;
        send_message.rank = {w_rank};
        send_message.buf = cur_dst->getData();
        message->send_message(send_message);

        auto vec_message = message->wait_message(/*cur_rank*/w_rank, /*streams_num*/w_size);

        for (int i=0; i<w_size; ++i) {
            const int recv_rank = vec_message[i].rank[0];
            if (recv_rank == w_rank) {
                continue;
            }
            // auto fp32_ptr = static_cast<float*>(vec_message[i].buf);
            // printf("[dbg] idx=%d, w_rank=%d, %f - %f - %f - %f - %f\n",i, w_rank, fp32_ptr[0], fp32_ptr[1], fp32_ptr[2], fp32_ptr[3], fp32_ptr[4]);
            auto new_ptr = static_cast<uint8_t*>(vec_message[i].buf);
            parallel_for(step, [&](int i){
                int src_offset = i * copySize;
                int dst_offset = i * copySize* 2 + recv_rank * copySize;
                cpu_parallel_memcpy(dst_ptr + dst_offset, new_ptr + src_offset, copySize);
            });
        }
    }

    if (tp_mode == 3) {
        // dst
        auto dst = getDstMemoryAtPort(0);
        auto dst_ptr = static_cast<uint8_t*>(dst->getData());
        const auto copySize = dst->getSize() / w_size;
        // cur dst
        auto cur_dst = memory[ARG_DST];
        auto cur_dst_ptr = static_cast<uint8_t*>(cur_dst->getData());
        // std::cout << "[dbg] target size: " << dst->getSize() << ", current size: " << copySize << "\n";
        // copy cur dst buffer to dst buffer. But cur dst buffer should be in dst buffer already.
        // TODO@Xiuchuan: may optimize here!
        // printf("[dbg] w_rank=%d, target address:%d, current address:%d\n", w_rank, dst_ptr + w_rank * copySize, cur_dst_ptr);
        cpu_parallel_memcpy(dst_ptr + w_rank * copySize, cur_dst_ptr, copySize);
        // sync with another stream's buffer
        ov::threading::MessageInfo send_message;
        send_message.msg_type = ov::threading::MsgType::TP;
        send_message.rank = {w_rank};
        send_message.buf = cur_dst->getData();
        message->send_message(send_message);
        // auto fp32_cur = cur_dst->getDataAs<float>();
        // printf("[dbg] current w_rank=%d, %f - %f - %f - %f - %f\n", w_rank,fp32_cur[0], fp32_cur[1], fp32_cur[2], fp32_cur[3], fp32_cur[4]);
        auto vec_message = message->wait_message(/*cur_rank*/w_rank, /*streams_num*/w_size);
        for (int i=0; i<w_size; ++i) {
            const int recv_rank = vec_message[i].rank[0];
            if (recv_rank == w_rank) {
                continue;
            }
            // auto fp32_ptr = static_cast<float*>(vec_message[i].buf);
            // printf("[dbg] idx=%d, w_rank=%d, %f - %f - %f - %f - %f\n",i, w_rank, fp32_ptr[0], fp32_ptr[1], fp32_ptr[2], fp32_ptr[3], fp32_ptr[4]);
            auto new_ptr = static_cast<uint8_t*>(vec_message[i].buf);
            cpu_parallel_memcpy(dst_ptr + recv_rank * copySize, new_ptr, copySize);
        }
        // No sync here. First done, first finish.
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

MemoryPtr FullyConnected::split_h(const MemoryPtr src, int dim, int w_rank, int w_size) {
    auto desc = src->getDescPtr();
    auto shape = src->getShape();
    auto dims = shape.getDims();
    auto prec = src->getPrecision();
    if (dim < 0) {
        dim += dims.size();
    }
    if (shape.isDynamic()) {
        // if the dim is dynamic, should return a dynamic dim without any change.
        // if the dim is static, should split it indeed.
        const auto& pshape = shape.toPartialShape();
        if (pshape[dim].is_dynamic()) {
            return src;
        }
        auto new_pshape = pshape;
        new_pshape[dim] = new_pshape[dim] / w_size;
        auto new_desc = std::make_shared<CpuBlockedMemoryDesc>(prec, Shape{new_pshape});
        MemoryPtr ptr = std::make_shared<Memory>(context->getEngine(), new_desc);
        return ptr;
    }
    // auto element_size = prec.size();
    VectorDims new_dims = dims;
    new_dims[dim] = dims[dim] / w_size;
    // const int stride = dims[dim] / w_size;
    auto new_desc = desc->cloneWithNewDims(new_dims, true);
    auto srcPtr = static_cast<uint8_t*>(src->getData());
    const size_t stride = src->getSize() / w_size;

    MemoryPtr ptr = std::make_shared<Memory>(context->getEngine(), new_desc, srcPtr + w_rank * stride);
    return ptr;
}


MemoryPtr FullyConnected::split_v(const MemoryPtr src, int dim, int w_rank, int w_size) {
    auto desc = src->getDescPtr();
    auto shape = src->getShape();
    auto dims = shape.getDims();
    auto prec = src->getPrecision();
    if (dim < 0) {
        dim += dims.size();
    }
    if (shape.isDynamic()) {
        const auto& pshape = shape.toPartialShape();
        if (pshape[dim].is_dynamic()) {
            OPENVINO_THROW("Can't split data with dynamic shapes");
        }
        auto new_pshape = pshape;
        new_pshape[dim] = pshape[dim] / w_size;
        auto new_desc = std::make_shared<CpuBlockedMemoryDesc>(prec, Shape{new_pshape});
        MemoryPtr ptr = std::make_shared<Memory>(context->getEngine(), new_desc);
        return ptr;
    }
    auto element_size = prec.size();
    VectorDims new_dims = dims;
    new_dims[dim] = dims[dim] / w_size;
    auto new_desc = desc->cloneWithNewDims(new_dims, true);
    MemoryPtr ptr = std::make_shared<Memory>(context->getEngine(), new_desc);
    // copy
    auto srcPtr = static_cast<uint8_t*>(src->getData());
    auto dstPtr = static_cast<uint8_t*>(ptr->getData());
    auto mem_size = src->getSize(); // total bytes
    auto channel_size = dims[dim] * element_size; // selected dim bytes
    const int step = (mem_size / channel_size); // the steps need to copy.
    const int stride = dims[dim] / w_size; // elements of half selected dim.
    const auto copySize = stride * element_size; // bytes of half selected dim.
    parallel_for(step, [&](int i){
        int dst_offset = i * copySize;
        int src_offset = i * copySize* 2 + w_rank * copySize;
        cpu_parallel_memcpy(dstPtr + dst_offset, srcPtr + src_offset, copySize);
    });
    return ptr;
}

void FullyConnected::createPrimitive() {
    // printf("[dbg] create Primitive\n");
    auto src = getSrcMemoryAtPort(DATA_ID);
    auto wgt = getSrcMemoryAtPort(WEIGHTS_ID);
    auto dst = getDstMemoryAtPort(0);
    // {
    //     printf("src shape is %s\n", src->getShape().toString().c_str());
    //     printf("weight is transposed? %d. and weight shape is %s\n", attrs.weightsNonTransposed, wgt->getShape().toString().c_str());
    //     printf("dst shape is %s\n", dst->getShape().toString().c_str());
    // }
    if (tp_mode == 1) {
        auto select_src= split_v(src, -1, w_rank, w_size);
        auto select_wgt = attrs.weightsNonTransposed ? split_h(wgt, -1, w_rank, w_size)
                          : split_v(wgt, -1, w_rank, w_size);
        // TODO: by default, we consider the weight is constant.
        // cache for later reuse.
        memory[ARG_SRC] = select_src;
        memory[ARG_WEI] = select_wgt;
        if (attrs.withBias && w_rank == 0) {
            auto bias = getSrcMemoryAtPort(BIAS_ID);
            memory[ARG_BIAS] = getSrcMemoryAtPort(BIAS_ID);
        } else {
            memory[ARG_BIAS] = MemoryDescUtils::makeEmptyMemory(context);
        }
        memory[ARG_DST] = getDstMemoryAtPort(0);
    } else if (tp_mode == 2) {
        // src
        memory[ARG_SRC] = getSrcMemoryAtPort(DATA_ID);
        
        // wgt
        // split N direction
        cached_splited_weight = attrs.weightsNonTransposed ? split_v(wgt, 0, w_rank, w_size)
                                : split_h(wgt, 0, w_rank, w_size);
        // std::cout << "[dbg] ### " << cached_splited_weight->getShape().toString() << "\n";
        // memory[ARG_WEI] = wgt;
        memory[ARG_WEI] = cached_splited_weight;

        // bias
        if (attrs.withBias && w_rank == 0) {
            auto bias = getSrcMemoryAtPort(BIAS_ID);
            auto select_bias = split_h(bias, 0, w_rank, w_size);
            cached_splited_bias = select_bias;
        } else {
            cached_splited_bias = MemoryDescUtils::makeEmptyMemory(context);
        }

        // memory[ARG_BIAS] = attrs.withBias ? getSrcMemoryAtPort(BIAS_ID) : MemoryDescUtils::makeEmptyMemory(context);
        memory[ARG_BIAS] = cached_splited_bias;

        // dst
        memory[ARG_DST] = getDstMemoryAtPort(0);
    } else if (tp_mode == 3) {
        /*
        // src
        auto select_src = split_h(src, 0, w_rank, w_size);
        memory[ARG_SRC] = select_src;

        // wgt
        // should NOT split weight at all.

        // bias
        if (attrs.withBias && w_rank == 0) {
            auto bias = getSrcMemoryAtPort(BIAS_ID);
            auto select_bias = split_v(bias, 0, w_rank, w_size);
            memory[ARG_BIAS] = select_bias;
        } else {
            memory[ARG_BIAS] = MemoryDescUtils::makeEmptyMemory(context);
        }

        // dst
        auto select_dst = split_h(dst, 0, w_rank, w_size);
        memory[ARG_DST] = select_dst;
        std::cout << "[dbg] src_shape:" << select_src->getShape().toString()
            << ", wgt_shape:" << wgt->getShape().toString()
            << ", dst_shape:" << select_dst->getShape().toString()
            << "\n";
        */
        // no change for create primitive.
        memory[ARG_SRC] = getSrcMemoryAtPort(DATA_ID);
        memory[ARG_WEI] = getSrcMemoryAtPort(WEIGHTS_ID);
        memory[ARG_BIAS] = attrs.withBias ? getSrcMemoryAtPort(BIAS_ID) : MemoryDescUtils::makeEmptyMemory(context);
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
    // printf("[dbg] finish Primitive\n");
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
