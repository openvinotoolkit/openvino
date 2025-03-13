// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fullyconnected.h"

#include <cpu/x64/cpu_isa_traits.hpp>
#include <memory>
#include <openvino/op/constant.hpp>

#include "common/cpu_convert.h"
#include "common/cpu_memcpy.h"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "executors/memory_arguments.hpp"
#include "fake_quantize.h"
#include "graph_context.h"
#include "input.h"
#include "memory_desc/blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/threading/cpu_message.hpp"
#include "ov_ops/fully_connected.hpp"
#include "ov_ops/fully_connected_compressed.hpp"
#include "ov_ops/fully_connected_quantized.hpp"
#include "ov_ops/fully_connected_quantized_legacy.hpp"
#include "post_ops.hpp"
#include "shape_inference/custom/fullyconnected.hpp"
#include "transformations/utils/utils.hpp"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"

using namespace dnnl;
using namespace ov::element;

namespace ov::intel_cpu::node {

ov::element::TypeVector FullyConnected::getSupportedCompressedWeightsTypes(bool apply_fp8) {
    using ov::element::Type_t;

    bool useMatmulPrim = false;
    CPU_DEBUG_CAP_ENABLE(useMatmulPrim = getEnvBool("OV_CPU_ENABLE_DNNL_MAMTUL_FOR_FC");)

    if (useMatmulPrim) {
        return {Type_t::u8, Type_t::i8};
    }
#if defined(OPENVINO_ARCH_X86_64)
    ov::element::TypeVector supportedDataTypes =
        {Type_t::u8, Type_t::i8, Type_t::u4, Type_t::i4, Type_t::nf4, Type_t::f4e2m1};
    if (apply_fp8) {
        supportedDataTypes.insert(supportedDataTypes.end(), {Type_t::f8e4m3, Type_t::f8e5m2});
    }
    return supportedDataTypes;
#else
    return {};
#endif
}

ov::element::TypeVector FullyConnected::getSupportedCompressedActivationsTypes() {
    using ov::element::Type_t;

    bool useMatmulPrim = false;
    CPU_DEBUG_CAP_ENABLE(useMatmulPrim = getEnvBool("OV_CPU_ENABLE_DNNL_MAMTUL_FOR_FC");)

    if (useMatmulPrim) {
        return {Type_t::f32, Type_t::f16};
    }
#if defined(OPENVINO_ARCH_X86_64)
    // @todo enable for bf16 as well
    // after EnforceInferencePrecision is replaced with ConvertPrecision
    return {Type_t::f32};
#else
    return {};
#endif
}

bool FullyConnected::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                          std::string& errorMessage) noexcept {
    try {
        if (!ov::is_type<const ov::op::internal::FullyConnected>(op)) {
            return false;
        }

        if (ov::is_type<const ov::op::internal::FullyConnected>(op)) {
            if (!ov::op::util::is_on_constant_path(op->input_value(BIAS))) {
                errorMessage = "Only Constant operation on 'bias' input is supported";
                return false;
            }
        }

        if (ov::is_type<const ov::op::internal::FullyConnectedCompressed>(op)) {
            if (!ov::op::util::is_on_constant_path(op->input_value(WEIGHT_SCALES)) ||
                !ov::op::util::is_on_constant_path(op->input_value(WEIGHT_ZERO_POINTS))) {
                errorMessage =
                    "Only Constant operation on 'weight scales', and 'weight zero points' inputs is supported";
                return false;
            }
        }
    } catch (...) {
        return false;
    }

    return true;
}

// @todo replace 'inferencePrecision' check with 'fc->get_input_element_type(0) == ov::element::bf16'
// after bf16 pipeline is moved to ConvertPrecision
bool FullyConnected::isSupportedCompressedOperation(const std::shared_ptr<ov::Node>& op,
                                                    size_t IC,
                                                    size_t OC,
                                                    size_t G,
                                                    ov::element::Type inferencePrecision) noexcept {
#if defined(OPENVINO_ARCH_X86_64)
    try {
        std::string errorMessage;
        if (!isSupportedOperation(op, errorMessage)) {
            return false;
        }

        if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
            return false;
        }

        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx) &&
            inferencePrecision == ov::element::bf16) {
            // OneDNN AMX IP implementation has limited shapes support due to performance considerations. As a
            // current solution conditions below are copied from OneDNN to make sure correct IP impl will be
            // used since fallback one doesn't support weights decompression feature.
            size_t simdWidth = 16;
            size_t vnniFactor = 2;
            size_t maxSize = 512;
            auto amxRow = vnniFactor * simdWidth;

            if ((IC <= amxRow && OC <= amxRow) || (IC <= maxSize && OC <= maxSize && IC % amxRow != 0)) {
                return false;
            }
        }

        if (IC % G != 0 || IC / G < 4 || OC == 1) {
            return false;
        }

        return true;
    } catch (...) {
        return false;
    }
    return true;
#else
    bool useMatmulPrim = false;
    CPU_DEBUG_CAP_ENABLE(useMatmulPrim = getEnvBool("OV_CPU_ENABLE_DNNL_MAMTUL_FOR_FC");)
    return useMatmulPrim;
#endif
}

void FullyConnected::initTensorParallelConfig(const GraphContext::CPtr& context) {
    if (context->getCPUStreamExecutor()) {
        if (!context->getCPUStreamExecutor()->get_rank().empty()) {
            // init tp_cfg.w_rank and tp_cfg.w_size
            tp_cfg.w_rank = context->getCPUStreamExecutor()->get_rank()[0];
            tp_cfg.w_size = ov::threading::message_manager()->get_num_sub_streams();
            tp_cfg.enable_tensor_parallel = tp_cfg.w_size > 1;
            tp_cfg.sub_memory = context->getSubMemory();
        }
    }
}

FullyConnected::FullyConnected(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, FCShapeInferFactory(op)) {
    std::string errorMessage;
    initTensorParallelConfig(context);
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    m_atoi[ARG_SRC] = DATA;
    m_atoi[ARG_WEI] = WEIGHTS;
    m_atoi[ARG_BIAS] = BIAS;

    auto mapArgToInput = [&op](std::unordered_map<size_t, size_t>& argToInput, size_t argId, size_t inputId) {
        if (op->get_input_size() > inputId && op->input(inputId).get_element_type() != ov::element::dynamic &&
            op->input(inputId).get_element_type() != ov::element::dynamic) {
            argToInput[argId] = inputId;
        }
    };

    if (ov::is_type<const ov::op::internal::FullyConnectedCompressed>(op)) {
        mapArgToInput(m_atoi, ARG_WEI | ARG_ATTR_SCALES, WEIGHT_SCALES);
        mapArgToInput(m_atoi, ARG_WEI | ARG_ATTR_ZERO_POINTS, WEIGHT_ZERO_POINTS);
        algorithm = Algorithm::FullyConnectedCompressed;
    } else if (ov::is_type<const ov::op::internal::FullyConnectedQuantizedLegacy>(op)) {
        mapArgToInput(m_atoi, ARG_DST_DEQ_SCALE, 3);
        algorithm = Algorithm::FullyConnectedQuantizedLegacy;
    } else if (ov::is_type<const ov::op::internal::FullyConnectedQuantized>(op)) {
        algorithm = Algorithm::FullyConnectedQuantized;
        OPENVINO_THROW_NOT_IMPLEMENTED("FullyConnectedQuantized is not implemented yet");
    } else {
        algorithm = Algorithm::FullyConnectedCommon;
    }
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
        const auto& dst_shape = dstMemoryBuffer->getShape();
        auto dst_desc = dstMemoryBuffer->getDescPtr();
        auto dims = dst_shape.getDims();
        if (dim < 0) {
            dim += dims.size();
        }
        CPU_NODE_ASSERT(static_cast<int>(dims[dim]) >= tp_cfg.w_size,
                        getName() + " dim[" + std::to_string(dim) + "] is " + std::to_string(dims[dim]) +
                            ", which is larger than w_size " + std::to_string(tp_cfg.w_size));
        auto splited_dim_vec = split_parts(dims[dim], tp_cfg.w_size);

        VectorDims new_dims = std::move(dims);
        new_dims[dim] = splited_dim_vec[tp_cfg.w_rank];
        auto memory_desc = dst_desc->cloneWithNewDims(new_dims, true);
        tp_cfg.cached_dst->redefineDesc(std::move(memory_desc));
        memory[ARG_DST] = tp_cfg.cached_dst;
    }
}

void FullyConnected::prepareParams() {
    needPrepareParamsForTensorParallel();

    executor->update(memory);
    // @todo avoid updating implementation type in scope of every prepareParams call
    getSelectedPrimitiveDescriptor()->setImplementationType(executor->implType());
}

void FullyConnected::initTensorParallelSync() {
    if (tp_cfg.enable_tensor_parallel) {
        tp_cfg.id = tp_cfg.sub_memory->get_memory_id(tp_cfg.w_rank);
        CPU_NODE_ASSERT(tp_cfg.id >= 0, "Tensor Parallel Config ID cannot be negative.");
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
                    const auto copySize = splited_dim_vec[idx] * prec.size();  // bytes of half selected dim.
                    const size_t unloop = 8;
                    size_t step = count / unloop;
                    parallel_for(step, [&](size_t i) {
                        cpu_memcpy(dst_ptr + idx * strideSize + (i * unloop) * channel_size,
                                   new_ptr + (i * unloop) * copySize,
                                   copySize);
                        cpu_memcpy(dst_ptr + idx * strideSize + (i * unloop + 1) * channel_size,
                                   new_ptr + (i * unloop + 1) * copySize,
                                   copySize);
                        cpu_memcpy(dst_ptr + idx * strideSize + (i * unloop + 2) * channel_size,
                                   new_ptr + (i * unloop + 2) * copySize,
                                   copySize);
                        cpu_memcpy(dst_ptr + idx * strideSize + (i * unloop + 3) * channel_size,
                                   new_ptr + (i * unloop + 3) * copySize,
                                   copySize);
                        cpu_memcpy(dst_ptr + idx * strideSize + (i * unloop + 4) * channel_size,
                                   new_ptr + (i * unloop + 4) * copySize,
                                   copySize);
                        cpu_memcpy(dst_ptr + idx * strideSize + (i * unloop + 5) * channel_size,
                                   new_ptr + (i * unloop + 5) * copySize,
                                   copySize);
                        cpu_memcpy(dst_ptr + idx * strideSize + (i * unloop + 6) * channel_size,
                                   new_ptr + (i * unloop + 6) * copySize,
                                   copySize);
                        cpu_memcpy(dst_ptr + idx * strideSize + (i * unloop + 7) * channel_size,
                                   new_ptr + (i * unloop + 7) * copySize,
                                   copySize);
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

void FullyConnected::execute(const dnnl::stream& strm) {
    initTensorParallelSync();

    executor->execute(memory);

    execTensorParallelSync();
}

void FullyConnected::executeDynamicImpl(const dnnl::stream& strm) {
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

    if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx)) {
        return false;
    }

    const auto constNode = std::dynamic_pointer_cast<Input>(weightsInput);
    if (!constNode) {
        return false;
    }

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
    attrs.withBias = getOriginalInputPrecisionAtPort(BIAS) != ov::element::dynamic;

    attrs.sparseWeights = useSparseWeightsDecompression(getParentEdgeAt(WEIGHTS)->getParent(),
                                                        getOriginalInputPrecisionAtPort(DATA),
                                                        context->getConfig().fcSparseWeiDecompressionRate);
    attrs.dynamicQuantizationGroupSize = context->getConfig().fcDynamicQuantizationGroupSize;
    attrs.modelType = context->getConfig().modelType;

    postOps = getPostOps(fusedWith);

    const auto& srcTypes = getOriginalInputPrecisions();
    auto dstTypes = getOriginalOutputPrecisions();
    // @todo graph optimizer should update original output precisions instead
    if (!fusedWith.empty()) {
        dstTypes = fusedWith.back()->getOriginalOutputPrecisions();
    }

    VecMemoryDescs srcDescs;
    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    for (size_t i = 0; i < srcTypes.size(); i++) {
        if (srcTypes[i] == element::dynamic) {
            srcDescs.push_back(MemoryDescUtils::makeEmptyDesc());
            continue;
        }
        const auto srcDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(srcTypes[i], getInputShapeAtPort(i));
        srcDescs.push_back(srcDesc);
    }

    VecMemoryDescs dstDescs;
    for (size_t i = 0; i < dstTypes.size(); i++) {
        const auto dstDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(dstTypes[i], getOutputShapeAtPort(i));
        dstDescs.push_back(dstDesc);
    }

    MemoryDescArgs descs{
        {ARG_SRC, srcDescs[DATA]},
        {ARG_WEI, srcDescs[WEIGHTS]},
        {ARG_BIAS, srcDescs[BIAS]},
        {ARG_DST, dstDescs[0]},
    };

    auto executionContext = std::make_shared<ExecutorContext>(context, getImplPriority(), privateWeightCache);
    factory = std::make_shared<ExecutorFactory<FCAttrs>>(attrs, postOps, executionContext, descs);
    const auto nodeDescriptors = factory->getProperMemoryDescriptors(descs);

    NodeConfig nodeConfig;
    nodeConfig.inConfs.resize(srcDescs.size());

    for (const auto& desc : nodeDescriptors) {
        if (m_atoi.count(desc.first)) {
            nodeConfig.inConfs[m_atoi[desc.first]] = desc.second;
        }
    }

    // add extra inputs bypassing proper memory descriptors
    // @todo pass all the input descriptors to getProperMemoryDescriptors and allow
    // to ignore extra input descriptors if necessery
    for (size_t i = 3; i < srcDescs.size(); i++) {
        nodeConfig.inConfs[i] = srcDescs[i];
    }

    const int inPlace = canBeInPlace() ? 0 : -1;
    nodeConfig.outConfs.emplace_back(nodeDescriptors.at(ARG_DST), BlockedMemoryDesc::FULL_MASK, inPlace);

    supportedPrimitiveDescriptors.emplace_back(nodeConfig, impl_desc_type::undef);
}

void FullyConnected::needSplitMemoryForTensorParallel() {
    if (tp_cfg.enable_tensor_parallel) {
        auto src = getSrcMemoryAtPort(DATA);
        auto wgt = getSrcMemoryAtPort(WEIGHTS);
        auto dst = getDstMemoryAtPort(0);
        // src
        memory[ARG_SRC] = getSrcMemoryAtPort(DATA);
        // wgt
        // split N direction
        tp_cfg.cached_splited_weight =
            attrs.weightsNonTransposed ? split_vertical(context->getEngine(), wgt, 0, tp_cfg.w_rank, tp_cfg.w_size)
                                       : split_horizontal(context->getEngine(), wgt, 0, tp_cfg.w_rank, tp_cfg.w_size);
        memory[ARG_WEI] = tp_cfg.cached_splited_weight;
        // bias
        if (attrs.withBias) {
            auto bias = getSrcMemoryAtPort(BIAS);
            auto select_bias = split_horizontal(context->getEngine(), bias, 0, tp_cfg.w_rank, tp_cfg.w_size);
            tp_cfg.cached_splited_bias = std::move(select_bias);
        } else {
            tp_cfg.cached_splited_bias = MemoryDescUtils::makeEmptyMemory(context);
        }
        memory[ARG_BIAS] = tp_cfg.cached_splited_bias;
        // dst
        memory[ARG_DST] = getDstMemoryAtPort(0);
        tp_cfg.cached_dst = split_horizontal(context->getEngine(), dst, -1, tp_cfg.w_rank, tp_cfg.w_size, false);

        if (memory.count(ARG_DST | ARG_ATTR_SCALES)) {
            memory[ARG_DST | ARG_ATTR_SCALES] = split_horizontal(context->getEngine(),
                                                                 memory[ARG_DST | ARG_ATTR_SCALES],
                                                                 0,
                                                                 tp_cfg.w_rank,
                                                                 tp_cfg.w_size);
        }

        if (memory.count(ARG_WEI | ARG_ATTR_SCALES)) {
            auto scale_mem = std::const_pointer_cast<IMemory>(memory[ARG_WEI | ARG_ATTR_SCALES]);
            memory[ARG_WEI | ARG_ATTR_SCALES] =
                attrs.weightsNonTransposed
                    ? split_vertical(context->getEngine(), scale_mem, 0, tp_cfg.w_rank, tp_cfg.w_size)
                    : split_horizontal(context->getEngine(), scale_mem, 0, tp_cfg.w_rank, tp_cfg.w_size);
        }

        if (memory.count(ARG_WEI | ARG_ATTR_ZERO_POINTS)) {
            auto zeropoint_mem = std::const_pointer_cast<IMemory>(memory[ARG_WEI | ARG_ATTR_ZERO_POINTS]);
            auto element_num = zeropoint_mem->getSize() / zeropoint_mem->getPrecision().size();
            if (element_num == 1) {
                tp_cfg.cached_zeropoint = zeropoint_mem;
            } else {
                tp_cfg.cached_zeropoint =
                    attrs.weightsNonTransposed
                        ? split_vertical(context->getEngine(), zeropoint_mem, 0, tp_cfg.w_rank, tp_cfg.w_size)
                        : split_horizontal(context->getEngine(), zeropoint_mem, 0, tp_cfg.w_rank, tp_cfg.w_size);
            }
        }
    }
}

void FullyConnected::needUpdateTensorParalelConfig() {
    // tensor parallel should be disabled in two conditions.
    // 1. weight shape is dynamic
    // 2. last dim can be splited.
    if (tp_cfg.enable_tensor_parallel) {
        const auto& shape = getSrcMemoryAtPort(WEIGHTS)->getShape();
        if (shape.isDynamic() || shape.getDims()[0] < static_cast<size_t>(tp_cfg.w_size)) {
            tp_cfg.enable_tensor_parallel = false;
        }
    }
}

void FullyConnected::createPrimitive() {
    needUpdateTensorParalelConfig();

    for (const auto& entry : m_atoi) {
        const auto argumentId = entry.first;
        const auto inputId = entry.second;
        memory[argumentId] = getSrcMemoryAtPort(inputId);
    }

    memory[ARG_DST] = getDstMemoryAtPort(0);

    needSplitMemoryForTensorParallel();
    // @todo should we preconfigure only for dynamic shapes?
    // Since for static shapes primitive is created in scope of compile_model() anyway
    executor = factory->make(memory);

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

}  // namespace ov::intel_cpu::node
