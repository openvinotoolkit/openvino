// Copyright (C) 2026 FUJITSU LIMITED
// SPDX-License-Identifier: Apache-2.0
//

#include "kleidiai_gathermatmul.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "memory_desc/blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/common/blocked_desc_creator.h"
#include "nodes/executors/dnnl/dnnl_utils.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/gathermatmul_config.hpp"
#include "nodes/executors/kleidiai/kleidiai_mm.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/precision_support.h"

namespace ov::intel_cpu {

bool GatherMatMulKleidiAIExecutor::useDynamicQuantizationImpl(const MemoryDescPtr& weightDesc) {
    if (!hasIntDotProductSupport() && !hasInt8MMSupport()) {
        return false;
    }
    return weightDesc->getPrecision() == element::i8 || weightDesc->getPrecision() == element::i4;
}

bool GatherMatMulKleidiAIExecutor::supports([[maybe_unused]] const GatherMatmulConfig& config) {
    return config.descs.at(ARG_WEI)->getPrecision() == element::f32 ||
           useDynamicQuantizationImpl(config.descs.at(ARG_WEI));
}

GatherMatMulKleidiAIExecutor::GatherMatMulKleidiAIExecutor(const GatherMatmulAttrs& attrs,
                                                           const MemoryArgs& memory,
                                                           const ExecutorContext::CPtr& context)
    : m_context(context) {
    auto srcMemoryDesc = memory.at(ARG_SRC)->getDescPtr();
    auto weiMemoryDesc = memory.at(ARG_WEI)->getDescPtr();
    auto biasMemoryDesc = memory.at(ARG_BIAS)->getDescPtr();

    const auto& weiDims = weiMemoryDesc->getShape().getStaticDims();
    auto weiPrec = weiMemoryDesc->getPrecision();
    auto SrcPrec = srcMemoryDesc->getPrecision();
    size_t N = weiDims[weiDims.size() - 2];
    size_t K = weiDims[weiDims.size() - 1];
    numExperts = weiDims[0];

    MemoryPtr m_scalesMemory = nullptr;

    OPENVINO_ASSERT(weiMemoryDesc->isDefined(), "Weights memory descriptor is not defined");
    OPENVINO_ASSERT(SrcPrec == ov::element::f32, "Activation currently supported only in f32");

    auto addBatchDim = [](const BlockedMemoryDescPtr& desc, size_t batchDim) -> DnnlMemoryDescPtr {
        const auto& weightsDims = desc->getShape().getStaticDims();
        const auto& weightsBlockDims = desc->getBlockDims();
        const auto& weightsOrder = desc->getOrder();
        VectorDims newDims = {batchDim};
        newDims.insert(newDims.end(), weightsDims.begin(), weightsDims.end());
        VectorDims newBlockDims = {batchDim};
        newBlockDims.insert(newBlockDims.end(), weightsBlockDims.begin(), weightsBlockDims.end());
        VectorDims newOrder(weightsOrder.size() + 1);
        newOrder[0] = 0;
        for (size_t i = 0; i < weightsOrder.size(); i++) {
            newOrder[i + 1] = weightsOrder[i] + 1;
        }
        auto targetDesc =
            std::make_shared<CpuBlockedMemoryDesc>(desc->getPrecision(), Shape(newDims), newBlockDims, newOrder);
        return MemoryDescUtils::convertToDnnlMemoryDesc(targetDesc);
    };

    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    if (attrs.isCompressedOperation) {
        const auto& scales = memory.at(ARG_SRC_3);
        if (scales && !scales->getDesc().empty()) {
            const auto& fullScalesShape = scales->getDesc().getShape().getStaticDims();
            if (1 == fullScalesShape.size()) {
                OPENVINO_THROW(" broadcastable scales shape not supported ");
            } else {
                auto expertScaleDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(ov::element::f32, Shape({N}));
                auto expectedScaleMemDesc = MemoryDescUtils::convertToDnnlMemoryDesc(expertScaleDesc);
                expectedScaleMemDesc =
                    addBatchDim(MemoryDescUtils::convertToBlockedMemoryDesc(expectedScaleMemDesc), numExperts);
                if (expectedScaleMemDesc->isCompatible(scales->getDesc())) {
                    m_scalesMemory = scales;
                } else {
                    m_scalesMemory = std::make_shared<Memory>(m_context->getEngine(), expectedScaleMemDesc);
                    m_scalesMemory->load(*scales, false, false);
                }
            }
        } else {
            OPENVINO_THROW(" Scales cannot be empty for GatherMatmulCompressed op ");
        }
    }

    auto expertWeiDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(weiPrec, Shape({N, K}));
    auto targetWeightsDesc = addBatchDim(expertWeiDesc, numExperts);
    auto m_weightsMemory = utils::prepareWeightsMemory(MemoryDescUtils::convertToDnnlMemoryDesc(weiMemoryDesc),
                                                       targetWeightsDesc,
                                                       memory.at(ARG_WEI),
                                                       context->getEngine(),
                                                       context->getRuntimeCache(),
                                                       context->getWeightsCache(),
                                                       context->getPrivateWeightCache(),
                                                       context->getThreadPool());

    MemoryDescArgs memDescArgs;
    memDescArgs[ARG_SRC] = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(SrcPrec, Shape({1, K}));
    memDescArgs[ARG_DST] = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(SrcPrec, Shape({1, N}));
    memDescArgs[ARG_WEI] = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(weiPrec, Shape({N, K}));
    if (biasMemoryDesc && !biasMemoryDesc->empty()) {
        memDescArgs[ARG_BIAS] = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(ov::element::f32, Shape({N}));
    } else {
        memDescArgs[ARG_BIAS] = MemoryDescUtils::makeEmptyDesc();
    }

    FCAttrs fcArgs;
    if (attrs.isCompressedOperation) {
        // set as uint64_max to enable dynamic quantization by default
        fcArgs.dynamicQuantizationGroupSize = std::numeric_limits<uint64_t>::max();
        memDescArgs[ARG_WEI | ARG_ATTR_SCALES] =
            creatorsMap.at(LayoutType::ncsp)->createSharedDesc(ov::element::f32, Shape({N}));
    }

    // As we call executors sequentially, execution-context can be shared
    memArgs.reserve(numExperts);
    for (size_t expert = 0; expert < numExperts; ++expert) {
        MemoryArgs mArgs;
        mArgs[ARG_WEI] = split_horizontal(context->getEngine(), m_weightsMemory, 0, expert, numExperts, true);
        // wei_shape shape after split becomes: [1, N, K] --> redefine desc to [N, K]
        mArgs[ARG_WEI]->redefineDesc(memDescArgs[ARG_WEI]);
        if (biasMemoryDesc && !biasMemoryDesc->empty()) {
            auto bias = memory.at(ARG_BIAS);
            mArgs[ARG_BIAS] = split_horizontal(context->getEngine(), bias, 0, expert, numExperts, true);
            mArgs[ARG_BIAS]->redefineDesc(memDescArgs[ARG_BIAS]);
        } else {
            mArgs[ARG_BIAS] = std::make_shared<Memory>(context->getEngine(), MemoryDescUtils::makeEmptyDesc());
        }
        if (attrs.isCompressedOperation) {
            mArgs[ARG_WEI | ARG_ATTR_SCALES] =
                split_horizontal(context->getEngine(), m_scalesMemory, 0, expert, numExperts, true);
            mArgs[ARG_WEI | ARG_ATTR_SCALES]->redefineDesc(memDescArgs[ARG_WEI | ARG_ATTR_SCALES]);
        }
        mArgs[ARG_SRC] = std::make_shared<Memory>(context->getEngine(), memDescArgs[ARG_SRC]);
        memArgs.emplace_back(mArgs);
        auto kai_executor = std::make_shared<MatMulKleidiAIExecutor>(fcArgs, mArgs, context);
        kai_executor->setKaiExecutorImplAsGatherMatmul();
        executor.push_back(kai_executor);
    }
}

bool GatherMatMulKleidiAIExecutor::update(const MemoryArgs& memory) {
    for (size_t expert = 0; expert < numExperts; ++expert) {
        memArgs[expert][ARG_SRC] = memory.at(ARG_SRC);
        memArgs[expert][ARG_DST] = memory.at(ARG_DST);
        executor[expert]->update(memArgs[expert]);
    }
    return true;
}

void GatherMatMulKleidiAIExecutor::execute(const MemoryArgs& memory) {
    const auto& indexMem = memory.at(ARG_SRC_1);
    auto index_offset = OffsetHelper::createOffsetHelper(indexMem);

    const auto& indexShape = indexMem->getStaticDims();
    size_t M = indexShape[0];
    size_t B = indexShape[1];

    // all the gather idx for corresponding m index
    const size_t gather_axis_size = numExperts;
    std::vector<std::pair<int32_t, int32_t>> gather_idx_map(gather_axis_size * M);
    std::vector<int32_t> elements_per_gather_indx(gather_axis_size, 0);
    for (size_t m = 0; m < M; m++) {
        const auto* gather_ids = static_cast<const int32_t*>(index_offset(m));
        for (size_t i = 0; i < B; i++) {
            int32_t gather_axis_index = gather_ids[i];
            OPENVINO_ASSERT(gather_axis_index >= 0 && static_cast<size_t>(gather_axis_index) < gather_axis_size,
                            "Invalid gather_id ",
                            gather_axis_index,
                            " for m ",
                            m);
            auto& index = elements_per_gather_indx[gather_axis_index];
            gather_idx_map[gather_axis_index * M + index] = {m, i};
            index++;
        }
    }

    for (size_t gather_axis_index = 0; gather_axis_index < gather_axis_size; gather_axis_index++) {
        const size_t num_valid_rows = elements_per_gather_indx[gather_axis_index];
        if (0 == num_valid_rows) {
            continue;
        }
        memArgs[gather_axis_index][ARG_SRC] = memory.at(ARG_SRC);
        memArgs[gather_axis_index][ARG_DST] = memory.at(ARG_DST);
        auto gather_idx_expertOffset = gather_idx_map.begin() + gather_axis_index * M;
        std::vector<std::pair<int32_t, int32_t>> kai_gather_idx(gather_idx_expertOffset,
                                                                gather_idx_expertOffset + num_valid_rows);
        executor[gather_axis_index]->set_gather_idx(kai_gather_idx);
        executor[gather_axis_index]->execute(memArgs[gather_axis_index]);
    }
}

}  // namespace ov::intel_cpu