// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transpose.h"

#include <oneapi/dnnl/dnnl_types.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "graph_context.h"
#include "memory_desc/blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/common/blocked_desc_creator.h"
#include "nodes/common/reorder_prim.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/transpose.hpp"
#include "nodes/executors/transpose_list.hpp"
#include "nodes/node_config.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/transpose.hpp"
#include "shape_inference/custom/transpose.hpp"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"
using namespace dnnl;

namespace ov::intel_cpu::node {

bool Transpose::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (none_of(op->get_type_info(), ov::op::v1::Transpose::get_type_info_static())) {
            errorMessage = "Node is not an instance of the Transpose operation from opset1.";
            return false;
        }

        if (op->get_input_node_ptr(INPUT_ORDER_IDX)->get_type_info() != ov::op::v0::Constant::get_type_info_static()) {
            // TODO: Support parameterized Order input for dynamic shapes.
            errorMessage = "Constant expected as the second input for static shapes.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Transpose::Transpose(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, TransposeShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    if (op->get_input_node_ptr(INPUT_ORDER_IDX)->get_type_info() == ov::op::v0::Constant::get_type_info_static()) {
        isInputOrderConst = true;
        order = ov::as_type<ov::op::v0::Constant>(op->get_input_node_ptr(INPUT_ORDER_IDX))->cast_vector<size_t>();

        if (order.empty()) {
            size_t rank = getInputShapeAtPort(INPUT_DATA_IDX).getRank();
            for (size_t i = 1LU; i <= rank; ++i) {
                order.emplace_back(rank - i);
            }
        }
    }
}

void Transpose::getSupportedDescriptors() {}

void Transpose::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    prec = getOriginalInputPrecisionAtPort(0);

    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();

    NodeConfig config;
    config.inConfs.resize(2);
    config.outConfs.resize(1);
    config.inConfs[INPUT_DATA_IDX].inPlace(-1);
    config.inConfs[INPUT_DATA_IDX].constant(false);
    config.inConfs[INPUT_ORDER_IDX].constant(isInputOrderConst);
    config.inConfs[INPUT_ORDER_IDX].setMemDesc(
        creatorsMap.at(LayoutType::ncsp)->createSharedDesc(ov::element::i32, getInputShapeAtPort(INPUT_ORDER_IDX)));
    config.outConfs[0].inPlace(isOptimized ? 0 : -1);
    config.outConfs[0].constant(false);
    transpose_context = std::make_shared<ExecutorContext>(context, getImplPriority());

    auto supportedPrimitiveDescriptorsBuilder = [this](const NodeConfig& config,
                                                       const TransposeParams& transposeParams) {
        std::vector<MemoryDescPtr> srcMemoryDescs;
        srcMemoryDescs.reserve(config.inConfs.size());
        for (const auto& inConf : config.inConfs) {
            srcMemoryDescs.emplace_back(inConf.getMemDesc());
        }
        std::vector<MemoryDescPtr> dstMemoryDescs;
        srcMemoryDescs.reserve(config.outConfs.size());
        dstMemoryDescs.reserve(config.outConfs.size());
        for (const auto& outConf : config.outConfs) {
            dstMemoryDescs.emplace_back(outConf.getMemDesc());
        }
        auto factory = std::make_shared<TransposeExecutorFactory>(transposeParams,
                                                                  srcMemoryDescs,
                                                                  dstMemoryDescs,
                                                                  transpose_context);
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown, factory);
    };

    const auto& inputDataShape = getInputShapeAtPort(INPUT_DATA_IDX);
    const auto& outputDataShape = getOutputShapeAtPort(0);
    if (any_of(inputDataShape.getRank(), 4U, 5U)) {
        config.inConfs[0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(prec, inputDataShape));
        config.outConfs[0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(prec, outputDataShape));
        supportedPrimitiveDescriptorsBuilder(config, transposeParams);
#if defined(OPENVINO_ARCH_X86_64)
        const auto& srcDims = inputDataShape.getDims();
        if (srcDims[1] != Shape::UNDEFINED_DIM && srcDims[1] % 8 == 0) {
            config.inConfs[0].setMemDesc(creatorsMap.at(LayoutType::nCsp8c)->createSharedDesc(prec, inputDataShape));
            supportedPrimitiveDescriptorsBuilder(config, transposeParams);
        }

        if (srcDims[1] != Shape::UNDEFINED_DIM && srcDims[1] % 16 == 0) {
            config.inConfs[0].setMemDesc(creatorsMap.at(LayoutType::nCsp16c)->createSharedDesc(prec, inputDataShape));
            supportedPrimitiveDescriptorsBuilder(config, transposeParams);
        }
#endif  // OPENVINO_ARCH_X86_64
        if (any_of(prec, ov::element::f32, ov::element::f16, ov::element::i8, ov::element::u8, ov::element::bf16)) {
            config.inConfs[0].setMemDesc(creatorsMap.at(LayoutType::nspc)->createSharedDesc(prec, inputDataShape));
            config.outConfs[0].setMemDesc(creatorsMap.at(LayoutType::nspc)->createSharedDesc(prec, outputDataShape));
            supportedPrimitiveDescriptorsBuilder(config, transposeParams);
        }
    } else {
        // general plain case
        config.inConfs[0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(prec, inputDataShape));
        config.outConfs[0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(prec, outputDataShape));
        supportedPrimitiveDescriptorsBuilder(config, transposeParams);
    }
}

bool Transpose::neverExecute() const {
    return isOptimized || getSelectedPrimitiveDescriptor()->hasZeroInputDimsAtPort(0);
}

bool Transpose::isExecutable() const {
    return !isOptimized && !isInputTensorAtPortEmpty(0);
}

bool Transpose::needPrepareParams() const {
    return inputShapesModified();
}

void Transpose::prepareParams() {
    if (isOptimized) {
        return;
    }

    if (performAsReorder) {
        //  Transpose(order={0,3,1,2}) can be performed as Reorder(acdb=>abcd)
        auto srcMemPtr = getSrcMemoryAtPort(INPUT_DATA_IDX);
        auto dstMemPtr = getDstMemoryAtPort(0);
        auto dstDesc = dstMemPtr->getDescWithType<DnnlMemoryDesc>()->getDnnlDesc();
        auto srcDesc = dnnl::memory::desc(dstDesc.get_dims(), dstDesc.get_data_type(), memory::format_tag::acdb);
        auto result = getReorderPrim(context->getParamsCache(), getEngine(), srcDesc, dstDesc);
        CPU_NODE_ASSERT(result, "reorder primitive descriptor was not found.");
        prim = result;

        getSelectedPrimitiveDescriptor()->setImplementationType(
            parse_impl_name(DnnlExtensionUtils::query_impl_info_str(prim.get_primitive_desc())));

        primArgs = {{DNNL_ARG_SRC, srcMemPtr->getPrimitive()}, {DNNL_ARG_DST, dstMemPtr->getPrimitive()}};
#ifdef CPU_DEBUG_CAPS
        if (prim) {
            const auto* pd = prim.get_primitive_desc();
            DEBUG_LOG("verbose##", getName(), "##", DnnlExtensionUtils::query_pd_info(pd), "\n");
        }
#endif
        return;
    }

    auto srcDesc = getParentEdgeAt(INPUT_DATA_IDX)->getMemory().getDescWithType<BlockedMemoryDesc>();
    transposeParams.permuteParams.src_block_dims = srcDesc->getBlockDims();
    auto dstDesc = getChildEdgeAt(0)->getMemory().getDescWithType<BlockedMemoryDesc>();
    transposeParams.permuteParams.dst_block_dims = dstDesc->getBlockDims();

    if (!isInputOrderConst) {
        const auto* orderPtr = getSrcDataAtPortAs<const int32_t>(0);
        auto orderLen = getSrcMemoryAtPort(0)->getSize();
        transposeParams.permuteParams.order.assign(orderPtr, orderPtr + orderLen);
    }

    auto engine = getEngine();
    auto builder =
        [&srcDesc, &dstDesc, this]([[maybe_unused]] const PermuteParams& key) -> std::shared_ptr<TransposeExecutor> {
        dnnl::primitive_attr attr;
        auto* selectedPD = getSelectedPrimitiveDescriptor();
        auto executor = selectedPD->getExecutorFactoryAs<TransposeExecutorFactory>()->makeExecutor(transposeParams,
                                                                                                   {srcDesc},
                                                                                                   {dstDesc},
                                                                                                   attr);
        return executor;
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(transposeParams.permuteParams, builder);

    CPU_NODE_ASSERT(result.first, "Primitive descriptor was not found.");

    execPtr = result.first;
}

void Transpose::createPrimitive() {
    if (isOptimized) {
        return;
    }

    auto dstMemPtr = getDstMemoryAtPort(0);
    auto srcMemPtr = getSrcMemoryAtPort(INPUT_DATA_IDX);
    CPU_NODE_ASSERT(dstMemPtr, "Destination memory is null.");
    CPU_NODE_ASSERT(srcMemPtr, "Input memory is null.");
    CPU_NODE_ASSERT(getSelectedPrimitiveDescriptor(), "Preferable primitive descriptor was not set.");

    if (getParentEdgeAt(INPUT_DATA_IDX)->getMemory().getDesc().hasLayoutType(LayoutType::ncsp) &&
        getChildEdgeAt(0)->getMemory().getDesc().hasLayoutType(LayoutType::ncsp) &&
        order == std::vector<size_t>{0, 3, 1, 2}) {
        performAsReorder = true;
    }

#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
    // Avoid using reference implementation of non-fp32 reorders on arm platforms
    if (prec != ov::element::f32) {
        performAsReorder = false;
    }
#endif

    if (!performAsReorder) {
        transposeParams.permuteParams.data_size =
            getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].getMemDesc()->getPrecision().size();
        if (isInputOrderConst) {
            transposeParams.permuteParams.order = order;
        }
        auto srcDesc = getParentEdgeAt(INPUT_DATA_IDX)->getMemory().getDescWithType<BlockedMemoryDesc>();
        transposeParams.permuteParams.src_block_order = srcDesc->getOrder();
        auto dstDesc = getChildEdgeAt(0)->getMemory().getDescWithType<BlockedMemoryDesc>();
        transposeParams.permuteParams.dst_block_order = dstDesc->getOrder();
    }

    if (inputShapesDefined() && isExecutable()) {
        prepareParams();
        updateLastInputDims();
    }
}

void Transpose::execute(const dnnl::stream& strm) {
    if (isOptimized) {
        return;
    }

    if (prim) {
        prim.execute(strm, primArgs);
    } else if (execPtr) {
        auto dstMemPtr = getDstMemoryAtPort(0);
        auto srcMemPtr = getSrcMemoryAtPort(INPUT_DATA_IDX);

        execPtr->exec({srcMemPtr}, {dstMemPtr});
    } else {
        CPU_NODE_THROW("Primitive was not created.");
    }
}

void Transpose::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

bool Transpose::created() const {
    return getType() == Type::Transpose;
}

}  // namespace ov::intel_cpu::node
