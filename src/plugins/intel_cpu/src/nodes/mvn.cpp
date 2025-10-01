// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn.h"

#include <algorithm>
#include <any>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <string>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "graph_context.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/common/blocked_desc_creator.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_factory.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/mvn_config.hpp"
#include "nodes/node_config.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/mvn.hpp"
#include "post_ops.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/general_utils.h"
#include "utils/precision_support.h"

namespace ov::intel_cpu::node {

bool MVN::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (op->get_output_partial_shape(0).rank().is_dynamic()) {
            errorMessage = "Unsupported dynamic input rank.";
            return false;
        }
        const auto& inDataRank = op->get_output_partial_shape(0).rank().get_length();
        if (inDataRank < 1 || inDataRank > 5) {
            errorMessage = "First input accepts ranks from 1 to 5. Actual: " + std::to_string(inDataRank);
            return false;
        }

        if (auto mvnOp = ov::as_type_ptr<const ov::op::v6::MVN>(op)) {
            auto axesOp = ov::as_type_ptr<ov::op::v0::Constant>(mvnOp->get_input_node_shared_ptr(1));
            if (!axesOp) {
                errorMessage = "Constant expected as the second input.";
                return false;
            }

            auto epsMode = mvnOp->get_eps_mode();
            if (epsMode != ov::op::MVNEpsMode::INSIDE_SQRT && epsMode != ov::op::MVNEpsMode::OUTSIDE_SQRT) {
                errorMessage = std::string("Just INSIDE_SQRT and OUTSIDE_SQRT epsilon mods are supported. Actual: ") +
                               std::to_string(static_cast<int>(epsMode));
                return false;
            }
            // Validates MVN node axes to check whether it can be executed on the current CPU implementation.
            // Supported cases:
            // 1D: axes: [0]
            // 2D: axes: [1]
            // 3D: axes: [1,2], [2]
            // 4D: axes: [1,2,3], [2,3]
            // 5D: axes: [1,2,3,4], [2,3,4]
            auto axesVal = axesOp->cast_vector<int>();
            for (int& axe : axesVal) {
                axe = axe < 0 ? axe + inDataRank : axe;
            }
            std::sort(axesVal.begin(), axesVal.end());
            if (inDataRank == 1) {
                if (axesVal.size() != 1 || axesVal[0] != 0) {
                    errorMessage = "Unsupported axes.";
                    return false;
                }
            } else {
                const bool rank_too_large = inDataRank > 5;
                const bool invalid_axes_size = static_cast<size_t>(inDataRank) != axesVal.size() + 1 &&
                                               static_cast<size_t>(inDataRank) != axesVal.size() + 2;
                if (rank_too_large || invalid_axes_size) {
                    errorMessage = "Unsupported axes.";
                    return false;
                }
                int value = inDataRank - 1;
                for (int i = axesVal.size() - 1; i >= 0; i--, value--) {
                    if (axesVal[i] != value) {
                        errorMessage = "Unsupported axes.";
                        return false;
                    }
                }
            }
        } else if (auto mvnOp = ov::as_type_ptr<const ov::op::v0::MVN>(op)) {
        } else {
            errorMessage = "Node is not an instance of the MVN operation.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MVN::MVN(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    mvnAttrs.epsMode_ = INSIDE_SQRT;
    if (auto mvnOp = ov::as_type_ptr<ov::op::v6::MVN>(op)) {
        mvnAttrs.normalizeVariance_ = mvnOp->get_normalize_variance();
        mvnAttrs.epsValue_ = mvnOp->get_eps();
        if (mvnOp->get_eps_mode() == ov::op::MVNEpsMode::OUTSIDE_SQRT) {
            mvnAttrs.epsMode_ = OUTSIDE_SQRT;
        }

        mvnAttrs.initAcrossChannels_ = false;
        const auto& inDataShapeSize = getInputShapeAtPort(0).getRank();
        if (inDataShapeSize == mvnOp->input_value(1).get_shape()[0] + 1 || inDataShapeSize == 1) {
            mvnAttrs.initAcrossChannels_ = true;
        }
    } else if (auto mvnOp = ov::as_type_ptr<ov::op::v0::MVN>(op)) {
        mvnAttrs.normalizeVariance_ = mvnOp->get_normalize_variance();
        mvnAttrs.epsValue_ = static_cast<float>(mvnOp->get_eps());
        mvnAttrs.initAcrossChannels_ = mvnOp->get_across_channels();
    } else {
        OPENVINO_THROW_NOT_IMPLEMENTED("Node is not an instance of MVN from the operation set v0 or v6");
    }
    mvnAttrs.execAcrossChannels_ = mvnAttrs.initAcrossChannels_;
}

void MVN::getSupportedDescriptors() {}

static inline bool isUnaryEltwise(const NodePtr& node) {
    return any_of(node->getAlgorithm(),
                  Algorithm::EltwiseRelu,
                  Algorithm::EltwiseGeluErf,
                  Algorithm::EltwiseGeluTanh,
                  Algorithm::EltwiseElu,
                  Algorithm::EltwiseSigmoid,
                  Algorithm::EltwiseClamp,
                  Algorithm::EltwiseTanh,
                  Algorithm::EltwiseSwish,
                  Algorithm::EltwiseHswish,
                  Algorithm::EltwiseMish,
                  Algorithm::EltwiseHsigmoid,
                  Algorithm::EltwiseRoundHalfToEven,
                  Algorithm::EltwiseRoundHalfAwayFromZero,
                  Algorithm::EltwiseAbs,
                  Algorithm::EltwiseSqrt,
                  Algorithm::EltwiseSoftRelu);
}

void MVN::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    ov::element::Type inputPrecision = getOriginalInputPrecisionAtPort(0);
    ov::element::Type outputPrecision = getOriginalOutputPrecisionAtPort(0);
    if (!hasHardwareSupport(outputPrecision)) {
        outputPrecision = ov::element::f32;
    }

    // For bf16 precision with blocked layouts, ensure proper precision handling
    // to avoid precision loss during computation
    if (inputPrecision == ov::element::bf16 && outputPrecision == ov::element::bf16) {
        // Check if using blocked layout
        bool willUseBlockedLayout = false;
        const auto rank = getInputShapeAtPort(0).getRank();
        const bool is_rank_4_or_5 = rank == 4 || rank == 5;
        const bool has_blocked_layouts = BlockedDescCreator::getCommonCreators().count(LayoutType::nCsp8c) != 0U ||
                                         BlockedDescCreator::getCommonCreators().count(LayoutType::nCsp16c) != 0U;
        if (is_rank_4_or_5 && has_blocked_layouts) {
            willUseBlockedLayout = true;
        }

        // If blocked layout, ensure computation is done in f32 internally
        if (willUseBlockedLayout) {
            // The executor will handle internal precision conversion
            // based on the type mapping we defined
        }
    }

    if (!fusedWith.empty()) {
        outputPrecision = fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0);
        onlyUnaryPostOps = true;
        for (auto& node : fusedWith) {
            if (isUnaryEltwise(node)) {
                continue;
            }
            onlyUnaryPostOps = false;
            break;
        }
    }

    // Create initial memory descriptors
    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    const bool enforce_formats = !memoryFormatFilter.input.empty() || !memoryFormatFilter.output.empty();
    // rank is not used in the following logic; remove to avoid unused warning
    bool prefer_nspc = false;  // Let factory enumerate; do not bias to nspc blindly for enforced filters
    auto srcDesc = creatorsMap.at(prefer_nspc ? LayoutType::nspc : LayoutType::ncsp)
                       ->createSharedDesc(inputPrecision, getInputShapeAtPort(0));
    auto dstDesc = creatorsMap.at(prefer_nspc ? LayoutType::nspc : LayoutType::ncsp)
                       ->createSharedDesc(outputPrecision, getOutputShapeAtPort(0));

    MemoryDescArgs descs;
    descs[ARG_SRC_0] = srcDesc;
    descs[ARG_DST] = dstDesc;

    // Init factory and preconfigure memory descriptors
    auto executionContext = std::make_shared<ExecutorContext>(context, getImplPriority());
    // Respect externally enforced memory formats (e.g., from tests) via memoryFormatFilter
    auto factory = std::make_shared<ExecutorFactory<MVNAttrs>>(mvnAttrs, executionContext, descs, memoryFormatFilter);
    const std::vector<MemoryDescArgs> nodeDescriptorsList = factory->getProperMemoryDescriptors(descs);

    // Build supported primitive descriptors; prefer nspc when formats are enforced
    std::vector<NodeDesc> nspc_spds;
    std::vector<NodeDesc> other_spds;
    for (const auto& nodeDescriptors : nodeDescriptorsList) {
        NodeConfig nodeConfig;
        nodeConfig.inConfs.resize(getParentEdges().size());

        const auto& outDesc = nodeDescriptors.at(ARG_DST);
        const auto outPrecision = outDesc->getPrecision();

        // Input config
        const int inPlace = (!isDynamicNode() && inputPrecision.size() == outPrecision.size() &&
                             getParentEdgeAt(0)->getParent()->getChildEdges().size() == 1 &&
                             !getParentEdgeAt(0)->getParent()->isConstant())
                                ? 0
                                : -1;

        nodeConfig.inConfs[0] = PortConfig(nodeDescriptors.at(ARG_SRC_0), BlockedMemoryDesc::SKIP_OFFSET_MASK, -1);
        nodeConfig.outConfs.emplace_back(outDesc, BlockedMemoryDesc::SKIP_OFFSET_MASK, inPlace);

        // Axes input if present
        if (getParentEdges().size() == 2) {
            nodeConfig.inConfs[1].setMemDesc(
                std::make_shared<CpuBlockedMemoryDesc>(ov::element::i32, getInputShapeAtPort(1)));
            nodeConfig.inConfs[1].constant(true);
        }

        const bool is_nspc = nodeDescriptors.at(ARG_SRC_0)->hasLayoutType(LayoutType::nspc);
        if (is_nspc) {
            nspc_spds.emplace_back(nodeConfig, ov::intel_cpu::impl_desc_type::undef);
        } else {
            other_spds.emplace_back(nodeConfig, ov::intel_cpu::impl_desc_type::undef);
        }
    }

    if (enforce_formats) {
        // If formats enforced, and nspc variants exist, keep only nspc to avoid accidental planar selection
        if (!nspc_spds.empty()) {
            supportedPrimitiveDescriptors.insert(supportedPrimitiveDescriptors.end(),
                                                 nspc_spds.begin(),
                                                 nspc_spds.end());
        } else {
            supportedPrimitiveDescriptors.insert(supportedPrimitiveDescriptors.end(),
                                                 other_spds.begin(),
                                                 other_spds.end());
        }
    } else {
        // Preserve original order when no filter enforced
        supportedPrimitiveDescriptors.insert(supportedPrimitiveDescriptors.end(), other_spds.begin(), other_spds.end());
        supportedPrimitiveDescriptors.insert(supportedPrimitiveDescriptors.end(), nspc_spds.begin(), nspc_spds.end());
    }
}

void MVN::prepareParams() {
    auto dstMemPtr = getDstMemoryAtPort(0);
    auto srcMemPtr = getSrcMemoryAtPort(0);
    if (!dstMemPtr || !dstMemPtr->isDefined()) {
        CPU_NODE_THROW("Destination memory is undefined.");
    }
    if (!srcMemPtr || !srcMemPtr->isDefined()) {
        CPU_NODE_THROW("Input memory is undefined.");
    }
    if (getSelectedPrimitiveDescriptor() == nullptr) {
        CPU_NODE_THROW("Preferable primitive descriptor is not set.");
    }

    const VectorDims in_dims = srcMemPtr->getStaticDims();
    transformTo5DCase(in_dims);
    mvnAttrs.shape5D = shape5D;

    auto* selectedPD = getSelectedPrimitiveDescriptor();
    // Derive layout from the selected primitive descriptor rather than runtime memory,
    // to honor the chosen memory format (e.g., NHWC) before memory is materialized.
    auto selectedInDesc = selectedPD->getConfig().inConfs[0].getMemDesc();
    if (selectedInDesc->hasLayoutType(LayoutType::ncsp)) {
        mvnAttrs.layout = MVNLayoutType::mvn_planar;
    } else if (selectedInDesc->hasLayoutType(LayoutType::nspc)) {
        mvnAttrs.layout = MVNLayoutType::mvn_by_channel;
    } else {
        mvnAttrs.layout = MVNLayoutType::mvn_block;
    }

    // Determine actual channel size based on logical 5D shape (N, C, D, H, W)
    // This is needed for post-ops to have correct channel dimension
    if (shape5D.size() >= 2) {
        mvnAttrs.actualChannelSize = shape5D[1];
    } else if (shape5D.size() == 1) {
        // For 1D case, the entire dimension might be channels
        mvnAttrs.actualChannelSize = mvnAttrs.initAcrossChannels_ ? 1 : shape5D[0];
    } else {
        mvnAttrs.actualChannelSize = 1;
    }

    // Populate post-ops from fused nodes (no adjustments)
    mvnAttrs.postOps = getPostOps(fusedWith);

    // Update executor with memory arguments
    MemoryArgs memoryArgs;
    memoryArgs[ARG_SRC_0] = getSrcMemoryAtPort(0);
    memoryArgs[ARG_DST] = getDstMemoryAtPort(0);

    {
        auto execCtx = std::make_shared<ExecutorContext>(context, getImplPriority());
        auto makeFactory =
            std::make_shared<ExecutorFactory<MVNAttrs>>(mvnAttrs,
                                                        execCtx,
                                                        MemoryDescArgs{{ARG_SRC_0, getSrcMemoryAtPort(0)->getDescPtr()},
                                                                       {ARG_DST, getDstMemoryAtPort(0)->getDescPtr()}});
        executorPtr = makeFactory->make(memoryArgs);
    }

    executorPtr->update(memoryArgs);
    selectedPD->setImplementationType(executorPtr->implType());
}

void MVN::createPrimitive() {
    MemoryArgs memoryArgs;
    memoryArgs[ARG_SRC_0] = getSrcMemoryAtPort(0);
    memoryArgs[ARG_DST] = getDstMemoryAtPort(0);

    auto execCtx = std::make_shared<ExecutorContext>(context, getImplPriority());
    // Build factory based on currently selected PD
    MemoryDescArgs descs;
    descs[ARG_SRC_0] = getSrcMemoryAtPort(0)->getDescPtr();
    descs[ARG_DST] = getDstMemoryAtPort(0)->getDescPtr();

    auto factory = std::make_shared<ExecutorFactory<MVNAttrs>>(mvnAttrs, execCtx, descs, memoryFormatFilter);
    executorPtr = factory->make(memoryArgs);

    Node::createPrimitive();
}

void MVN::transformTo5DCase(const VectorDims& shape) {
    size_t rank = shape.size();
    // for 1 and 2 rank, if initAcrossChannels_ is true, adjust shape to fully vectorize under the unified 5d procedure.
    // otherwise there is not enough data in spatial dimension to process in one kernel.
    switch (rank) {
    case 1:  // C
        if (mvnAttrs.initAcrossChannels_) {
            shape5D = {1, 1, 1, 1, shape[0]};
            mvnAttrs.execAcrossChannels_ = false;
            break;
        } else {
            shape5D = {1, shape[0], 1, 1, 1};
            break;
        }
    case 2:  // NC
        if (mvnAttrs.initAcrossChannels_) {
            shape5D = {1, shape[0], 1, shape[1], 1};
            mvnAttrs.execAcrossChannels_ = false;
            break;
        } else {
            shape5D = {shape[0], shape[1], 1, 1, 1};
            break;
        }
    case 3: {
        shape5D = {shape[0], shape[1], 1, shape[2], 1};
        break;
    }
    case 4: {
        shape5D = {shape[0], shape[1], 1, shape[2], shape[3]};
        break;
    }
    case 5: {
        shape5D = {shape[0], shape[1], shape[2], shape[3], shape[4]};
        break;
    }
    default: {
        CPU_NODE_THROW("doesn't support planar layout with rank: ", shape.size());
    }
    }
}

void MVN::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

void MVN::execute([[maybe_unused]] const dnnl::stream& strm) {
    if (executorPtr) {
        MemoryArgs memoryArgs;
        memoryArgs[ARG_SRC_0] = getSrcMemoryAtPort(0);
        memoryArgs[ARG_DST] = getDstMemoryAtPort(0);

        executorPtr->execute(memoryArgs);
    } else {
        CPU_NODE_THROW("Primitive wasn't created");
    }
}

bool MVN::canFuse(const NodePtr& node) const {
    if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::sse41)) {
        return false;
    }
    // limit post-ops to unary when shape transformed on channel
    // 1D only fused with unary
    int inputRank = getInputShapeAtPort(0).getRank();
    bool unaryEltwise = isUnaryEltwise(node);
    if ((inputRank == 1 && !unaryEltwise) || (inputRank == 2 && !unaryEltwise && mvnAttrs.initAcrossChannels_)) {
        return false;
    }

    return canFuseSimpleOperation(node);
}

bool MVN::created() const {
    return getType() == Type::MVN;
}

}  // namespace ov::intel_cpu::node
