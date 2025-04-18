// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_uni_eltwise.hpp"
#include "dnnl_extension_utils.h"
#include "emitters/plugin/x64/jit_bf16_emitters.hpp"
#include "fake_quantize.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "utils/cpu_utils.hpp"

using namespace dnnl;

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

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
                if (inDataRank > 5 || (static_cast<size_t>(inDataRank) != axesVal.size() + 1 &&
                                       static_cast<size_t>(inDataRank) != axesVal.size() + 2)) {
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
        mvnAttrs.epsValue_ = mvnOp->get_eps();
        mvnAttrs.initAcrossChannels_ = mvnOp->get_across_channels();
    } else {
        OPENVINO_THROW_NOT_IMPLEMENTED("Node is not an instance of MVN from the operation set v0 or v6");
    }
    mvnAttrs.execAcrossChannels_ = getAcrossChannels();
}

void MVN::getSupportedDescriptors() {}

static inline bool isUnaryEltwise(const NodePtr& node) {
    return one_of(node->getAlgorithm(),
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

    if (!fusedWith.empty()) {
        outputPrecision = fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0);
        onlyUnaryPostOps = true;
        for (auto& node : fusedWith) {
            if (isUnaryEltwise(node)) {
                continue;
            } else {
                onlyUnaryPostOps = false;
                break;
            }
        }
    }

    const auto& srcTypes = getOriginalInputPrecisions();
    auto dstTypes = getOriginalOutputPrecisions();
    // @todo graph optimizer should update original output precisions instead
    if (!fusedWith.empty()) {
        dstTypes = fusedWith.back()->getOriginalOutputPrecisions();
    }


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
            {ARG_DST, dstDescs[0]},
    };

    if (one_of(descs.at(ARG_SRC)->getShape().getRank(), 1lu, 2lu) && getAcrossChannels()) {
        mvnAttrs.execAcrossChannels_ = false;
    }
    mvnAttrs.src_prc = descs.at(ARG_SRC)->getPrecision();
    mvnAttrs.dst_prc = descs.at(ARG_DST)->getPrecision();

    mvnAttrs.postOps = getPostOps(fusedWith);
//    mvnAttrs.fusedWith = fusedWith;
    auto executionContext = std::make_shared<ExecutorContext>(context, getImplPriority(), privateWeightCache);
    factory = std::make_shared<ExecutorFactory<MVNAttrs>>(mvnAttrs, executionContext, descs);
    const auto nodeDescriptors = factory->getProperMemoryDescriptors(descs);

    // TODO [DS]: inplace
    bool canBeInplace = !isDynamicNode() && (inputPrecision.size() == outputPrecision.size()) &&
                        (getParentEdgeAt(0)->getParent()->getChildEdges().size() == 1) &&
                        !getParentEdgeAt(0)->getParent()->isConstant();

    const size_t inputsNum = getParentEdges().size();
    for (auto& nodeDesc : nodeDescriptors) {
        NodeConfig nodeConfig;
        nodeConfig.inConfs.emplace_back(nodeDesc.at(ARG_SRC));
        nodeConfig.outConfs.emplace_back(nodeDesc.at(ARG_DST));
        nodeConfig.inConfs.resize(inputsNum);
        nodeConfig.outConfs.resize(1);
        nodeConfig.inConfs[0].constant(false);
        nodeConfig.outConfs[0].constant(false);
        nodeConfig.inConfs[0].inPlace(-1);
        nodeConfig.outConfs[0].inPlace(canBeInplace ? 0 : -1);
        if (inputsNum == 2) {
            nodeConfig.inConfs[1].setMemDesc(
                    std::make_shared<CpuBlockedMemoryDesc>(ov::element::i32, getInputShapeAtPort(1)));
            nodeConfig.inConfs[1].constant(true);
        }

        // TODO: move canBeInplace to requiresFallbackCommon
        if (nodeDesc.at(ARG_SRC)->hasLayoutType(LayoutType::ncsp) && canBeInplace) {
            nodeConfig.inConfs[0].inPlace(0);
        }

        supportedPrimitiveDescriptors.emplace_back(nodeConfig, impl_desc_type::undef);
    }
}

ExecutorPtr MVN::createExecutor() {
    const auto& new_executor = factory->make(memory);
    getSelectedPrimitiveDescriptor()->setImplementationType(new_executor->implType());
    return new_executor;
}

void MVN::prepareParams() {
    if (!memory[ARG_DST] || !memory[ARG_DST]->isDefined()) {
        OPENVINO_THROW("Destination memory is undefined.");
    }
    if (!memory[ARG_SRC] || !memory[ARG_SRC]->isDefined()) {
        OPENVINO_THROW("Input memory is undefined.");
    }
    if (getSelectedPrimitiveDescriptor() == nullptr) {
        OPENVINO_THROW("Preferable primitive descriptor is not set.");
    }
    executor = createExecutor();
    executor->update(memory);
}

void MVN::createPrimitive() {
    memory[ARG_SRC] = getSrcMemoryAtPort(0);
    memory[ARG_DST] = getDstMemoryAtPort(0);

    // @todo should we preconfigure only for dynamic shapes?
    // Since for static shapes primitive is created in scope of compile_model() anyway
    executor = factory->make(memory);

    Node::createPrimitive();
}

void MVN::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

void MVN::execute(const dnnl::stream& strm) {
    if (executor) {
        executor->execute(memory);
    } else {
        OPENVINO_THROW("Can't execute Interpolate node. Primitive didn't created");
    }
}

bool MVN::canFuse(const NodePtr& node) const {
    if (!mayiuse(cpu::x64::sse41)) {
        return false;
    }
    // limit post ops to unary when shape transformed on channel
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
