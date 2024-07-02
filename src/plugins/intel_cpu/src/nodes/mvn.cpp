// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn.h"

#include <algorithm>
#include <string>
#include <vector>
#include <memory>

#include "fake_quantize.h"
#include "eltwise.h"
#include "dnnl_extension_utils.h"

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_uni_eltwise.hpp"

#include <openvino/opsets/opset6.hpp>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "utils/cpu_utils.hpp"

using namespace dnnl;

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
MVNJitExecutor(const MVNAttrs& mvnAttrs,
               const dnnl::primitive_attr &attr);
namespace ov {
namespace intel_cpu {
namespace node {
namespace {

struct MVNKey {
    MVNAttrs mvnAttrs;
    dnnl::primitive_attr attr;

    size_t hash() const;
    bool operator==(const MVNKey& rhs) const;
};

size_t MVNKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;
    seed = hash_combine(seed, mvnAttrs.initAcrossChannels_);
    seed = hash_combine(seed, mvnAttrs.execAcrossChannels_);
    seed = hash_combine(seed, mvnAttrs.normalizeVariance_);
    seed = hash_combine(seed, mvnAttrs.epsValue_);
    seed = hash_combine(seed, mvnAttrs.epsMode_);
    seed = hash_combine(seed, mvnAttrs.src_prc.hash());
    seed = hash_combine(seed, mvnAttrs.dst_prc.hash());
    seed = hash_combine(seed, mvnAttrs.layout);
    seed = hash_combine(seed, get_attr_hash(*attr.get()));
    return seed;
}

bool MVNKey::operator==(const MVNKey& rhs) const {
    bool retVal = true;
    retVal = retVal &&
             mvnAttrs.initAcrossChannels_ == rhs.mvnAttrs.initAcrossChannels_ &&
             mvnAttrs.execAcrossChannels_ == rhs.mvnAttrs.execAcrossChannels_ &&
             mvnAttrs.normalizeVariance_ == rhs.mvnAttrs.normalizeVariance_ &&
             mvnAttrs.epsValue_ == rhs.mvnAttrs.epsValue_ &&
             mvnAttrs.epsMode_ == rhs.mvnAttrs.epsMode_ &&
             mvnAttrs.src_prc == rhs.mvnAttrs.src_prc &&
             mvnAttrs.dst_prc == rhs.mvnAttrs.dst_prc &&
             mvnAttrs.layout == rhs.mvnAttrs.layout;
    retVal = retVal && *attr.get() == *rhs.attr.get();
    return retVal;
}
} // namespace



//////////////////////////////////////////////////////////////////////////////////

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
            if (epsMode != ov::op::MVNEpsMode::INSIDE_SQRT &&
                    epsMode != ov::op::MVNEpsMode::OUTSIDE_SQRT) {
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
            for (int& axe : axesVal)
                axe = axe < 0 ? axe + inDataRank : axe;
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

MVN::MVN(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
        : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
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
        if (inDataShapeSize == mvnOp->input_value(1).get_shape()[0] + 1 || inDataShapeSize == 1)
            mvnAttrs.initAcrossChannels_ = true;
    } else if (auto mvnOp = ov::as_type_ptr<ov::op::v0::MVN>(op)) {
        mvnAttrs.normalizeVariance_ = mvnOp->get_normalize_variance();
        mvnAttrs.epsValue_ = mvnOp->get_eps();
        mvnAttrs.initAcrossChannels_ = mvnOp->get_across_channels();
    } else {
        OPENVINO_THROW_NOT_IMPLEMENTED("Node is not an instance of MVN from the operation set v0 or v6");
    }
    mvnAttrs.execAcrossChannels_ = mvnAttrs.initAcrossChannels_;
}

void MVN::getSupportedDescriptors() {}

static inline bool isUnaryEltwise(const NodePtr& node) {
    return one_of(node->getAlgorithm(), Algorithm::EltwiseRelu,
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
    if (!supportedPrimitiveDescriptors.empty())
        return;

    ov::element::Type inputPrecision = getOriginalInputPrecisionAtPort(0);
    ov::element::Type outputPrecision = getOriginalOutputPrecisionAtPort(0);
    if (!hasHardwareSupport(outputPrecision))
        outputPrecision = ov::element::f32;

    if (!fusedWith.empty()) {
        outputPrecision = fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0);
        onlyUnaryPostOps = true;
        for (auto &node : fusedWith) {
            if (isUnaryEltwise(node)) {
                continue;
            } else {
                onlyUnaryPostOps = false;
                break;
            }
        }
    }
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
    // ref with float planar and no fusion
    if (!mayiuse(cpu::x64::sse41)) {
        inputPrecision = outputPrecision = ov::element::f32;
    }
#endif
//Output precision has to be equal to input precision in ACL MVN
#if defined(OV_CPU_WITH_ACL)
    outputPrecision = inputPrecision;
#endif
    // TODO [DS]: inplace
    bool canBeInplace = !isDynamicNode() && (inputPrecision.size() == outputPrecision.size()) &&
                        (getParentEdgeAt(0)->getParent()->getChildEdges().size() == 1) &&
                        !getParentEdgeAt(0)->getParent()->isConstant();

    const size_t inputsNum = getParentEdges().size();
    NodeConfig config;
    config.inConfs.resize(inputsNum);
    config.outConfs.resize(1);
    config.inConfs[0].constant(false);
    config.outConfs[0].constant(false);
    config.inConfs[0].inPlace(-1);
    config.outConfs[0].inPlace(canBeInplace ? 0 : -1);
    if (inputsNum == 2) {
        config.inConfs[1].setMemDesc(std::make_shared<CpuBlockedMemoryDesc>(ov::element::i32, getInputShapeAtPort(1)));
        config.inConfs[1].constant(true);
    }

    auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    auto pushDesc = [&](LayoutType format, impl_desc_type impl_type, bool useAclExecutor = false) {
        config.inConfs[0].setMemDesc(creatorsMap.at(format)->createSharedDesc(inputPrecision, getInputShapeAtPort(0)));
        config.outConfs[0].setMemDesc(creatorsMap.at(format)->createSharedDesc(outputPrecision, getOutputShapeAtPort(0)));

        if (useAclExecutor) {
            std::vector<MemoryDescPtr> srcMemoryDescs;
            for (size_t i = 0; i < config.inConfs.size(); i++) {
                srcMemoryDescs.push_back(config.inConfs[i].getMemDesc());
            }
            std::vector<MemoryDescPtr> dstMemoryDescs;
            for (size_t i = 0; i < config.outConfs.size(); i++) {
                dstMemoryDescs.push_back(config.outConfs[i].getMemDesc());
            }

            auto factory = std::make_shared<MVNExecutorFactory>(mvnAttrs, srcMemoryDescs, dstMemoryDescs,
                                                                        std::make_shared<ExecutorContext>(context, getImplPriority()));
            if (!factory->isEmpty()) {
                supportedPrimitiveDescriptors.push_back({config, impl_type, factory});
            }
        } else {
            supportedPrimitiveDescriptors.push_back({config, impl_type});
        }
    };

#if defined(OV_CPU_WITH_ACL)
        pushDesc(LayoutType::nspc, undef, true);
        pushDesc(LayoutType::ncsp, undef, true);
        canUseAclExecutor = !supportedPrimitiveDescriptors.empty();
        if (canUseAclExecutor)
            return;
        else
            // Reference MVN implementation does not support fp16, so set fp32 explicitly
            inputPrecision = outputPrecision = ov::element::f32;
#endif // OV_CPU_WITH_ACL

    impl_desc_type impl_type;
    if (mayiuse(cpu::x64::avx512_core)) {
        impl_type = impl_desc_type::jit_avx512;
    } else if (mayiuse(cpu::x64::avx2)) {
        impl_type = impl_desc_type::jit_avx2;
    } else if (mayiuse(cpu::x64::sse41)) {
        impl_type = impl_desc_type::jit_sse42;
    } else {
        impl_type = impl_desc_type::ref;
    }

    if (mayiuse(cpu::x64::sse41)) {
        // nspc
        if (getInputShapeAtPort(0).getRank() == 4 || getInputShapeAtPort(0).getRank() == 5) {
            pushDesc(LayoutType::nspc, impl_type);
        }
        // blk
        if (impl_desc_type::jit_avx512 == impl_type) {
            if (getInputShapeAtPort(0).getRank() == 4 || getInputShapeAtPort(0).getRank() == 5) {
                pushDesc(LayoutType::nCsp16c, impl_type);
            }
        } else if (impl_desc_type::jit_avx2 ==  impl_type || impl_desc_type::jit_sse42 == impl_type) {
            if (getInputShapeAtPort(0).getRank() == 4 || getInputShapeAtPort(0).getRank() == 5) {
                pushDesc(LayoutType::nCsp8c, impl_type);
            }
        }
    }

    // planar
    if (canBeInplace)
        config.inConfs[0].inPlace(0);
    pushDesc(LayoutType::ncsp, impl_type);
}

void MVN::prepareParams() {
    auto dstMemPtr = getDstMemoryAtPort(0);
    auto srcMemPtr = getSrcMemoryAtPort(0);
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        OPENVINO_THROW("Destination memory didn't allocate.");
    if (!srcMemPtr || !srcMemPtr->isAllocated())
        OPENVINO_THROW("Input memory didn't allocate.");
    if (getSelectedPrimitiveDescriptor() == nullptr)
        OPENVINO_THROW("Preferable primitive descriptor is not set.");

    const VectorDims in_dims = srcMemPtr->getStaticDims();
    transformTo5DCase(in_dims);

#if defined(OPENVINO_ARCH_X86_64)
    // New shape5D always need prepare via transformTo5DCase(), which is need in exec().
    // MVN itself and unary post ops is totally shape agnostic, execPtr can be reused directly w/o recompilation and setPostOps when shape is changed.
    // As key have not shape, if shape changes and new post ops attr is also the same, execPtr can still hit.
    // If new shape(channel changes) impact post ops attr, such as entry.quantization.offset, entry.depthwise.offset, entry.quantization.per_channel,
    // which is participate in compilation, even postOpsData is passed in runtime, still need recompilation.
    if (execPtr != nullptr && (fusedWith.empty() || onlyUnaryPostOps)) {
        return;
    }
#endif

    auto selectedPD = getSelectedPrimitiveDescriptor();
    mvnAttrs.src_prc = selectedPD->getConfig().inConfs[0].getMemDesc()->getPrecision();
    mvnAttrs.dst_prc = selectedPD->getConfig().outConfs[0].getMemDesc()->getPrecision();
    if (getParentEdgeAt(0)->getMemory().getDesc().hasLayoutType(LayoutType::ncsp)) {
        mvnAttrs.layout = MVNLayoutType::mvn_planar;
    } else if (getParentEdgeAt(0)->getMemory().getDesc().hasLayoutType(LayoutType::nspc)) {
        mvnAttrs.layout = MVNLayoutType::mvn_by_channel;
    } else {
        mvnAttrs.layout = MVNLayoutType::mvn_block;
    }

    if (canUseAclExecutor) {
        std::vector<MemoryDescPtr> srcMemoryDescs;
        for (size_t i = 0; i < getParentEdges().size(); i++) {
            srcMemoryDescs.push_back(getSrcMemoryAtPort(i)->getDescPtr());
        }
        std::vector<MemoryDescPtr> dstMemoryDescs;
        dstMemoryDescs.push_back(getDstMemoryAtPort(0)->getDescPtr());

        auto selectedPD = getSelectedPrimitiveDescriptor();
        aclExecPtr = selectedPD->getExecutorFactoryAs<MVNExecutorFactory>()->makeExecutor(mvnAttrs, srcMemoryDescs, dstMemoryDescs, {});
        selectedPD->setImplementationType(aclExecPtr->getImplType());

        return;
    }

    MVNKey key = {mvnAttrs, dnnl::primitive_attr()};
    setPostOps(key.attr, true);

    auto builder = [&](const MVNKey& key) -> std::shared_ptr<MVNExecutorBase> {
        std::shared_ptr<MVNExecutorBase> executor;
        if (mayiuse(cpu::x64::sse41)) {
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
            executor = std::make_shared<MVNJitExecutor>(key.mvnAttrs, key.attr);
#endif
        } else {
            executor = std::make_shared<MVNRefExecutor>(key.mvnAttrs);
        }
        return executor;
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, builder);
    execPtr = result.first;
}

void MVN::transformTo5DCase(const VectorDims& shape) {
    size_t rank = shape.size();
    // for 1 and 2 rank, if initAcrossChannels_ is true, adjust shape to fully vectorize under unified 5d procedure.
    // otherwise there are not enough data in spatial dimension to process in one kernel.
    switch (rank) {
        case 1 :  // C
            if (mvnAttrs.initAcrossChannels_) {
                shape5D = {1, 1, 1, 1, shape[0]};
                mvnAttrs.execAcrossChannels_ = false;
                break;
            } else {
                shape5D = {1, shape[0], 1, 1, 1};
                break;
            }
        case 2 :  // NC
            if (mvnAttrs.initAcrossChannels_) {
                shape5D = {1, shape[0], 1, shape[1], 1};
                mvnAttrs.execAcrossChannels_ = false;
                break;
            } else {
                shape5D = {shape[0], shape[1], 1, 1, 1};
                break;
            }
        case 3 : { shape5D = {shape[0], shape[1], 1, shape[2], 1}; break; }
        case 4 : { shape5D = {shape[0], shape[1], 1, shape[2], shape[3]}; break; }
        case 5 : { shape5D = {shape[0], shape[1], shape[2], shape[3], shape[4]}; break; }
        default: {
            OPENVINO_THROW("MVN layer with name '",
                           getName(),
                           "' doesn't support planar layout with rank: ",
                           shape.size());
        }
    }
}

void MVN::setPostOps(dnnl::primitive_attr &attr, bool initWeights) {
    dnnl::post_ops ops;
    postOpsDataPtrs.clear();
    for (auto &node : fusedWith) {
        auto* fakeQuantizeNode = dynamic_cast<FakeQuantize *>(node.get());
        if (fakeQuantizeNode) {
            fakeQuantizeNode->appendPostOps(ops, {}, postOpsDataPtrs);
            continue;
        }

        auto* eltwiseNode = dynamic_cast<Eltwise *>(node.get());
        if (eltwiseNode) {
            eltwiseNode->appendPostOps(ops, shape5D, postOpsDataPtrs);
            continue;
        }
        OPENVINO_THROW("Fusing of ",
                       NameFromType(node->getType()),
                       " operation to ",
                       NameFromType(this->getType()),
                       " node is not implemented");
    }
    attr.set_post_ops(ops);
}

void MVN::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

void MVN::execute(dnnl::stream strm) {
    auto dstMemPtr = getDstMemoryAtPort(0);
    auto srcMemPtr = getSrcMemoryAtPort(0);

    if (execPtr) {
        uint8_t *dst_data = dstMemPtr->getDataAs<uint8_t>();
        uint8_t *src_data = srcMemPtr->getDataAs<uint8_t>();
        execPtr->exec(src_data, dst_data, postOpsDataPtrs.data(), shape5D);
    } else if (aclExecPtr) {
        aclExecPtr->exec({srcMemPtr}, {dstMemPtr}, postOpsDataPtrs.data());
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
    if ((inputRank == 1 && !unaryEltwise) ||
        (inputRank == 2 && !unaryEltwise && mvnAttrs.initAcrossChannels_)) {
        return false;
    }

    return canFuseSimpleOperation(node);
}

bool MVN::created() const {
    return getType() == Type::MVN;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
