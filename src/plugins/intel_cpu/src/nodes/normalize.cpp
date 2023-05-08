// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "normalize.h"

#include <ie_parallel.hpp>

#include "fake_quantize.h"
#include "eltwise.h"
#include "utils/bfloat16.hpp"
#include "utils/general_utils.h"
#include <dnnl_extension_utils.h>
#include <cpu/x64/injectors/jit_uni_quantization_injector.hpp>
#include "nodes/common/cpu_convert.h"
#include <selective_build.h>

#include <ngraph/opsets/opset1.hpp>
#include <utils/shape_inference/shape_inference_pass_through.hpp>

using namespace InferenceEngine;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

#define THROW_ERROR IE_THROW() << "NormalizeL2 layer with name '" << getName() << "' "

namespace ov {
namespace intel_cpu {
namespace node {

bool NormalizeL2::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        auto norm = ov::as_type_ptr<const ngraph::op::v0::NormalizeL2>(op);
        if (!norm) {
            errorMessage = "Only opset1 NormalizeL2 operation is supported";
            return false;
        }

        const auto inputRank = norm->get_input_partial_shape(DATA).size();
        if (inputRank < 2 || inputRank > 4) {
            errorMessage = "Doesn't support 'data' input with rank: " + std::to_string(inputRank);
            return false;
        }

        auto axesNode = ov::as_type_ptr<const ngraph::op::v0::Constant>(norm->get_input_node_shared_ptr(AXES));
        if (!axesNode) {
            errorMessage = "Supports only constant 'axes' input";
            return false;
        }

        if (axesNode->get_type_info() != ov::op::v0::Constant::get_type_info_static()) {
            // TODO [DS]: Add 'axes' input dynamism support
            errorMessage = "Doesn't support dynamic 'axes' input";
            return false;
        }

        const auto isSupportedAxes = [](const std::vector<size_t> &axes, const size_t inputRank) {
            if (axes.size() == 1 && axes[0] == 1) {
                return true;
            } else if (axes.size() == inputRank - 1) {
                auto sortAxes = axes;
                std::sort(sortAxes.begin(), sortAxes.end());
                for (size_t i = 0; i < sortAxes.size(); i++) {
                    if (sortAxes[i] != i + 1)
                        return false;
                }
                return true;
            }
            return false;
        };

        const auto axes = axesNode->cast_vector<size_t>();
        if (!isSupportedAxes(axes, inputRank) && ngraph::shape_size(axesNode->get_shape()) != 0) {
            errorMessage = "Doesn't support reduction axes: " + vec2str(axes);
            return false;
        }

        const auto mode = norm->get_eps_mode();
        if (!one_of(mode, ngraph::op::EpsMode::ADD, ngraph::op::EpsMode::MAX)) {
            errorMessage = "Doesn't support eps_mode: " + ngraph::as_string(mode);
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

NormalizeL2::NormalizeL2(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context) :
        Node(op, context, PassThroughShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    if (inputShapes.size() != 2 || outputShapes.size() != 1)
        THROW_ERROR << " has incorrect number of input/output edges";

    if (getInputShapeAtPort(DATA).getRank() > 4 || getInputShapeAtPort(DATA).getRank() < 2) {
        THROW_ERROR << "has invalid input shape. Normalize supports from 2D to 4D blobs.";
    }

    auto norm = ov::as_type_ptr<const ngraph::op::v0::NormalizeL2>(op);
    attrs.axisSet = norm->get_reduction_axes();
    attrs.eps = norm->get_eps();
    attrs.epsMode = norm->get_eps_mode() == ngraph::op::EpsMode::MAX ? NormEpsMode::MAX : NormEpsMode::ADD;
    attrs.across_spatial = ngraph::shape_size(op->get_input_shape(AXES)) != 1;
    // One of the corner cases is when axes is an empty list,
    // then we divide each input element by itself resulting value 1 for all non-zero elements
    attrs.cornerCase = ngraph::shape_size(op->get_input_shape(AXES)) == 0;
}

void NormalizeL2::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    Precision inputPrecision = getOriginalInputPrecisionAtPort(DATA);
    Precision outputPrecision = getOriginalOutputPrecisionAtPort(DATA);

    if (!fusedWith.empty()) {
        attrs.isFusing = true;
        outputPrecision = fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0);
    }

    if (inputPrecision == Precision::BF16 || outputPrecision == Precision::BF16) {
        if (!mayiuse(avx512_core))
            inputPrecision = outputPrecision = Precision::FP32;
        else
            inputPrecision = outputPrecision = Precision::BF16;
    }

    if (!one_of(inputPrecision, Precision::FP32, Precision::BF16, Precision::I8, Precision::U8)) {
        THROW_ERROR << "has unsupported input precision: " << inputPrecision;
    }
    if (!one_of(outputPrecision, Precision::FP32, Precision::BF16, Precision::I8, Precision::U8)) {
        THROW_ERROR << "has unsupported output precision: " << outputPrecision;
    }

    attrs.input_prec = inputPrecision;
    attrs.output_prec = outputPrecision;
    attrs.src_data_size = inputPrecision.size();
    attrs.dst_data_size = outputPrecision.size();

    // TODO [DS]: inplace
    bool canBeInplace = !isDynamicNode() && attrs.src_data_size == attrs.dst_data_size &&
                        getParentEdgeAt(DATA)->getParent()->getChildEdges().size() == 1;

    NodeConfig config;
    config.inConfs.resize(2);
    config.outConfs.resize(1);
    config.outConfs[0].inPlace(canBeInplace ? 0 : -1);

    auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    auto pushDesc = [&](LayoutType format, impl_desc_type impl_type) {
        auto a = creatorsMap.at(format)->createSharedDesc(inputPrecision, getInputShapeAtPort(DATA));
        config.inConfs[0].setMemDesc(std::move(a));
        a = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(InferenceEngine::Precision::I32, getInputShapeAtPort(AXES));
        config.inConfs[1].setMemDesc(std::move(a));
        a = creatorsMap.at(format)->createSharedDesc(outputPrecision, getOutputShapeAtPort(DATA));
        config.outConfs[0].setMemDesc(std::move(a));

        std::vector<MemoryDescPtr> srcMemoryDescs;
        for (int i = 0; i < config.inConfs.size(); i++) {
            srcMemoryDescs.push_back(config.inConfs[i].getMemDesc());
        }
        std::vector<MemoryDescPtr> dstMemoryDescs;
        for (int i = 0; i < config.outConfs.size(); i++) {
            dstMemoryDescs.push_back(config.outConfs[i].getMemDesc());
        }
        auto factory = std::make_shared<NormalizeL2ExecutorFactory>(attrs, srcMemoryDescs, dstMemoryDescs,
                                                                    std::make_shared<ExecutorContext>(context, getPrimitivesPriority()));
        supportedPrimitiveDescriptors.push_back({config, impl_type, factory});
    };

    attrs.implDescType = impl_desc_type::unknown;

    // only plain layout support when w/o sse42
    if (getInputShapeAtPort(DATA).getRank() == 4 && !attrs.cornerCase) {
        if (mayiuse(cpu::x64::sse41)) {
            pushDesc(LayoutType::nspc, attrs.implDescType);
            if (mayiuse(cpu::x64::avx512_core)) {
                pushDesc(LayoutType::nCsp16c, attrs.implDescType);
            } else {
                pushDesc(LayoutType::nCsp8c, attrs.implDescType);
            }
        }
    }
    if (canBeInplace)
        config.inConfs[0].inPlace(0);
    pushDesc(LayoutType::ncsp, attrs.implDescType);
}

bool NormalizeL2::canFuse(const NodePtr& node) const {
    return !attrs.cornerCase && canFuseSimpleOperation(node);
}

void NormalizeL2::setPostOps(dnnl::primitive_attr& kernel_attrs, const VectorDims& dims, bool initWeights) {
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
            eltwiseNode->appendPostOps(ops, dims, postOpsDataPtrs);
            continue;
        }

        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType()) << " node is not implemented";
    }

    kernel_attrs.set_post_ops(ops);
}

void NormalizeL2::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(DATA)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(DATA)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        THROW_ERROR << "can't get destination memory";
    if (!srcMemPtr || !srcMemPtr->isAllocated())
        THROW_ERROR << "can't get input memory";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_ERROR << "has nullable preferable primitive descriptor";

    if (!attrs.cornerCase) {
        if (srcMemPtr->getDesc().hasLayoutType(LayoutType::ncsp)) {
            attrs.layout = LayoutType::ncsp;
        } else if (srcMemPtr->getDesc().hasLayoutType(LayoutType::nCsp8c)) {
            attrs.layout = LayoutType::nCsp8c;
        } else if (srcMemPtr->getDesc().hasLayoutType(LayoutType::nCsp16c)) {
            attrs.layout = LayoutType::nCsp16c;
        } else if (srcMemPtr->getDesc().hasLayoutType(LayoutType::nspc)) {
            attrs.layout = LayoutType::nspc;
        } else {
            THROW_ERROR << "has selected layout which is not supported";
        }
    }

    if (inputShapesDefined()) {
        if (needPrepareParams())
            prepareParams();
        updateLastInputDims();
    }
}

bool NormalizeL2::isExecutable() const {
    return !isInputTensorAtPortEmpty(0);
}

void NormalizeL2::prepareParams() {
    const auto& dims = getParentEdgeAt(DATA)->getMemoryPtr()->getStaticDims();

    setPostOps(kernel_attrs, dims, true);

    attrs.vectorDims = dims;
    NormalizeKey key = {attrs, kernel_attrs};

    auto builder = [this](const NormalizeKey& key) -> std::shared_ptr<NormalizeL2Executor> {
        std::vector<MemoryDescPtr> srcMemoryDescs;
        for (int i = 0; i < getOriginalInputsNumber(); i++) {
            srcMemoryDescs.push_back(getParentEdgeAt(i)->getMemoryPtr()->getDescPtr());
        }
        std::vector<MemoryDescPtr> dstMemoryDescs;
        for (int i = 0; i < getOriginalOutputsNumber(); i++) {
            dstMemoryDescs.push_back(getChildEdgeAt(i)->getMemoryPtr()->getDescPtr());
        }

        auto selectedPD = getSelectedPrimitiveDescriptor();
        auto currExecPtr = selectedPD->getExecutorFactoryAs<NormalizeL2ExecutorFactory>()->makeExecutor(attrs, srcMemoryDescs, dstMemoryDescs, kernel_attrs);
        selectedPD->setImplementationType(currExecPtr->getImplType());
        return currExecPtr;
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, builder);

    if (!result.first) {
        IE_THROW() << "Primitive descriptor was not found for node " << getName() << ".";
    }

    execPtr = result.first;
}

void NormalizeL2::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

void NormalizeL2::execute(dnnl::stream strm) {
    if (!execPtr)
        THROW_ERROR << "doesn't have a compiled executor.";
    execPtr->exec({getParentEdgeAt(DATA)->getMemoryPtr()}, {getChildEdgeAt(DATA)->getMemoryPtr()}, postOpsDataPtrs.data());
}

bool NormalizeL2::created() const {
    return getType() == Type::NormalizeL2;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
