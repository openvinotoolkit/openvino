// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert.h"

#include "common/blocked_desc_creator.h"
#include "dnnl_extension_utils.h"
#include "openvino/opsets/opset1.hpp"
#include "shape_inference/shape_inference_pass_through.hpp"

using namespace dnnl;

namespace ov {
namespace intel_cpu {
namespace node {

bool Convert::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto convert = std::dynamic_pointer_cast<const ov::opset1::Convert>(op);
        if (!convert) {
            errorMessage = "Only opset1 Convert operation is supported";
            return false;
        }

        auto srcPrc = op->get_input_element_type(0);
        auto dstPrc = op->get_output_element_type(0);
        if (!CommonConvertExecutor::isSupported(srcPrc, dstPrc)) {
            errorMessage = "cpu_convert can't convert from: " + srcPrc.to_string() + " precision to: " + dstPrc.to_string();
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Convert::Convert(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
        : Node(op, context, PassThroughShapeInferFactory()) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "Convert node with name '" + getName() + "'";
    } else {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    auto convert = ov::as_type_ptr<const ov::opset1::Convert>(op);
    convertParams.origPrc = convert->get_destination_type();
}

Convert::Convert(const Shape &shape, const ov::element::Type &inPrc, const ov::element::Type &outPrc,
                 const std::string &nodeName, const GraphContext::CPtr context)
    : Node("Convert", {shape}, {shape}, {inPrc}, {outPrc}, nodeName, context) {
    convertParams.origPrc = outPrc;

    isDynamic = shape.isDynamic();
    if (isDynamicNode()) {
        shapeInference = std::make_shared<ShapeInferPassThrough>();
    }

    errorPrefix = "Convert node with name '" + getName() + "'";
}

void Convert::getSupportedDescriptors() {
    // if tensor descriptors are set via setDescs method we need to update the inDims/outDims data
    // from correspond tensor descriptors.
    if (outputShapes.empty())
        outputShapes.push_back(output->getShape());
    if (inputShapes.empty())
        inputShapes.push_back(input->getShape());
    if (getParentEdges().size() != 1)
        OPENVINO_THROW(errorPrefix, " has incorrect number of input edges");
    if (getChildEdges().empty())
        OPENVINO_THROW(errorPrefix, " has incorrect number of output edges");
}

bool Convert::isSupportedDesc(const MemoryDesc &desc) {
    bool isSupported = desc.getType() & MemoryDescType::Blocked;
    if (desc.getType() == MemoryDescType::DnnlBlocked)
        isSupported &= desc.as<const DnnlMemoryDesc>()->hasEmptyExtraData();
    return isSupported;
}

void Convert::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    NodeConfig config;
    PortConfig dataIn;
    PortConfig dataConfigOut;

    bool canInitExternalDesc = false;
    if (input && output) {
        canInitExternalDesc = true;
        canInitExternalDesc &= isSupportedDesc(*input);
        canInitExternalDesc &= isSupportedDesc(*output);
    }

    auto supportedPrimitiveDescriptorsBuilder = [this](NodeConfig config) {
        MemoryDescPtr srcMemoryDesc = config.inConfs[0].getMemDesc();
        MemoryDescPtr dstMemoryDesc = config.outConfs[0].getMemDesc();
        convertParams.srcPrc = srcMemoryDesc->getPrecision();
        convertParams.dstPrc = dstMemoryDesc->getPrecision();
        auto factory = std::make_shared<ConvertExecutorFactory>(convertParams, srcMemoryDesc, dstMemoryDesc,
                                                                std::make_shared<ExecutorContext>(context, getImplPriority()));
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown, factory);
    };

    // if input and output pointers are not null and not contain extra data, then the inp/output tensor descriptors were set using setDescs method, so
    // they should be used as the actual descriptors.
    if (canInitExternalDesc) {
        dataIn.setMemDesc(input);
        config.inConfs.push_back(dataIn);

        // inp/out layouts must be the same
        dataConfigOut.setMemDesc(config.inConfs[0].getMemDesc());
        dataConfigOut.setMemDesc(dataConfigOut.getMemDesc()->cloneWithNewPrecision(output->getPrecision()));
        config.outConfs.push_back(dataConfigOut);
        supportedPrimitiveDescriptorsBuilder(config);
    } else if (inputShapes.size() == 1 && outputShapes.size() == 1) {
        const Shape& insShape = getInputShapeAtPort(0);
        auto insPrecision = getOriginalInputPrecisionAtPort(0);
        const Shape& outputShape = getOutputShapeAtPort(0);
        auto outPrecision = getOriginalOutputPrecisionAtPort(0);

        config.inConfs.push_back(dataIn);
        config.outConfs.push_back(dataConfigOut);

        auto creators = BlockedDescCreator::getCommonCreators();

        // As long as convert is placed right before the output, only planar layout makes sense since the output tensor
        // is always in a planar layout (ngraph limitation), so there is no reason to convert in any other layout.
        bool hasOutputChild = false;
        for (auto& childEdge : getChildEdgesAtPort(0)) {
            if (Type::Output == childEdge->getChild()->getType()) {
                hasOutputChild = true;
                break;
            }
        }
        auto range = hasOutputChild
                         ? BlockedDescCreator::makeFilteredRange(creators, insShape.getRank(), {LayoutType::ncsp})
                         : BlockedDescCreator::makeFilteredRange(creators, insShape.getRank());

        for (auto itr = range.first; itr != range.second; ++itr) {
            config.inConfs[0].setMemDesc(std::make_shared<CpuBlockedMemoryDesc>(itr->second->createDesc(insPrecision, insShape)));
            config.outConfs[0].setMemDesc(std::make_shared<CpuBlockedMemoryDesc>(itr->second->createDesc(outPrecision, outputShape)));

            supportedPrimitiveDescriptorsBuilder(config);
        }
    } else {
        OPENVINO_THROW(errorPrefix, " has incorrect number of input/output edges");
    }
}

void Convert::prepareParams() {
    auto& parentMem = getParentEdgeAt(0)->getMemory();
    convertParams.size = parentMem.getDescWithType<BlockedMemoryDesc>()->getPaddedElementsCount();

    auto selectedPD = getSelectedPrimitiveDescriptor();
    MemoryDescPtr srcDesc = getSrcMemoryAtPort(0)->getDescPtr();
    MemoryDescPtr dstDesc = getDstMemoryAtPort(0)->getDescPtr();
    execPtr = selectedPD->getExecutorFactoryAs<ConvertExecutorFactory>()->makeExecutor(convertParams,
                                                                                       srcDesc,
                                                                                       dstDesc,
                                                                                       {});
    selectedPD->setImplementationType(execPtr->implType());
}

void Convert::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

void Convert::execute(dnnl::stream strm) {
    auto& parentMem = getParentEdgeAt(0)->getMemory();
    auto& childMem = getChildEdgeAt(0)->getMemory();

    const auto parentPaddElemCount = parentMem.getDescWithType<BlockedMemoryDesc>()->getPaddedElementsCount();
    const auto childPaddElemCount = childMem.getDescWithType<BlockedMemoryDesc>()->getPaddedElementsCount();

    if (parentPaddElemCount != childPaddElemCount)
        OPENVINO_THROW(errorPrefix, " has different elements number in input and output buffers");

    MemoryCPtr srcMemory = getSrcMemoryAtPort(0);
    MemoryPtr dstMemory = getDstMemoryAtPort(0);
    execPtr->exec({srcMemory}, {dstMemory});
}

bool Convert::created() const {
    return getType() == Type::Convert;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
