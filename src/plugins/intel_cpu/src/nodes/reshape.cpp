// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape.h"
#include "utils.hpp"
#include <string>
#include <dnnl_types.h>
#include <dnnl_extension_utils.h>
#include <openvino/opsets/opset1.hpp>
#include <ie_ngraph_utils.hpp>
#include <utils/shape_inference/static_shape.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include "utils/shape_inference/shape_inference_cpu.hpp"

#include "common/cpu_memcpy.h"

using namespace dnnl;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

bool Reshape::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!std::dynamic_pointer_cast<const ov::opset1::Reshape>(op) &&
            !std::dynamic_pointer_cast<const ov::opset1::Squeeze>(op) &&
                !std::dynamic_pointer_cast<const ov::opset1::Unsqueeze>(op)) {
            errorMessage = "Only opset1 Reshape, Squeeze, Unsqueeze operations are supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

namespace {
class ReshapeShapeInfer : public ShapeInferEmptyPads {
public:
    ReshapeShapeInfer(bool specialZero) : m_specialZero(specialZero) {}
    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        static constexpr size_t RESHAPE_SRC = 0, RESHAPE_PATTERN = 1;
        const auto& inputShape = input_shapes[RESHAPE_SRC].get();
        const size_t inputShapeSize = inputShape.size();
        const auto memPtr = data_dependency.at(RESHAPE_PATTERN);
        const auto data = memPtr->getData();
        const auto& dims = memPtr->getStaticDims();
        const auto outputPatternSize = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<Dim>());
        std::vector<int64_t> outPattern = ov::get_raw_data_as<int64_t>(
                                              InferenceEngine::details::convertPrecision(memPtr->getDesc().getPrecision()),
                                              data,
                                              outputPatternSize,
                                              ov::util::Cast<int64_t>());
        VectorDims outputShape(outputPatternSize);
        size_t outputProduct = 1;
        int32_t minusOneIdx = -1;
        int32_t minusOneCount = 0;
        for (int32_t i = 0; i < outputPatternSize; ++i) {
            if (outPattern[i] == 0 && m_specialZero && i < static_cast<int32_t>(inputShapeSize)) {
                outputShape[i] = inputShape[i];
            } else if (outPattern[i] == -1) {
                minusOneIdx = i;
                minusOneCount++;
            } else {
                outputShape[i] = outPattern[i];
                outputProduct *= outputShape[i];
            }
        }
        size_t inputProduct = 1;
        for (size_t i = 0; i < inputShapeSize; ++i) {
            if (static_cast<int>(i) < outputPatternSize && outPattern[i] == 0 && m_specialZero)
                continue;
            inputProduct *= inputShape[i];
        }
        if (minusOneIdx >= 0) {
            if (outputProduct != 0) {
                outputShape[minusOneIdx] = inputProduct / outputProduct;
                outputProduct *= outputShape[minusOneIdx];
            } else {
                outputShape[minusOneIdx] = 0;
            }
        }
        if (minusOneCount > 1  || inputProduct != outputProduct) {
            IE_THROW(Unexpected) << "[cpu]reshape: the shape of input data conflicts with the reshape pattern";
        }
        return {{std::move(outputShape)}, ShapeInferStatus::success};
    }
    port_mask_t get_port_mask() const override {
        return PortMask(1);
    }

private:
    bool m_specialZero;
};

class SqueezeShapeInfer : public ShapeInferEmptyPads {
public:
    SqueezeShapeInfer() {}
    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        static constexpr size_t SQUEEZE_SRC = 0, SQUEEZE_PATTERN = 1;
        const auto& inputShape = input_shapes[SQUEEZE_SRC].get();
        const size_t inputShapeSize = inputShape.size();
        auto itr = data_dependency.find(SQUEEZE_PATTERN);
        VectorDims outputShape;
        outputShape.reserve(inputShapeSize);
        if (itr != data_dependency.end()) {
            const auto memPtr = data_dependency.at(SQUEEZE_PATTERN);
            const auto data = memPtr->getData();
            const auto& dims = memPtr->getStaticDims();
            const auto outputPatternSize = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<Dim>());
            std::vector<int64_t> outPattern = ov::get_raw_data_as<int64_t>(
                                                  InferenceEngine::details::convertPrecision(memPtr->getDesc().getPrecision()),
                                                  data,
                                                  outputPatternSize,
                                                  ov::util::Cast<int64_t>());
            std::vector<bool> removeMask(inputShapeSize, false);
            bool existError = false;
            for (int i = 0; i < outputPatternSize; i++) {
                if (outPattern[i] < 0) {
                    outPattern[i] = inputShapeSize + outPattern[i];
                }
                if (outPattern[i] >= 0 && outPattern[i] < static_cast<int64_t>(inputShapeSize)) {
                    removeMask[outPattern[i]] = true;
                } else {
                    existError = true;
                    break;
                }
            }
            for (size_t i = 0; i < inputShapeSize; i++) {
                if (!removeMask[i]) {
                    outputShape.push_back(inputShape[i]);
                } else if (inputShape[i] != 1) {
                    existError = true;
                    break;
                }
            }
            if (existError) {
                IE_THROW(Unexpected) << "[cpu]squeeze: the shape of input data conflict with the squeeze pattern";
            }
        } else {
            for (size_t i = 0; i < inputShapeSize; i++) {
                if (inputShape[i] != 1) {
                    outputShape.push_back(inputShape[i]);
                }
            }
        }
        return {{std::move(outputShape)}, ShapeInferStatus::success};
    }
    port_mask_t get_port_mask() const override {
        return PortMask(1);
    }
};

class UnsqueezeShapeInfer : public ShapeInferEmptyPads {
public:
    UnsqueezeShapeInfer() {}
    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        static constexpr size_t UNSQUEEZE_SRC = 0, UNSQUEEZE_PATTERN = 1;
        const auto& inputShape = input_shapes[UNSQUEEZE_SRC].get();
        const size_t inputShapeSize = inputShape.size();
        const auto memPtr = data_dependency.at(UNSQUEEZE_PATTERN);
        const auto data = memPtr->getData();
        const auto& dims = memPtr->getStaticDims();
        const auto outputPatternSize = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<Dim>());
        std::vector<int64_t> outPattern = ov::get_raw_data_as<int64_t>(
                                              InferenceEngine::details::convertPrecision(memPtr->getDesc().getPrecision()),
                                              data,
                                              outputPatternSize,
                                              ov::util::Cast<int64_t>());
        size_t outputShapeSize = inputShapeSize + outputPatternSize;
        VectorDims outputShape(outputShapeSize, 0);
        bool existError = false;
        for (int i = 0; i < outputPatternSize; i++) {
            if (outPattern[i] < 0) {
                outPattern[i] = outputShapeSize + outPattern[i];
            }
            if (outPattern[i] >= 0 && outPattern[i] < static_cast<int64_t>(outputShapeSize)) {
                outputShape[outPattern[i]] = 1;
            } else {
                existError = true;
                break;
            }
        }
        for (size_t i = 0, y = 0; i < outputShapeSize; i++) {
            if (outputShape[i] == 0) {
                if (y < inputShapeSize) {
                    outputShape[i] = inputShape[y];
                    y++;
                } else {
                    existError = true;
                    break;
                }
            }
        }
        if (existError) {
            IE_THROW(Unexpected) << "[cpu]unsqueeze: the shape of input data conflicts with the unsqueeze pattern";
        }
        return {{std::move(outputShape)}, ShapeInferStatus::success};
    }
    port_mask_t get_port_mask() const override {
        return PortMask(1);
    }
};

class ReshapeShapeInferFactory : public ShapeInferFactory {
public:
    ReshapeShapeInferFactory(std::shared_ptr<ov::Node> op) : m_op(op) {}
    ShapeInferPtr makeShapeInfer() const override {
        if (const auto reshapeOp = ov::as_type_ptr<const ov::op::v1::Reshape>(m_op)) {
            return std::make_shared<ReshapeShapeInfer>(reshapeOp->get_special_zero());
        } else if (ov::is_type<ov::op::v0::Squeeze>(m_op)) {
            return std::make_shared<SqueezeShapeInfer>();
        } else if (ov::is_type<ov::op::v0::Unsqueeze>(m_op)) {
            return std::make_shared<UnsqueezeShapeInfer>();
        } else {
            IE_THROW(Unexpected) << "[cpu]reshape: " << m_op->get_type_name() << "is not implemented";
        }
    }
private:
    std::shared_ptr<ov::Node> m_op;
};
} // namespace

Reshape::Reshape(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context) :
        Node(op, context, ReshapeShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = std::string(op->get_type_name()) + " node with name '" + getName() + "'";

    if (isDynamicNode()) {
        auto checkSecondInput = [](const std::shared_ptr<ngraph::Node>& op, const std::string opType) {
            if (op->get_input_partial_shape(1).is_dynamic()) {
                IE_THROW() << "CPU plug-in doesn't support " << opType << " node with non static second input";
            }
        };

        if (std::dynamic_pointer_cast<const ov::opset1::Reshape>(op)) {
            checkSecondInput(op, "Reshape");
        } else if (std::dynamic_pointer_cast<const ov::opset1::Squeeze>(op)) {
            if (op->get_input_size() == 1)
                IE_THROW() << "CPU plug-in doesn't support Squeeze node with inputs num equal 1";
            checkSecondInput(op, "Squeeze");
        } else if (std::dynamic_pointer_cast<const ov::opset1::Unsqueeze>(op)) {
            checkSecondInput(op, "Unsqueeze");
        } else {
            IE_THROW() << "Unsupported operation type via reshape node";
        }
    }
}

bool Reshape::needShapeInfer() const {
    if (inputShapesModified()) {
        return true;
    }
    const auto& mem = getParentEdgesAtPort(1)[0]->getMemory();
    if (lastSecondInputValues.empty()) {
        lastSecondInputValues.resize(mem.getStaticDims()[0], 0);
    }
    const int32_t *sndInput = reinterpret_cast<const int32_t *>(mem.getData());
    for (size_t i = 0; i < lastSecondInputValues.size(); i++) {
        if (lastSecondInputValues[i] != sndInput[i]) {
            for (size_t i = 0; i < lastSecondInputValues.size(); i++) {
                lastSecondInputValues[i] = sndInput[i];
            }
            return true;
        }
    }
    return false;
}

void Reshape::getSupportedDescriptors() {
    if (getParentEdges().size() != 1 && getParentEdges().size() != 2)
        IE_THROW() << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        IE_THROW() << "Incorrect number of output edges for layer " << getName();
}

void Reshape::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision inPrec = getOriginalInputPrecisionAtPort(0);
    InferenceEngine::Precision outPrec = getOriginalOutputPrecisionAtPort(0);
    InferenceEngine::Precision secondInPrc = InferenceEngine::Precision::I32;

    // Current reshape implementation is simple memory reinterpret,
    // same precision on input and output is required
    if (inPrec != outPrec)
        inPrec = outPrec;

    bool canBeInPlace = true;

    // CVS-81059 : disable inPlace in following case since it won't be satisfied by framework
    if (!isConstant() && getParentEdgeAt(0)->getParent()->isConstant())
        canBeInPlace = false;

    NodeConfig config;
    config.inConfs.resize(getParentEdges().size());
    auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        config.inConfs[i].inPlace(0 == i && canBeInPlace ? 0 : -1);
        config.inConfs[i].constant(false);
        config.inConfs[i].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc((i > 0 ? secondInPrc : inPrec), getInputShapeAtPort(i)));
    }
    config.outConfs.resize(1);
    config.outConfs[0].inPlace(canBeInPlace ? 0 : -1);
    config.outConfs[0].constant(false);
    config.outConfs[0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(outPrec, getOutputShapeAtPort(0)));
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
}

void Reshape::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

void Reshape::execute(dnnl::stream strm) {
    auto srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    auto dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();

    auto srcPtr = static_cast<uint8_t*>(srcMemPtr->getData());
    auto dstPtr = static_cast<uint8_t*>(dstMemPtr->getData());

    if (dstPtr != srcPtr) {
        cpu_memcpy(dstPtr, srcPtr, dstMemPtr->getSize());
    }
}

bool Reshape::isExecutable() const {
    bool inPlaceEnabled = false;
    if (auto prim_desc = getSelectedPrimitiveDescriptor()) {
        auto& config = prim_desc->getConfig();
        if (config.inConfs[0].inPlace() >= 0 ||
            config.outConfs[0].inPlace() >= 0) {
            inPlaceEnabled = true;
        }
    }
    return !inPlaceEnabled;
}

bool Reshape::created() const {
    return getType() == Type::Reshape;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
