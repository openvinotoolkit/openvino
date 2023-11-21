// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <extension_mngr.h>
#include <dnnl_extension_utils.h>
#include "generic.h"
#include <vector>
#include <string>
#include <blob_factory.hpp>
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"

using namespace dnnl;

namespace ov {
namespace intel_cpu {
namespace node {

namespace {
/**
 * Dummy Shape Inference while Generic op doesn't support Dynamism
 *
 */
class GenericShapeInfer : public ShapeInferEmptyPads {
public:
    GenericShapeInfer() = default;
    Result infer(
        const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
        const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        OPENVINO_THROW("Unexpected: Generic operations doesn't support shape inference.");
        return {{}, ShapeInferStatus::skip};
    }

    port_mask_t get_port_mask() const override { return EMPTY_PORT_MASK; }
};

class GenericShapeInferFactory : public ShapeInferFactory {
public:
    ShapeInferPtr makeShapeInfer() const override {
        return std::make_shared<GenericShapeInfer>();
    }
};
} // namespace

Generic::Generic(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, GenericShapeInferFactory()), ngraphOp(op) {
}

void Generic::getSupportedDescriptors() {
    if (impls.empty()) {
        OPENVINO_THROW("Cannot get generic primitive for layer: ", getName(), " with type: ", getTypeStr());
    }
}

NodeConfig Generic::convertLayerToNodeConfig(const InferenceEngine::LayerConfig &layerConfig) {
    NodeConfig config;
    config.inConfs.resize(layerConfig.inConfs.size());
    for (size_t i = 0; i < layerConfig.inConfs.size(); i++) {
        config.inConfs[i].inPlace(layerConfig.inConfs[i].inPlace);
        config.inConfs[i].constant(layerConfig.inConfs[i].constant);
        config.inConfs[i].setMemDesc(MemoryDescUtils::convertToDnnlBlockedMemoryDesc(layerConfig.inConfs[i].desc).clone());
    }
    config.outConfs.resize(layerConfig.outConfs.size());
    for (size_t i = 0; i < layerConfig.outConfs.size(); i++) {
        config.outConfs[i].inPlace(layerConfig.outConfs[i].inPlace);
        config.outConfs[i].constant(layerConfig.outConfs[i].constant);
        config.outConfs[i].setMemDesc(MemoryDescUtils::convertToDnnlBlockedMemoryDesc(layerConfig.outConfs[i].desc).clone());
    }
    return config;
}

InferenceEngine::LayerConfig Generic::convertNodeToLayerConfig(const NodeConfig &nodeConfig) {
    InferenceEngine::LayerConfig config;
    config.inConfs.resize(nodeConfig.inConfs.size());
    for (size_t i = 0; i < nodeConfig.inConfs.size(); i++) {
        config.inConfs[i].inPlace = nodeConfig.inConfs[i].inPlace();
        config.inConfs[i].constant = nodeConfig.inConfs[i].constant();
        config.inConfs[i].desc = MemoryDescUtils::convertToTensorDesc(*nodeConfig.inConfs[i].getMemDesc());
    }
    config.outConfs.resize(nodeConfig.outConfs.size());
    for (size_t i = 0; i < nodeConfig.outConfs.size(); i++) {
        config.outConfs[i].inPlace = nodeConfig.outConfs[i].inPlace();
        config.outConfs[i].constant = nodeConfig.outConfs[i].constant();
        config.outConfs[i].desc = MemoryDescUtils::convertToTensorDesc(*nodeConfig.outConfs[i].getMemDesc());
    }
    return config;
}

void Generic::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::ResponseDesc resp;
    for (auto &impl : impls) {
        std::vector<InferenceEngine::LayerConfig> configs;
        auto rc = impl->getSupportedConfigurations(configs, &resp);
        if (rc != InferenceEngine::OK) {
            OPENVINO_THROW(resp.msg);
        }

        for (auto& config : configs) {
            supportedPrimitiveDescriptors.emplace_back(convertLayerToNodeConfig(config), impl_desc_type::unknown);
        }
    }
    if (impls.empty()) {
        OPENVINO_THROW("Layer ", getName(), " hasn't available configurations!");
    }
}

void Generic::createPrimitive() {
}

void Generic::execute(dnnl::stream strm) {
    if (!impls.empty()) {
        execLayer();
    } else {
        OPENVINO_THROW("Descriptor for generic primitive doesn't exist");
    }
}

bool Generic::created() const {
    return Type::Generic == getType();
}

bool Generic::created(const ExtensionManager::Ptr &extMgr) {
    if (ngraphOp && extMgr) {
        // We should save extension manager in order to avoid situation when
        // it will destroyed before extensibility primitives
        auto impl = extMgr->CreateImplementation(ngraphOp);
        if (auto execImpl = std::dynamic_pointer_cast<InferenceEngine::ILayerExecImpl>(impl))
            impls.emplace_back(execImpl);

        if (impls.empty())
            return false;

        setType(Type::Generic);
    }
    return created();
}

void Generic::cleanup() {
    Node::cleanup();
}

void Generic::execLayer() {
    std::vector<InferenceEngine::Blob::Ptr> inputs;
    std::vector<InferenceEngine::Blob::CPtr> constInputs;
    std::vector<InferenceEngine::TensorDesc> inputDescs;
    std::vector<InferenceEngine::SizeVector> execOutputShapes;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto inputBlob = MemoryDescUtils::interpretAsBlob(getParentEdgeAt(i)->getMemory());
        inputs.push_back(inputBlob);
        constInputs.push_back(inputBlob);
        // TODO: Ask the right dims using getShape() from previous node
        inputDescs.push_back(inputs[inputs.size() - 1]->getTensorDesc());
    }

    std::vector<InferenceEngine::Blob::Ptr> outputs;
    for (size_t i = 0; i < outputShapes.size(); i++) {
        outputs.push_back(MemoryDescUtils::interpretAsBlob(getChildEdgesAtPort(i)[0]->getMemory()));
    }
    InferenceEngine::ResponseDesc resp;
    InferenceEngine::StatusCode rc = impls[0]->execute(inputs, outputs, &resp);
    if (rc != InferenceEngine::OK) {
        OPENVINO_THROW(this->getTypeStr(), ":", this->getName(), ": ", resp.msg);
    }
}

void Generic::initDescriptor(const NodeConfig &config) {
    NodeConfig rightConfig = config;
    InferenceEngine::StatusCode rc;
    InferenceEngine::ResponseDesc resp;

    InferenceEngine::ILayerExecImpl::Ptr selectedImpl;
    for (size_t k = 0, t = 0; k < impls.size(); k++) {
        std::vector<InferenceEngine::LayerConfig> configs;
        rc = impls[k]->getSupportedConfigurations(configs, &resp);
        if (rc != InferenceEngine::OK) {
            OPENVINO_THROW(resp.msg);
        }
        for (size_t j = 0; j < configs.size(); j++, t++) {
            if (t == static_cast<size_t>(selectedPrimitiveDescriptorIndex)) {
                selectedImpl = impls[k];
            }
        }
    }

    for (size_t j = 0; j < rightConfig.inConfs.size(); j++) {
        // TODO: we need to better recognize cases with possible inplace conficts
        if (getParentEdgeAt(j)->getParent()->getType() != Type::Split &&
            getParentEdgeAt(j)->getParent()->getChildEdges().size() > 1) {
            rightConfig.inConfs[j].inPlace(-1);
        }
    }
    for (auto &outConf : rightConfig.outConfs) {
        if (outConf.inPlace() < static_cast<int>(getParentEdges().size()) &&
            getParentEdgeAt(static_cast<size_t>(outConf.inPlace()))->getParent()->getChildEdges().size() > 1) {
            outConf.inPlace(-1);
        }
    }


    impls.clear();
    impls.emplace_back(selectedImpl);
    auto ieConfig = convertNodeToLayerConfig(rightConfig);
    rc = impls[0]->init(ieConfig, &resp);
    if (rc != InferenceEngine::OK) {
        OPENVINO_THROW(resp.msg);
    }
    rightConfig = convertLayerToNodeConfig(ieConfig);
    auto descriptor = getSelectedPrimitiveDescriptor();
    if (descriptor != nullptr) {
        descriptor->setConfig(rightConfig);
    }
    bool isConst = !rightConfig.inConfs.empty() || !rightConfig.outConfs.empty();
    for (const auto &inConf : rightConfig.inConfs) {
        isConst = isConst && inConf.constant();
    }
    for (const auto &outConf : rightConfig.outConfs) {
        isConst = isConst && outConf.constant();
    }
    if (isConst) {
        constant = ConstantType::Const;
    }
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
