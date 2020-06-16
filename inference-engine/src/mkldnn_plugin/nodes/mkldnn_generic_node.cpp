// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mkldnn_extension_mngr.h>
#include <mkldnn_extension_utils.h>
#include "mkldnn_generic_node.h"
#include <vector>
#include <string>
#include <blob_factory.hpp>

using namespace mkldnn;
using namespace MKLDNNPlugin;

MKLDNNGenericNode::MKLDNNGenericNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(layer, eng, cache) {
    params = layer->params;
    blobs = layer->blobs;
}

void MKLDNNGenericNode::getSupportedDescriptors() {
    if (!extFactory && impls.empty()) {
        std::string type = getCnnLayer() ? getCnnLayer()->type : "Generic";
        THROW_IE_EXCEPTION << "Cannot get generic primitive for layer: " << getName() << " with type: " << type;
    }
}

void MKLDNNGenericNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::ResponseDesc resp;
    if (impls.empty()) {
        if (!extFactory)
            THROW_IE_EXCEPTION << "Descriptor for generic primitive doesn't exist";

        std::vector<InferenceEngine::ILayerImpl::Ptr> impls_no_exec;

        InferenceEngine::StatusCode rc = extFactory->getImplementations(impls_no_exec, &resp);
        for (const auto& impl : impls_no_exec) {
            if (auto exec_impl = std::dynamic_pointer_cast<InferenceEngine::ILayerExecImpl>(impl)) {
                impls.emplace_back(exec_impl);
            }
        }
        if (rc != InferenceEngine::OK) {
            THROW_IE_EXCEPTION << resp.msg;
        }
    }

    for (auto &impl : impls) {
        std::vector<InferenceEngine::LayerConfig> configs;
        auto rc = impl->getSupportedConfigurations(configs, &resp);
        if (rc != InferenceEngine::OK) {
            THROW_IE_EXCEPTION << resp.msg;
        }

        for (auto& config : configs) {
            supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
        }
    }
    if (impls.empty()) {
        THROW_IE_EXCEPTION << "Layer " << getName() << " hasn't available configurations!";
    }
}

void MKLDNNGenericNode::createPrimitive() {
    if (extFactory || !impls.empty()) {
        return;
    }
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";
}

void MKLDNNGenericNode::execute(mkldnn::stream strm) {
    if (!impls.empty()) {
        execLayer();
    } else {
        THROW_IE_EXCEPTION << "Descriptor for generic primitive doesn't exist";
    }
}

bool MKLDNNGenericNode::created() const {
    return Generic == getType();
}

bool MKLDNNGenericNode::created(const MKLDNNExtensionManager::Ptr &extMgr) {
    if (getCnnLayer() && extMgr) {
        // We should save extension manager in order to avoid situation when
        // it will destroyed before extensibility primitives
        if (getCnnLayer()->getNode()) {
            auto impl = extMgr->CreateImplementation(getCnnLayer()->getNode());
            if (auto execImpl = std::dynamic_pointer_cast<InferenceEngine::ILayerExecImpl>(impl))
                impls.emplace_back(execImpl);
        }
        if (impls.empty()) {
            extFactory = extMgr->CreateExtensionFactory(getCnnLayer());
        }

        if (extFactory || !impls.empty())
            setType(Generic);
    }
    return created();
}

void MKLDNNGenericNode::cleanup() {
    MKLDNNNode::cleanup();
    extFactory.reset();
}

void MKLDNNGenericNode::execLayer() {
    bool isDynBatch = dynBatchLim > 0;
    std::vector<InferenceEngine::Blob::Ptr> inputs;
    std::vector<InferenceEngine::Blob::CPtr> constInputs;
    std::vector<InferenceEngine::TensorDesc> inputDescs;
    std::vector<InferenceEngine::SizeVector> outputShapes;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto inputBlob = getParentEdgeAt(i)->getBlob();
        inputs.push_back(inputBlob);
        constInputs.push_back(inputBlob);
        if (isDynBatch && dynBatchLim >= inputs[inputs.size() - 1]->getTensorDesc().getDims()[0]) {
            isDynBatch = false;
        } else {
            // TODO: Ask the right dims using getShape() from previous node
            inputDescs.push_back(inputs[inputs.size() - 1]->getTensorDesc());
            if (inputDescs[inputDescs.size() - 1].getDims().size() > 0)
                inputDescs[inputDescs.size() - 1].getDims()[0] = static_cast<size_t>(batchToProcess());
        }
    }

    if (isDynBatch) {
        // TODO: use ngraph-based extension mechnism if needed to recompute shape
        isDynBatch = false;
    }

    if (isDynBatch) {
        for (size_t i = 0; i < inputs.size(); i++) {
            auto td = inputs[i]->getTensorDesc();
            td.setDims(inputDescs[i].getDims());
            inputs[i] = make_blob_with_precision(td, getParentEdgeAt(i)->getMemory().GetData());
        }
    }
    std::vector<InferenceEngine::Blob::Ptr> outputs;
    for (size_t i = 0; i < outDims.size(); i++) {
        if (isDynBatch) {
            auto out_edge = getChildEdgesAtPort(i)[0];
            auto td = out_edge->getBlob()->getTensorDesc();
            td.setDims(outputShapes[i]);
            outputs.push_back(make_blob_with_precision(td, out_edge->getMemory().GetData()));
        } else {
            outputs.push_back(getChildEdgesAtPort(i)[0]->getBlob());
        }
    }
    InferenceEngine::ResponseDesc resp;
    InferenceEngine::StatusCode rc = impls[0]->execute(inputs, outputs, &resp);
    if (rc != InferenceEngine::OK) {
        THROW_IE_EXCEPTION << resp.msg;
    }
}

InferenceEngine::LayerConfig convert_layout_config(const MKLDNNLayoutConfig& config) {
    InferenceEngine::LayerConfig res;

    res.dynBatchSupport = config.dynBatchSupport;

    auto convert_port_info = [] (const MKLDNNPortConfig& port_desc) {
        InferenceEngine::DataConfig res;
        res.inPlace  = port_desc.inPlace;
        res.constant = port_desc.constant;
        res.desc     = port_desc.desc;
        return res;
    };

    for (auto info : config.inConfs)
        res.inConfs.push_back(convert_port_info(info));

    for (auto info : config.outConfs)
        res.inConfs.push_back(convert_port_info(info));

    return res;
}

void MKLDNNGenericNode::initDescriptor(const MKLDNNLayoutConfig& config) {
    auto rightConfig = config;
    InferenceEngine::StatusCode rc;
    InferenceEngine::ResponseDesc resp;

    InferenceEngine::ILayerExecImpl::Ptr selectedImpl;
    for (size_t k = 0, t = 0; k < impls.size(); k++) {
        std::vector<InferenceEngine::LayerConfig> configs;
        rc = impls[k]->getSupportedConfigurations(configs, &resp);
        if (rc != InferenceEngine::OK) {
            THROW_IE_EXCEPTION << resp.msg;
        }
        for (size_t j = 0; j < configs.size(); j++, t++) {
            if (t == selectedPrimitiveDescriptorIndex) {
                selectedImpl = impls[k];
            }
        }
    }

    for (size_t j = 0; j < rightConfig.inConfs.size(); j++) {
        // TODO: we need to better recognize cases with possible inplace conficts
        if (getParentEdgeAt(j)->getParent()->getType() != Split &&
            getParentEdgeAt(j)->getParent()->getChildEdges().size() > 1) {
            rightConfig.inConfs[j].inPlace = -1;
        }
    }
    for (auto &outConf : rightConfig.outConfs) {
        if (outConf.inPlace < getParentEdges().size() &&
            getParentEdgeAt(static_cast<size_t>(outConf.inPlace))->getParent()->getChildEdges().size() > 1) {
            outConf.inPlace = -1;
        }
    }


    impls.clear();
    impls.emplace_back(selectedImpl);
    auto ie_config = convert_layout_config(rightConfig);
    rc = impls[0]->init(ie_config, &resp);
    if (rc != InferenceEngine::OK) {
        THROW_IE_EXCEPTION << resp.msg;
    }

    auto descriptor = getSelectedPrimitiveDescriptor();
    if (descriptor != nullptr) {
        descriptor->getConfig() = rightConfig;
    }
    bool isConst = !rightConfig.inConfs.empty() || !rightConfig.outConfs.empty();
    for (const auto &inConf : rightConfig.inConfs) {
        isConst = isConst && inConf.constant;
    }
    for (const auto &outConf : rightConfig.outConfs) {
        isConst = isConst && outConf.constant;
    }
    if (isConst) {
        constant = ConstantType::Const;
    }
}

REG_MKLDNN_PRIM_FOR(MKLDNNGenericNode, Generic);
