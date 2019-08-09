// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_activation_node.h"
#include "desc_iterator.hpp"
#include <ie_layers.h>
#include <algorithm>
#include <string>
#include <mkldnn_extension_utils.h>
#include "details/caseless.hpp"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

// TODO: (ichuraev) I don't fully sure that names of types and parameters are correct for square, abs, sqrt, linear, bounded_relu and soft_relu
caseless_map<std::string, std::function<void(GenericLayer*, mkldnn::algorithm&, float&, float&)>> MKLDNNActivationNode::initializers = {
        {"relu", [](GenericLayer* activationLayer, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = activationLayer->GetParamAsFloat("negative_slope", 0.0f);
            beta = 0.0f;
            algorithm = eltwise_relu;
        }},
        {"elu", [](GenericLayer* activationLayer, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = activationLayer->GetParamAsFloat("alpha", 1.0f);
            beta = 0.0f;
            algorithm = eltwise_elu;
        }},
        {"tanh", [](GenericLayer* activationLayer, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = 0.0f;
            beta = 0.0f;
            algorithm = eltwise_tanh;
        }},
        {"logistic", [](GenericLayer* activationLayer, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = 0.0f;
            beta = 0.0f;
            algorithm = eltwise_logistic;
        }},
        {"square", [](GenericLayer* activationLayer, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = 0.0f;
            beta = 0.0f;
            algorithm = eltwise_square;
        }},
        {"abs", [](GenericLayer* activationLayer, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = 0.0f;
            beta = 0.0f;
            algorithm = eltwise_abs;
        }},
        {"sqrt", [](GenericLayer* activationLayer, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = 0.0f;
            beta = 0.0f;
            algorithm = eltwise_sqrt;
        }},
        {"linear", [](GenericLayer* activationLayer, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = activationLayer->GetParamAsFloat("alpha", 1.0f);
            beta = activationLayer->GetParamAsFloat("beta", 0.0f);
            algorithm = eltwise_linear;
        }},
        {"bounded_relu", [](GenericLayer* activationLayer, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = activationLayer->GetParamAsFloat("alpha", 0.0f);
            beta = 0.0f;
            algorithm = eltwise_bounded_relu;
        }},
        {"soft_relu", [](GenericLayer* activationLayer, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = 0.0f;
            beta = 0.0f;
            algorithm = eltwise_soft_relu;
        }},
        {"relu6", [](GenericLayer* activationLayer, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = activationLayer->GetParamAsFloat("n", 6.0f);
            beta = 0.0f;
            algorithm = eltwise_bounded_relu;
        }},
        {"clamp", [](GenericLayer* activationLayer, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = activationLayer->GetParamAsFloat("max", 1.0f);
            beta = activationLayer->GetParamAsFloat("min", 0.0f);
            algorithm = eltwise_clamp;
        }},
        {"exp", [](GenericLayer* activationLayer, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = 0.0f;
            beta = 0.0f;
            algorithm = eltwise_exp;
        }},
        {"not", [](GenericLayer* activationLayer, mkldnn::algorithm& algorithm, float& alpha, float& beta) {
            alpha = 0.0f;
            beta = 0.0f;
            algorithm = eltwise_not;
        }}
};

MKLDNNActivationNode::MKLDNNActivationNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, int socket) : MKLDNNNode(layer, eng, socket) {}

void MKLDNNActivationNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (!getChildEdges().size())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();

    auto parentOutDims = getParentEdgeAt(0)->getDims();

    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();

    // FIXME: MKLDNN doesn't support not inputs with number of dimensions less than 4 for activation
    while (parentOutDims.ndims() < 4)
        parentOutDims.push_back(1);
    for (auto format : getAvailableFormatsForDims(parentOutDims)) {
        MKLDNNMemoryDesc in_candidate(parentOutDims, MKLDNNExtensionUtils::IEPrecisionToDataType(precision), format);
        createDescriptor({in_candidate}, {});
    }
}

void MKLDNNActivationNode::createPrimitive() {
    if (prim)
        return;

    auto prim_desc = createPrimitiveDescriptor<eltwise_forward::primitive_desc, eltwise_forward::desc>();

    prim.reset(new eltwise_forward(prim_desc, getParentEdgeAt(0)->getMemory().GetPrimitive(),
                                getChildEdgeAt(0)->getMemory().GetPrimitive()));
}

bool MKLDNNActivationNode::created() const {
    return getType() == Activation;
}

void MKLDNNActivationNode::initValues() {
    GenericLayer* activationLayer = getCnnLayer().get();
    if (activationLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get CNNLayer.";

    std::string type = activationLayer->type;
    CaselessEq<std::string> comparator;
    if (comparator(type, "activation"))
        type = activationLayer->GetParamAsString("type");
    if (comparator(type, "sigmoid"))
        type = "logistic";

    if (initializers.find(type) == initializers.end())
        THROW_IE_EXCEPTION << "Node " << getName() << " has unsupported activation primitive: "
                           << activationLayer->type << " : " << type;
    initializers[type](activationLayer, algorithm, alpha, beta);
    initialized = true;
}

void MKLDNNActivationNode::createDescriptor(const std::vector<InferenceEngine::TensorDesc> &inputDesc,
                                            const std::vector<InferenceEngine::TensorDesc> &outputDesc) {
    MKLDNNMemoryDesc inDesc(inputDesc[0]);
    MKLDNNDescriptor desc(std::shared_ptr<eltwise_forward::desc>(
            new eltwise_forward::desc(prop_kind::forward_scoring, getAlgorithm(), inDesc, getAlpha(), getBeta())));
    descs.push_back(desc);
}

void MKLDNNActivationNode::initOptimalPrimitiveDescriptor() {
    auto config = getSelectedPrimitiveDescriptor()->getConfig();
    if (isInitConfig(config))
        return;

    if (config.inConfs.size() != 1 || config.outConfs.size() != 1 ||
            (!isUninitTensorDesc(config.inConfs[0].desc) &&
                    !isUninitTensorDesc(config.outConfs[0].desc) && config.inConfs[0].desc != config.outConfs[0].desc))
        THROW_IE_EXCEPTION << "Layer " << getName() << " has incorrect selected config!";

    if (!isUninitTensorDesc(config.inConfs[0].desc)) {
        config.outConfs[0].desc = config.inConfs[0].desc;
    } else if (!isUninitTensorDesc(config.outConfs[0].desc)) {
        config.inConfs[0].desc = config.outConfs[0].desc;
    } else {
        config.outConfs[0].desc = config.inConfs[0].desc = getConfiguredInputDesc(config, 0);
    }

    initDescriptor(config);
}

MKLDNNMemoryDesc MKLDNNActivationNode::getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    InferenceEngine::TensorDesc desc = MKLDNNMemoryDesc(primitive_desc_it.src_primitive_desc(idx).desc());

    auto parentOutDims = getParentEdgeAt(idx)->getDims().ToSizeVector();

    SizeVector blocked_dims, order, dimOffsets, strides;
    size_t offset = desc.getBlockingDesc().getOffsetPadding();

    for (size_t i = 0; i < desc.getBlockingDesc().getStrides().size(); i++) {
        if (desc.getBlockingDesc().getOrder()[i] >= parentOutDims.size())
            continue;

        blocked_dims.push_back(desc.getBlockingDesc().getBlockDims()[i]);
        order.push_back(desc.getBlockingDesc().getOrder()[i]);
        dimOffsets.push_back(desc.getBlockingDesc().getOffsetPaddingToData()[i]);
        strides.push_back(desc.getBlockingDesc().getStrides()[i]);
    }
    if (desc.getLayout() == InferenceEngine::Layout::ANY)
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            parentOutDims,
                                                            desc.getLayout()));
    else
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            parentOutDims,
                                                            {blocked_dims, order, offset, dimOffsets, strides}));
}

MKLDNNMemoryDesc MKLDNNActivationNode::getDstMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    InferenceEngine::TensorDesc desc = MKLDNNMemoryDesc(primitive_desc_it.dst_primitive_desc(idx).desc());

    auto childInDims = getChildEdgeAt(idx)->getDims().ToSizeVector();

    SizeVector blocked_dims, order, dimOffsets, strides;
    size_t offset = desc.getBlockingDesc().getOffsetPadding();

    for (size_t i = 0; i < desc.getBlockingDesc().getStrides().size(); i++) {
        if (desc.getBlockingDesc().getOrder()[i] >= childInDims.size())
            continue;

        blocked_dims.push_back(desc.getBlockingDesc().getBlockDims()[i]);
        order.push_back(desc.getBlockingDesc().getOrder()[i]);
        dimOffsets.push_back(desc.getBlockingDesc().getOffsetPaddingToData()[i]);
        strides.push_back(desc.getBlockingDesc().getStrides()[i]);
    }
    if (desc.getLayout() == InferenceEngine::Layout::ANY)
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            childInDims,
                                                            desc.getLayout()));
    else
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            childInDims,
                                                            {blocked_dims, order, offset, dimOffsets, strides}));
}
