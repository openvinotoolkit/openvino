// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_softmax_node.h"

#include <string>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNSoftMaxNode::MKLDNNSoftMaxNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(op, eng, cache) {
    const auto softmaxOp = ngraph::as_type_ptr<ngraph::op::v1::Softmax>(op);
    if (softmaxOp) {
        axis = softmaxOp->get_axis();
    } else {
        IE_THROW(NotImplemented)
                << "CPU Softmax node doesn't support ngraph operation " << op->get_type_name() << " with name " << op->get_friendly_name();
    }
}

void MKLDNNSoftMaxNode::getSupportedDescriptors() {
    if (descs.size())
        return;

    InferenceEngine::Precision precision = getOriginalInputPrecisionAtPort(0);
    if (precision != InferenceEngine::Precision::FP32 && precision != InferenceEngine::Precision::BF16)
        precision = InferenceEngine::Precision::FP32;
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    if (getParentEdges().size() != 1)
        IE_THROW() << "Incorrect number of input edges for layer " << getName();
    if (!getChildEdges().size())
        IE_THROW() << "Incorrect number of output edges for layer " << getName();

    if (getParentEdgeAt(0)->getDims().ndims() == 3) {
        MKLDNNMemoryDesc in_candidate(getParentEdgeAt(0)->getDims(), inputDataType, memory::format_tag::abc);
        createDescriptor({in_candidate}, {});
    }

    for (auto format : getAvailableFormatsForDims(getParentEdgeAt(0)->getDims())) {
        MKLDNNDims dims = getParentEdgeAt(0)->getDims();
        if (MKLDNNMemoryDesc(dims, inputDataType, format).blocksExtended())
            continue;

        MKLDNNMemoryDesc in_candidate(dims, inputDataType, format);

        createDescriptor({in_candidate}, {});
    }
}

void MKLDNNSoftMaxNode::createPrimitive() {
    if (prim)
        return;

    memory::desc in_candidate = getParentEdgeAt(0)->getMemory().GetDescriptor();
    MKLDNNDescriptor desc(std::shared_ptr<softmax_forward::desc>(
            new softmax_forward::desc(prop_kind::forward_scoring, in_candidate, axis)));
    descs[0] = desc;
    std::shared_ptr<softmax_forward::desc> selected_desc_ptr = descs[0];

    const PrimitiveDescInfo *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set for node " << getName() << ".";

    auto prim_desc = softmax_forward::primitive_desc(*selected_desc_ptr, getEngine());
    primitive_desc_iterator itpd = descs[0].createPrimitiveDescriptorIterator(getEngine());

    while (itpd) {
        impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());
        auto primitiveDescriptor = getSelectedPrimitiveDescriptor();
        if ((primitiveDescriptor != nullptr) && (impl_type == primitiveDescriptor->getImplementationType())) {
            prim_desc = itpd.get();
            break;
        }
        if (!itpd.next_impl())
            break;
    }

    prim.reset(new softmax_forward(prim_desc));

    auto src = getParentEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
    auto dst = getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
    primArgs = {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst}};
}

bool MKLDNNSoftMaxNode::created() const {
    return getType() == Softmax;
}

void MKLDNNSoftMaxNode::initOptimalPrimitiveDescriptor() {
    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set.";
    auto config = selected_pd->getConfig();
    if (isInitConfig(config))
        return;

    if (config.inConfs.size() != 1 || config.outConfs.size() != 1 ||
            (!isUninitTensorDesc(config.inConfs[0].desc) &&
                    !isUninitTensorDesc(config.outConfs[0].desc) && config.inConfs[0].desc != config.outConfs[0].desc))
        IE_THROW() << "Layer " << getName() << " has incorrect selected config!";

    if (!isUninitTensorDesc(config.inConfs[0].desc)) {
        config.outConfs[0].desc = config.inConfs[0].desc;
    } else if (!isUninitTensorDesc(config.outConfs[0].desc)) {
        config.inConfs[0].desc = config.outConfs[0].desc;
    } else {
        config.outConfs[0].desc = config.inConfs[0].desc = getConfiguredInputDesc(config, 0);
    }

    initDescriptor(config);
}

void MKLDNNSoftMaxNode::createDescriptor(const std::vector<InferenceEngine::TensorDesc> &inputDesc,
                                         const std::vector<InferenceEngine::TensorDesc> &outputDesc) {
    MKLDNNMemoryDesc in_candidate(inputDesc[0]);

    MKLDNNDescriptor desc(std::shared_ptr<softmax_forward::desc>(
            new softmax_forward::desc(prop_kind::forward_scoring, in_candidate, axis)));
    descs.push_back(desc);
}
REG_MKLDNN_PRIM_FOR(MKLDNNSoftMaxNode, Softmax);
