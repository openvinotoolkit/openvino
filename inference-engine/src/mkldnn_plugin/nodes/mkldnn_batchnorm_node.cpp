// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_batchnorm_node.h"
#include "mkldnn_depthwise_node.h"
#include <mkldnn_extension_utils.h>
#include "ie_memcpy.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNBatchNormalizationNode::MKLDNNBatchNormalizationNode(const InferenceEngine::CNNLayerPtr& layer,
                                                           const mkldnn::engine& eng, int socket)
        : MKLDNNNode(layer, eng, socket) {
    internalBlobDesc.emplace_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return GetVarianceDesc(primitive_desc_it.fetch());
    });
    internalBlobDesc.emplace_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return GetMeanDesc(primitive_desc_it.fetch());
    });

    internalBlobDesc.emplace_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        if (!fusedWithScale())
            return MKLDNNMemoryDesc();
        return GetScaleShiftWeightsDesc(primitive_desc_it.fetch());
    });
}

void MKLDNNBatchNormalizationNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;
    auto * bnLayer = dynamic_cast<BatchNormalizationLayer*>(getCnnLayer().get());
    if (bnLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert batch normalization layer.";
    if (bnLayer->_weights == nullptr || bnLayer->_biases == nullptr) {
        THROW_IE_EXCEPTION << "Weights/biases are empty for layer: " << bnLayer->name
                           << " used in MKLDNN node: " << getName() << "\n"
                           << "Use the second argumemt of InferenceEngine::Core::ReadNetwork"
                           << " to load them from .bin part of the IR";
    }

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (!getChildEdges().size())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();

    eps = bnLayer->epsilon;

    size_t variancesSize = MKLDNNDims(bnLayer->_weights->getTensorDesc().getDims()).size();
    size_t meansSize = MKLDNNDims(bnLayer->_biases->getTensorDesc().getDims()).size();

    if (variancesSize != meansSize && variancesSize != 1)
        THROW_IE_EXCEPTION << "Incorrect weights and biases sizes!";

    internalBlobs.push_back(createInternalBlob(bnLayer->_weights->getTensorDesc().getDims(), true));
    internalBlobs.push_back(createInternalBlob(bnLayer->_biases->getTensorDesc().getDims(), false));

    auto parentOutDims = getParentEdgeAt(0)->getDims();

    if (fusedWith.size() > 1)
        THROW_IE_EXCEPTION << "BatchNorm fusion is possible with only one layer!";

    for (const auto &node : fusedWith) {
        auto * scshLayer = dynamic_cast<ScaleShiftLayer*>(node->getCnnLayer().get());
        if (scshLayer == nullptr)
            THROW_IE_EXCEPTION << "Cannot cast to the ScaleShift layer to fuse with BatchNorm.";

        size_t C = static_cast<size_t>(getChildEdgeAt(0)->getDims()[1]);
        SizeVector mkldnn_weights = {2, C};
        TensorDesc desc(scshLayer->_weights->getTensorDesc().getPrecision(), mkldnn_weights, InferenceEngine::NC);
        InferenceEngine::TBlob<float>::Ptr internalBlob = InferenceEngine::make_shared_blob<float>(desc);
        internalBlob->allocate();
        float * data = internalBlob->buffer();
        if (data == nullptr)
            THROW_IE_EXCEPTION << "Cannot get memory!";

        InferenceEngine::Blob::Ptr blb = scshLayer->_weights;
        if (blb == nullptr)
            THROW_IE_EXCEPTION << "Cannot get weights blob for node " << getName() << ".";

        size_t weightsByteSize = blb->byteSize();
        ie_memcpy(data, internalBlob->byteSize(), blb->buffer(), weightsByteSize);
        data += blb->size();
        blb = scshLayer->_biases;

        if (blb == nullptr) {
            memset(data, 0, weightsByteSize);
        } else {
            if (weightsByteSize != blb->byteSize())
                THROW_IE_EXCEPTION << "ScaleShift has incorrect weights!";
            ie_memcpy(data, internalBlob->byteSize(), blb->buffer(), weightsByteSize);
        }
        internalBlobs.push_back(internalBlob);
    }

    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    for (auto format : getAvailableFormatsForDims(parentOutDims)) {
        MKLDNNMemoryDesc in_candidate(parentOutDims, inputDataType, format);
        createDescriptor({in_candidate}, {});
    }
}

MKLDNNMemoryDesc MKLDNNBatchNormalizationNode::GetVarianceDesc(const memory::primitive_desc &primitive_desc) const {
    memory::primitive_desc aprimitive_desc;
    mkldnn_primitive_desc_t bndesc;
    mkldnn_batch_normalization_desc_t *p;
    error::wrap_c_api(mkldnn_primitive_desc_query(
            primitive_desc.get(), mkldnn::convert_to_c(batch_normalization_d), 0, &p),
                      "could not get a batch-normalization descriptor");
    const_mkldnn_primitive_desc_t const_bndesc =
            (p->flags & use_global_stats) ?
            mkldnn_primitive_desc_query_pd(primitive_desc.get(),
                                                  mkldnn::convert_to_c(src_pd), 2) :
            mkldnn_primitive_desc_query_pd(primitive_desc.get(),
                                                  mkldnn::convert_to_c(dst_pd), 2);
    error::wrap_c_api(mkldnn_primitive_desc_clone(&bndesc,
                                                         const_bndesc),
                      "could not clone a variance primitive descriptor");
    aprimitive_desc.reset(bndesc);
    return MKLDNNMemoryDesc(aprimitive_desc.desc());
}

MKLDNNMemoryDesc MKLDNNBatchNormalizationNode::GetMeanDesc(const memory::primitive_desc &primitive_desc) const {
    memory::primitive_desc aprimitive_desc;
    mkldnn_primitive_desc_t bndesc;
    mkldnn_batch_normalization_desc_t *p;
    error::wrap_c_api(mkldnn_primitive_desc_query(
            primitive_desc.get(), mkldnn::convert_to_c(batch_normalization_d), 0, &p),
                      "could not get a batch-normalization descriptor");
    const_mkldnn_primitive_desc_t const_bndesc =
            (p->flags & use_global_stats) ?
            mkldnn_primitive_desc_query_pd(primitive_desc.get(),
                                                  mkldnn::convert_to_c(src_pd), 1) :
            mkldnn_primitive_desc_query_pd(primitive_desc.get(),
                                                  mkldnn::convert_to_c(dst_pd), 1);
    error::wrap_c_api(mkldnn_primitive_desc_clone(&bndesc,
                                                         const_bndesc),
                      "could not clone a mean primitive descriptor");
    aprimitive_desc.reset(bndesc);
    return MKLDNNMemoryDesc(aprimitive_desc.desc());
}

MKLDNNMemoryDesc MKLDNNBatchNormalizationNode::GetScaleShiftWeightsDesc(const memory::primitive_desc &primitive_desc) const {
    memory::primitive_desc adesc;
    mkldnn_primitive_desc_t bndesc;
    const_mkldnn_primitive_desc_t const_bndesc =
            mkldnn_primitive_desc_query_pd(primitive_desc.get(),
                                           mkldnn::convert_to_c(weights_pd), 0);
    error::wrap_c_api(mkldnn_primitive_desc_clone(&bndesc,
                                                  const_bndesc),
                      "could not clone a weights primitive descriptor");
    adesc.reset(bndesc);
    return MKLDNNMemoryDesc(adesc.desc());
}

bool MKLDNNBatchNormalizationNode::created() const {
    return getType() == BatchNormalization;
}

void MKLDNNBatchNormalizationNode::createPrimitive() {
    if (prim)
        return;

    if (fusedWithScale()) {
        auto prim_desc = createPrimitiveDescriptor<batch_normalization_forward::primitive_desc,
                batch_normalization_forward::desc>();
        prim.reset(new batch_normalization_forward(prim_desc,
                                                   getParentEdgeAt(0)->getMemory().GetPrimitive(),
                                                   (const primitive::at) internalBlobMemory[1]->GetPrimitive(),
                                                   (const primitive::at) internalBlobMemory[0]->GetPrimitive(),
                                                   (const primitive::at) internalBlobMemory[2]->GetPrimitive(),
                                                   getChildEdgeAt(0)->getMemory().GetPrimitive()));
    }  else {
        auto prim_desc = createPrimitiveDescriptor<batch_normalization_forward::primitive_desc,
                batch_normalization_forward::desc>();
        prim.reset(new batch_normalization_forward(prim_desc,
                                                   getParentEdgeAt(0)->getMemory().GetPrimitive(),
                                                   (const primitive::at) internalBlobMemory[1]->GetPrimitive(),
                                                   (const primitive::at) internalBlobMemory[0]->GetPrimitive(),
                                                   getChildEdgeAt(0)->getMemory().GetPrimitive()));
    }
}

void MKLDNNBatchNormalizationNode::createDescriptor(const std::vector<InferenceEngine::TensorDesc> &inputDesc,
                                                    const std::vector<InferenceEngine::TensorDesc> &outputDesc) {
    MKLDNNMemoryDesc inDesc(inputDesc[0]);
    if (inDesc.getDims().ndims() == 2) {
        // Make it 4D
        MKLDNNDims dims = inDesc.getDims();
        dims.push_back(1);  // H
        dims.push_back(1);  // W
        auto format = memory::nchw;
        inDesc = MKLDNNMemoryDesc(dims, inDesc.getDataType(), format);
    }

    unsigned flag = mkldnn_use_global_stats;
    if (fusedWithScale())
        flag |= mkldnn_use_scaleshift;
    MKLDNNDescriptor desc(std::shared_ptr<batch_normalization_forward::desc>(
            new batch_normalization_forward::desc(prop_kind::forward_scoring, inDesc, eps,
                                                  flag)));
    descs.push_back(desc);
}

void MKLDNNBatchNormalizationNode::initOptimalPrimitiveDescriptor() {
    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";
    auto config = selected_pd->getConfig();
    if (isInitConfig(config))
        return;

    if (config.inConfs.size() != 1 || config.outConfs.size() != 1 || (!isUninitTensorDesc(config.inConfs[0].desc) &&
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

void MKLDNNBatchNormalizationNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    // BN primitive doesn't support strides
    for (auto& desc : descs) {
        primitive_desc_iterator itpd = desc.createPrimitiveDescriptorIterator(getEngine());
        while (itpd.is_not_end()) {
            InferenceEngine::LayerConfig config;
            config.dynBatchSupport = true;
            for (size_t i = 0; i < desc.inputNumbers(); i++) {
                InferenceEngine::DataConfig dataConfig;
                dataConfig.inPlace = -1;
                dataConfig.constant = false;
                dataConfig.desc = getSrcMemDesc(itpd, i);
                config.inConfs.push_back(dataConfig);
            }

            std::vector<memory::format> outFormats;
            for (size_t i = 0; i < desc.outputNumbers(); i++) {
                InferenceEngine::DataConfig dataConfig;
                dataConfig.inPlace = canBeInPlace() ? 0 : -1;
                dataConfig.constant = false;
                dataConfig.desc = getDstMemDesc(itpd, i);
                config.outConfs.push_back(dataConfig);

                outFormats.emplace_back(static_cast<memory::format>(itpd.dst_primitive_desc().desc().data.format));
            }
            impl_desc_type impl_type = parse_impl_name(itpd.get_impl_info_str());

            supportedPrimitiveDescriptors.emplace_back(config, impl_type, outFormats);
            itpd++;
        }
    }
}

MKLDNNMemoryDesc MKLDNNBatchNormalizationNode::getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it,
                                                             size_t idx) {
    TensorDesc desc = MKLDNNMemoryDesc(primitive_desc_it.src_primitive_desc(idx).desc());

    if (getParentEdgeAt(0)->getDims().ndims() == 2 && desc.getLayout() == InferenceEngine::Layout::NCHW) {
        desc.reshape(getParentEdgeAt(idx)->getDims().ToSizeVector(), InferenceEngine::Layout::NC);
        return MKLDNNMemoryDesc(desc);
    }
    if (desc.getLayout() == InferenceEngine::Layout::ANY)
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getParentEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getLayout()));
    else
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getParentEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getBlockingDesc()));
}

MKLDNNMemoryDesc MKLDNNBatchNormalizationNode::getDstMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it,
                                                             size_t idx) {
    TensorDesc desc =  MKLDNNMemoryDesc(primitive_desc_it.dst_primitive_desc(idx).desc());

    if (getParentEdgeAt(0)->getDims().ndims() == 2 && desc.getLayout() == InferenceEngine::Layout::NCHW) {
        desc.reshape(getParentEdgeAt(idx)->getDims().ToSizeVector(), InferenceEngine::Layout::NC);
        return MKLDNNMemoryDesc(desc);
    }
    if (desc.getLayout() == InferenceEngine::Layout::ANY)
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getChildEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getLayout()));
    else
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getChildEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getBlockingDesc()));
}

REG_MKLDNN_PRIM_FOR(MKLDNNBatchNormalizationNode, BatchNormalization);
