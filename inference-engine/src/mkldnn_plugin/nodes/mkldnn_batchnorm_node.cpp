// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_batchnorm_node.h"
#include <mkldnn_extension_utils.h>
#include "common/cpu_memcpy.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNBatchNormalizationNode::MKLDNNBatchNormalizationNode(const InferenceEngine::CNNLayerPtr& layer,
                                                           const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache) {
    internalBlobDesc.emplace_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return GetVarianceDesc(primitive_desc_it);
    });
    internalBlobDesc.emplace_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return GetMeanDesc(primitive_desc_it);
    });

    internalBlobDesc.emplace_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        if (!fusedWithScale())
            return MKLDNNMemoryDesc();
        return GetScaleShiftWeightsDesc(primitive_desc_it);
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
        cpu_memcpy_s(data, internalBlob->byteSize(), blb->buffer(), weightsByteSize);
        data += blb->size();
        blb = scshLayer->_biases;

        if (blb == nullptr) {
            memset(data, 0, weightsByteSize);
        } else {
            if (weightsByteSize != blb->byteSize())
                THROW_IE_EXCEPTION << "ScaleShift has incorrect weights!";
            cpu_memcpy_s(data, internalBlob->byteSize(), blb->buffer(), weightsByteSize);
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

static MKLDNNMemoryDesc get_bn_mdesc_by_index(const mkldnn::primitive_desc_iterator &primitive_desc, int idx) {
    mkldnn_batch_normalization_desc_t *p;
    error::wrap_c_api(mkldnn_primitive_desc_query(
            primitive_desc.get(), mkldnn::convert_to_c(mkldnn::query::batch_normalization_d), 0, &p),
                      "could not get a batch-normalization descriptor");
    auto bndesc =
            (p->flags & mkldnn::convert_to_c(mkldnn::normalization_flags::use_global_stats)) ?
            primitive_desc.src_desc(idx) : primitive_desc.dst_desc(idx);

    return MKLDNNMemoryDesc {bndesc};
}

MKLDNNMemoryDesc MKLDNNBatchNormalizationNode::GetVarianceDesc(const mkldnn::primitive_desc &primitive_desc) const {
    // TODO: rewrite with using stat_desc
    return get_bn_mdesc_by_index(primitive_desc, 2);
}

MKLDNNMemoryDesc MKLDNNBatchNormalizationNode::GetMeanDesc(const mkldnn::primitive_desc &primitive_desc) const {
    return get_bn_mdesc_by_index(primitive_desc, 1);
}

MKLDNNMemoryDesc MKLDNNBatchNormalizationNode::GetScaleShiftWeightsDesc(const mkldnn::primitive_desc &primitive_desc) const {
    return MKLDNNMemoryDesc(primitive_desc.weights_desc(0));
}

bool MKLDNNBatchNormalizationNode::created() const {
    return getType() == BatchNormalization;
}

void MKLDNNBatchNormalizationNode::createPrimitive() {
    if (prim)
        return;

    auto prim_desc = createPrimitiveDescriptor<batch_normalization_forward::primitive_desc,
            batch_normalization_forward::desc>();
    prim.reset(new batch_normalization_forward(prim_desc));

    auto src = getParentEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
    auto dst = getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();

    const auto &mean = internalBlobMemory[1]->GetPrimitive();
    const auto &var = internalBlobMemory[0]->GetPrimitive();

    if (convert_to_c(flag) & dnnl_use_scaleshift) {
        const auto &sclshft = internalBlobMemory[2]->GetPrimitive();
        primArgs = {{DNNL_ARG_SRC, src},
                    {DNNL_ARG_MEAN, mean},
                    {DNNL_ARG_VARIANCE, var},
                    {DNNL_ARG_SCALE_SHIFT, sclshft},
                    {DNNL_ARG_DST, dst}};
    } else {
        primArgs = {{DNNL_ARG_SRC, src},
                    {DNNL_ARG_MEAN, mean},
                    {DNNL_ARG_VARIANCE, var},
                    {DNNL_ARG_DST, dst}};
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
        auto format = memory::format_tag::nchw;
        inDesc = MKLDNNMemoryDesc(dims, inDesc.getDataType(), format);
    }

    flag = normalization_flags::use_global_stats;
    if (fusedWithScale())
        flag |= normalization_flags::use_scale_shift;

    MKLDNNDescriptor desc(std::shared_ptr<batch_normalization_forward::desc>(
            new mkldnn::batch_normalization_forward::desc(prop_kind::forward_scoring, inDesc, eps,
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
        while (static_cast<bool>(itpd)) {
            InferenceEngine::LayerConfig config;
            config.dynBatchSupport = true;
            for (size_t i = 0; i < desc.inputNumbers(); i++) {
                InferenceEngine::DataConfig dataConfig;
                dataConfig.inPlace = -1;
                dataConfig.constant = false;
                dataConfig.desc = getSrcMemDesc(itpd, i);
                config.inConfs.push_back(dataConfig);
            }

            for (size_t i = 0; i < desc.outputNumbers(); i++) {
                InferenceEngine::DataConfig dataConfig;
                dataConfig.inPlace = canBeInPlace() ? 0 : -1;
                dataConfig.constant = false;
                dataConfig.desc = getDstMemDesc(itpd, i);
                config.outConfs.push_back(dataConfig);
            }
            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());

            supportedPrimitiveDescriptors.emplace_back(config, impl_type);
            if (!itpd.next_impl())
                break;
        }
    }
}

MKLDNNMemoryDesc MKLDNNBatchNormalizationNode::getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it,
                                                             size_t idx) {
    TensorDesc desc = MKLDNNMemoryDesc(primitive_desc_it.src_desc(idx));

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
    TensorDesc desc =  MKLDNNMemoryDesc(primitive_desc_it.dst_desc(idx));

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
