// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_def_conv_node.h"
#include "mkldnn_reorder_node.h"
#include "mkldnn_input_node.h"
#include "mkldnn_activation_node.h"
#include "desc_iterator.hpp"
#include "mkldnn_eltwise_node.h"
#include "mkldnn_depthwise_node.h"
#include <ie_layers.h>
#include <string>
#include <vector>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <ie_layers_internal.hpp>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNDeformableConvolutionNode::MKLDNNDeformableConvolutionNode(const InferenceEngine::CNNLayerPtr& layer,
                                                                 const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache) {
    internalBlobDesc.emplace_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(0).desc());
    });
    internalBlobDesc.emplace_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        if (!withBiases)
            return MKLDNNMemoryDesc();
        return MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(1).desc());
    });
}

void MKLDNNDeformableConvolutionNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    auto * defConvLayer = dynamic_cast<DeformableConvolutionLayer*>(getCnnLayer().get());
    if (defConvLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert deformable convolution layer.";

    if (getParentEdges().size() != 2)
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();

    if (getParentEdgeAt(0)->getDims().ndims() != 4) {
        THROW_IE_EXCEPTION << "Deformable convolution layer. Unsupported mode. Only 4D blobs are supported as input.";
    }

    isMerged = (!getMergeWith().empty());  // grouped convolution was constructed from split->concat subgraph
    isGrouped = defConvLayer->_group != 1;  // group info available from IR
    if (isMerged && isGrouped)
        THROW_IE_EXCEPTION << "Deformable convolution initialization. Group splitted mode are used together with direct group specification.";

    // default values. Can be replaced in next steps
    size_t groupNum = defConvLayer->_group;
    size_t groupIC = defConvLayer->input()->getDims()[1];
    size_t groupOC = defConvLayer->_out_depth;

    isDW = groupNum == groupOC && groupNum == groupIC;

    if (isMerged) {
        groupNum = getMergeWith().size() + 1;
    }
    if (isGrouped) {
        groupIC /= groupNum;
        groupOC /= groupNum;
    }

    weightDims.clear();
    weightDims.push_back(groupOC);
    weightDims.push_back(groupIC);
    for (int i = 1; i <= defConvLayer->_kernel.size(); i++) {
        weightDims.push_back(defConvLayer->_kernel[defConvLayer->_kernel.size() - i]);
    }
    biasesDims = { groupOC * groupNum };

    if (isGrouped || isMerged) weightDims.insert(weightDims.begin(), groupNum);

    withBiases = (defConvLayer->_biases != nullptr && defConvLayer->_biases->size() != 0);

    internalBlobs.push_back(createInternalBlob(weightDims, true));
    if (withBiases) {
        internalBlobs.push_back(createInternalBlob(biasesDims, false));
    }

    invertVectorCopyUtoI(defConvLayer->_stride, stride);
    deformable_group = defConvLayer->_deformable_group;
    for (int i = 1; i <= defConvLayer->_dilation.size(); i++) {
        dilation.push_back(static_cast<int>(defConvLayer->_dilation[defConvLayer->_dilation.size() - i] - 1));
    }

    auto allPads = getPaddings(*defConvLayer);
    invertVectorCopyUtoI(allPads.begin, paddingL);
    invertVectorCopyUtoI(allPads.end, paddingR);

    MKLDNNDims weightsDims = MKLDNNDims(weightDims);

    for (int i = 0; i < paddingR.size(); i++) {
        int with_group = (isGrouped || isMerged) ? 1 : 0;
        int krn = weightsDims[with_group + 2 + i];
        int src = getParentEdgeAt(0)->getDims()[2 + i];
        int dst = getChildEdgeAt(0)->getDims()[2 + i];

        krn = (krn - 1)*(dilation[i] + 1) + 1;
        int calc_dst = (src - krn + paddingL[i]) / stride[i] + 1;
        paddingR[i] = (dst - calc_dst) * stride[i];
    }

    MKLDNNMemoryDesc in_candidate(getParentEdgeAt(0)->getDims(), memory::f32, memory::nhwc);
    MKLDNNMemoryDesc offset_candidate(getParentEdgeAt(1)->getDims(), memory::f32, memory::nchw);
    MKLDNNMemoryDesc out_candidate(getChildEdgeAt(0)->getDims(), memory::f32, memory::nhwc);
    createDescriptor({in_candidate, offset_candidate}, {out_candidate});
}

void MKLDNNDeformableConvolutionNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    mkldnn::primitive_attr attr;

    for (auto& desc : descs) {
        auto itpd = desc.createPrimitiveDescriptorIterator(getEngine(), attr);
        while (itpd.is_not_end()) {
            InferenceEngine::LayerConfig config;
            config.dynBatchSupport = true;
            for (size_t i = 0; i < desc.inputNumbers(); i++) {
                InferenceEngine::DataConfig dataConfig;
                dataConfig.inPlace = -1;
                dataConfig.constant = false;
                dataConfig.desc = getSrcMemDesc(itpd, i);
                if (!isGrouped)
                    dataConfig.desc = MKLDNNExtensionUtils::getUninitTensorDesc(dataConfig.desc);
                config.inConfs.push_back(dataConfig);
            }

            std::vector<memory::format> outFormats;
            for (size_t i = 0; i < desc.outputNumbers(); i++) {
                InferenceEngine::DataConfig dataConfig;

                dataConfig.constant = false;
                dataConfig.desc = getDstMemDesc(itpd, i);
                if (!isGrouped)
                    dataConfig.desc = MKLDNNExtensionUtils::getUninitTensorDesc(dataConfig.desc);
                config.outConfs.push_back(dataConfig);

                outFormats.emplace_back(static_cast<memory::format>(itpd.dst_primitive_desc().desc().data.format));
            }
            impl_desc_type impl_type = parse_impl_name(itpd.get_impl_info_str());

            supportedPrimitiveDescriptors.emplace_back(config, impl_type, outFormats);
            itpd++;
        }
    }
}


void MKLDNNDeformableConvolutionNode::createPrimitive() {
    if (prim)
        return;

    mkldnn::primitive_attr attr;

    auto prim_desc = createPrimitiveDescriptor<deformable_convolution_forward::primitive_desc,
            deformable_convolution_forward::desc>(attr);

    std::vector<primitive::at> src_p;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        src_p.push_back(getParentEdgeAt(i)->getMemoryPtr()->GetPrimitive());
    }

    if (internalBlobMemory.size() > 1) {
        prim.reset(new deformable_convolution_forward(prim_desc, src_p,
                                           internalBlobMemory[0]->GetPrimitive(),
                                           internalBlobMemory[1]->GetPrimitive(),
                                           getChildEdgeAt(0)->getMemory().GetPrimitive()));
    } else {
        prim.reset(new deformable_convolution_forward(prim_desc, src_p,
                                           internalBlobMemory[0]->GetPrimitive(),
                                           getChildEdgeAt(0)->getMemory().GetPrimitive()));
    }
}

bool MKLDNNDeformableConvolutionNode::created() const {
    return getType() == DeformableConvolution;
}

void MKLDNNDeformableConvolutionNode::createDescriptor(const std::vector<InferenceEngine::TensorDesc> &inputDesc,
                                             const std::vector<InferenceEngine::TensorDesc> &outputDesc) {
    std::vector<memory::desc> srcs;
    srcs.push_back(MKLDNNMemoryDesc(inputDesc[0]));
    srcs.push_back(MKLDNNMemoryDesc(inputDesc[1]));
    TensorDesc inDesc = inputDesc[0], offsetDesc = inputDesc[1], outDesc = outputDesc[0];
    mkldnn::memory::data_type wdt = MKLDNNExtensionUtils::IEPrecisionToDataType(inDesc.getPrecision());
    mkldnn::memory::data_type bdt = MKLDNNExtensionUtils::IEPrecisionToDataType(inDesc.getPrecision());

    MKLDNNMemoryDesc in_candidate(inDesc);
    MKLDNNMemoryDesc offset_candidate(offsetDesc);
    MKLDNNMemoryDesc out_candidate(outDesc);
    if (((isGrouped && !isDW) || isMerged) && (in_candidate.blocksExtended() || offset_candidate.blocksExtended() || out_candidate.blocksExtended()))
        return;

    MKLDNNDims blocked_weightDims(weightDims);
    MKLDNNDims blocked_biasesDims(biasesDims);
    MKLDNNMemoryDesc wgh_candidate{blocked_weightDims, wdt, memory::any};

    for (auto alg : {algorithm::deformable_convolution_direct}) {
        std::shared_ptr<mkldnn::deformable_convolution_forward::desc> def_conv_desc;
        if (withBiases) {
            MKLDNNMemoryDesc bias_candidate{blocked_biasesDims, bdt, memory::any};

            def_conv_desc.reset(new deformable_convolution_forward::desc(prop_kind::forward_scoring, alg, srcs, wgh_candidate, bias_candidate, out_candidate,
                                                          stride, dilation, paddingL, paddingR, padding_kind::zero, deformable_group));
        } else {
            def_conv_desc.reset(new deformable_convolution_forward::desc(prop_kind::forward_scoring, alg, srcs, wgh_candidate, out_candidate, stride, dilation,
                                                          paddingL, paddingR, padding_kind::zero, deformable_group));
        }

        descs.emplace_back(def_conv_desc);
    }
}

void MKLDNNDeformableConvolutionNode::initDescriptor(const InferenceEngine::LayerConfig& config) {
    auto* selectedPD = getSelectedPrimitiveDescriptor();
    if (!selectedPD) {
        return;
    }
    bool addedNewDesc = false;
    if (config.inConfs[0].desc.getPrecision() == InferenceEngine::Precision::FP32 &&
            config.inConfs[1].desc.getPrecision() == InferenceEngine::Precision::FP32 &&
            config.outConfs[0].desc.getPrecision() == InferenceEngine::Precision::FP32) {
        addedNewDesc = true;
        createDescriptor({config.inConfs[0].desc, config.inConfs[1].desc}, {config.outConfs[0].desc});
    }

    mkldnn::primitive_attr attr;

    InferenceEngine::LayerConfig rightConfig = selectedPD->getConfig();
    size_t selected_count = 0;
    for (size_t i = 0; i < descs.size(); i++) {
        const auto& desc = descs[i];
        auto itpd = desc.createPrimitiveDescriptorIterator(getEngine(), attr);
        while (itpd.is_not_end()) {
            InferenceEngine::LayerConfig cfg;
            cfg.dynBatchSupport = true;
            for (size_t j = 0; j < desc.inputNumbers(); j++) {
                InferenceEngine::DataConfig dataConfig;
                dataConfig.inPlace = -1;
                dataConfig.constant = false;
                dataConfig.desc = getSrcMemDesc(itpd, j);
                cfg.inConfs.push_back(dataConfig);
            }

            for (size_t j = 0; j < desc.outputNumbers(); j++) {
                InferenceEngine::DataConfig dataConfig;
                dataConfig.inPlace = -1;
                dataConfig.constant = false;
                dataConfig.desc = getDstMemDesc(itpd, j);

                cfg.outConfs.push_back(dataConfig);
            }
            impl_desc_type impl_type = parse_impl_name(itpd.get_impl_info_str());

            if (selected_count == selectedPrimitiveDescriptorIndex) {
                if (impl_type != selectedPD->getImplementationType()) {
                    THROW_IE_EXCEPTION << "Cannot get the original layer configuration!";
                }
                rightConfig = cfg;
            }
            if (i == descs.size() - 1 && addedNewDesc) {
                if (impl_type == selectedPD->getImplementationType()) {
                    rightConfig = config;
                }
            }
            selected_count++;
            itpd++;
        }
    }
    selectedPD->getConfig() = rightConfig;
}
REG_MKLDNN_PRIM_FOR(MKLDNNDeformableConvolutionNode, DeformableConvolution);
