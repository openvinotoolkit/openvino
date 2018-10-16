// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_conv_node.h"
#include "mkldnn_reorder_node.h"
#include "mkldnn_input_node.h"
#include "mkldnn_activation_node.h"
#include "desc_iterator.hpp"
#include "mkldnn_eltwise_node.h"
#include <ie_layers.h>
#include <string>
#include <vector>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNConvolutionNode::MKLDNNConvolutionNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng)
        : MKLDNNNode(layer, eng), withBiases(false) {
    internalBlobDesc.emplace_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(0).desc());
    });
    internalBlobDesc.emplace_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        if (!withBiases)
            return MKLDNNMemoryDesc();
        return MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(1).desc());
    });
}

void MKLDNNConvolutionNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
    precision = getCnnLayer()->outData[0]->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    auto * convLayer = dynamic_cast<ConvolutionLayer*>(getCnnLayer().get());
    if (convLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert convolution layer.";

    if (getParentEdges().size() != 1 &&
        ((getType() != Convolution_Sum && getType() != Convolution_Sum_Activation) || getParentEdges().size() != 2))
        THROW_IE_EXCEPTION << "Incorrect number of input edges.";
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges.";

    if (getParentEdgeAt(0)->getDims().ndims() != 4) {
        THROW_IE_EXCEPTION << "Convolution layer. Unsupported mode, only 4D blob as input.";
    }

    isMerged = (!getMergeWith().empty());  // grouped convolution was constructed from split->concat subgraph
    isGrouped = convLayer->_group != 1;    // group info available from IR
    if (isMerged && isGrouped)
        THROW_IE_EXCEPTION << "Convolution initialization. Group splitted mode are used together with direct group specification.";

    // default values. Can be replaced in next steps
    size_t groupNum = convLayer->_group;
    size_t IC = convLayer->input()->getDims()[1];
    size_t groupIC = IC;
    size_t groupOC = convLayer->_out_depth;

    isDW = groupNum == groupOC && groupNum == groupIC;

    if (isMerged) {
        groupNum = getMergeWith().size() + 1;
    }
    if (isGrouped) {
        groupIC /= groupNum;
        groupOC /= groupNum;
    }

    weightDims = { groupOC, groupIC, convLayer->_kernel_y, convLayer->_kernel_x};
    biasesDims = { groupOC * groupNum};

    if (isGrouped || isMerged) weightDims.insert(weightDims.begin(), groupNum);

    withBiases = (convLayer->_biases != nullptr && convLayer->_biases->size() != 0);

    internalBlobs.push_back(createInternalBlob(weightDims, true));
    if (withBiases) {
        internalBlobs.push_back(createInternalBlob(biasesDims, false));
    }

    stride = {static_cast<int>(convLayer->_stride_y), static_cast<int>(convLayer->_stride_x)};
    dilation = {static_cast<int>(convLayer->_dilation_y) - 1, static_cast<int>(convLayer->_dilation_x) - 1};
    paddingL = {static_cast<int>(convLayer->_padding_y), static_cast<int>(convLayer->_padding_x)};
    paddingR = {0, 0};

    MKLDNNDims weightsDims = MKLDNNDims(weightDims);

    for (int i = 0; i < 2; i++) {
        int with_group = (isGrouped || isMerged) ? 1 : 0;
        int krn = weightsDims[with_group + 2 + i];
        int src = getParentEdgeAt(0)->getDims()[2 + i];
        int dst = getChildEdgeAt(0)->getDims()[2 + i];

        krn = (krn - 1)*(dilation[i] + 1) + 1;
        int calc_dst = (src - krn + paddingL[i]) / stride[i] + 1;
        paddingR[i] = (dst - calc_dst) * stride[i];
    }

    withSum = getType() == Convolution_Sum || getType() == Convolution_Sum_Activation;

    for (auto &node : fusedWith) {
        auto *convolutionNode = dynamic_cast<MKLDNNConvolutionNode *>(node.get());
        if (convolutionNode) {
            auto *convLayer = reinterpret_cast<ConvolutionLayer *>(convolutionNode->getCnnLayer().get());
            dw_conv_ih = convolutionNode->inDims[0][2];
            dw_conv_iw = convolutionNode->inDims[0][3];
            dw_conv_oc = convLayer->_out_depth;
            dw_conv_kh = convLayer->_kernel_y;
            dw_conv_kw = convLayer->_kernel_x;
            dw_conv_sh = convLayer->_stride_y;
            dw_conv_sw = convLayer->_stride_x;
        }
    }

    MKLDNNMemoryDesc in_candidate(getParentEdgeAt(0)->getDims(), inputDataType, memory::nchw);
    MKLDNNMemoryDesc out_candidate(getChildEdgeAt(0)->getDims(), outputDataType, memory::nchw);
    createDescriptor({in_candidate}, {out_candidate});

    if (IC == 3 || IC == 1) {
        out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::nChw16c);
        createDescriptor({in_candidate}, {out_candidate});
        out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::nChw8c);
        createDescriptor({in_candidate}, {out_candidate});
    } else {
        in_candidate = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), outputDataType, memory::nChw16c);
        out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::nChw16c);
        createDescriptor({in_candidate}, {out_candidate});
        in_candidate = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), outputDataType, memory::nChw8c);
        out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::nChw8c);
        createDescriptor({in_candidate}, {out_candidate});
    }
}


void MKLDNNConvolutionNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    mkldnn::post_ops ops;
    for (auto &node : fusedWith) {
        auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(node.get());
        if (eltwiseNode) {
            ops.append_sum(1.0);
            continue;
        }

        auto* activationNode = dynamic_cast<MKLDNNActivationNode *>(node.get());
        if (activationNode) {
            ops.append_eltwise(1.0, activationNode->getAlgorithm(), activationNode->getAlpha(),
                               activationNode->getBeta());
            continue;
        }

        auto* convolutionNode = dynamic_cast<MKLDNNConvolutionNode *>(node.get());
        if (convolutionNode) {
            ops.append_dw_conv(dw_conv_ih, dw_conv_iw, dw_conv_kh, dw_conv_kw, dw_conv_sh, dw_conv_sw,
                               nullptr, nullptr);

            for (auto &dwConvFusedNode : convolutionNode->fusedWith) {
                auto* dwConvActivationNode = dynamic_cast<MKLDNNActivationNode *>(dwConvFusedNode.get());
                if (dwConvActivationNode) {
                    ops.append_eltwise(1.0, dwConvActivationNode->getAlgorithm(), dwConvActivationNode->getAlpha(),
                                       dwConvActivationNode->getBeta());
                }
            }

            continue;
        }
    }

    mkldnn::primitive_attr attr;
    attr.set_post_ops(ops);

    for (auto& desc : descs) {
        try {
            primitive_desc_iterator itpd = desc.createPrimitiveDescriptorIterator(getEngine(), attr);
            do {
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

                for (size_t i = 0; i < desc.outputNumbers(); i++) {
                    InferenceEngine::DataConfig dataConfig;
                    if (withSum) {
                        dataConfig.inPlace = 1;
                    }

                    dataConfig.constant = false;
                    dataConfig.desc = getDstMemDesc(itpd, i);
                    if (!isGrouped)
                        dataConfig.desc = MKLDNNExtensionUtils::getUninitTensorDesc(dataConfig.desc);
                    config.outConfs.push_back(dataConfig);

                    if (withSum) {
                        dataConfig.inPlace = -1;
                        config.inConfs.push_back(dataConfig);
                    }
                }
                impl_desc_type impl_type = parse_impl_name(itpd.get_impl_info_str());

                supportedPrimitiveDescriptors.emplace_back(config, impl_type);
            } while (itpd.next());
        } catch (std::exception& e) {
            // it throw exception in case of no implementation found
            continue;
        }
    }
}


void MKLDNNConvolutionNode::createPrimitive() {
    if (prim)
        return;

    mkldnn::post_ops ops;
    for (auto &node : fusedWith) {
        auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(node.get());
        if (eltwiseNode) {
            ops.append_sum(1.0);
            continue;
        }

        auto* activationNode = dynamic_cast<MKLDNNActivationNode *>(node.get());
        if (activationNode) {
            ops.append_eltwise(1.0, activationNode->getAlgorithm(), activationNode->getAlpha(),
                               activationNode->getBeta());
            continue;
        }

        auto* convolutionNode = dynamic_cast<MKLDNNConvolutionNode *>(node.get());
        if (convolutionNode) {
            auto* convLayer = reinterpret_cast<ConvolutionLayer*>(convolutionNode->getCnnLayer().get());

            DWConvInternalBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
            MKLDNNDims dwWeightsDims({dw_conv_oc, 1, 1, dw_conv_kh, dw_conv_kw});
            DWConvInternalBlobMemory[0]->Create(dwWeightsDims, memory::data_type::f32, memory::format::Goihw8g);

            DWConvInternalBlobMemory[0]->SetData(memory::data_type::f32, memory::goihw, convLayer->_weights->buffer(),
                               dwWeightsDims.size() * MKLDNNExtensionUtils::sizeOfDataType(memory::data_type::f32));

            DWConvInternalBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
            MKLDNNDims dwBiasesDims({dw_conv_oc});
            DWConvInternalBlobMemory[1]->Create(dwBiasesDims, memory::data_type::f32, memory::format::x);
            DWConvInternalBlobMemory[1]->SetData(memory::data_type::f32, memory::x, convLayer->_biases->buffer(),
                               dwBiasesDims.size() * MKLDNNExtensionUtils::sizeOfDataType(memory::data_type::f32));
            ops.append_dw_conv(dw_conv_ih, dw_conv_iw, dw_conv_kh, dw_conv_kw, dw_conv_sh, dw_conv_sw,
                               (const float *)DWConvInternalBlobMemory[0]->GetData(),
                               (const float *)DWConvInternalBlobMemory[1]->GetData());

            for (auto &dwConvFusedNode : convolutionNode->fusedWith) {
                auto* dwConvActivationNode = dynamic_cast<MKLDNNActivationNode *>(dwConvFusedNode.get());
                if (dwConvActivationNode) {
                    ops.append_eltwise(1.0, dwConvActivationNode->getAlgorithm(), dwConvActivationNode->getAlpha(),
                                       dwConvActivationNode->getBeta());
                }
            }

            continue;
        }
    }

    mkldnn::primitive_attr attr;
    attr.set_post_ops(ops);

    auto prim_desc = createPrimitiveDescriptor<convolution_forward::primitive_desc,
            convolution_forward::desc>(attr);

    if (internalBlobMemory.size() > 1) {
        prim.reset(new convolution_forward(prim_desc,
                                           getParentEdgeAt(0)->getMemory().GetPrimitive(),
                                           internalBlobMemory[0]->GetPrimitive(),
                                           internalBlobMemory[1]->GetPrimitive(),
                                           getChildEdgeAt(0)->getMemory().GetPrimitive()));
    } else {
        prim.reset(new convolution_forward(prim_desc,
                                           getParentEdgeAt(0)->getMemory().GetPrimitive(),
                                           internalBlobMemory[0]->GetPrimitive(),
                                           getChildEdgeAt(0)->getMemory().GetPrimitive()));
    }
}

bool MKLDNNConvolutionNode::created() const {
    return getType() == Convolution || getType() == Convolution_Sum_Activation ||
           getType() == Convolution_Activation || getType() == Convolution_Sum;
}

void MKLDNNConvolutionNode::createDescriptor(const std::vector<InferenceEngine::TensorDesc> &inputDesc,
                                             const std::vector<InferenceEngine::TensorDesc> &outputDesc) {
    MKLDNNMemoryDesc in_candidate(inputDesc[0]);
    MKLDNNMemoryDesc out_candidate(outputDesc[0]);
    auto in_fmt = in_candidate.getFormat();
    auto out_fmt = out_candidate.getFormat();

    int O_IND = (isGrouped || isMerged) ? 1 : 0;
    int I_IND = (isGrouped || isMerged) ? 2 : 1;

    // grouping and autoblicking is not compatible
    if (((isGrouped && !isDW) || isMerged) && (in_candidate.blocksExtended() || out_candidate.blocksExtended()))
        return;

    MKLDNNDims blocked_weightDims(weightDims);
    MKLDNNDims blocked_biasesDims(biasesDims);

    if (!isGrouped && !isMerged) {
        if (in_fmt == memory::nChw16c) {
            blocked_weightDims[I_IND] = rnd_up(blocked_weightDims[I_IND], 16);
        } else if (in_fmt == memory::nChw8c) {
            blocked_weightDims[I_IND] = rnd_up(blocked_weightDims[I_IND], 8);
        }

        if (out_fmt == memory::nChw16c) {
            blocked_weightDims[O_IND] = rnd_up(blocked_weightDims[O_IND], 16);
            blocked_biasesDims[0] = rnd_up(blocked_biasesDims[0], 16);
        } else if (out_fmt == memory::nChw8c) {
            blocked_weightDims[O_IND] = rnd_up(blocked_weightDims[O_IND], 8);
            blocked_biasesDims[0] = rnd_up(blocked_biasesDims[0], 8);
        }
    } else if (isDW) {
        if (out_fmt != in_fmt)
            return;

        if (in_fmt == memory::nChw16c) {
            blocked_weightDims[0] = rnd_up(blocked_weightDims[0], 16);
            blocked_biasesDims[0] = rnd_up(blocked_biasesDims[0], 16);
        } else if (in_fmt == memory::nChw8c) {
            blocked_weightDims[0] = rnd_up(blocked_weightDims[0], 8);
            blocked_biasesDims[0] = rnd_up(blocked_biasesDims[0], 8);
        }
    }

    MKLDNNMemoryDesc wgh_candidate{blocked_weightDims, in_candidate.getDataType(), memory::any};

    for (auto alg : {algorithm::convolution_winograd, algorithm::convolution_direct}) {
        std::shared_ptr<mkldnn::convolution_forward::desc> conv_desc;
        if (withBiases) {
            MKLDNNMemoryDesc bias_candidate{blocked_biasesDims, in_candidate.getDataType(), memory::any};

            conv_desc.reset(new convolution_forward::desc(prop_kind::forward_scoring, alg, in_candidate,
                                                          wgh_candidate, bias_candidate, out_candidate,
                                                          stride, dilation, paddingL, paddingR, padding_kind::zero));
        } else {
            conv_desc.reset(new convolution_forward::desc(prop_kind::forward_scoring, alg, in_candidate,
                                                          wgh_candidate, out_candidate, stride, dilation,
                                                          paddingL, paddingR, padding_kind::zero));
        }

        descs.emplace_back(conv_desc);
    }
}

void MKLDNNConvolutionNode::initDescriptor(const InferenceEngine::LayerConfig& config) {
    auto* selectedPD = getSelectedPrimitiveDescriptor();
    if (!selectedPD) {
        return;
    }
    bool addedNewDesc = false;
    if (config.inConfs[0].desc.getPrecision() == InferenceEngine::Precision::FP32 &&
        config.outConfs[0].desc.getPrecision() == InferenceEngine::Precision::FP32) {
        addedNewDesc = true;
        createDescriptor({config.inConfs[0].desc}, {config.outConfs[0].desc});
    }

    mkldnn::post_ops ops;
    for (auto &node : fusedWith) {
        auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(node.get());
        if (eltwiseNode) {
            ops.append_sum(1.0);
            continue;
        }

        auto* activationNode = dynamic_cast<MKLDNNActivationNode *>(node.get());
        if (activationNode) {
            ops.append_eltwise(1.0, activationNode->getAlgorithm(), activationNode->getAlpha(),
                               activationNode->getBeta());
            continue;
        }

        auto* convolutionNode = dynamic_cast<MKLDNNConvolutionNode *>(node.get());
        if (convolutionNode) {
            ops.append_dw_conv(dw_conv_ih, dw_conv_iw, dw_conv_kh, dw_conv_kw, dw_conv_sh, dw_conv_sw,
                               nullptr, nullptr);

            for (auto &dwConvFusedNode : convolutionNode->fusedWith) {
                auto* dwConvActivationNode = dynamic_cast<MKLDNNActivationNode *>(dwConvFusedNode.get());
                if (dwConvActivationNode) {
                    ops.append_eltwise(1.0, dwConvActivationNode->getAlgorithm(), dwConvActivationNode->getAlpha(),
                                       dwConvActivationNode->getBeta());
                }
            }

            continue;
        }
    }

    mkldnn::primitive_attr attr;
    attr.set_post_ops(ops);

    InferenceEngine::LayerConfig rightConfig = selectedPD->getConfig();
    size_t selected_count = 0;
    for (size_t i = 0; i < descs.size(); i++) {
        const auto& desc = descs[i];
        try {
            primitive_desc_iterator itpd = desc.createPrimitiveDescriptorIterator(getEngine(), attr);
            do {
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
                    if (withSum) {
                        cfg.inConfs.push_back(dataConfig);
                        dataConfig.inPlace = 1;
                    }
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
            } while (itpd.next());
        } catch (std::exception& e) {
            continue;
        }
    }
    selectedPD->getConfig() = rightConfig;
}
