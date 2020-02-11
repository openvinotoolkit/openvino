// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_bin_conv_node.h"
#include "mkldnn_reorder_node.h"
#include "mkldnn_input_node.h"
#include "mkldnn_activation_node.h"
#include "desc_iterator.hpp"
#include "mkldnn_eltwise_node.h"
#include "mkldnn_depthwise_node.h"
#include "mkldnn_quantize_node.h"
#include "mkldnn_conv_node.h"
#include <ie_layers.h>
#include <string>
#include <vector>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <ie_layers_internal.hpp>
#include "cpu_isa_traits.hpp"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNBinaryConvolutionNode::MKLDNNBinaryConvolutionNode(const InferenceEngine::CNNLayerPtr& layer,
                                                         const mkldnn::engine& eng, int socket)
        : MKLDNNNode(layer, eng, socket) {
    internalBlobDesc.emplace_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(0).desc());
    });
}

void MKLDNNBinaryConvolutionNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    auto* binConvLayer = dynamic_cast<BinaryConvolutionLayer*>(getCnnLayer().get());
    if (binConvLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert convolution layer.";

    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();

    if ((getParentEdgeAt(0)->getDims().ndims() < 4) || (getParentEdgeAt(0)->getDims().ndims() > 5)) {
        THROW_IE_EXCEPTION << "Convolution layer. Unsupported mode. Only 4D and 5D blobs are supported as input.";
    }

    isMerged = (!getMergeWith().empty());  // grouped convolution was constructed from split->concat subgraph
    isGrouped = binConvLayer->_group != 1;  // group info available from IR
    if (isMerged && isGrouped)
        THROW_IE_EXCEPTION << "Convolution initialization. Group splitted mode are used together with direct group specification.";

    // default values. Can be replaced in next steps
    size_t groupNum = binConvLayer->_group;
    pad_value = binConvLayer->_pad_value;
    size_t groupIC = binConvLayer->_in_depth;
    size_t groupOC = binConvLayer->_out_depth;

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
    for (int i = 1; i <= binConvLayer->_kernel.size(); i++) {
        weightDims.push_back(binConvLayer->_kernel[binConvLayer->_kernel.size() - i]);
    }
    biasesDims = { groupOC * groupNum };

    if (isGrouped || isMerged) weightDims.insert(weightDims.begin(), groupNum);

    internalBlobs.push_back(createInternalBlob(weightDims, true));

    Blob::Ptr weights = this->getCnnLayer()->blobs.find("weights")->second;

    invertVectorCopyUtoI(binConvLayer->_stride, stride);
    for (int i = 1; i <= binConvLayer->_dilation.size(); i++) {
        dilation.push_back(static_cast<int>(binConvLayer->_dilation[binConvLayer->_dilation.size() - i]) - 1);
    }

    auto allPads = getPaddings(*binConvLayer);
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

    withSum = isFusedWith(Eltwise);
    withDWConv = isFusedWith(Convolution);
    withBinarization = isFusedWith(Quantize);
    for (auto &node : fusedWith) {
#if defined (COMPILED_CPU_MKLDNN_CONV_NODE)
        auto* convolutionNode = dynamic_cast<MKLDNNConvolutionNode*>(node.get());
        if (convolutionNode) {
            auto *convLayer = reinterpret_cast<ConvolutionLayer*>(convolutionNode->getCnnLayer().get());
            dw_conv_ih = convolutionNode->inDims[0][convolutionNode->inDims[0].ndims() - 2];
            dw_conv_iw = convolutionNode->inDims[0][convolutionNode->inDims[0].ndims() - 1];
            dw_conv_oc = convLayer->_out_depth;
            for (int i = 0; i < convLayer->_kernel.size(); i++) {
                dw_conv_kernel.push_back(convLayer->_kernel[i]);
            }
            for (int i = 0; i < convLayer->_stride.size(); i++) {
                dw_conv_strides.push_back(convLayer->_stride[i]);
            }
            dw_conv_in_dt = MKLDNNExtensionUtils::IEPrecisionToDataType(convLayer->outData[0]->getPrecision());
        }
#endif
    }

    int expectedInputEdgesNum = 1 + isFusedWith(Eltwise);
    for (int i = 0; i < fusedWith.size(); i++) {
        auto *convolutionNode = dynamic_cast<MKLDNNConvolutionNode *>(fusedWith[i].get());
        if (convolutionNode) {
            expectedInputEdgesNum += convolutionNode->getBaseIntputsNumber() - 1;
        }
    }

    if (getParentEdges().size() != expectedInputEdgesNum)
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();

    auto inputDataType = memory::bin;
    auto outputDataType = withBinarization ? memory::bin : memory::f32;

    MKLDNNMemoryDesc in_candidate = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::nhwc);
    MKLDNNMemoryDesc out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::nhwc);
    createDescriptor({in_candidate}, {out_candidate});
}

void MKLDNNBinaryConvolutionNode::setPostOps(mkldnn::primitive_attr &attr, bool initWeights = false) {
    int blob_idx = 0;
    mkldnn::post_ops ops;

    for (auto &node : fusedWith) {
#if defined (COMPILED_CPU_MKLDNN_ELTWISE_NODE)
        auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(node.get());
        if (eltwiseNode) {
            if (eltwiseNode->getCnnLayer()->precision == Precision::I8) {
                auto it = eltwiseNode->getCnnLayer()->blobs.find("eltwise-sum-scale");
                if (it != eltwiseNode->getCnnLayer()->blobs.end()) {
                    // currently there is the only one scale while we need scale by channel :(
                    ops.append_sum(it->second->buffer().as<float*>()[0]);
                }
            } else {
                ops.append_sum(1.0);
            }
            continue;
        }
#endif

#if defined(COMPILED_CPU_MKLDNN_ACTIVATION_NODE)
        auto* activationNode = dynamic_cast<MKLDNNActivationNode *>(node.get());
        if (activationNode) {
            ops.append_eltwise(1.0, activationNode->getAlgorithm(), activationNode->getAlpha(),
                               activationNode->getBeta());
            continue;
        }
#endif

#if defined (COMPILED_CPU_MKLDNN_DEPTHWISE_NODE)
        auto* depthwiseNode = dynamic_cast<MKLDNNDepthwiseNode *>(node.get());
        if (depthwiseNode) {
            auto* depthwiseLayer = reinterpret_cast<WeightableLayer*>(depthwiseNode->getCnnLayer().get());

            if (initWeights) {
                MKLDNNDims depthwiseDims({static_cast<ptrdiff_t>(rnd_up(biasesDims[0], 16))});

                PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
                PostOpsIntBlobMemory[blob_idx]->Create(depthwiseDims, memory::data_type::f32, memory::format::x);

                PostOpsIntBlobMemory[blob_idx]->SetData(memory::data_type::f32, memory::x,
                                                             depthwiseLayer->_weights->buffer(),
                                                             depthwiseLayer->_weights->size() *
                                                             MKLDNNExtensionUtils::sizeOfDataType(memory::data_type::f32));

                if (depthwiseNode->isBroadcast()) {
                    float broadcastValue = static_cast<float *>(PostOpsIntBlobMemory[blob_idx]->GetData())[0];
                    for (int i = 1; i < PostOpsIntBlobMemory[blob_idx]->GetPrimitiveDescriptor().desc().data.dims[0]; i++) {
                        static_cast<float *>(PostOpsIntBlobMemory[blob_idx]->GetData())[i] = broadcastValue;
                    }
                }

                if (depthwiseNode->getAlgorithm() == depthwise_scale_shift) {
                    PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
                    PostOpsIntBlobMemory[blob_idx + 1]->Create(depthwiseDims, memory::data_type::f32,
                                                                memory::format::x);
                    PostOpsIntBlobMemory[blob_idx + 1]->SetData(memory::data_type::f32, memory::x,
                                                                 depthwiseLayer->_biases->buffer(),
                                                                 depthwiseLayer->_biases->size() *
                                                                 MKLDNNExtensionUtils::sizeOfDataType(memory::data_type::f32));

                    if (depthwiseNode->isBroadcast()) {
                        float broadcastValue = static_cast<float *>(PostOpsIntBlobMemory[blob_idx + 1]->GetData())[0];
                        for (int i = 1; i < PostOpsIntBlobMemory[blob_idx + 1]->GetPrimitiveDescriptor().desc().data.dims[0]; i++) {
                            static_cast<float *>(PostOpsIntBlobMemory[blob_idx + 1]->GetData())[i] = broadcastValue;
                        }
                    }

                    ops.append_depthwise(depthwiseNode->getAlgorithm(),
                                         (const float *) PostOpsIntBlobMemory[blob_idx]->GetData(),
                                         (const float *) PostOpsIntBlobMemory[blob_idx + 1]->GetData());

                    blob_idx += 2;
                } else {
                    ops.append_depthwise(depthwiseNode->getAlgorithm(),
                                         (const float *) PostOpsIntBlobMemory[blob_idx]->GetData(),
                                         nullptr);

                    blob_idx += 1;
                }
            } else {
                ops.append_depthwise(depthwiseNode->getAlgorithm(),
                                     nullptr,
                                     nullptr);
            }

            continue;
        }
#endif

#if defined (COMPILED_CPU_MKLDNN_QUANTIZE_NODE)
        auto* quantizeNode = dynamic_cast<MKLDNNQuantizeNode *>(node.get());
        if (quantizeNode) {
            if (initWeights) {
                MKLDNNDims binarizationDims({static_cast<ptrdiff_t>(rnd_up(biasesDims[0], 16))});

                PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
                PostOpsIntBlobMemory[blob_idx]->Create(binarizationDims, memory::data_type::f32, memory::format::x);

                PostOpsIntBlobMemory[blob_idx]->SetData(memory::data_type::f32, memory::x,
                                                        quantizeNode->getBinarizationTresholdsPtr(),
                                                        quantizeNode->getBinarizationTresholdsSize() *
                                                        MKLDNNExtensionUtils::sizeOfDataType(memory::data_type::f32));

                PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
                PostOpsIntBlobMemory[blob_idx+1]->Create(binarizationDims, memory::data_type::f32, memory::format::x);

                PostOpsIntBlobMemory[blob_idx+1]->SetData(memory::data_type::f32, memory::x,
                                                        quantizeNode->getBinarizationOutputMaskPtr(),
                                                        quantizeNode->getBinarizationOutputMaskSize() *
                                                        MKLDNNExtensionUtils::sizeOfDataType(memory::data_type::f32));

                ops.append_binarization(binarization_depthwise, (const float*)PostOpsIntBlobMemory[blob_idx+0]->GetData(),
                                                                (const float*)PostOpsIntBlobMemory[blob_idx+1]->GetData());

                blob_idx += 2;
            } else {
                ops.append_binarization(binarization_depthwise, nullptr, nullptr);
            }
        }
#endif

#if defined(COMPILED_CPU_MKLDNN_CONV_NODE)
        auto* convolutionNode = dynamic_cast<MKLDNNConvolutionNode *>(node.get());
        if (convolutionNode) {
            auto* convLayer = reinterpret_cast<ConvolutionLayer*>(convolutionNode->getCnnLayer().get());
            if (initWeights) {
                if (convolutionNode->getBaseIntputsNumber() == 1) {
                    auto w_fmt = mkldnn::impl::cpu::mayiuse(impl::cpu::cpu_isa_t::avx512_common)
                                 ? memory::format::Goihw16g : memory::format::Goihw8g;

                    PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
                    MKLDNNDims dwWeightsDims(
                            {dw_conv_oc, (ptrdiff_t) 1, (ptrdiff_t) 1, dw_conv_kernel[Y_AXIS], dw_conv_kernel[X_AXIS]});
                    PostOpsIntBlobMemory[blob_idx]->Create(dwWeightsDims, memory::data_type::f32, w_fmt);

                    PostOpsIntBlobMemory[blob_idx]->SetData(memory::data_type::f32, memory::goihw,
                                                            convLayer->_weights->buffer(),
                                                            dwWeightsDims.size() *
                                                            MKLDNNExtensionUtils::sizeOfDataType(
                                                                    memory::data_type::f32));

                    PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
                    MKLDNNDims dwBiasesDims({dw_conv_oc});
                    PostOpsIntBlobMemory[blob_idx + 1]->Create(dwBiasesDims, memory::data_type::f32,
                                                               memory::format::x);
                    PostOpsIntBlobMemory[blob_idx + 1]->SetData(memory::data_type::f32, memory::x,
                                                                convLayer->_biases->buffer(),
                                                                dwBiasesDims.size() *
                                                                MKLDNNExtensionUtils::sizeOfDataType(
                                                                        memory::data_type::f32));
                    ops.append_dw_conv(dw_conv_ih, dw_conv_iw, dw_conv_kernel[Y_AXIS], dw_conv_kernel[X_AXIS],
                                       dw_conv_strides[Y_AXIS], dw_conv_strides[X_AXIS],
                                       mkldnn::memory::convert_to_c(dw_conv_in_dt),
                                       (const float *) PostOpsIntBlobMemory[blob_idx]->GetData(),
                                       (const float *) PostOpsIntBlobMemory[blob_idx + 1]->GetData());

                    blob_idx += 2;
                } else {
                    ops.append_dw_conv(dw_conv_ih, dw_conv_iw, dw_conv_kernel[Y_AXIS], dw_conv_kernel[X_AXIS],
                                       dw_conv_strides[Y_AXIS], dw_conv_strides[X_AXIS],
                                       mkldnn::memory::convert_to_c(dw_conv_in_dt),
                                       static_cast<float *>(getParentEdgeAt(1)->getMemory().GetData()),
                                       static_cast<float *>(getParentEdgeAt(2)->getMemory().GetData()));
                }
            } else {
                ops.append_dw_conv(dw_conv_ih, dw_conv_iw, dw_conv_kernel[Y_AXIS], dw_conv_kernel[X_AXIS],
                                   dw_conv_strides[Y_AXIS], dw_conv_strides[X_AXIS],
                                   mkldnn::memory::convert_to_c(dw_conv_in_dt),
                                   nullptr,
                                   nullptr);
            }

            continue;
        }
#endif
    }

    attr.set_post_ops(ops);
}

void MKLDNNBinaryConvolutionNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    mkldnn::primitive_attr attr;
    setPostOps(attr);

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

            if (withDWConv) {
                auto weightsPrc = memory::data_type::f32;
                auto biasPrc = memory::data_type::f32;

                MKLDNNDims dwWeightsDims({dw_conv_oc, (ptrdiff_t)1, (ptrdiff_t)1, dw_conv_kernel[Y_AXIS], dw_conv_kernel[X_AXIS]});
                MKLDNNDims dwBiasesDims({dw_conv_oc});
                auto w_fmt = mkldnn::impl::cpu::mayiuse(impl::cpu::cpu_isa_t::avx512_common)
                             ? memory::format::Goihw16g : memory::format::Goihw8g;

                InferenceEngine::DataConfig dataConfig;
                dataConfig.inPlace = -1;
                dataConfig.constant = false;
                dataConfig.desc = MKLDNNMemoryDesc(dwWeightsDims, weightsPrc, w_fmt);
                config.inConfs.push_back(dataConfig);

                dataConfig.desc = MKLDNNMemoryDesc(dwBiasesDims, biasPrc, memory::format::x);
                config.inConfs.push_back(dataConfig);
            }

            std::vector<memory::format> outFormats;
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
                outFormats.emplace_back(static_cast<memory::format>(itpd.dst_primitive_desc().desc().data.format));

                if (withSum) {
                    dataConfig.inPlace = -1;
                    config.inConfs.push_back(dataConfig);
                }
            }
            impl_desc_type impl_type = parse_impl_name(itpd.get_impl_info_str());

            supportedPrimitiveDescriptors.emplace_back(config, impl_type, outFormats);
            itpd++;
        }
    }
}


void MKLDNNBinaryConvolutionNode::createPrimitive() {
    if (prim)
        return;

    mkldnn::primitive_attr attr;
    setPostOps(attr, true);

    auto prim_desc = createPrimitiveDescriptor<binary_convolution_forward::primitive_desc,
            binary_convolution_forward::desc>(attr);

    prim.reset(new binary_convolution_forward(prim_desc,
                                       getParentEdgeAt(0)->getMemory().GetPrimitive(),
                                       internalBlobMemory[0]->GetPrimitive(),
                                       getChildEdgeAt(0)->getMemory().GetPrimitive()));
}

bool MKLDNNBinaryConvolutionNode::created() const {
    return getType() == BinaryConvolution;
}

void MKLDNNBinaryConvolutionNode::createDescriptor(const std::vector<InferenceEngine::TensorDesc> &inputDesc,
                                                   const std::vector<InferenceEngine::TensorDesc> &outputDesc) {
    TensorDesc inDesc = inputDesc[0], outDesc = outputDesc[0];
    mkldnn::memory::data_type wdt = MKLDNNExtensionUtils::IEPrecisionToDataType(inDesc.getPrecision());

    MKLDNNMemoryDesc in_candidate(inDesc);
    MKLDNNMemoryDesc out_candidate(outDesc);

    // grouping and autoblocking is not compatible
    if (((isGrouped && !isDW) || isMerged) && (in_candidate.blocksExtended() || out_candidate.blocksExtended()))
        return;

    MKLDNNDims blocked_weightDims(weightDims);
    MKLDNNDims blocked_biasesDims(biasesDims);
    MKLDNNMemoryDesc wgh_candidate{blocked_weightDims, wdt, memory::any};

    std::shared_ptr<mkldnn::binary_convolution_forward::desc> bin_conv_desc;
    bin_conv_desc.reset(new binary_convolution_forward::desc(prop_kind::forward_scoring, algorithm::binary_convolution_direct,
                                                             in_candidate, wgh_candidate, out_candidate, stride, dilation,
                                                             paddingL, paddingR, pad_value));

    descs.emplace_back(bin_conv_desc);
}

void MKLDNNBinaryConvolutionNode::initDescriptor(const InferenceEngine::LayerConfig& config) {
    auto* selectedPD = getSelectedPrimitiveDescriptor();
    if (!selectedPD) {
        return;
    }

    createDescriptor({config.inConfs[0].desc}, {config.outConfs[0].desc});

    mkldnn::primitive_attr attr;
    setPostOps(attr);

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

            if (withDWConv) {
                auto weightsPrc = memory::data_type::f32;
                auto biasPrc = memory::data_type::f32;

                MKLDNNDims dwWeightsDims({dw_conv_oc, (ptrdiff_t)1, (ptrdiff_t)1, dw_conv_kernel[Y_AXIS], dw_conv_kernel[X_AXIS]});
                MKLDNNDims dwBiasesDims({dw_conv_oc});
                auto w_fmt = mkldnn::impl::cpu::mayiuse(impl::cpu::cpu_isa_t::avx512_common)
                             ? memory::format::Goihw16g : memory::format::Goihw8g;

                InferenceEngine::DataConfig dataConfig;
                dataConfig.inPlace = -1;
                dataConfig.constant = false;
                dataConfig.desc = MKLDNNMemoryDesc(dwWeightsDims, weightsPrc, w_fmt);
                cfg.inConfs.push_back(dataConfig);

                dataConfig.desc = MKLDNNMemoryDesc(dwBiasesDims, biasPrc, memory::format::x);
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
            if (i == descs.size() - 1) {
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

REG_MKLDNN_PRIM_FOR(MKLDNNBinaryConvolutionNode, BinaryConvolution);
