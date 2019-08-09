// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_conv_node.h"
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

MKLDNNConvolutionNode::MKLDNNConvolutionNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, int socket)
        : MKLDNNNode(layer, eng, socket), withBiases(false), withSum(false),  dw_conv_iw(0), dw_conv_ih(0),
        dw_conv_oc(0), dw_conv_in_dt(memory::data_type::data_undef), isDW(false), isMerged(false), withActivation(false),
        convLayer(nullptr), isGrouped(false) {
    internalBlobDesc.emplace_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(0).desc());
    });
    internalBlobDesc.emplace_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        if (!withBiases)
            return MKLDNNMemoryDesc();
        return MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(1).desc());
    });

    auto ws = layer->blobs.find("w-scale");
    if (ws != layer->blobs.end()) {
        wScale = ws->second;
    }

    // Trying to find oi-scale
    if (getCnnLayer()->type == "Convolution" && getCnnLayer()->precision == Precision::I8) {
        auto ois = layer->blobs.find("oi-scale");
        if ((getCnnLayer()->outData[0]->getPrecision() == Precision::I8 || getCnnLayer()->outData[0]->getPrecision() == Precision::U8)
            && ois == layer->blobs.end()) {
            THROW_IE_EXCEPTION << "Internal error of graph quantization - mismatch of intermediate scales and next layer type for convolution "
                << getCnnLayer()->name;
        }
        if (ois != layer->blobs.end()) {
            // If we can find an oi-scale, then the next layer has to be an INT8.
            oScale = ois->second;
        }
    }
}

void MKLDNNConvolutionNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    if (precision == InferenceEngine::Precision::U16) {
        precision = InferenceEngine::Precision::FP32;
    }
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
    precision = getCnnLayer()->outData[0]->getPrecision();
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    auto * convLayer = dynamic_cast<ConvolutionLayer*>(getCnnLayer().get());
    if (convLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert convolution layer.";

    if (getParentEdges().size() != 1 &&
        (!isFusedWith(Eltwise) || getParentEdges().size() != 2))
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();

    if ((getParentEdgeAt(0)->getDims().ndims() < 4) || (getParentEdgeAt(0)->getDims().ndims() > 5)) {
        THROW_IE_EXCEPTION << "Convolution layer. Unsupported mode. Only 4D and 5D blobs are supported as input.";
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

    weightDims.clear();
    weightDims.push_back(groupOC);
    weightDims.push_back(groupIC);
    for (int i = 1; i <= convLayer->_kernel.size(); i++) {
        weightDims.push_back(convLayer->_kernel[convLayer->_kernel.size() - i]);
    }
    biasesDims = { groupOC * groupNum };

    if (isGrouped || isMerged) weightDims.insert(weightDims.begin(), groupNum);

    withBiases = (convLayer->_biases != nullptr && convLayer->_biases->size() != 0);

    internalBlobs.push_back(createInternalBlob(weightDims, true));
    if (withBiases) {
        internalBlobs.push_back(createInternalBlob(biasesDims, false));
    }

    Blob::Ptr weights = this->getCnnLayer()->blobs.find("weights")->second;
    if (weights->getTensorDesc().getPrecision() == Precision::I8) {
        // The weights blob has incorrect dims, so we have to fix it
        TensorDesc wdesc = internalBlobs[0]->getTensorDesc();
        wdesc.setPrecision(Precision::I8);
        InferenceEngine::TBlob<int8_t>::Ptr reshapedInt8Weights =
                InferenceEngine::TBlob<int8_t>::Ptr(
                        new InferenceEngine::TBlob<int8_t>(wdesc, static_cast<int8_t*>(weights->buffer()), weights->byteSize()));

        internalBlobs[0] = reshapedInt8Weights;
        if (withBiases) {
            Blob::Ptr biases = this->getCnnLayer()->blobs.find("biases")->second;
            TensorDesc bdesc = internalBlobs[1]->getTensorDesc();
            bdesc.setPrecision(Precision::I32);
            InferenceEngine::TBlob<int32_t>::Ptr reshapedInt32Biases =
                    InferenceEngine::TBlob<int32_t>::Ptr(
                            new InferenceEngine::TBlob<int32_t>(bdesc, static_cast<int32_t*>(biases->buffer()), biases->byteSize()));
            internalBlobs[1] = reshapedInt32Biases;
        }
    }

    invertVectorCopyUtoI(convLayer->_stride, stride);
    for (int i = 1; i <= convLayer->_dilation.size(); i++) {
        dilation.push_back(static_cast<int>(convLayer->_dilation[convLayer->_dilation.size() - i]) - 1);
    }

    auto allPads = getPaddings(*convLayer);
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

    for (auto &node : fusedWith) {
        auto *convolutionNode = dynamic_cast<MKLDNNConvolutionNode *>(node.get());
        if (convolutionNode) {
            auto *convLayer = reinterpret_cast<ConvolutionLayer *>(convolutionNode->getCnnLayer().get());
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
    }

    if (this->getCnnLayer()->precision == Precision::I8) {
        MKLDNNMemoryDesc in_candidate = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::nhwc);
        MKLDNNMemoryDesc out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::nhwc);
        createDescriptor({in_candidate}, {out_candidate});
    } else {
        // If the weights aren't quantized, the only precision we support is FP32
        inputDataType = memory::f32;
        outputDataType = memory::f32;

        Layout layout = convLayer->input()->getLayout();

        if (layout == NCHW || layout == NHWC) {
            MKLDNNMemoryDesc in_candidate(getParentEdgeAt(0)->getDims(), inputDataType,
                    layout == NCHW ? memory::nchw : memory::nhwc);
            MKLDNNMemoryDesc out_candidate(getChildEdgeAt(0)->getDims(), outputDataType,
                    layout == NCHW ? memory::nchw : memory::nhwc);
            createDescriptor({in_candidate}, {out_candidate});

            if (IC == 3 || IC == 1) {
                out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::nChw16c);
                createDescriptor({in_candidate}, {out_candidate});
                out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::nChw8c);
                createDescriptor({in_candidate}, {out_candidate});
            } else {
                in_candidate = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::nChw16c);
                out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::nChw16c);
                createDescriptor({in_candidate}, {out_candidate});
                in_candidate = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::nChw8c);
                out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::nChw8c);
                createDescriptor({in_candidate}, {out_candidate});
            }
        } else if (layout == NCDHW || layout == NDHWC) {
            MKLDNNMemoryDesc in_candidate(getParentEdgeAt(0)->getDims(), inputDataType,
                    layout == NCDHW ? memory::ncdhw : memory::ndhwc);
            MKLDNNMemoryDesc out_candidate(getChildEdgeAt(0)->getDims(), outputDataType,
                    layout == NCDHW ? memory::ncdhw : memory::ndhwc);
            createDescriptor({in_candidate}, {out_candidate});

            if (IC == 3 || IC == 1) {
                out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::nCdhw16c);
                createDescriptor({in_candidate}, {out_candidate});
                out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::nCdhw8c);
                createDescriptor({in_candidate}, {out_candidate});
            } else {
                in_candidate = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::nCdhw16c);
                out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::nCdhw16c);
                createDescriptor({in_candidate}, {out_candidate});
                in_candidate = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::nCdhw8c);
                out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::nCdhw8c);
                createDescriptor({in_candidate}, {out_candidate});
            }
        }
    }
}

void MKLDNNConvolutionNode::setPostOps(mkldnn::primitive_attr &attr, bool initWeights = false) {
    int blob_idx = 0;
    mkldnn::post_ops ops;

    for (auto &node : fusedWith) {
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

        auto* activationNode = dynamic_cast<MKLDNNActivationNode *>(node.get());
        if (activationNode) {
            ops.append_eltwise(1.0, activationNode->getAlgorithm(), activationNode->getAlpha(),
                               activationNode->getBeta());
            continue;
        }

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

        auto* convolutionNode = dynamic_cast<MKLDNNConvolutionNode *>(node.get());
        if (convolutionNode) {
            auto* convLayer = reinterpret_cast<ConvolutionLayer*>(convolutionNode->getCnnLayer().get());

            auto weightsPrc = MKLDNNExtensionUtils::IEPrecisionToDataType(convLayer->precision);
            auto biasPrc = memory::data_type::s32;

            if (initWeights) {
                PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
                MKLDNNDims dwWeightsDims({dw_conv_oc, (ptrdiff_t)1, (ptrdiff_t)1, dw_conv_kernel[Y_AXIS], dw_conv_kernel[X_AXIS]});
                PostOpsIntBlobMemory[blob_idx]->Create(dwWeightsDims, weightsPrc, memory::format::Goihw8g);

                Blob::Ptr weights = convLayer->blobs.find("weights")->second;
                Blob::Ptr biases = convLayer->blobs.find("biases")->second;

                PostOpsIntBlobMemory[blob_idx]->SetData(weightsPrc, memory::goihw, weights->buffer(),
                                                        dwWeightsDims.size() * MKLDNNExtensionUtils::sizeOfDataType(weightsPrc));

                PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
                MKLDNNDims dwBiasesDims({dw_conv_oc});
                PostOpsIntBlobMemory[blob_idx + 1]->Create(dwBiasesDims, biasPrc, memory::format::x);
                PostOpsIntBlobMemory[blob_idx + 1]->SetData(biasPrc, memory::x, biases->buffer(),
                                                            dwBiasesDims.size() * MKLDNNExtensionUtils::sizeOfDataType(biasPrc));
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
                                   nullptr,
                                   nullptr);
            }

            if (convolutionNode->wScale != nullptr) {
                float* wScaleData = static_cast<float*>(convolutionNode->wScale->buffer());

                std::vector<float> oScaleDataVector;
                std::vector<float> oShiftDataVector;
                if (convolutionNode->getCnnLayer()->precision == Precision::I8 &&
                    convolutionNode->getCnnLayer()->outData[0]->getPrecision() != Precision::FP32) {
                    float *oScaleData = static_cast<float *>(convolutionNode->oScale->buffer());

                    for (size_t c = 0; c < convolutionNode->wScale->size(); c++) {
                        oScaleDataVector.push_back(wScaleData[c] / oScaleData[c]);
                        oShiftDataVector.push_back(0.f);
                    }
                } else {
                    for (size_t c = 0; c < convolutionNode->wScale->size(); c++) {
                        oScaleDataVector.push_back(wScaleData[c]);
                        oShiftDataVector.push_back(0.f);
                    }
                }

                MKLDNNDims oScaleDims({static_cast<ptrdiff_t>(rnd_up(biasesDims[0], 16))});

                PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
                PostOpsIntBlobMemory[blob_idx]->Create(oScaleDims, memory::data_type::f32, memory::format::x);
                PostOpsIntBlobMemory[blob_idx]->SetData(memory::data_type::f32, memory::x, &oScaleDataVector[0],
                                                        oScaleDataVector.size() * MKLDNNExtensionUtils::sizeOfDataType(memory::data_type::f32));

                PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
                PostOpsIntBlobMemory[blob_idx + 1]->Create(oScaleDims, memory::data_type::f32, memory::format::x);
                PostOpsIntBlobMemory[blob_idx + 1]->SetData(memory::data_type::f32, memory::x, &oShiftDataVector[0],
                                                            oShiftDataVector.size() * MKLDNNExtensionUtils::sizeOfDataType(memory::data_type::f32));

                ops.append_depthwise(depthwise_scale_shift,
                                     (const float *)PostOpsIntBlobMemory[blob_idx]->GetData(),
                                     (const float *)PostOpsIntBlobMemory[blob_idx + 1]->GetData());

                blob_idx += 2;
            }

            for (auto &dwConvFusedNode : convolutionNode->fusedWith) {
                auto* dwConvActivationNode = dynamic_cast<MKLDNNActivationNode *>(dwConvFusedNode.get());
                if (dwConvActivationNode) {
                    ops.append_eltwise(1.0, dwConvActivationNode->getAlgorithm(), dwConvActivationNode->getAlpha(),
                                       dwConvActivationNode->getBeta());

                    continue;
                }

                auto* dwConvDepthwiseNode = dynamic_cast<MKLDNNDepthwiseNode *>(dwConvFusedNode.get());
                if (dwConvDepthwiseNode) {
                    auto* depthwiseLayer = reinterpret_cast<WeightableLayer*>(dwConvDepthwiseNode->getCnnLayer().get());

                    if (initWeights) {
                        MKLDNNDims depthwiseDims({static_cast<ptrdiff_t>(rnd_up(biasesDims[0], 16))});

                        PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
                        PostOpsIntBlobMemory[blob_idx]->Create(depthwiseDims, memory::data_type::f32, memory::format::x);

                        PostOpsIntBlobMemory[blob_idx]->SetData(memory::data_type::f32, memory::x,
                                                                     depthwiseLayer->_weights->buffer(),
                                                                     depthwiseLayer->_weights->size() *
                                                                     MKLDNNExtensionUtils::sizeOfDataType(memory::data_type::f32));

                        if (dwConvDepthwiseNode->isBroadcast()) {
                            float broadcastValue = static_cast<float *>(PostOpsIntBlobMemory[blob_idx]->GetData())[0];
                            for (int i = 1; i < PostOpsIntBlobMemory[blob_idx]->GetPrimitiveDescriptor().desc().data.dims[0]; i++) {
                                static_cast<float *>(PostOpsIntBlobMemory[blob_idx]->GetData())[i] = broadcastValue;
                            }
                        }

                        if (dwConvDepthwiseNode->getAlgorithm() == depthwise_scale_shift) {
                            PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
                            PostOpsIntBlobMemory[blob_idx + 1]->Create(depthwiseDims, memory::data_type::f32,
                                                                        memory::format::x);
                            PostOpsIntBlobMemory[blob_idx + 1]->SetData(memory::data_type::f32, memory::x,
                                                                         depthwiseLayer->_biases->buffer(),
                                                                         depthwiseLayer->_biases->size() *
                                                                         MKLDNNExtensionUtils::sizeOfDataType(memory::data_type::f32));

                            if (dwConvDepthwiseNode->isBroadcast()) {
                                float broadcastValue = static_cast<float *>(PostOpsIntBlobMemory[blob_idx + 1]->GetData())[0];
                                for (int i = 1; i < PostOpsIntBlobMemory[blob_idx + 1]->GetPrimitiveDescriptor().desc().data.dims[0]; i++) {
                                    static_cast<float *>(PostOpsIntBlobMemory[blob_idx + 1]->GetData())[i] = broadcastValue;
                                }
                            }

                            ops.append_depthwise(dwConvDepthwiseNode->getAlgorithm(),
                                                 (const float *) PostOpsIntBlobMemory[blob_idx]->GetData(),
                                                 (const float *) PostOpsIntBlobMemory[blob_idx + 1]->GetData());

                            blob_idx += 2;
                        } else {
                            ops.append_depthwise(dwConvDepthwiseNode->getAlgorithm(),
                                                 (const float *) PostOpsIntBlobMemory[blob_idx]->GetData(),
                                                 nullptr);

                            blob_idx += 1;
                        }
                    } else {
                        ops.append_depthwise(dwConvDepthwiseNode->getAlgorithm(),
                                             nullptr,
                                             nullptr);
                    }

                    continue;
                }
            }

            continue;
        }
    }

    attr.set_post_ops(ops);
}

void MKLDNNConvolutionNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    mkldnn::primitive_attr attr;
    setPostOps(attr);

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

                    if (withSum) {
                        dataConfig.inPlace = -1;
                        config.inConfs.push_back(dataConfig);
                    }

                    outFormats.emplace_back(static_cast<memory::format>(itpd.dst_primitive_desc().desc().data.format));
                }
                impl_desc_type impl_type = parse_impl_name(itpd.get_impl_info_str());

                supportedPrimitiveDescriptors.emplace_back(config, impl_type, outFormats);
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

    mkldnn::primitive_attr attr;
    setPostOps(attr, true);
    addScaleToPrimitiveAttr(attr);

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
    return getType() == Convolution;
}

void MKLDNNConvolutionNode::createDescriptor(const std::vector<InferenceEngine::TensorDesc> &inputDesc,
                                             const std::vector<InferenceEngine::TensorDesc> &outputDesc) {
    TensorDesc inDesc = inputDesc[0], outDesc = outputDesc[0];
    mkldnn::memory::data_type wdt = MKLDNNExtensionUtils::IEPrecisionToDataType(inDesc.getPrecision());
    mkldnn::memory::data_type bdt = MKLDNNExtensionUtils::IEPrecisionToDataType(inDesc.getPrecision());

    Blob::Ptr weights = this->getCnnLayer()->blobs.find("weights")->second;

    if (weights->getTensorDesc().getPrecision() == Precision::I8) {
        wdt = memory::s8;
        bdt = memory::s32;

        Precision outPrec;
        if (getCnnLayer()->outData[0]->getPrecision() == Precision::FP32) {
            outPrec = Precision::FP32;
        } else {
            // define precision accordninly normalizer
            // TODO(amalyshe) do we need to have separate flow for last in int8 chain or not?
            outPrec = outDesc.getPrecision();
        }

        inDesc = TensorDesc(inDesc.getPrecision() , inputDesc[0].getDims(), inputDesc[0].getBlockingDesc());
        outDesc = TensorDesc(outPrec, outputDesc[0].getDims(), outputDesc[0].getBlockingDesc());
    }

    MKLDNNMemoryDesc in_candidate(inDesc);
    MKLDNNMemoryDesc out_candidate(outDesc);

    auto in_fmt = in_candidate.getFormat();
    auto out_fmt = out_candidate.getFormat();

    int O_IND = (isGrouped || isMerged) ? 1 : 0;
    int I_IND = (isGrouped || isMerged) ? 2 : 1;

    // grouping and autoblocking is not compatible
    if (((isGrouped && !isDW) || isMerged) && (in_candidate.blocksExtended() || out_candidate.blocksExtended()))
        return;

    MKLDNNDims blocked_weightDims(weightDims);
    MKLDNNDims blocked_biasesDims(biasesDims);
    MKLDNNMemoryDesc wgh_candidate{blocked_weightDims, wdt, memory::any};

    for (auto alg : {algorithm::convolution_winograd, algorithm::convolution_direct}) {
        std::shared_ptr<mkldnn::convolution_forward::desc> conv_desc;
        if (withBiases) {
            MKLDNNMemoryDesc bias_candidate{blocked_biasesDims, bdt, memory::any};

            conv_desc.reset(new convolution_forward::desc(prop_kind::forward_scoring, alg,
                                                          in_candidate, wgh_candidate, bias_candidate, out_candidate,
                                                          stride, dilation, paddingL, paddingR, padding_kind::zero));
        } else {
            conv_desc.reset(new convolution_forward::desc(prop_kind::forward_scoring, alg,
                                                          in_candidate, wgh_candidate, out_candidate, stride, dilation,
                                                          paddingL, paddingR, padding_kind::zero));
        }

        descs.emplace_back(conv_desc);
    }
}

void MKLDNNConvolutionNode::addScaleToPrimitiveAttr(mkldnn::primitive_attr attr) const {
    bool scaled = false;
    if (wScale != nullptr) {
       float* wScaleData = static_cast<float*>(wScale->buffer());

       std::vector<float> oScaleDataVector;
       if (getCnnLayer()->precision == Precision::I8 && getCnnLayer()->outData[0]->getPrecision() != Precision::FP32) {
           float *oScaleData = static_cast<float *>(oScale->buffer());

           for (size_t c = 0; c < wScale->size(); c++) {
               oScaleDataVector.push_back(wScaleData[c] / oScaleData[c]);
           }
       } else {
           for (size_t c = 0; c < wScale->size(); c++) {
               oScaleDataVector.push_back(wScaleData[c]);
           }
       }

       attr.set_int_output_round_mode(mkldnn::round_nearest);
       attr.set_output_scales(1 << 1 /*through C dim*/, oScaleDataVector);
    }
}

void MKLDNNConvolutionNode::initDescriptor(const InferenceEngine::LayerConfig& config) {
    auto* selectedPD = getSelectedPrimitiveDescriptor();
    if (!selectedPD) {
        return;
    }
    bool addedNewDesc = false;
    /*if (config.inConfs[0].desc.getPrecision() == InferenceEngine::Precision::FP32 &&
            config.outConfs[0].desc.getPrecision() == InferenceEngine::Precision::FP32) {*/
        addedNewDesc = true;
        createDescriptor({config.inConfs[0].desc}, {config.outConfs[0].desc});
    //}

    mkldnn::primitive_attr attr;
    setPostOps(attr);
    addScaleToPrimitiveAttr(attr);

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
