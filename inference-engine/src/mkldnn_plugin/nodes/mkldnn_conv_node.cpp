// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_conv_node.h"
#include "mkldnn_reorder_node.h"
#include "mkldnn_input_node.h"
#include "mkldnn_activation_node.h"
#include "desc_iterator.hpp"
#include "mkldnn_eltwise_node.h"
#include "mkldnn_depthwise_node.h"
#include "mkldnn_quantize_node.h"
#include "mkldnn_pooling_node.h"
#include "mkldnn_concat_node.h"
#include <ie_layers.h>
#include <string>
#include <vector>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <ie_layers_internal.hpp>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNConvolutionNode::MKLDNNConvolutionNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache), withBiases(false), withSum(false), withDWConv(false), isDW(false), isMerged(false),
          isGrouped(false), dw_conv_oc(0), dw_conv_ih(0), dw_conv_iw(0), dw_conv_in_dt(memory::data_type::data_undef),
          groupNum(1lu), baseInputsNumber(1), eltwisePrecision(Precision::FP32) {
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

    if (getCnnLayer()->type == "Convolution") {
        baseInputsNumber = getCnnLayer().get()->insData.size();
    }
}

mkldnn::memory::data_type MKLDNNConvolutionNode::precisionToDataType(InferenceEngine::Precision prec) {
    // MKLDNN Plugin doesn't support U16 layout so upcast to FP32 in this case
    if (prec == Precision::U16)
        prec = Precision::FP32;

    return MKLDNNExtensionUtils::IEPrecisionToDataType(prec);
}

bool MKLDNNConvolutionNode::canBeExecutedInInt8() {
    auto * convLayer = dynamic_cast<ConvolutionLayer*>(getCnnLayer().get());
    if (convLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert convolution layer.";

    if (baseInputsNumber > 1) {
        auto inputDataType = precisionToDataType(getCnnLayer()->insData[0].lock()->getPrecision());
        if (!inputZeroPoints.empty())
            inputDataType = memory::u8;

        auto weightsDataType = precisionToDataType(Precision::FP32);
        if (baseInputsNumber > 1) {
            weightsDataType = precisionToDataType(getCnnLayer()->insData[1].lock()->getPrecision());
            if (!weightsZeroPoints.empty())
                weightsDataType = memory::s8;
        }

        return (inputDataType == mkldnn_s8 || inputDataType == mkldnn_u8) && weightsDataType == mkldnn_s8;
    } else {
        return this->getCnnLayer()->precision == Precision::I8;
    }
}

InferenceEngine::Precision MKLDNNConvolutionNode::fusedEltwisePrecision(MKLDNNEltwiseNode *eltwiseNode, int findex) {
    InferenceEngine::Precision eltwisePrecision;
    auto parent0 = eltwiseNode->getCnnLayer()->insData[0].lock()->getCreatorLayer().lock();
    auto parent1 = eltwiseNode->getCnnLayer()->insData[1].lock()->getCreatorLayer().lock();

    auto fusedParent = findex != 0 ? fusedWith[findex - 1].get()->getCnnLayer() : this->getCnnLayer();
    eltwisePrecision = fusedParent == parent0 ? eltwiseNode->getCnnLayer()->insData[1].lock()->getPrecision() :
        eltwiseNode->getCnnLayer()->insData[0].lock()->getPrecision();
    return eltwisePrecision;
}

void MKLDNNConvolutionNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    auto * convLayer = dynamic_cast<ConvolutionLayer*>(getCnnLayer().get());
    if (convLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert convolution layer.";

    auto inputDataType = precisionToDataType(getCnnLayer()->insData[0].lock()->getPrecision());
    if (!inputZeroPoints.empty())
        inputDataType = memory::u8;

    auto outputDataType = precisionToDataType(getCnnLayer()->outData[0]->getPrecision());
    eltwisePrecision = MKLDNNExtensionUtils::DataTypeToIEPrecision(outputDataType);
    if (baseInputsNumber > 1) {
        if (!fusedWith.empty()) {
            auto lastFusedLayer = fusedWith[fusedWith.size() - 1].get()->getCnnLayer();
            if (lastFusedLayer) {
                outputDataType = precisionToDataType(lastFusedLayer->outData[0]->getPrecision());
                eltwisePrecision = MKLDNNExtensionUtils::DataTypeToIEPrecision(outputDataType);
            }
        }

        // We need to make sure that convolution output and second input of fused Eltwise operation
        // have equal precision sizes since they use the same physical memory. In case precisions are different we upscale to FP32.
        if (outputDataType != memory::f32 && outputDataType != memory::bf16 && isFusedWith(Eltwise)) {
            for (int i = 0; i < fusedWith.size(); i++) {
                auto *eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(fusedWith[i].get());
                if (eltwiseNode) {
                    eltwisePrecision = fusedEltwisePrecision(eltwiseNode, i);
                    if (MKLDNNExtensionUtils::DataTypeToIEPrecision(outputDataType).size() != eltwisePrecision.size()) {
                        eltwisePrecision = Precision::FP32;
                        outputDataType = memory::f32;
                    }
                    break;
                }
            }
        }
    }

    int expectedInputEdgesNum = baseInputsNumber + isFusedWith(Eltwise);
    for (int i = 0; i < fusedWith.size(); i++) {
        auto *convolutionNode = dynamic_cast<MKLDNNConvolutionNode *>(fusedWith[i].get());
        if (convolutionNode) {
            expectedInputEdgesNum += convolutionNode->getBaseIntputsNumber() - 1;
        }
    }

    if (getParentEdges().size() != expectedInputEdgesNum)
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
    groupNum = convLayer->_group;
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

    withBiases = (convLayer->_biases != nullptr && convLayer->_biases->size() != 0) || baseInputsNumber == 3;

    if (baseInputsNumber == 1) {
        internalBlobs.push_back(createInternalBlob(weightDims, true, isGrouped));

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
    }

    invertVectorCopyUtoI(convLayer->_stride, stride);
    for (int i = 1; i <= convLayer->_dilation.size(); i++) {
        dilation.push_back(static_cast<int>(convLayer->_dilation[convLayer->_dilation.size() - i]) - 1);
    }

    auto allPads = getPaddings(*convLayer);
    invertVectorCopyUtoI(allPads.begin, paddingL);
    invertVectorCopyUtoI(allPads.end, paddingR);

    MKLDNNDims weightsDims = MKLDNNDims(weightDims);

    withSum = isFusedWith(Eltwise);
    withDWConv = isFusedWith(Convolution);

    for (int i = 0; i < fusedWith.size(); i++) {
        auto *convolutionNode = dynamic_cast<MKLDNNConvolutionNode *>(fusedWith[i].get());
        if (convolutionNode) {
            auto *convLayer = reinterpret_cast<ConvolutionLayer *>(convolutionNode->getCnnLayer().get());
            dw_conv_ih = convolutionNode->inDims[0][convolutionNode->inDims[0].ndims() - 2];
            dw_conv_iw = convolutionNode->inDims[0][convolutionNode->inDims[0].ndims() - 1];
            dw_conv_oc = convLayer->_out_depth;
            for (int j = 0; j < convLayer->_kernel.size(); j++) {
                dw_conv_kernel.push_back(convLayer->_kernel[j]);
            }
            for (int j = 0; j < convLayer->_stride.size(); j++) {
                dw_conv_strides.push_back(convLayer->_stride[j]);
            }

            if (canBeExecutedInInt8()) {
                if (i == 0) {
                    dw_conv_in_dt = precisionToDataType(getCnnLayer()->outData[0]->getPrecision());
                } else {
                    dw_conv_in_dt = precisionToDataType(fusedWith[i - 1].get()->getCnnLayer()->outData[0]->getPrecision());
                }
            } else {
                dw_conv_in_dt = memory::f32;
            }

            for (int j = 0; j < paddingR.size(); j++) {
                int with_group = (isGrouped || isMerged) ? 1 : 0;
                int krn = weightsDims[with_group + 2 + j];
                int src = getParentEdgeAt(0)->getDims()[2 + j];
                int dst = getChildEdgeAt(0)->getDims()[2 + j];

                krn = (krn - 1)*(dilation[j] + 1) + 1;
                int calc_dst = (src - krn + paddingL[j]) / stride[j] + 1;
                paddingR[j] = (dst - calc_dst) * stride[j];
            }
        }
    }

    MKLDNNMemoryDesc in_candidate, out_candidate;
    if (canBeExecutedInInt8()) {
        in_candidate = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType,
                getParentEdgeAt(0)->getDims().ndims() == 5 ? memory::ndhwc : memory::nhwc);
        out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType,
                getParentEdgeAt(0)->getDims().ndims() == 5 ? memory::ndhwc : memory::nhwc);
        createDescriptor({in_candidate}, {out_candidate});
    } else {
        inputDataType = (convLayer->input()->getPrecision() == Precision::BF16
        && !(isGrouped && getParentEdgeAt(0)->getDims().ndims() == 5)) ? memory::bf16 : memory::f32;
        outputDataType = (convLayer->outData[0]->getPrecision() == Precision::BF16
        && !(isGrouped && getParentEdgeAt(0)->getDims().ndims() == 5)) ? memory::bf16 : memory::f32;
        eltwisePrecision = Precision::FP32;
        for (int i = 0; i < fusedWith.size(); i++) {
            auto *eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(fusedWith[i].get());
            if (eltwiseNode) {
                eltwisePrecision = fusedEltwisePrecision(eltwiseNode, i);
                // TODO(amalyshe): there might be situation when convolution can be executed in BF16,
                // output is required in FP32 but eltwise inplace tensor would be in BF16
                // currently we forcedly change output to the BF16 that will add reoreder after the node
                // Another situation can be when we mark output as FP32 and Eltwise asPrecison (which stand
                // for input of inplace tensor precision) to FP32. This will add reorder for that in-place tensor
                // bofore the fused convolution. This behaviour might be more correct regarding expected markup
                // of the graph but performance of first and second approaches might be different. Need to verify
                outputDataType = eltwisePrecision == Precision::BF16 ? memory::bf16 : memory::f32;
            }
        }
        // correction for cases of FP32 input - we do not have FP32 convolution supported BF16 output
        if (inputDataType == memory::f32
            && (outputDataType == memory::bf16 || eltwisePrecision == Precision::BF16)) {
            outputDataType = memory::f32;
            eltwisePrecision = Precision::FP32;
        }

        Layout layout = convLayer->input()->getLayout();

        if (layout == NCHW || layout == NHWC) {
            if (IC == 3 || IC == 1) {
                in_candidate = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType,
                                                layout == NCHW ? memory::nchw : memory::nhwc);
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

            in_candidate = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType,
                    layout == NCHW ? memory::nchw : memory::nhwc);
            out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType,
                    layout == NCHW ? memory::nchw : memory::nhwc);
            createDescriptor({in_candidate}, {out_candidate});
        } else if (layout == NCDHW || layout == NDHWC) {
            in_candidate = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType,
                    layout == NCDHW ? memory::ncdhw : memory::ndhwc);
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

            in_candidate = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType,
                    layout == NCDHW ? memory::ncdhw : memory::ndhwc);
            out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType,
                    layout == NCDHW ? memory::ncdhw : memory::ndhwc);
            createDescriptor({in_candidate}, {out_candidate});
        }
    }
}

void MKLDNNConvolutionNode::setPostOps(mkldnn::primitive_attr &attr, bool initWeights = false) {
    int blob_idx = 0;
    mkldnn::post_ops ops;

    for (auto &node : fusedWith) {
        if (node->getType() == Split || node->getType() == Concatenation)
            continue;

#if defined (COMPILED_CPU_MKLDNN_ELTWISE_NODE)
        auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(node.get());
        if (eltwiseNode) {
            if (eltwiseNode->getCnnLayer()->precision == Precision::I8) {
                auto it = eltwiseNode->getCnnLayer()->blobs.find("eltwise-sum-scale");
                if (it != eltwiseNode->getCnnLayer()->blobs.end()) {
                    // currently there is the only one scale while we need scale by channel :(
                    ops.append_sum(it->second->buffer().as<float*>()[0], mkldnn::memory::convert_to_c(precisionToDataType(eltwisePrecision)));
                }
            } else {
                ops.append_sum(1.0, mkldnn::memory::convert_to_c(precisionToDataType(eltwisePrecision)));
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
                PostOpsIntBlobMemory[blob_idx]->FillZero();
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
                    PostOpsIntBlobMemory[blob_idx + 1]->FillZero();
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

        auto* quantizeNode = dynamic_cast<MKLDNNQuantizeNode *>(node.get());
        if (quantizeNode) {
            quantizeNode->appendPostOps(ops);
            continue;
        }

        auto* convolutionNode = dynamic_cast<MKLDNNConvolutionNode *>(node.get());
        if (convolutionNode) {
            if (initWeights) {
                if (convolutionNode->getBaseIntputsNumber() == 1) {
                    auto* convLayer = reinterpret_cast<ConvolutionLayer*>(convolutionNode->getCnnLayer().get());

                    auto weightsPrc = precisionToDataType(convLayer->precision);
                    auto biasPrc = memory::data_type::s32;

                    PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
                    MKLDNNDims dwWeightsDims({dw_conv_oc, (ptrdiff_t)1, (ptrdiff_t)1, dw_conv_kernel[Y_AXIS], dw_conv_kernel[X_AXIS]});
                    PostOpsIntBlobMemory[blob_idx]->Create(dwWeightsDims, weightsPrc, memory::format::Goihw8g);
                    PostOpsIntBlobMemory[blob_idx]->FillZero();

                    Blob::Ptr weights = convLayer->blobs.find("weights")->second;
                    Blob::Ptr biases = convLayer->blobs.find("biases")->second;

                    PostOpsIntBlobMemory[blob_idx]->SetData(weightsPrc, memory::goihw, weights->buffer(),
                                                            dwWeightsDims.size() * MKLDNNExtensionUtils::sizeOfDataType(weightsPrc));

                    PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
                    MKLDNNDims dwBiasesDims({dw_conv_oc});
                    PostOpsIntBlobMemory[blob_idx + 1]->Create(dwBiasesDims, biasPrc, memory::format::x);
                    PostOpsIntBlobMemory[blob_idx + 1]->FillZero();
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
                                       static_cast<float *>(getParentEdgeAt(
                                               baseInputsNumber + 0)->getMemory().GetData()),
                                       static_cast<float *>(getParentEdgeAt(
                                               baseInputsNumber + 1)->getMemory().GetData()));
                }
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
                PostOpsIntBlobMemory[blob_idx]->FillZero();
                PostOpsIntBlobMemory[blob_idx]->SetData(memory::data_type::f32, memory::x, &oScaleDataVector[0],
                                                        oScaleDataVector.size() * MKLDNNExtensionUtils::sizeOfDataType(memory::data_type::f32));

                PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
                PostOpsIntBlobMemory[blob_idx + 1]->Create(oScaleDims, memory::data_type::f32, memory::format::x);
                PostOpsIntBlobMemory[blob_idx + 1]->FillZero();
                PostOpsIntBlobMemory[blob_idx + 1]->SetData(memory::data_type::f32, memory::x, &oShiftDataVector[0],
                                                            oShiftDataVector.size() * MKLDNNExtensionUtils::sizeOfDataType(memory::data_type::f32));

                ops.append_depthwise(depthwise_scale_shift,
                                     (const float *)PostOpsIntBlobMemory[blob_idx]->GetData(),
                                     (const float *)PostOpsIntBlobMemory[blob_idx + 1]->GetData());

                blob_idx += 2;
            }

            continue;
        }

        THROW_IE_EXCEPTION << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType()) << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

void MKLDNNConvolutionNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    mkldnn::primitive_attr attr;
    addZeroPoints(attr);
    setPostOps(attr);

    bool containJitImpl = false;

    for (auto& desc : descs) {
        if (containJitImpl && isPossibleToSkipInitConfig(desc))
            continue;
        auto itpd = desc.createPrimitiveDescriptorIterator(getEngine(), attr);
        while (itpd.is_not_end()) {
            InferenceEngine::LayerConfig config;
            config.dynBatchSupport = true;
            for (size_t i = 0; i < descInputNumbers(desc); i++) {
                InferenceEngine::DataConfig dataConfig;
                dataConfig.inPlace = -1;
                dataConfig.constant = false;
                dataConfig.desc = getSrcMemDesc(itpd, i);
                if (!isGrouped)
                    dataConfig.desc = MKLDNNExtensionUtils::getUninitTensorDesc(dataConfig.desc);
                config.inConfs.push_back(dataConfig);
            }

            if (withDWConv && baseInputsNumber > 1) {
                auto weightsPrc = precisionToDataType(dw_conv_in_dt == mkldnn_u8 ? Precision::I8 : Precision::FP32);
                auto biasPrc = memory::data_type::f32;

                MKLDNNDims dwWeightsDims({dw_conv_oc, (ptrdiff_t)1, (ptrdiff_t)1, dw_conv_kernel[Y_AXIS], dw_conv_kernel[X_AXIS]});
                MKLDNNDims dwBiasesDims({dw_conv_oc});

                InferenceEngine::DataConfig dataConfig;
                dataConfig.inPlace = -1;
                dataConfig.constant = false;
                dataConfig.desc = MKLDNNMemoryDesc(dwWeightsDims, weightsPrc, memory::format::Goihw8g);
                config.inConfs.push_back(dataConfig);

                dataConfig.desc = MKLDNNMemoryDesc(dwBiasesDims, biasPrc, memory::format::x);
                config.inConfs.push_back(dataConfig);
            }

            std::vector<memory::format> outFormats;
            for (size_t i = 0; i < descOutputNumbers(desc); i++) {
                InferenceEngine::DataConfig dataConfig;
                if (withSum) {
                    dataConfig.inPlace = getParentEdges().size() - 1;
                }

                dataConfig.constant = false;
                dataConfig.desc = getDstMemDesc(itpd, i);
                if (!(isGrouped || isMerged))
                    dataConfig.desc = MKLDNNExtensionUtils::getUninitTensorDesc(dataConfig.desc);
                config.outConfs.push_back(dataConfig);

                if (withSum) {
                    dataConfig.inPlace = -1;
                    dataConfig.desc.setPrecision(eltwisePrecision);
                    config.inConfs.push_back(dataConfig);
                }

                outFormats.emplace_back(static_cast<memory::format>(itpd.dst_primitive_desc().desc().data.format));
            }
            impl_desc_type impl_type = parse_impl_name(itpd.get_impl_info_str());
            if (impl_type & jit)
                containJitImpl = true;

            supportedPrimitiveDescriptors.emplace_back(config, impl_type, outFormats);
            itpd++;
        }
    }
}


void MKLDNNConvolutionNode::createPrimitive() {
    if (prim)
        return;

    mkldnn::primitive_attr attr;
    addZeroPoints(attr);
    setPostOps(attr, true);
    addScaleToPrimitiveAttr(attr);

    auto prim_desc = createPrimitiveDescriptor<convolution_forward::primitive_desc,
            convolution_forward::desc>(attr);

    if (withBiases) {
        prim.reset(new convolution_forward(prim_desc,
                                           getParentEdgeAt(0)->getMemory().GetPrimitive(),
                                           getWeights(),
                                           getBias(),
                                           getChildEdgeAt(0)->getMemory().GetPrimitive()));
    } else {
        prim.reset(new convolution_forward(prim_desc,
                                           getParentEdgeAt(0)->getMemory().GetPrimitive(),
                                           getWeights(),
                                           getChildEdgeAt(0)->getMemory().GetPrimitive()));
    }
}

bool MKLDNNConvolutionNode::created() const {
    return getType() == Convolution;
}

void MKLDNNConvolutionNode::createDescriptor(const std::vector<InferenceEngine::TensorDesc> &inputDesc,
                                             const std::vector<InferenceEngine::TensorDesc> &outputDesc) {
    TensorDesc inDesc = inputDesc[0], outDesc = outputDesc[0];

    mkldnn::memory::data_type wdt = precisionToDataType(inDesc.getPrecision());
    mkldnn::memory::data_type bdt = precisionToDataType(inDesc.getPrecision());
    if (inDesc.getPrecision() == Precision::BF16) {
        bdt = mkldnn::memory::data_type::f32;
    }

    if (inDesc.getPrecision() == Precision::U8 || inDesc.getPrecision() == Precision::I8) {
        wdt = memory::s8;
        bdt = baseInputsNumber == 3 ? precisionToDataType(getCnnLayer()->insData[2].lock()->getPrecision()) : memory::s32;
    }

    if (baseInputsNumber == 1) {
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

            inDesc = TensorDesc(inDesc.getPrecision(), inputDesc[0].getDims(), inputDesc[0].getBlockingDesc());
            outDesc = TensorDesc(outPrec, outputDesc[0].getDims(), outputDesc[0].getBlockingDesc());
        }
    }

    MKLDNNMemoryDesc in_candidate(inDesc);
    MKLDNNMemoryDesc out_candidate(outDesc);

    // grouping and autoblocking is not compatible
    if (((isGrouped && !isDW) || isMerged) && (in_candidate.blocksExtended() || out_candidate.blocksExtended()))
        return;

    MKLDNNDims blocked_weightDims(weightDims);
    MKLDNNDims blocked_biasesDims(biasesDims);
    MKLDNNMemoryDesc wgh_candidate{blocked_weightDims, wdt, memory::any};

    std::vector<algorithm> algorithms;
    // We cannot map wino_format on tensor descriptor for now
    if (getBaseIntputsNumber() == 1) {
        algorithms.push_back(algorithm::convolution_winograd);
    }
    algorithms.push_back(algorithm::convolution_direct);

    for (auto alg : algorithms) {
        try {
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
        } catch (...) {
            THROW_IE_EXCEPTION << "Cannot create convolution forward descriptor for layer: " << getName();
        }
    }
}

void MKLDNNConvolutionNode::addZeroPoints(mkldnn::primitive_attr& attr) const {
    if (!inputZeroPoints.empty())
        attr.set_input_zero_points(1 << 1 /*through C dim*/, inputZeroPoints);

    if (!weightsZeroPoints.empty())
        attr.set_weights_zero_points(1 << 1 /*through C dim*/, weightsZeroPoints);

    if (!outputCompensation.empty())
        attr.set_output_compensations(1 << 1 /*through C dim*/, outputCompensation);
}

void MKLDNNConvolutionNode::addScaleToPrimitiveAttr(mkldnn::primitive_attr attr) const {
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

    // Strided blobs feature support.
    // Works only for FP32 convolutions for now.
    bool isStridedBlobsSupported = true;
    for (auto &insData : getCnnLayer()->insData) {
        if (insData.lock()->getPrecision() != InferenceEngine::Precision::FP32
            && insData.lock()->getPrecision() != InferenceEngine::Precision::BF16) {
            isStridedBlobsSupported = false;
            break;
        }
    }

    // TODO: fix strided blobs feature support for dynamic weights
    if (baseInputsNumber != 1) {
        isStridedBlobsSupported = false;
    }

    if (isStridedBlobsSupported) {
        createDescriptor({config.inConfs[0].desc}, {config.outConfs[0].desc});
    }

    mkldnn::primitive_attr attr;
    addZeroPoints(attr);
    setPostOps(attr);
    addScaleToPrimitiveAttr(attr);

    InferenceEngine::LayerConfig rightConfig = selectedPD->getConfig();
    size_t selected_count = 0;

    bool containJitImpl = false;

    for (size_t i = 0; i < descs.size(); i++) {
        auto& desc = descs[i];
        if (containJitImpl && isPossibleToSkipInitConfig(desc))
            continue;
        auto itpd = desc.createPrimitiveDescriptorIterator(getEngine(), attr);
        while (itpd.is_not_end()) {
            InferenceEngine::LayerConfig cfg;
            cfg.dynBatchSupport = true;
            for (size_t j = 0; j < descInputNumbers(desc); j++) {
                InferenceEngine::DataConfig dataConfig;
                dataConfig.inPlace = -1;
                dataConfig.constant = false;
                dataConfig.desc = getSrcMemDesc(itpd, j);
                cfg.inConfs.push_back(dataConfig);
            }

            if (withDWConv && baseInputsNumber > 1) {
                auto weightsPrc = precisionToDataType(dw_conv_in_dt == mkldnn_u8 ? Precision::I8 : Precision::FP32);
                auto biasPrc = memory::data_type::f32;

                MKLDNNDims dwWeightsDims({dw_conv_oc, (ptrdiff_t)1, (ptrdiff_t)1, dw_conv_kernel[Y_AXIS], dw_conv_kernel[X_AXIS]});
                MKLDNNDims dwBiasesDims({dw_conv_oc});

                InferenceEngine::DataConfig dataConfig;
                dataConfig.inPlace = -1;
                dataConfig.constant = false;
                dataConfig.desc = MKLDNNMemoryDesc(dwWeightsDims, weightsPrc, memory::format::Goihw8g);
                cfg.inConfs.push_back(dataConfig);

                dataConfig.desc = MKLDNNMemoryDesc(dwBiasesDims, biasPrc, memory::format::x);
                cfg.inConfs.push_back(dataConfig);
            }

            for (size_t j = 0; j < descOutputNumbers(desc); j++) {
                InferenceEngine::DataConfig dataConfig;
                dataConfig.inPlace = -1;
                dataConfig.constant = false;
                dataConfig.desc = getDstMemDesc(itpd, j);
                if (withSum) {
                    auto eltwiseConfig = dataConfig;
                    eltwiseConfig.desc.setPrecision(eltwisePrecision);
                    cfg.inConfs.push_back(eltwiseConfig);
                    dataConfig.inPlace = getParentEdges().size() - 1;
                }

                cfg.outConfs.push_back(dataConfig);
            }
            impl_desc_type impl_type = parse_impl_name(itpd.get_impl_info_str());
            if (impl_type & jit)
                containJitImpl = true;

            if (selected_count == selectedPrimitiveDescriptorIndex) {
                if (impl_type != selectedPD->getImplementationType()) {
                    THROW_IE_EXCEPTION << "Cannot get the original layer configuration!";
                }
                rightConfig = cfg;
            }
            if (i == descs.size() - 1 && isStridedBlobsSupported) {
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

void MKLDNNConvolutionNode::filterSupportedPrimitiveDescriptors() {
    MKLDNNNode::filterSupportedPrimitiveDescriptors();
    // We also need to filter descs in Convolution node
    filterSupportedDescriptors();
}

void MKLDNNConvolutionNode::filterSupportedDescriptors() {
    if (!inputMemoryFormatsFilter.empty() || !outputMemoryFormatsFilter.empty()) {
        if (inputMemoryFormatsFilter.size() > 1 || outputMemoryFormatsFilter.size() > 1) {
            THROW_IE_EXCEPTION << "Incorrect number of input or output memory formats for Convolution node";
        }
        auto itd = descs.begin();
        while (itd != descs.end()) {
            bool isSuitableDesc = true;
            if (!inputMemoryFormatsFilter.empty()) {
                auto src_fmt = std::shared_ptr<mkldnn::convolution_forward::desc>(*itd)->data.src_desc.format;
                if (src_fmt != inputMemoryFormatsFilter[0])
                    isSuitableDesc = false;
            }
            if (!outputMemoryFormatsFilter.empty()) {
                auto dst_fmt = std::shared_ptr<mkldnn::convolution_forward::desc>(*itd)->data.dst_desc.format;
                if (dst_fmt != outputMemoryFormatsFilter[0])
                    isSuitableDesc = false;
            }
            if (!isSuitableDesc) {
                itd = descs.erase(itd);
            } else {
                itd++;
            }
        }
    }
}

bool MKLDNNConvolutionNode::isPossibleToSkipInitConfig(MKLDNNDescriptor &desc) {
    //  WA: In some cases, we can predict in advance the type of primitive that will be called in the future.
    //  In particular, isPossibleToSkipInitConfig() checks whether we can skip the creation of primitives with
    //  gemm implementation, which significantly increase the network load time.
    if (!inputMemoryFormatsFilter.empty() || !outputMemoryFormatsFilter.empty())
        return false;

    if (getCnnLayer()->params.find("PrimitivesPriority") != getCnnLayer()->params.end())
        return false;

    //  Here we check that we will not delete jit_planar_conv primitive by mistake.
    //  It requires:
    //      1) strides equal 1;
    //      2) not grouped;
    //      3) first dim of weights is not 1.
    bool isPossibleJitPlanar = true;
    if (isGrouped || weightDims[0] != 1)
        isPossibleJitPlanar = false;
    for (int i = 0; i < stride.size(); i++)
        if (stride[i] != 1)
            isPossibleJitPlanar = false;

    std::shared_ptr<mkldnn::convolution_forward::desc> convDesc(desc);
    auto srcMemFmt = convDesc->data.src_desc.format;
    auto dstMemFmt = convDesc->data.dst_desc.format;
    auto srcDataType = convDesc->data.src_desc.data_type;
    auto dstDataType = convDesc->data.dst_desc.data_type;
    bool isPlanarFloatConv = (srcMemFmt == memory::nchw || srcMemFmt == memory::ncdhw)
                             && (dstMemFmt == memory::nchw || dstMemFmt == memory::ncdhw)
                             && srcDataType == memory::f32
                             && dstDataType == memory::f32;

    return !isPossibleJitPlanar && isPlanarFloatConv;
}

MKLDNNMemoryDesc MKLDNNConvolutionNode::getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    InferenceEngine::TensorDesc desc = idx > 0 ? MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(idx - 1).desc())
                                               : MKLDNNMemoryDesc(primitive_desc_it.src_primitive_desc(idx).desc());

    if (desc.getLayout() == InferenceEngine::Layout::ANY) {
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getParentEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getLayout()));
    } else {
        if (getParentEdgeAt(idx)->getDims().ToSizeVector().size() != *std::max_element(desc.getBlockingDesc().getOrder().begin(),
                                                                                       desc.getBlockingDesc().getOrder().end()) + 1) {
            auto old_dims = getParentEdgeAt(idx)->getDims().ToSizeVector();
            auto new_dims = InferenceEngine::SizeVector({groupNum, div_up(old_dims[0], groupNum)});
            for (int i = 1; i < old_dims.size(); i++) {
                new_dims.push_back(old_dims[i]);
            }

            auto td = InferenceEngine::TensorDesc(desc.getPrecision(),
                                                  new_dims,
                                                  desc.getBlockingDesc());
            if (new_dims.size() == desc.getBlockingDesc().getBlockDims().size()) {
                td.setLayout(BLOCKED);
            }
            return MKLDNNMemoryDesc(td);
        } else {
            return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                                getParentEdgeAt(idx)->getDims().ToSizeVector(),
                                                                desc.getBlockingDesc()));
        }
    }
}

const mkldnn::memory& MKLDNNConvolutionNode::getWeights() const {
    return baseInputsNumber > 1 ? getParentEdgeAt(1)->getMemory().GetPrimitive() : internalBlobMemory[0]->GetPrimitive();
}

const mkldnn::memory& MKLDNNConvolutionNode::getBias() const {
    return baseInputsNumber > 2 ? getParentEdgeAt(2)->getMemory().GetPrimitive() : internalBlobMemory[1]->GetPrimitive();
}

REG_MKLDNN_PRIM_FOR(MKLDNNConvolutionNode, Convolution);
