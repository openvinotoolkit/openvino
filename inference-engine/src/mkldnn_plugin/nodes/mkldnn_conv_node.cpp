// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_conv_node.h"
#include "mkldnn_reorder_node.h"
#include "mkldnn_input_node.h"
#include "mkldnn_eltwise_node.h"
#include "mkldnn_fake_quantize_node.h"
#include "mkldnn_pooling_node.h"
#include "mkldnn_concat_node.h"
#include <string>
#include <vector>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <utils/general_utils.h>
#include <ngraph/ops.hpp>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNConvolutionNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ngraph::is_type<ngraph::op::v1::Convolution>(op) && !ngraph::is_type<ngraph::op::v1::GroupConvolution>(op)) {
            errorMessage = "Only opset1 Convolution and GroupConvolution operations are supported";
            return false;
        }
        size_t ndims = op->get_input_shape(0).size();
        if ((ndims < 4) || (ndims > 5)) {
            errorMessage = "Doesn't support 'data' input with rank: " + std::to_string(ndims);
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

MKLDNNConvolutionNode::MKLDNNConvolutionNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(op, eng, cache), withBiases(false), withSum(false), withDWConv(false),
          isGrouped(false), /* dw_conv_oc(0), dw_conv_ih(0), dw_conv_iw(0), dw_conv_in_dt(memory::data_type::undef), */
          groupNum(1lu), eltwisePrecision(Precision::FP32) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    isPrimitivesPriorityDefined = op->get_rt_info().count("PrimitivesPriority") != 0;

    auto convolutionOp = ngraph::as_type_ptr<ngraph::op::v1::Convolution>(op);
    auto groupConvolutionOp = ngraph::as_type_ptr<ngraph::op::v1::GroupConvolution>(op);

    if (convolutionOp) {
        algorithm = ConvolutionCommon;

        for (int i = 0; i < convolutionOp->get_strides().size(); i++) {
            stride.push_back(static_cast<ptrdiff_t>(convolutionOp->get_strides()[i]));
        }
        for (int i = 0; i < convolutionOp->get_dilations().size(); i++) {
            dilation.push_back(static_cast<ptrdiff_t>(convolutionOp->get_dilations()[i]) - 1);
        }
        paddingL = convolutionOp->get_pads_begin();
        paddingR = convolutionOp->get_pads_end();
    } else if (groupConvolutionOp) {
        algorithm = ConvolutionGrouped;

        for (int i = 0; i < groupConvolutionOp->get_strides().size(); i++) {
            stride.push_back(static_cast<ptrdiff_t>(groupConvolutionOp->get_strides()[i]));
        }
        for (int i = 0; i < groupConvolutionOp->get_dilations().size(); i++) {
            dilation.push_back(static_cast<ptrdiff_t>(groupConvolutionOp->get_dilations()[i]) - 1);
        }
        paddingL = groupConvolutionOp->get_pads_begin();
        paddingR = groupConvolutionOp->get_pads_end();
    }
}

bool MKLDNNConvolutionNode::canBeExecutedInInt8() const {
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(0));
    if (!inputZeroPoints.empty())
        inputDataType = memory::data_type::u8;

    auto weightsDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(1));
    if (!weightsZeroPoints.empty())
        weightsDataType = memory::data_type::s8;

    return one_of(inputDataType, memory::data_type::u8, memory::data_type::s8) && weightsDataType == memory::data_type::s8;
}

InferenceEngine::Precision MKLDNNConvolutionNode::fusedEltwisePrecision(const MKLDNNNodePtr& fusingNode) const {
    InferenceEngine::Precision eltwisePrecision;

    int fusingPort = fusingNode->getFusingPort();
    if (fusingPort == 0) {
        eltwisePrecision = fusingNode->getOriginalInputPrecisionAtPort(1);
    } else if (fusingPort == 1) {
        eltwisePrecision = fusingNode->getOriginalInputPrecisionAtPort(0);
    } else {
        IE_THROW() << "Cannot determine Eltwise post op precision for Convolution node with name '" << getName() << "'";
    }

    return eltwisePrecision;
}

void MKLDNNConvolutionNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    if (getAlgorithm() == ConvolutionCommon) {
        groupNum = 1;
        isGrouped = false;

        weightDims = inDims[1].ToSizeVector();

        IC = weightDims[1];
        groupIC = IC;
        groupOC = weightDims[0];

        biasesDims = { groupOC };
    } else if (getAlgorithm() == ConvolutionGrouped) {
        weightDims = inDims[1].ToSizeVector();

        groupNum = weightDims[0];
        isGrouped = true;

        groupIC = weightDims[2];
        IC = groupIC * groupNum;
        groupOC = weightDims[1];

        biasesDims = {groupOC * groupNum};
    }

    withBiases = getOriginalInputsNumber() == 3;

    withSum = false;
    int expectedInputEdgesNum = static_cast<int>(getOriginalInputsNumber());
    for (int i = 0; i < fusedWith.size(); i++) {
        if (fusedWith[i]->getType() == Convolution) {
            expectedInputEdgesNum += static_cast<int>(fusedWith[i]->getOriginalInputsNumber()) - 1;
        }

        if (fusedWith[i]->getAlgorithm() == EltwiseAdd) {
            auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(fusedWith[i].get());
            if (eltwiseNode && eltwiseNode->isSpecialConvolutionAddFusing()) {
                withSum = true;
                expectedInputEdgesNum++;
            }
        }
    }

    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(0));
    if (!inputZeroPoints.empty())
        inputDataType = memory::data_type::u8;

    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(getOriginalOutputPrecisionAtPort(0));
    eltwisePrecision = MKLDNNExtensionUtils::DataTypeToIEPrecision(outputDataType);
    if (!fusedWith.empty()) {
        outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0));
        eltwisePrecision = MKLDNNExtensionUtils::DataTypeToIEPrecision(outputDataType);
    }

    // We need to make sure that convolution output and second input of fused Eltwise operation
    // have equal precision sizes since they use the same physical memory. In case precisions are different we upscale to FP32.
    if (outputDataType != memory::data_type::f32 && outputDataType != memory::data_type::bf16 && withSum) {
        for (int i = 0; i < fusedWith.size(); i++) {
            if (fusedWith[i]->getAlgorithm() == EltwiseAdd) {
                auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(fusedWith[i].get());
                if (eltwiseNode && eltwiseNode->isSpecialConvolutionAddFusing()) {
                    eltwisePrecision = fusedEltwisePrecision(fusedWith[i]);
                    if (MKLDNNExtensionUtils::DataTypeToIEPrecision(outputDataType).size() != eltwisePrecision.size()) {
                        eltwisePrecision = Precision::FP32;
                        outputDataType = memory::data_type::f32;
                    }
                    break;
                }
            }
        }
    }

    if (getParentEdges().size() != expectedInputEdgesNum)
        IE_THROW() << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        IE_THROW() << "Incorrect number of output edges for layer " << getName();

    int ndims = getParentEdgesAtPort(0)[0]->getDims().ndims();
    MKLDNNDims weightsDims = MKLDNNDims(weightDims);

    withDWConv = isFusedWith(Convolution);

    // TODO: fusing with Convolution is not ported yet
//    for (int i = 0; i < fusedWith.size(); i++) {
//        auto *convolutionNode = dynamic_cast<MKLDNNConvolutionNode *>(fusedWith[i].get());
//        if (convolutionNode) {
//            auto *convLayer = reinterpret_cast<ConvolutionLayer *>(convolutionNode->getCnnLayer().get());
//            dw_conv_ih = convolutionNode->inDims[0][convolutionNode->inDims[0].ndims() - 2];
//            dw_conv_iw = convolutionNode->inDims[0][convolutionNode->inDims[0].ndims() - 1];
//            dw_conv_oc = convLayer->_out_depth;
//            for (int j = 0; j < convLayer->_kernel.size(); j++) {
//                dw_conv_kernel.push_back(convLayer->_kernel[j]);
//            }
//            for (int j = 0; j < convLayer->_stride.size(); j++) {
//                dw_conv_strides.push_back(convLayer->_stride[j]);
//            }
//
//            if (canBeExecutedInInt8()) {
//                if (i == 0) {
//                    dw_conv_in_dt = precisionToDataType(getCnnLayer()->outData[0]->getPrecision());
//                } else {
//                    dw_conv_in_dt = precisionToDataType(fusedWith[i - 1].get()->getCnnLayer()->outData[0]->getPrecision());
//                }
//            } else {
//                dw_conv_in_dt = memory::data_type::f32;
//            }
//
//            for (int j = 0; j < paddingR.size(); j++) {
//                int with_group = (isGrouped || isMerged) ? 1 : 0;
//                int krn = weightsDims[with_group + 2 + j];
//                int src = getParentEdgeAt(0)->getDims()[2 + j];
//                int dst = getChildEdgeAt(0)->getDims()[2 + j];
//
//                krn = (krn - 1)*(dilation[j] + 1) + 1;
//                int calc_dst = (src - krn + paddingL[j]) / stride[j] + 1;
//                paddingR[j] = (dst - calc_dst) * stride[j];
//            }
//        }
//    }

    MKLDNNMemoryDesc in_candidate, out_candidate;
    if (canBeExecutedInInt8()) {
        in_candidate = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, ndims == 5 ? memory::format_tag::ndhwc
                                                                                                 : memory::format_tag::nhwc);
        out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, ndims == 5 ? memory::format_tag::ndhwc
                                                                                                  : memory::format_tag::nhwc);
        createDescriptor({in_candidate}, {out_candidate});
    } else {
        inputDataType = (getOriginalInputPrecisionAtPort(0) == Precision::BF16 && !(isGrouped && ndims == 5)) ? memory::data_type::bf16
                                                                                                           : memory::data_type::f32;
        outputDataType = (getOriginalOutputPrecisionAtPort(0) == Precision::BF16 && !(isGrouped && ndims == 5)) ? memory::data_type::bf16
                                                                                                             : memory::data_type::f32;
        eltwisePrecision = Precision::FP32;
        for (int i = 0; i < fusedWith.size(); i++) {
            if (fusedWith[i]->getAlgorithm() == EltwiseAdd) {
                auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(fusedWith[i].get());
                if (eltwiseNode && eltwiseNode->isSpecialConvolutionAddFusing()) {
                    eltwisePrecision = fusedEltwisePrecision(fusedWith[i]);
                    // TODO(amalyshe): there might be situation when convolution can be executed in BF16,
                    // output is required in FP32 but eltwise inplace tensor would be in BF16
                    // currently we forcedly change output to the BF16 that will add reoreder after the node
                    // Another situation can be when we mark output as FP32 and Eltwise asPrecison (which stand
                    // for input of inplace tensor precision) to FP32. This will add reorder for that in-place tensor
                    // bofore the fused convolution. This behaviour might be more correct regarding expected markup
                    // of the graph but performance of first and second approaches might be different. Need to verify
                    outputDataType = eltwisePrecision == Precision::BF16 ? memory::data_type::bf16 : memory::data_type::f32;
                }
            }
        }
        // correction for cases of FP32 input - we do not have FP32 convolution supported BF16 output
        if (inputDataType == memory::data_type::f32
            && (outputDataType == memory::data_type::bf16 || eltwisePrecision == Precision::BF16)) {
            outputDataType = memory::data_type::f32;
            eltwisePrecision = Precision::FP32;
        }

        if (ndims == 4) {
            if (IC == 1 && groupOC == 1) {
                in_candidate = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::format_tag::nchw);
                out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::format_tag::nchw);
                createDescriptor({in_candidate}, {out_candidate});
            } else if (IC == 3 || IC == 1) {
                in_candidate = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::format_tag::nchw);
                out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::format_tag::nChw16c);
                createDescriptor({in_candidate}, {out_candidate});
                out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::format_tag::nChw8c);
                createDescriptor({in_candidate}, {out_candidate});
            } else {
                in_candidate = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::format_tag::nChw16c);
                out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::format_tag::nChw16c);
                createDescriptor({in_candidate}, {out_candidate});
                in_candidate = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::format_tag::nChw8c);
                out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::format_tag::nChw8c);
                createDescriptor({in_candidate}, {out_candidate});
            }

            in_candidate = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::format_tag::nchw);
            out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::format_tag::nchw);
            createDescriptor({in_candidate}, {out_candidate});
        } else if (ndims == 5) {
            if (IC == 1 && groupOC == 1) {
                in_candidate = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::format_tag::ncdhw);
                out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::format_tag::ncdhw);
                createDescriptor({in_candidate}, {out_candidate});
            } else if (IC == 3 || IC == 1) {
                in_candidate = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::format_tag::ncdhw);
                out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::format_tag::nCdhw16c);
                createDescriptor({in_candidate}, {out_candidate});
                out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::format_tag::nCdhw8c);
                createDescriptor({in_candidate}, {out_candidate});
            } else {
                in_candidate = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::format_tag::nCdhw16c);
                out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::format_tag::nCdhw16c);
                createDescriptor({in_candidate}, {out_candidate});
                in_candidate = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::format_tag::nCdhw8c);
                out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::format_tag::nCdhw8c);
                createDescriptor({in_candidate}, {out_candidate});
            }

            in_candidate = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::format_tag::ncdhw);
            out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::format_tag::ncdhw);
            createDescriptor({in_candidate}, {out_candidate});
        }
    }
}

void MKLDNNConvolutionNode::setPostOps(mkldnn::primitive_attr &attr, bool initWeights = false) const {
    mkldnn::post_ops ops;

    for (auto &node : fusedWith) {
        if (node->getType() == Split || node->getType() == Concatenation)
            continue;

        auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(node.get());
        if (eltwiseNode) {
            if (eltwiseNode->isSpecialConvolutionAddFusing())
                ops.append_sum(1.0, MKLDNNExtensionUtils::IEPrecisionToDataType(eltwisePrecision));
            else
                eltwiseNode->appendPostOps(ops);
            continue;
        }

        auto* fakeQuantizeNode = dynamic_cast<MKLDNNFakeQuantizeNode *>(node.get());
        if (fakeQuantizeNode) {
            fakeQuantizeNode->appendPostOps(ops);
            continue;
        }

        // auto* convolutionNode = dynamic_cast<MKLDNNConvolutionNode *>(node.get());
        // if (convolutionNode) {
        //     if (initWeights) {
        //         if (convolutionNode->getBaseIntputsNumber() == 1) {
        //             auto* convLayer = reinterpret_cast<ConvolutionLayer*>(convolutionNode->getCnnLayer().get());

        //             auto weightsPrc = precisionToDataType(convLayer->precision);
        //             auto biasPrc = memory::data_type::s32;

        //             PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
        //             MKLDNNDims dwWeightsDims({dw_conv_oc, (ptrdiff_t)1, (ptrdiff_t)1, dw_conv_kernel[Y_AXIS], dw_conv_kernel[X_AXIS]});
        //             PostOpsIntBlobMemory[blob_idx]->Create(dwWeightsDims, weightsPrc, memory::format_tag::Goihw8g);
        //             PostOpsIntBlobMemory[blob_idx]->FillZero();

        //             Blob::Ptr weights = convLayer->blobs.find("weights")->second;
        //             Blob::Ptr biases = convLayer->blobs.find("biases")->second;

        //             PostOpsIntBlobMemory[blob_idx]->SetData(weightsPrc, memory::format_tag::goihw, weights->buffer(),
        //                                                     dwWeightsDims.size() * MKLDNNExtensionUtils::sizeOfDataType(weightsPrc));

        //             PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
        //             MKLDNNDims dwBiasesDims({dw_conv_oc});
        //             PostOpsIntBlobMemory[blob_idx + 1]->Create(dwBiasesDims, biasPrc, memory::format_tag::x);
        //             PostOpsIntBlobMemory[blob_idx + 1]->FillZero();
        //             PostOpsIntBlobMemory[blob_idx + 1]->SetData(biasPrc, memory::format_tag::x, biases->buffer(),
        //                                                         dwBiasesDims.size() * MKLDNNExtensionUtils::sizeOfDataType(biasPrc));
        //             // todo: rewrite onto append_dw_k3s2p1
        //             ops.append_dw_conv(dw_conv_ih, dw_conv_iw, dw_conv_kernel[Y_AXIS], dw_conv_kernel[X_AXIS],
        //                                dw_conv_strides[Y_AXIS], dw_conv_strides[X_AXIS],
        //                                mkldnn::memory::convert_to_c(dw_conv_in_dt),
        //                                static_cast<const float *>(PostOpsIntBlobMemory[blob_idx]->GetData()),
        //                                static_cast<const float *>(PostOpsIntBlobMemory[blob_idx + 1]->GetData()));

        //             blob_idx += 2;
        //         } else {
        //             // todo: rewrite onto append_dw_k3s2p1
        //             ops.append_dw_conv(dw_conv_ih, dw_conv_iw, dw_conv_kernel[Y_AXIS], dw_conv_kernel[X_AXIS],
        //                                dw_conv_strides[Y_AXIS], dw_conv_strides[X_AXIS],
        //                                mkldnn::memory::convert_to_c(dw_conv_in_dt),
        //                                static_cast<const float *>(getParentEdgeAt(
        //                                        baseInputsNumber + 0)->getMemory().GetData()),
        //                                static_cast<const float *>(getParentEdgeAt(
        //                                        baseInputsNumber + 1)->getMemory().GetData()));
        //         }
        //     } else {
        //         // todo: rewrite onto append_dw_k3s2p1
        //         ops.append_dw_conv(dw_conv_ih, dw_conv_iw, dw_conv_kernel[Y_AXIS], dw_conv_kernel[X_AXIS],
        //                            dw_conv_strides[Y_AXIS], dw_conv_strides[X_AXIS],
        //                            mkldnn::memory::convert_to_c(dw_conv_in_dt),
        //                            nullptr,
        //                            nullptr);
        //     }

        //     if (convolutionNode->wScale != nullptr) {
        //         float* wScaleData = static_cast<float*>(convolutionNode->wScale->buffer());

        //         std::vector<float> oScaleDataVector;
        //         std::vector<float> oShiftDataVector;
        //         if (convolutionNode->getCnnLayer()->precision == Precision::I8 &&
        //             convolutionNode->getCnnLayer()->outData[0]->getPrecision() != Precision::FP32) {
        //             float *oScaleData = static_cast<float *>(convolutionNode->oScale->buffer());

        //             for (size_t c = 0; c < convolutionNode->wScale->size(); c++) {
        //                 oScaleDataVector.push_back(wScaleData[c] / oScaleData[c]);
        //                 oShiftDataVector.push_back(0.f);
        //             }
        //         } else {
        //             for (size_t c = 0; c < convolutionNode->wScale->size(); c++) {
        //                 oScaleDataVector.push_back(wScaleData[c]);
        //                 oShiftDataVector.push_back(0.f);
        //             }
        //         }

        //         MKLDNNDims oScaleDims({static_cast<ptrdiff_t>(rnd_up(biasesDims[0], 16))});

        //         PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
        //         PostOpsIntBlobMemory[blob_idx]->Create(oScaleDims, memory::data_type::f32, memory::format_tag::x);
        //         PostOpsIntBlobMemory[blob_idx]->FillZero();
        //         PostOpsIntBlobMemory[blob_idx]->SetData(memory::data_type::f32, memory::format_tag::x, &oScaleDataVector[0],
        //                                                 oScaleDataVector.size() * MKLDNNExtensionUtils::sizeOfDataType(memory::data_type::f32));

        //         PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
        //         PostOpsIntBlobMemory[blob_idx + 1]->Create(oScaleDims, memory::data_type::f32, memory::format_tag::x);
        //         PostOpsIntBlobMemory[blob_idx + 1]->FillZero();
        //         PostOpsIntBlobMemory[blob_idx + 1]->SetData(memory::data_type::f32, memory::format_tag::x, &oShiftDataVector[0],
        //                                                     oShiftDataVector.size() * MKLDNNExtensionUtils::sizeOfDataType(memory::data_type::f32));

        //         ops.append_depthwise(mkldnn::algorithm::depthwise_scale_shift,
        //                              static_cast<const float *>(PostOpsIntBlobMemory[blob_idx]->GetData()),
        //                              static_cast<const float *>(PostOpsIntBlobMemory[blob_idx + 1]->GetData()));

        //         blob_idx += 2;
        //     }
        //     continue;
        // }

        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType()) << " node is not implemented";
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
        while (static_cast<bool>(itpd)) {
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

// TODO: fusing with Convolution is not ported yet
//            if (withDWConv) {
//                auto weightsPrc = precisionToDataType(dw_conv_in_dt == mkldnn_u8 ? Precision::I8 : Precision::FP32);
//                auto biasPrc = memory::data_type::f32;
//
//                MKLDNNDims dwWeightsDims({dw_conv_oc, (ptrdiff_t)1, (ptrdiff_t)1, dw_conv_kernel[Y_AXIS], dw_conv_kernel[X_AXIS]});
//                MKLDNNDims dwBiasesDims({dw_conv_oc});
//
//                InferenceEngine::DataConfig dataConfig;
//                dataConfig.inPlace = -1;
//                dataConfig.constant = false;
//                dataConfig.desc = MKLDNNMemoryDesc(dwWeightsDims, weightsPrc, memory::format_tag::Goihw8g);
//                config.inConfs.push_back(dataConfig);
//
//                dataConfig.desc = MKLDNNMemoryDesc(dwBiasesDims, biasPrc, memory::format_tag::x);
//                config.inConfs.push_back(dataConfig);
//            }

            for (size_t i = 0; i < descOutputNumbers(desc); i++) {
                InferenceEngine::DataConfig dataConfig;
                if (withSum) {
                    dataConfig.inPlace = getParentEdges().size() - 1;
                }

                dataConfig.constant = false;
                dataConfig.desc = getDstMemDesc(itpd, i);
                if (!isGrouped)
                    dataConfig.desc = MKLDNNExtensionUtils::getUninitTensorDesc(dataConfig.desc);
                config.outConfs.push_back(dataConfig);

                if (withSum) {
                    dataConfig.inPlace = -1;
                    dataConfig.desc.setPrecision(eltwisePrecision);
                    config.inConfs.push_back(dataConfig);
                }
            }
            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());
            if (impl_type & jit)
                containJitImpl = true;

            supportedPrimitiveDescriptors.emplace_back(config, impl_type);
            if (!itpd.next_impl())
                break;
        }
    }
}


void MKLDNNConvolutionNode::createPrimitive() {
    if (prim)
        return;

    mkldnn::primitive_attr attr;
    addZeroPoints(attr);
    setPostOps(attr, true);

    auto prim_desc = createPrimitiveDescriptor<convolution_forward::primitive_desc,
            convolution_forward::desc>(attr);

    prim.reset(new convolution_forward(prim_desc));

    auto src = getParentEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
    auto wei = getParentEdgesAtPort(1)[0]->getMemoryPtr()->GetPrimitive();
    auto dst = getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
    if (withBiases) {
        auto bias = getParentEdgesAtPort(2)[0]->getMemoryPtr()->GetPrimitive();
        primArgs = {{DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, wei}, {DNNL_ARG_BIAS, bias}, {DNNL_ARG_DST, dst}};
    } else {
        primArgs = {{DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, wei}, {DNNL_ARG_DST, dst}};
    }
}

bool MKLDNNConvolutionNode::created() const {
    return getType() == Convolution;
}

void MKLDNNConvolutionNode::createDescriptor(const std::vector<InferenceEngine::TensorDesc> &inputDesc,
                                             const std::vector<InferenceEngine::TensorDesc> &outputDesc) {
    TensorDesc inDesc = inputDesc[0], outDesc = outputDesc[0];

    memory::data_type wdt = MKLDNNExtensionUtils::IEPrecisionToDataType(inDesc.getPrecision());
    memory::data_type bdt = memory::data_type::f32;

    if (inDesc.getPrecision() == Precision::U8 || inDesc.getPrecision() == Precision::I8) {
        wdt = memory::data_type::s8;
    }

    MKLDNNMemoryDesc in_candidate(inDesc);
    MKLDNNMemoryDesc out_candidate(outDesc);

    MKLDNNDims blocked_weightDims(weightDims);
    MKLDNNDims blocked_biasesDims(biasesDims);
    MKLDNNMemoryDesc wgh_candidate{blocked_weightDims, wdt, memory::format_tag::any};

    std::vector<mkldnn::algorithm> algorithms;

    // TODO [NM]: We cannot map wino_format on tensor descriptor for now
    // algorithms.push_back(algorithm::convolution_winograd);
    algorithms.push_back(mkldnn::algorithm::convolution_direct);

    for (auto alg : algorithms) {
        try {
            std::shared_ptr<mkldnn::convolution_forward::desc> conv_desc;
            if (withBiases) {
                MKLDNNMemoryDesc bias_candidate{blocked_biasesDims, bdt, memory::format_tag::any};

                conv_desc.reset(new convolution_forward::desc(prop_kind::forward_scoring, alg,
                            in_candidate, wgh_candidate, bias_candidate, out_candidate,
                            mkldnn::memory::dims(stride.begin(), stride.end()),
                            mkldnn::memory::dims(dilation.begin(), dilation.end()),
                            mkldnn::memory::dims(paddingL.begin(), paddingL.end()),
                            mkldnn::memory::dims(paddingR.begin(), paddingR.end())));
            } else {
                conv_desc.reset(new convolution_forward::desc(prop_kind::forward_scoring, alg,
                            in_candidate, wgh_candidate, out_candidate,
                            mkldnn::memory::dims(stride.begin(), stride.end()),
                            mkldnn::memory::dims(dilation.begin(), dilation.end()),
                            mkldnn::memory::dims(paddingL.begin(), paddingL.end()),
                            mkldnn::memory::dims(paddingR.begin(), paddingR.end())));
            }

            descs.emplace_back(conv_desc);
        } catch (...) {
            IE_THROW() << "Cannot create convolution forward descriptor for layer: " << getName();
        }
    }
}

void MKLDNNConvolutionNode::addZeroPoints(mkldnn::primitive_attr& attr) const {
    if (!inputZeroPoints.empty())
        attr.set_input_zero_points(1 << 1 /*through C dim*/, inputZeroPoints);

    if (!weightsZeroPoints.empty())
        attr.set_weights_zero_points(1 << 1 /*through C dim*/, weightsZeroPoints);

    if (!outputCompensation.empty()) {
        attr.set_output_compensations(1 << 1 /*through C dim*/, outputCompensation);
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

    // TODO [NM]: refactor via using global executionPrecision.
    if (canBeExecutedInInt8()) {
        isStridedBlobsSupported = false;
    }
//    // TODO [NM]: fix strided blobs feature support for dynamic weights
//    if (getOriginalInputsNumber() != 1) {
//        isStridedBlobsSupported = false;
//    }

    if (isStridedBlobsSupported) {
        createDescriptor({config.inConfs[0].desc}, {config.outConfs[0].desc});
    }

    mkldnn::primitive_attr attr;
    addZeroPoints(attr);
    setPostOps(attr);

    InferenceEngine::LayerConfig rightConfig = selectedPD->getConfig();
    size_t selected_count = 0;

    bool containJitImpl = false;

    for (size_t i = 0; i < descs.size(); i++) {
        auto& desc = descs[i];
        if (containJitImpl && isPossibleToSkipInitConfig(desc))
            continue;
        auto itpd = desc.createPrimitiveDescriptorIterator(getEngine(), attr);
        while (static_cast<bool>(itpd)) {
            InferenceEngine::LayerConfig cfg;
            cfg.dynBatchSupport = true;
            for (size_t j = 0; j < descInputNumbers(desc); j++) {
                InferenceEngine::DataConfig dataConfig;
                dataConfig.inPlace = -1;
                dataConfig.constant = false;
                dataConfig.desc = getSrcMemDesc(itpd, j);
                cfg.inConfs.push_back(dataConfig);
            }

            // TODO: fusing with Convolution is not ported yet
//            if (withDWConv) {
//                auto weightsPrc = precisionToDataType(dw_conv_in_dt == mkldnn_u8 ? Precision::I8 : Precision::FP32);
//                auto biasPrc = memory::data_type::f32;
//
//                MKLDNNDims dwWeightsDims({dw_conv_oc, (ptrdiff_t)1, (ptrdiff_t)1, dw_conv_kernel[Y_AXIS], dw_conv_kernel[X_AXIS]});
//                MKLDNNDims dwBiasesDims({dw_conv_oc});
//
//                InferenceEngine::DataConfig dataConfig;
//                dataConfig.inPlace = -1;
//                dataConfig.constant = false;
//                dataConfig.desc = MKLDNNMemoryDesc(dwWeightsDims, weightsPrc, memory::format_tag::Goihw8g);
//                cfg.inConfs.push_back(dataConfig);
//
//                dataConfig.desc = MKLDNNMemoryDesc(dwBiasesDims, biasPrc, memory::format_tag::x);
//                cfg.inConfs.push_back(dataConfig);
//            }

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
            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());
            if (impl_type & jit)
                containJitImpl = true;

            if (selected_count == selectedPrimitiveDescriptorIndex) {
                if (impl_type != selectedPD->getImplementationType()) {
                    IE_THROW() << "Cannot get the original layer configuration!";
                }
                rightConfig = cfg;
            }
            if (i == descs.size() - 1 && isStridedBlobsSupported) {
                if (impl_type == selectedPD->getImplementationType()) {
                    rightConfig = config;
                }
            }
            selected_count++;
            if (!itpd.next_impl())
                break;
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
            IE_THROW() << "Incorrect number of input or output memory formats for Convolution node";
        }
        auto itd = descs.begin();
        while (itd != descs.end()) {
            bool isSuitableDesc = true;
            if (!inputMemoryFormatsFilter.empty()) {
                MKLDNNMemoryDesc src_tdesc(std::shared_ptr<mkldnn::convolution_forward::desc>(*itd)->data.src_desc);
                isSuitableDesc &= src_tdesc.isSame(inputMemoryFormatsFilter[0]);
            }
            if (!outputMemoryFormatsFilter.empty()) {
                MKLDNNMemoryDesc dst_tdesc(std::shared_ptr<mkldnn::convolution_forward::desc>(*itd)->data.dst_desc);
                isSuitableDesc &= dst_tdesc.isSame(outputMemoryFormatsFilter[0]);
            }
            if (!isSuitableDesc) {
                itd = descs.erase(itd);
            } else {
                itd++;
            }
        }
    }
}

bool MKLDNNConvolutionNode::isPossibleToSkipInitConfig(MKLDNNDescriptor &desc) const {
    //  WA: In some cases, we can predict in advance the type of primitive that will be called in the future.
    //  In particular, isPossibleToSkipInitConfig() checks whether we can skip the creation of primitives with
    //  gemm implementation, which significantly increase the network load time.
    if (!inputMemoryFormatsFilter.empty() || !outputMemoryFormatsFilter.empty())
        return false;

    if (isPrimitivesPriorityDefined)
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
    auto srcMemDesc = MKLDNNMemoryDesc {convDesc->data.src_desc};
    auto dstMemDesc = MKLDNNMemoryDesc {convDesc->data.dst_desc};
    auto srcDataType = convDesc->data.src_desc.data_type;
    auto dstDataType = convDesc->data.dst_desc.data_type;
    bool isPlanarFloatConv = srcMemDesc.isPlainFormat()
                             && dstMemDesc.isPlainFormat()
                             && srcDataType == memory::data_type::f32
                             && dstDataType == memory::data_type::f32;

    return !isPossibleJitPlanar && isPlanarFloatConv;
}

MKLDNNMemoryDesc MKLDNNConvolutionNode::getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    InferenceEngine::TensorDesc desc = idx > 0 ? MKLDNNMemoryDesc(primitive_desc_it.weights_desc(idx - 1))
                                               : MKLDNNMemoryDesc(primitive_desc_it.src_desc(idx));

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

InferenceEngine::Precision MKLDNNConvolutionNode::getRuntimePrecision() const {
    std::vector<InferenceEngine::Precision> inputPrecisions;
    // Don't take bias precision into account
    size_t inputsNumLimit = 2;
    for (size_t i = 0; i < std::min(getParentEdges().size(), inputsNumLimit); i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (parentEdge && parentEdge->getStatus() == MKLDNNEdge::Status::Validated) {
            inputPrecisions.emplace_back(MKLDNNExtensionUtils::DataTypeToIEPrecision((parentEdge->getMemoryPtr()->GetDataType())));
        }
    }

    return MKLDNNExtensionUtils::getMaxPrecision(inputPrecisions);
}

REG_MKLDNN_PRIM_FOR(MKLDNNConvolutionNode, Convolution);
