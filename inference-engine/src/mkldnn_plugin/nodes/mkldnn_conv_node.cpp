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
#include "cpu/x64/cpu_isa_traits.hpp"
#include <string>
#include <vector>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <utils/general_utils.h>
#include <ngraph/ops.hpp>
#include <cpu/x64/jit_generator.hpp>
#include "common/cpu_convert.h"
#include <cpu_memory_desc_utils.h>

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
          isGrouped(false), dw_conv_oc(0), dw_conv_ih(0), dw_conv_iw(0), dw_conv_in_dt(memory::data_type::undef),
          groupNum(1lu), eltwisePrecision(Precision::FP32) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    auto convolutionOp = ngraph::as_type_ptr<ngraph::op::v1::Convolution>(op);
    auto groupConvolutionOp = ngraph::as_type_ptr<ngraph::op::v1::GroupConvolution>(op);

    if (convolutionOp) {
        algorithm = ConvolutionCommon;

        groupNum = 1;
        isGrouped = false;

        weightDims = convolutionOp->input_value(1).get_shape();

        IC = weightDims[1];
        groupIC = IC;
        groupOC = weightDims[0];

        biasesDims = { groupOC };

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

        groupNum = groupConvolutionOp->input_value(1).get_shape()[0];
        isGrouped = true;

        weightDims = groupConvolutionOp->input_value(1).get_shape();

        groupIC = weightDims[2];
        IC = groupIC * groupNum;
        groupOC = weightDims[1];

        biasesDims = {groupOC * groupNum};

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

    withBiases = getOriginalInputsNumber() == 3;

    if (!implPriorities.empty()) {
        isPrimitivesPriorityDefined = true;
        // winograd support only constant weights and bias
        isWino = std::find(implPriorities.begin(), implPriorities.end(), impl_desc_type::jit_avx512_winograd) != implPriorities.end() &&
                 mkldnn::impl::cpu::x64::mayiuse(mkldnn::impl::cpu::x64::avx512_common) && !canBeExecutedInInt8() &&
                 getParentEdgeAt(1)->getParent()->isConstant() && getParentEdgeAt(1)->getParent()->getType() == Input &&
                 (withBiases ? (getParentEdgeAt(2)->getParent()->isConstant() && getParentEdgeAt(2)->getParent()->getType() == Input) : true);
    }

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

    int ndims = getParentEdgesAtPort(0)[0]->getShape().getRank();
    MKLDNNDims weightsDims = MKLDNNDims(weightDims);

    withDWConv = isFusedWith(Convolution);

    for (int i = 0; i < fusedWith.size(); i++) {
        auto *convolutionNode = dynamic_cast<MKLDNNConvolutionNode *>(fusedWith[i].get());
        if (convolutionNode) {
            auto& inActivationDims = convolutionNode->inputShapes[0].getStaticDims();
            dw_conv_ih = inActivationDims[convolutionNode->inputShapes[0].getRank() - 2];
            dw_conv_iw = inActivationDims[convolutionNode->inputShapes[0].getRank() - 1];

            auto& outDims = convolutionNode->outputShapes[0].getStaticDims();
            dw_conv_oc = outDims[1];

            const auto &dwWeightsDims = convolutionNode->inputShapes[1].getStaticDims();
            dw_conv_kernel.push_back(dwWeightsDims[dwWeightsDims.size() - 1]);
            dw_conv_kernel.push_back(dwWeightsDims[dwWeightsDims.size() - 2]);
            dw_conv_strides = convolutionNode->getStride();

            if (canBeExecutedInInt8()) {
                if (i == 0) {
                    dw_conv_in_dt = MKLDNNExtensionUtils::IEPrecisionToDataType(getOriginalOutputPrecisionAtPort(0));
                } else {
                    dw_conv_in_dt = MKLDNNExtensionUtils::IEPrecisionToDataType(fusedWith[i - 1]->getOriginalOutputPrecisionAtPort(0));
                }
            } else {
                dw_conv_in_dt = memory::data_type::f32;
            }

            for (int j = 0; j < paddingR.size(); j++) {
                int with_group = isGrouped ? 1 : 0;
                int krn = weightsDims[with_group + 2 + j];
                int src = getParentEdgeAt(0)->getShape().getStaticDims()[2 + j];
                int dst = getChildEdgeAt(0)->getShape().getStaticDims()[2 + j];

                krn = (krn - 1)*(dilation[j] + 1) + 1;
                int calc_dst = (src - krn + paddingL[j]) / stride[j] + 1;
                paddingR[j] = (dst - calc_dst) * stride[j];
            }
        }
    }

    MemoryDescPtr in_candidate, out_candidate;
    if (canBeExecutedInInt8()) {
        //  We have to extend convolution_x8s8s32x from oneDNN to support BF16 output data type
        if (outputDataType == memory::data_type::bf16)
            outputDataType = memory::data_type::f32;
        if (eltwisePrecision == Precision::BF16)
            eltwisePrecision = Precision::FP32;
        in_candidate = MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(getParentEdgeAt(0)->getShape().getStaticMklDims(),
                                                     inputDataType, ndims == 5 ? memory::format_tag::ndhwc : memory::format_tag::nhwc);
        out_candidate = MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(getChildEdgeAt(0)->getShape().getStaticMklDims(),
                                                      outputDataType, ndims == 5 ? memory::format_tag::ndhwc : memory::format_tag::nhwc);
        createDescriptor({ in_candidate.get() }, { out_candidate.get() });
    } else {
        inputDataType = (getOriginalInputPrecisionAtPort(0) == Precision::BF16
                && !(isDepthWise() && ndims == 5)) ? memory::data_type::bf16 : memory::data_type::f32;
        outputDataType = (getOriginalOutputPrecisionAtPort(0) == Precision::BF16
                && !(isDepthWise() && ndims == 5)) ? memory::data_type::bf16 : memory::data_type::f32;
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
                    eltwisePrecision = MKLDNNExtensionUtils::DataTypeToIEPrecision(outputDataType);
                }
            }
        }
        // correction for cases of FP32 input - we do not have FP32 convolution supported BF16 output
        if (inputDataType == memory::data_type::f32
            && (outputDataType == memory::data_type::bf16 || eltwisePrecision == Precision::BF16)) {
            outputDataType = memory::data_type::f32;
            eltwisePrecision = Precision::FP32;
        }

        if (one_of(ndims, 4, 5)) {
            memory::format_tag ncsp = ndims == 4 ? memory::format_tag::nchw : memory::format_tag::ncdhw;
            memory::format_tag nspc = ndims == 4 ? memory::format_tag::nhwc : memory::format_tag::ndhwc;
            memory::format_tag nCsp16c = ndims == 4 ? memory::format_tag::nChw16c : memory::format_tag::nCdhw16c;
            memory::format_tag nCsp8c = ndims == 4 ? memory::format_tag::nChw8c : memory::format_tag::nCdhw8c;

            auto inputDims = getParentEdgeAt(0)->getShape().getStaticMklDims();
            auto outputDims = getChildEdgeAt(0)->getShape().getStaticMklDims();

            if (IC == 1 && groupOC == 1) {
                in_candidate = MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(inputDims, inputDataType, ncsp);
                out_candidate = MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(outputDims, outputDataType, ncsp);
                createDescriptor({ in_candidate.get() }, { out_candidate.get() });
            } else if (IC < 4) {
                in_candidate = MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(inputDims, inputDataType, ncsp);
                out_candidate = MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(outputDims, outputDataType, nCsp16c);
                createDescriptor({ in_candidate.get() }, { out_candidate.get() });
                out_candidate = MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(outputDims, outputDataType, nCsp8c);
                createDescriptor({ in_candidate.get() }, { out_candidate.get() });
            } else {
                in_candidate = MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(inputDims, inputDataType, nCsp16c);
                out_candidate = MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(outputDims, outputDataType, nCsp16c);
                createDescriptor({ in_candidate.get() }, { out_candidate.get() });
                in_candidate = MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(inputDims, inputDataType, nCsp8c);
                out_candidate = MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(outputDims, outputDataType, nCsp8c);
                createDescriptor({ in_candidate.get() }, { out_candidate.get() });
            }

            in_candidate = MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(inputDims, inputDataType, ncsp);
            out_candidate = MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(outputDims, outputDataType, ncsp);
            createDescriptor({ in_candidate.get() }, { out_candidate.get() });

            if (inputDataType != memory::data_type::bf16 && isNspcAvailable()) {
                in_candidate = MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(inputDims, inputDataType, nspc);
                out_candidate = MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(outputDims, outputDataType, nspc);
                createDescriptor({ in_candidate.get() }, { out_candidate.get() });
            }
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

        auto* convolutionNode = dynamic_cast<MKLDNNConvolutionNode *>(node.get());
        if (convolutionNode) {
            if (initWeights) {
                // todo: rewrite onto append_dw_k3s2p1
                ops.append_dw_conv(dw_conv_ih, dw_conv_iw, dw_conv_kernel[Y_AXIS], dw_conv_kernel[X_AXIS],
                                   dw_conv_strides[Y_AXIS], dw_conv_strides[X_AXIS],
                                   mkldnn::memory::convert_to_c(dw_conv_in_dt),
                                   static_cast<const float *>(getParentEdgeAt(
                                           getOriginalInputsNumber() + 0)->getMemory().GetData()),
                                   static_cast<const float *>(getParentEdgeAt(
                                           getOriginalInputsNumber() + 1)->getMemory().GetData()));
            } else {
                // todo: rewrite onto append_dw_k3s2p1
                ops.append_dw_conv(dw_conv_ih, dw_conv_iw, dw_conv_kernel[Y_AXIS], dw_conv_kernel[X_AXIS],
                                   dw_conv_strides[Y_AXIS], dw_conv_strides[X_AXIS],
                                   mkldnn::memory::convert_to_c(dw_conv_in_dt),
                                   nullptr,
                                   nullptr);
            }
            continue;
        }

        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType()) << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

void MKLDNNConvolutionNode::selectOptimalPrimitiveDescriptor() {
    selectPreferPrimitiveDescriptor(getPrimitivesPriority(), true);
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
            NodeConfig config;
            config.dynBatchSupport = true;
            for (size_t i = 0; i < descInputNumbers(desc); i++) {
                PortConfig dataConfig;
                dataConfig.inPlace = -1;
                dataConfig.constant = false;
                if (isGrouped)
                    dataConfig.desc = getSrcMemDesc(itpd, i);
                else
                    dataConfig.desc = MemoryDescUtils::applyUndefinedOffset(*getSrcMemDesc(itpd, i));

                config.inConfs.push_back(dataConfig);
            }

            if (withDWConv) {
                auto weightsPrc = MKLDNNExtensionUtils::IEPrecisionToDataType(dw_conv_in_dt == mkldnn_u8 ? Precision::I8 : Precision::FP32);
                auto biasPrc = memory::data_type::f32;

                MKLDNNDims dwWeightsDims({dw_conv_oc, (ptrdiff_t)1, (ptrdiff_t)1, dw_conv_kernel[Y_AXIS], dw_conv_kernel[X_AXIS]});
                MKLDNNDims dwBiasesDims({dw_conv_oc});

                PortConfig dataConfig;
                dataConfig.inPlace = -1;
                dataConfig.constant = false;
                dataConfig.desc = MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(dwWeightsDims, weightsPrc, memory::format_tag::Goihw8g);
                config.inConfs.push_back(dataConfig);

                dataConfig.desc = MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(dwBiasesDims, biasPrc, memory::format_tag::x);
                config.inConfs.push_back(dataConfig);
            }

            for (size_t i = 0; i < descOutputNumbers(desc); i++) {
                PortConfig dataConfig;
                if (withSum) {
                    dataConfig.inPlace = getParentEdges().size() - 1;
                }

                dataConfig.constant = false;

                if (isGrouped)
                    dataConfig.desc = getDstMemDesc(itpd, i);
                else
                    dataConfig.desc = MemoryDescUtils::applyUndefinedOffset(*getDstMemDesc(itpd, i));

                config.outConfs.push_back(dataConfig);

                if (withSum) {
                    dataConfig.inPlace = -1;
                    dataConfig.desc->setPrecision(eltwisePrecision);
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
    auto dst = getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
    if (withBiases)
        primArgs = {{DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, getWeights()}, {DNNL_ARG_BIAS, getBias()}, {DNNL_ARG_DST, dst}};
    else
        primArgs = {{DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, getWeights()}, {DNNL_ARG_DST, dst}};
}

bool MKLDNNConvolutionNode::created() const {
    return getType() == Convolution;
}

void MKLDNNConvolutionNode::createDescriptor(const std::vector<const MemoryDesc*>& inputDesc,
                                             const std::vector<const MemoryDesc*>& outputDesc) {
    auto inDesc = MemoryDescUtils::convertToMKLDNNMemoryDesc(*inputDesc[0]);
    auto outDesc = MemoryDescUtils::convertToMKLDNNMemoryDesc(*outputDesc[0]);

    memory::data_type wdt = MKLDNNExtensionUtils::IEPrecisionToDataType(inDesc.getPrecision());
    memory::data_type bdt = memory::data_type::f32;

    if (inDesc.getPrecision() == Precision::U8 || inDesc.getPrecision() == Precision::I8) {
        wdt = memory::data_type::s8;
    }

    MKLDNNDims blocked_weightDims(weightDims);
    MKLDNNDims blocked_biasesDims(biasesDims);
    mkldnn::memory::desc wgh_candidate(blocked_weightDims, wdt, memory::format_tag::any);

    std::vector<mkldnn::algorithm> algorithms;

    if (isWinograd())
        algorithms.push_back(mkldnn::algorithm::convolution_winograd);
    algorithms.push_back(mkldnn::algorithm::convolution_direct);

    for (auto alg : algorithms) {
        try {
            std::shared_ptr<mkldnn::convolution_forward::desc> conv_desc;
            if (withBiases) {
                mkldnn::memory::desc bias_candidate(blocked_biasesDims, bdt, memory::format_tag::any);

                conv_desc.reset(new convolution_forward::desc(prop_kind::forward_scoring, alg,
                            inDesc, wgh_candidate, bias_candidate, outDesc,
                            mkldnn::memory::dims(stride.begin(), stride.end()),
                            mkldnn::memory::dims(dilation.begin(), dilation.end()),
                            mkldnn::memory::dims(paddingL.begin(), paddingL.end()),
                            mkldnn::memory::dims(paddingR.begin(), paddingR.end())));
            } else {
                conv_desc.reset(new convolution_forward::desc(prop_kind::forward_scoring, alg,
                            inDesc, wgh_candidate, outDesc,
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

void MKLDNNConvolutionNode::initDescriptor(const NodeConfig& config) {
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
    // TODO [NM]: fix strided blobs feature support for dynamic weights
    // if (getOriginalInputsNumber() != 1) {
    //     isStridedBlobsSupported = false;
    // }

    if (isStridedBlobsSupported) {
        createDescriptor({config.inConfs[0].desc.get()}, {config.outConfs[0].desc.get()});
    }

    mkldnn::primitive_attr attr;
    addZeroPoints(attr);
    setPostOps(attr);

    auto rightConfig = selectedPD->getConfig();
    size_t selected_count = 0;

    bool containJitImpl = false;

    for (size_t i = 0; i < descs.size(); i++) {
        auto& desc = descs[i];
        if (containJitImpl && isPossibleToSkipInitConfig(desc))
            continue;
        auto itpd = desc.createPrimitiveDescriptorIterator(getEngine(), attr);
        while (static_cast<bool>(itpd)) {
            NodeConfig cfg;
            cfg.dynBatchSupport = true;
            for (size_t j = 0; j < descInputNumbers(desc); j++) {
                PortConfig dataConfig;
                dataConfig.inPlace = -1;
                dataConfig.constant = false;
                dataConfig.desc = getSrcMemDesc(itpd, j);
                cfg.inConfs.push_back(dataConfig);
            }

            if (withDWConv) {
                auto weightsPrc = MKLDNNExtensionUtils::IEPrecisionToDataType(dw_conv_in_dt == mkldnn_u8 ? Precision::I8 : Precision::FP32);
                auto biasPrc = memory::data_type::f32;

                MKLDNNDims dwWeightsDims({dw_conv_oc, (ptrdiff_t)1, (ptrdiff_t)1, dw_conv_kernel[Y_AXIS], dw_conv_kernel[X_AXIS]});
                MKLDNNDims dwBiasesDims({dw_conv_oc});

                PortConfig dataConfig;
                dataConfig.inPlace = -1;
                dataConfig.constant = false;
                dataConfig.desc = MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(dwWeightsDims, weightsPrc, memory::format_tag::Goihw8g);
                cfg.inConfs.push_back(dataConfig);

                dataConfig.desc = MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(dwBiasesDims, biasPrc, memory::format_tag::x);
                cfg.inConfs.push_back(dataConfig);
            }

            for (size_t j = 0; j < descOutputNumbers(desc); j++) {
                PortConfig dataConfig;
                dataConfig.inPlace = -1;
                dataConfig.constant = false;
                dataConfig.desc = getDstMemDesc(itpd, j);
                if (withSum) {
                    auto eltwiseConfig = dataConfig;
                    eltwiseConfig.desc->setPrecision(eltwisePrecision);
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
    selectedPD->setConfig(rightConfig);
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
    bool isPlanarFloatConv = srcMemDesc.checkGeneralLayout(GeneralLayout::ncsp)
                             && dstMemDesc.checkGeneralLayout(GeneralLayout::ncsp)
                             && srcDataType == memory::data_type::f32
                             && dstDataType == memory::data_type::f32;

    return !isPossibleJitPlanar && isPlanarFloatConv;
}

std::unique_ptr<MKLDNNMemoryDesc> MKLDNNConvolutionNode::getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    auto desc = idx > 0 ? MKLDNNMemoryDesc(primitive_desc_it.weights_desc(idx - 1)) : MKLDNNMemoryDesc(primitive_desc_it.src_desc(idx));

    if (getParentEdgeAt(idx)->getShape().getRank() != desc.getShape().getRank()) {
        auto old_dims = getParentEdgeAt(idx)->getShape().getDims();
        auto new_dims = InferenceEngine::SizeVector({groupNum, div_up(old_dims[0], groupNum)});
        for (int i = 1; i < old_dims.size(); i++) {
            new_dims.push_back(old_dims[i]);
        }

        return MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(MKLDNNDims(new_dims), desc.getDataType(), desc.getFormat());
    } else {
        return MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(std::move(desc));
    }
}

bool MKLDNNConvolutionNode::canFuse(const MKLDNNNodePtr& node) const {
    return canFuseSimpleOperation(node);
}

const mkldnn::memory& MKLDNNConvolutionNode::getWeights() const {
    return getParentEdgeAt(1)->getMemory().GetPrimitive();
}

const mkldnn::memory& MKLDNNConvolutionNode::getBias() const {
    return getParentEdgeAt(2)->getMemory().GetPrimitive();
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

bool MKLDNNConvolutionNode::isNspcAvailable() const {
    using impl::cpu::x64::mayiuse;

    // do not use in non-quantized networks until it is enforced externally
    if (!isInQuantizedGraph) {
        auto predicate = [](memory::format_tag tag) {
            return one_of(tag, memory::format_tag::nwc, memory::format_tag::nhwc, memory::format_tag::ndhwc);
        };
        if (std::none_of(inputMemoryFormatsFilter.begin(), inputMemoryFormatsFilter.end(), predicate)) {
            return false;
        }
    }

    // A bunch of heuristics are designed to cut off not optimal nspc convolution applications
    auto inpDims = getParentEdgeAt(0)->getShape().getStaticDims();
    auto outDims = getChildEdgeAt(0)->getShape().getStaticDims();
    auto ndims = inpDims.size();

    if (isDepthWise()) {
        // 1d equivalent cases are painfully slow
        if (1 == inpDims[inpDims.size() - 2]) {
            return false;
        }
    } else {
        // it was empirically observed that the nspc convolutions perform much slower than the blocked ones if the channels number more than the specific value
        size_t spatialRank = ndims - 2; //two means batch dim plus channels dim

        bool is1x1 = false;

        if (!isGrouped) {
            auto weightDimsReversItr = weightDims.crbegin();
            auto inpDimsReversItr = inpDims.crbegin();
            auto outDimsReversItr = outDims.crbegin();
            auto paddingLreversItr = paddingL.crbegin();
            auto paddingRreversItr = paddingR.crbegin();

            for (size_t i = 0; i < spatialRank; ++i) {
                is1x1 = true
                        && *(weightDimsReversItr++) == 1
                        && *(inpDimsReversItr++) == *(outDimsReversItr++)
                        && *(paddingLreversItr++) == 0
                        && *(paddingRreversItr++) == 0;
            }
        }

        // if the activation field size is 1x1 the avx512 1x1 nspc convolution pollutes caches so that the layer after the convolution performs slow
        if (mayiuse(impl::cpu::x64::avx512_common) && is1x1) {
            auto end = inpDims.rbegin();
            std::advance(end, spatialRank);
            if (std::all_of(inpDims.rbegin(), end, [](size_t x) { return 1 == x; })) {
                return false;
            }
        }

        unsigned thresholdNumChannels = 128u; // for avx and below
        if (is1x1) {
            thresholdNumChannels = 2048u;
        } else if (mayiuse(impl::cpu::x64::avx512_common)) {
            thresholdNumChannels = 512u;
        }

        size_t OC = outDims[1];
        if (std::max(IC, OC) >= thresholdNumChannels) {
            return false;
        }
        if (!mayiuse(impl::cpu::x64::avx)) {
            // SSE41 nspc convolutions do not support ic and oc tails yet and the blocked implementation will be much better than gemm
            if ((IC % 8) || (OC % 8)) {
                return false;
            }
        }
    }

    return true;
}

InferenceEngine::Blob::Ptr MKLDNNConvolutionNode::createInternalBlob(InferenceEngine::SizeVector dims, size_t edgeNum, bool isGrouped) {
    const auto constNode = std::dynamic_pointer_cast<MKLDNNInputNode>(getParentEdgeAt(edgeNum)->getParent());
    if (!constNode) {
        IE_THROW() << "Cannot cast " << edgeNum << " input to Input node for " << getName() << ".";
    }
    auto blb = constNode->getMemoryPtr();
    if (blb == nullptr)
        IE_THROW() << "Cannot get const blob for node " << getName() << ".";

    auto const elementsCount = blb->GetElementsCount();

    InferenceEngine::TensorDesc desc(InferenceEngine::Precision::FP32, dims, getWeightsLayoutByDims(dims, isGrouped));

    Blob::Ptr internalBlob = InferenceEngine::make_shared_blob<float>(desc);
    internalBlob->allocate();

    if (internalBlob->size() != elementsCount) {
        IE_THROW() << "Created internal blob and const blob has different size for node: " << getName() << ".";
    }

    cpu_convert(blb->GetPtr(),
                internalBlob->buffer(),
                MKLDNNExtensionUtils::DataTypeToIEPrecision(blb->GetDataType()),
                internalBlob->getTensorDesc().getPrecision(),
                elementsCount);

    return internalBlob;
}

REG_MKLDNN_PRIM_FOR(MKLDNNConvolutionNode, Convolution);
