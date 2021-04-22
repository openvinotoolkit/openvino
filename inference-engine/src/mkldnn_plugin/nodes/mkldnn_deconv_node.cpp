// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_deconv_node.h"
#include <legacy/ie_layers.h>
#include <mkldnn.hpp>
#include <string>
#include <vector>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <legacy/ie_layers_internal.hpp>
#include <nodes/common/cpu_memcpy.h>
#include "ie_parallel.hpp"
#include "utils/general_utils.h"
#include <cpu/x64/cpu_isa_traits.hpp>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNDeconvolutionNode::MKLDNNDeconvolutionNode(const InferenceEngine::CNNLayerPtr& layer,
                                                 const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(layer, eng, cache) {
    internalBlobDesc.emplace_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return MKLDNNMemoryDesc(primitive_desc_it.weights_desc(0));
    });
    internalBlobDesc.emplace_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return MKLDNNMemoryDesc(primitive_desc_it.weights_desc(1));
    });
}

InferenceEngine::Blob::Ptr MKLDNNDeconvolutionNode::createWeiBlobAsIO(InferenceEngine::SizeVector dims) {
    auto checkSize = [](size_t dst_size, size_t src_size) {
        if (dst_size < src_size) {
            IE_THROW() << "Cannot create internal buffer. Buffer can be overrun.";
        }
    };
    auto * wLayer = dynamic_cast<InferenceEngine::WeightableLayer*>(getCnnLayer().get());
    if (wLayer == nullptr)
        IE_THROW() << "Cannot get weightable layer for node " << getName() << ".";

    InferenceEngine::Blob::Ptr blb = wLayer->_weights;

    if (blb == nullptr)
        IE_THROW() << "Cannot get internal blob layer for node " << getName() << ".";

    InferenceEngine::SizeVector dimsForBlockedDesc{dims};
    dimsForBlockedDesc.insert(dimsForBlockedDesc.begin() + 2 + withGroups, dimsForBlockedDesc[0 + withGroups]);
    dimsForBlockedDesc.erase(dimsForBlockedDesc.begin() + withGroups);

    InferenceEngine::SizeVector orderForBlockedDesc;
    if (withGroups) {
        orderForBlockedDesc = {0, 2, 1};
    } else {
        orderForBlockedDesc = {1, 0};
    }
    for (int i = 2 + withGroups; i < dimsForBlockedDesc.size(); i++)
        orderForBlockedDesc.push_back(i);

    BlockingDesc blkDesc(dimsForBlockedDesc, orderForBlockedDesc);

    InferenceEngine::TensorDesc desc(blb->getTensorDesc().getPrecision(), dims, blkDesc);

    auto fillInternalBlob = [&](char *data, size_t intBuffSize) {
        size_t offset = blb->byteSize();
        checkSize(intBuffSize, offset);
        cpu_memcpy_s(data, intBuffSize, blb->buffer(), blb->byteSize());
        data += blb->byteSize();
        for (const auto &merged : getMergeWith()) {
            wLayer = dynamic_cast<InferenceEngine::WeightableLayer*>(merged->getCnnLayer().get());
            if (wLayer == nullptr)
                IE_THROW() << "Cannot convert merged weightable layer for node "
                           << getName() << ".";
            blb = wLayer->_weights;

            if (blb == nullptr)
                IE_THROW() << "Cannot get internal blob layer for node " << getName() << ".";
            offset += blb->byteSize();
            checkSize(intBuffSize, offset);
            cpu_memcpy_s(data, intBuffSize, blb->buffer(), blb->byteSize());
            data += blb->byteSize();
        }
    };

    Blob::Ptr internalBlob = InferenceEngine::make_shared_blob<int8_t>(desc);
    internalBlob->allocate();
    char *data = internalBlob->buffer();
    size_t intBuffSize = internalBlob->byteSize();

    fillInternalBlob(data, intBuffSize);

    return internalBlob;
}

bool MKLDNNDeconvolutionNode::canBeExecutedInInt8() {
    if (!impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_common))
        return false;

    if (!fusedWith.empty())
        return false;

    for (int i = 0; i < kernel.size(); i++) {
        if (kernel[i] < stride[i])
            return false;
    }

    if (withGroups && !isDW && (IC % 16 != 0 || OC % 16 != 0))
        return false;

    auto * deconvLayer = dynamic_cast<DeconvolutionLayer*>(getCnnLayer().get());
    if (deconvLayer == nullptr)
        IE_THROW() << "Cannot convert deconvolution layer.";

    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(getCnnLayer()->insData[0].lock()->getPrecision());
    if (deconvLayer->_weights == nullptr)
        return false;
    auto weightsDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(deconvLayer->_weights->getTensorDesc().getPrecision());

    if (isDW && (inputDataType == dnnl_s8 || kernel.size() == 3))
        return false;

    return (inputDataType == dnnl_s8 || inputDataType == dnnl_u8) && weightsDataType == dnnl_s8;
}

void MKLDNNDeconvolutionNode::getSupportedDescriptors() {
    if (!descs_fwd.empty() && !descs_bwd.empty())
        return;

    if (getParentEdges().empty() || getParentEdges().size() > 3)
        IE_THROW() << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        IE_THROW() << "Incorrect number of output edges for layer " << getName();

    auto * deconvLayer = dynamic_cast<DeconvolutionLayer*>(getCnnLayer().get());
    if (deconvLayer == nullptr)
        IE_THROW() << "Cannot convert deconvolution layer.";
    if (getParentEdges().size() == 1 && deconvLayer->_weights == nullptr) {
        IE_THROW() << "Weights are empty for layer: " << deconvLayer->name
                   << " used in MKLDNN node: " << getName() << "\n"
                   << "Use the second argumemt of InferenceEngine::Core::ReadNetwork"
                   << " to load them from .bin part of the IR";
    }

    groupNum = deconvLayer->_group;
    withGroups = groupNum > 1;
    IC = deconvLayer->input()->getTensorDesc().getDims()[1] / groupNum;
    OC = deconvLayer->_out_depth / groupNum;
    isDW = withGroups && IC == 1 && OC == 1;

    invertVectorCopyUtoI(deconvLayer->_kernel, kernel);
    invertVectorCopyUtoI(deconvLayer->_stride, stride);
    for (int i = 1; i <= deconvLayer->_dilation.size(); i++) {
        dilation.push_back(static_cast<int>(deconvLayer->_dilation[deconvLayer->_dilation.size() - i]) - 1);
    }
    auto allPads = getPaddings(*deconvLayer);
    invertVectorCopyUtoI(allPads.begin, paddingL);
    invertVectorCopyUtoI(allPads.end, paddingR);

    isInt8 = canBeExecutedInInt8();

    InferenceEngine::Precision inPrecision = getCnnLayer()->insData[0].lock()->getPrecision();
    InferenceEngine::Precision outPrecision = getCnnLayer()->outData[0]->getPrecision();
    if (!isInt8) {
        if (inPrecision != InferenceEngine::Precision::FP32 && inPrecision != InferenceEngine::Precision::BF16)
            inPrecision = InferenceEngine::Precision::FP32;
        if (outPrecision != InferenceEngine::Precision::FP32 && outPrecision != InferenceEngine::Precision::BF16)
            outPrecision = InferenceEngine::Precision::FP32;
    }
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(inPrecision);
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(outPrecision);
    if (inputDataType == memory::data_type::bf16 || outputDataType == memory::data_type::bf16)
       inputDataType = outputDataType = memory::data_type::bf16;

    SizeVector weightDims;
    if (isInt8) {
        weightDims = {OC, IC};
    } else {
        /* Original layout format for deconv weights is iohw (from Caffe).
         * We specify oihw, but mean iohw, because there are no more
         * suitable format in MKLDNN.
         */
        weightDims = {IC, OC};
    }
    if (withGroups)
        weightDims.insert(weightDims.begin(), groupNum);

    for (int i = 0; i < kernel.size(); i++) {
        weightDims.push_back(kernel[i]);
    }

    weightsDims = MKLDNNDims(weightDims);

    for (int i = 0; i < paddingR.size(); i++) {
        int with_group = (withGroups) ? 1 : 0;
        int krn = weightsDims[with_group + 2 + i];
        int src = getChildEdgeAt(0)->getDims()[2 + i];
        int dst = getParentEdgeAt(0)->getDims()[2 + i];

        krn = (krn - 1)*(dilation[i] + 1) + 1;
        int calc_dst = (src - krn + paddingL[i]) / stride[i] + 1;
        paddingR[i] = (dst - calc_dst) * stride[i];
    }

    withBiases = (deconvLayer->_biases != nullptr && deconvLayer->_biases->size() != 0) || getParentEdges().size() == 3;
    if (withBiases && !isInt8) {
        Blob::Ptr biases;

        if (getParentEdges().size() == 3) {
            auto biasLayer = getParentEdgesAtPort(2)[0]->getParent()->getCnnLayer();
            if (biasLayer->type != "Const")
                IE_THROW() << "Deconvolution layer with name '" << getName() << "' doesn't support non-constant biases";
            biases = biasLayer->blobs["custom"];
        } else {
            biases = deconvLayer->_biases;
        }

        //  WA: we add bias as depthwise post op
        setBiasAsPostOp(biases);
    }

    if (isInt8) {
        if (getParentEdges().size() == 1) {
            //  WA: if int8 deconvolution is supported, we create internal weights blob in IO format
            internalBlobs.push_back(createWeiBlobAsIO(weightDims));
            if (withBiases) {
                internalBlobs.push_back(createInternalBlob({ OC * groupNum }, false));
            }
        }
    } else {
        if (getParentEdges().size() == 1)
            internalBlobs.push_back(createInternalBlob(weightDims, true));
    }

    if (isInt8) {
        auto format = getParentEdgeAt(0)->getDims().ndims() == 5 ? dnnl::memory::format_tag::ndhwc : dnnl::memory::format_tag::nhwc;
        MKLDNNMemoryDesc in_candidate(getParentEdgeAt(0)->getDims(), inputDataType, format);
        MKLDNNMemoryDesc out_candidate(getChildEdgeAt(0)->getDims(), outputDataType, format);
        createDescriptor({in_candidate}, {out_candidate});
    } else {
        for (auto format : getAvailableFormatsForDims(getParentEdgeAt(0)->getDims())) {
            MKLDNNMemoryDesc in_candidate(getParentEdgeAt(0)->getDims(), inputDataType, format);
            MKLDNNMemoryDesc out_candidate(getChildEdgeAt(0)->getDims(), outputDataType, format);
            createDescriptor({in_candidate}, {out_candidate});
        }
    }
}

void MKLDNNDeconvolutionNode::setBiasAsPostOp(const InferenceEngine::Blob::Ptr& biases) {
    mkldnn::post_ops ops;
    auto depthwiseSize = static_cast<ptrdiff_t>(rnd_up(biases->size(), 16));

    PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
    PostOpsIntBlobMemory[0]->Create({depthwiseSize}, memory::data_type::f32, memory::format_tag::x);
    PostOpsIntBlobMemory[0]->FillZero();
    std::vector<float> weights(depthwiseSize, 1.0f);
    std::fill(weights.begin() + biases->size(), weights.end(), 0.0f);
    PostOpsIntBlobMemory[0]->SetData(memory::data_type::f32, memory::format_tag::x, weights.data(),
            weights.size() * MKLDNNExtensionUtils::sizeOfDataType(memory::data_type::f32));

    PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
    PostOpsIntBlobMemory[1]->Create({depthwiseSize}, memory::data_type::f32, memory::format_tag::x);
    PostOpsIntBlobMemory[1]->FillZero();
    auto biases_ptr = biases->buffer().as<float*>();
    std::vector<float> bias(depthwiseSize, 0.0f);
    std::copy(biases_ptr, biases_ptr + biases->size(), bias.begin());

    InferenceEngine::Precision biasPrecision = biases->getTensorDesc().getPrecision();
    auto biasDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(biasPrecision);
    PostOpsIntBlobMemory[1]->SetData(biasDataType, memory::format_tag::x, bias.data(),
             bias.size() * MKLDNNExtensionUtils::sizeOfDataType(memory::data_type::f32));

    ops.append_depthwise(algorithm::depthwise_scale_shift,
                         (const float *) PostOpsIntBlobMemory[0]->GetData(),
                         (const float *) PostOpsIntBlobMemory[1]->GetData());

    attr.set_post_ops(ops);
}

void MKLDNNDeconvolutionNode::filterSupportedPrimitiveDescriptors() {
    MKLDNNNode::filterSupportedPrimitiveDescriptors();
    filterSupportedDescriptors();
}

void MKLDNNDeconvolutionNode::filterSupportedDescriptors() {
    if (!inputMemoryFormatsFilter.empty() || !outputMemoryFormatsFilter.empty()) {
        if (inputMemoryFormatsFilter.size() > 1 || outputMemoryFormatsFilter.size() > 1) {
            IE_THROW() << "Incorrect number of input or output memory formats for Deconvolution node";
        }
        auto itd = descs.begin();
        while (itd != descs.end()) {
            bool isSuitableDesc = true;
            if (!inputMemoryFormatsFilter.empty()) {
                if (isInt8) {
                    auto src_tdesc = MKLDNNMemoryDesc(std::shared_ptr<dnnl::deconvolution_forward::desc>(*itd)->data.src_desc);
                    isSuitableDesc &= src_tdesc.isSame(inputMemoryFormatsFilter[0]);
                } else {
                    auto src_tdesc = MKLDNNMemoryDesc(std::shared_ptr<mkldnn::convolution_backward_data::desc>(*itd)->data.diff_src_desc);
                    isSuitableDesc &= src_tdesc.isSame(inputMemoryFormatsFilter[0]);
                }
            }
            if (!outputMemoryFormatsFilter.empty()) {
                if (isInt8) {
                    auto dst_tdesc = MKLDNNMemoryDesc(std::shared_ptr<mkldnn::deconvolution_forward::desc>(*itd)->data.dst_desc);
                    isSuitableDesc &= dst_tdesc.isSame(outputMemoryFormatsFilter[0]);
                } else {
                    auto dst_tdesc = MKLDNNMemoryDesc(std::shared_ptr<mkldnn::convolution_backward_data::desc>(*itd)->data.diff_dst_desc);
                    isSuitableDesc &= dst_tdesc.isSame(outputMemoryFormatsFilter[0]);
                }
            }
            if (!isSuitableDesc) {
                itd = descs.erase(itd);
            } else {
                itd++;
            }
        }
    }
}

bool MKLDNNDeconvolutionNode::created() const {
    return getType() == Deconvolution;
}

void MKLDNNDeconvolutionNode::createPrimitive() {
    if (prim)
        return;

    if (isInt8) {
        auto prim_desc = createPrimitiveDescriptor<deconvolution_forward::primitive_desc,
                deconvolution_forward::desc>(attr);

        prim.reset(new deconvolution_forward(prim_desc));

        auto src = getParentEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
        auto dst = getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
        if (withBiases) {
            primArgs = {{DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, getWeights()}, {DNNL_ARG_DST, dst}, {DNNL_ARG_BIAS, getBiases()}};
        } else {
            primArgs = {{DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, getWeights()}, {DNNL_ARG_DST, dst}};
        }
    } else {
        auto prim_desc = createPrimitiveDescriptor<convolution_backward_data::primitive_desc,
                convolution_backward_data::desc, convolution_forward::primitive_desc>(attr);

        prim.reset(new convolution_backward_data(prim_desc));

        auto src = getParentEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
        auto dst = getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
        primArgs = {{DNNL_ARG_DIFF_DST, src}, {DNNL_ARG_WEIGHTS, getWeights()}, {DNNL_ARG_DIFF_SRC, dst}};
    }
}

void MKLDNNDeconvolutionNode::createDescriptor(const std::vector<InferenceEngine::TensorDesc> &inputDesc,
                                               const std::vector<InferenceEngine::TensorDesc> &outputDesc) {
    MKLDNNMemoryDesc in_candidate(inputDesc[0]);
    MKLDNNMemoryDesc out_candidate(outputDesc[0]);

    // grouping and autoblicking is not compatible
    if ((withGroups && !isDW) && (in_candidate.blocksExtended() || out_candidate.blocksExtended()))
        return;

    auto convert = [] (const std::vector<ptrdiff_t>& orig_dims) {
        return memory::dims(orig_dims.begin(), orig_dims.end());
    };

    if (isInt8) {
        MKLDNNMemoryDesc wgh_candidate{weightsDims, memory::data_type::s8, memory::format_tag::any};
        std::shared_ptr<mkldnn::deconvolution_forward::desc> deconv_desc;
        if (withBiases) {
            auto * deconvLayer = dynamic_cast<DeconvolutionLayer*>(getCnnLayer().get());
            auto biasDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(deconvLayer->_biases->getTensorDesc().getPrecision());
            MKLDNNMemoryDesc bias_candidate{{weightsDims[0]}, biasDataType, memory::format_tag::any};

            deconv_desc.reset(new deconvolution_forward::desc(prop_kind::forward_inference, algorithm::deconvolution_direct,
                                                               in_candidate, wgh_candidate, bias_candidate, out_candidate,
                                                               convert(stride), convert(dilation),
                                                               convert(paddingL), convert(paddingR)));
        } else {
            deconv_desc.reset(new deconvolution_forward::desc(prop_kind::forward_inference, algorithm::deconvolution_direct,
                                                               in_candidate, wgh_candidate, out_candidate,
                                                               convert(stride), convert(dilation),
                                                               convert(paddingL), convert(paddingR)));
        }
        descs.emplace_back(deconv_desc);
    } else {
        MKLDNNMemoryDesc wgh_candidate{weightsDims, in_candidate.getDataType(), memory::format_tag::any};
        for (auto alg : {algorithm::convolution_winograd, algorithm::convolution_direct}) {
            std::shared_ptr<mkldnn::convolution_forward::desc> conv_desc;
            conv_desc.reset(new convolution_forward::desc(prop_kind::forward_inference, alg,
                                                          out_candidate, wgh_candidate, in_candidate,
                                                          convert(stride),
                                                          convert(dilation),
                                                          convert(paddingL),
                                                          convert(paddingR)));

            std::shared_ptr<mkldnn::convolution_backward_data::desc> deconv_desc;
            deconv_desc.reset(new convolution_backward_data::desc(alg, out_candidate, wgh_candidate,
                                                                  in_candidate,
                                                                  convert(stride),
                                                                  convert(dilation),
                                                                  convert(paddingL),
                                                                  convert(paddingR)));
            descs_fwd.push_back(conv_desc);
            descs_bwd.push_back(deconv_desc);

            auto fwd_conv_pd = std::make_shared<convolution_forward::primitive_desc>(*conv_desc, getEngine(), true);
            if (fwd_conv_pd->get(true) == nullptr)
                continue;

            descs.emplace_back(deconv_desc, fwd_conv_pd);
        }
    }
}

MKLDNNMemoryDesc MKLDNNDeconvolutionNode::getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    InferenceEngine::TensorDesc desc = idx > 0 ? MKLDNNMemoryDesc(primitive_desc_it.weights_desc(idx - 1))
            : isInt8 ? MKLDNNMemoryDesc(primitive_desc_it.src_desc(idx)) : MKLDNNMemoryDesc(primitive_desc_it.diff_dst_desc(idx));

    if (desc.getLayout() == InferenceEngine::Layout::ANY) {
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getParentEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getLayout()));
    } else {
        if (getParentEdgeAt(idx)->getDims().ToSizeVector().size() != *std::max_element(desc.getBlockingDesc().getOrder().begin(),
                                                                                       desc.getBlockingDesc().getOrder().end()) + 1) {
            auto old_dims = getParentEdgeAt(idx)->getDims().ToSizeVector();
            auto new_dims = weightsDims.ToSizeVector();

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

MKLDNNMemoryDesc MKLDNNDeconvolutionNode::getDstMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    InferenceEngine::TensorDesc desc = isInt8 ? MKLDNNMemoryDesc(primitive_desc_it.dst_desc(idx))
            : MKLDNNMemoryDesc(primitive_desc_it.diff_src_desc(idx));
    if (desc.getLayout() == InferenceEngine::Layout::ANY)
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getChildEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getLayout()));
    else
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getChildEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getBlockingDesc()));
}

const mkldnn::memory& MKLDNNDeconvolutionNode::getWeights() const {
    return getParentEdges().size() > 1 ? getParentEdgeAt(1)->getMemory().GetPrimitive() : internalBlobMemory[0]->GetPrimitive();
}

const mkldnn::memory& MKLDNNDeconvolutionNode::getBiases() const {
    return internalBlobMemory[1]->GetPrimitive();
}

InferenceEngine::Precision MKLDNNDeconvolutionNode::getRuntimePrecision() const {
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

REG_MKLDNN_PRIM_FOR(MKLDNNDeconvolutionNode, Deconvolution);
