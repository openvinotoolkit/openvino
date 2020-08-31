// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_deconv_node.h"
#include "desc_iterator.hpp"
#include <legacy/ie_layers.h>
#include <mkldnn.hpp>
#include <string>
#include <vector>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <legacy/ie_layers_internal.hpp>
#include "ie_parallel.hpp"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNDeconvolutionNode::MKLDNNDeconvolutionNode(const InferenceEngine::CNNLayerPtr& layer,
                                                 const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(layer, eng, cache) {
    internalBlobDesc.emplace_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(0).desc());
    });
}

void MKLDNNDeconvolutionNode::getSupportedDescriptors() {
    if (!descs_fwd.empty() && !descs_bwd.empty())
        return;

    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
    precision = getCnnLayer()->outData[0]->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    if (getParentEdges().empty() || getParentEdges().size() > 3)
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();

    auto * deconvLayer = dynamic_cast<DeconvolutionLayer*>(getCnnLayer().get());
    if (deconvLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert deconvolution layer.";
    if (getParentEdges().size() == 1 && deconvLayer->_weights == nullptr) {
        THROW_IE_EXCEPTION << "Weights are empty for layer: " << deconvLayer->name
                           << " used in MKLDNN node: " << getName() << "\n"
                           << "Use the second argumemt of InferenceEngine::Core::ReadNetwork"
                           << " to load them from .bin part of the IR";
    }
    withGroups = (deconvLayer->_group > 1);
    isDW = withGroups && deconvLayer->_group == deconvLayer->_out_depth &&
            deconvLayer->_group == deconvLayer->input()->getDims()[1];

    bool withBiases = (deconvLayer->_biases != nullptr && deconvLayer->_biases->size() != 0) || getParentEdges().size() == 3;
    if (withBiases) {
        Blob::Ptr biases;

        if (getParentEdges().size() == 3) {
            auto biasLayer = getParentEdgesAtPort(2)[0]->getParent()->getCnnLayer();
            if (biasLayer->type != "Const")
                THROW_IE_EXCEPTION << "Deconvolution layer with name '" << getName() << "' doesn't support non-constant biases";
            biases = biasLayer->blobs["custom"];
        } else {
            biases = deconvLayer->_biases;
        }

        //  WA: we add bias as depthwise post op
        setBiasAsPostOp(biases);
    }

    /* Original layout format for deconv weights is iohw (from Caffe).
     * We specify oihw, but mean iohw, because there are no more
     * suitable format in MKLDNN.
     */
    SizeVector weightDims;
    if (withGroups) {
        weightDims = {
                deconvLayer->_group,
                deconvLayer->input()->getTensorDesc().getDims()[1] / deconvLayer->_group,
                deconvLayer->_out_depth / deconvLayer->_group,
        };
        groupNum = deconvLayer->_group;
    } else {
        weightDims = {
                deconvLayer->input()->getTensorDesc().getDims()[1],
                deconvLayer->_out_depth
        };
    }
    for (int i = 1; i <= deconvLayer->_kernel.size(); i++) {
        weightDims.push_back(deconvLayer->_kernel[deconvLayer->_kernel.size() - i]);
    }

    if (getParentEdges().size() == 1)
        internalBlobs.push_back(createInternalBlob(weightDims, true));

    invertVectorCopyUtoI(deconvLayer->_stride, stride);
    for (int i = 1; i <= deconvLayer->_dilation.size(); i++) {
        dilation.push_back(static_cast<int>(deconvLayer->_dilation[deconvLayer->_dilation.size() - i]) - 1);
    }
    auto allPads = getPaddings(*deconvLayer);
    invertVectorCopyUtoI(allPads.begin, paddingL);
    invertVectorCopyUtoI(allPads.end, paddingR);

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

    for (auto format : getAvailableFormatsForDims(getParentEdgeAt(0)->getDims())) {
        MKLDNNMemoryDesc in_candidate(getParentEdgeAt(0)->getDims(), inputDataType, format);
        MKLDNNMemoryDesc out_candidate(getChildEdgeAt(0)->getDims(), outputDataType, format);
        createDescriptor({in_candidate}, {out_candidate});
    }
}

void MKLDNNDeconvolutionNode::setBiasAsPostOp(const InferenceEngine::Blob::Ptr& biases) {
    mkldnn::post_ops ops;
    MKLDNNDims depthwiseDims({static_cast<ptrdiff_t>(rnd_up(biases->size(), 16))});

    PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
    PostOpsIntBlobMemory[0]->Create(depthwiseDims, memory::data_type::f32, memory::format::x);
    PostOpsIntBlobMemory[0]->FillZero();
    std::vector<float> weights(biases->size());
    for (int i = 0; i < biases->size(); i++) {
        weights[i] = 1;
    }
    PostOpsIntBlobMemory[0]->SetData(memory::data_type::f32, memory::x, &weights[0],
            biases->size() * MKLDNNExtensionUtils::sizeOfDataType(memory::data_type::f32));

    PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
    PostOpsIntBlobMemory[1]->Create(depthwiseDims, memory::data_type::f32, memory::format::x);
    PostOpsIntBlobMemory[1]->FillZero();
    PostOpsIntBlobMemory[1]->SetData(memory::data_type::f32, memory::x, biases->buffer(),
            biases->size() * MKLDNNExtensionUtils::sizeOfDataType(memory::data_type::f32));

    ops.append_depthwise(depthwise_scale_shift,
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
            THROW_IE_EXCEPTION << "Incorrect number of input or output memory formats for Deconvolution node";
        }
        auto itd = descs.begin();
        while (itd != descs.end()) {
            bool isSuitableDesc = true;
            if (!inputMemoryFormatsFilter.empty()) {
                auto src_fmt = std::shared_ptr<mkldnn::convolution_backward_data::desc>(*itd)->data.src_desc.format;
                if (src_fmt != inputMemoryFormatsFilter[0])
                    isSuitableDesc = false;
            }
            if (!outputMemoryFormatsFilter.empty()) {
                auto dst_fmt = std::shared_ptr<mkldnn::convolution_backward_data::desc>(*itd)->data.dst_desc.format;
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

void MKLDNNDeconvolutionNode::execute(mkldnn::stream strm) {
    if (prim) {
        strm.submit({*prim});
    }
}

bool MKLDNNDeconvolutionNode::created() const {
    return getType() == Deconvolution;
}

void MKLDNNDeconvolutionNode::createPrimitive() {
    if (prim)
        return;

    auto prim_desc = createPrimitiveDescriptor<convolution_backward_data::primitive_desc,
            convolution_backward_data::desc, convolution_forward::primitive_desc>(attr);

    prim.reset(new convolution_backward_data(prim_desc,
            getParentEdgeAt(0)->getMemory().GetPrimitive(),
            getWeights(),
            getChildEdgeAt(0)->getMemory().GetPrimitive()));
}

void MKLDNNDeconvolutionNode::createDescriptor(const std::vector<InferenceEngine::TensorDesc> &inputDesc,
                                               const std::vector<InferenceEngine::TensorDesc> &outputDesc) {
    MKLDNNMemoryDesc in_candidate(inputDesc[0]);
    MKLDNNMemoryDesc out_candidate(outputDesc[0]);

    // grouping and autoblicking is not compatible
    if ((withGroups && !isDW) && (in_candidate.blocksExtended() || out_candidate.blocksExtended()))
        return;

    MKLDNNMemoryDesc wgh_candidate{weightsDims, in_candidate.getDataType(), memory::any};
    for (auto alg : {algorithm::convolution_winograd, algorithm::convolution_direct}) {
        try {
            std::shared_ptr<mkldnn::convolution_forward::desc> conv_desc;
            conv_desc.reset(new convolution_forward::desc(prop_kind::forward_inference, alg,
                                                          out_candidate, wgh_candidate, in_candidate, stride, dilation,
                                                          paddingL, paddingR, padding_kind::zero));

            std::shared_ptr<mkldnn::convolution_backward_data::desc> deconv_desc;
            deconv_desc.reset(new convolution_backward_data::desc(alg, out_candidate, wgh_candidate,
                                                        in_candidate, stride, dilation, paddingL, paddingR,
                                                        padding_kind::zero));
            descs_fwd.push_back(conv_desc);
            descs_bwd.push_back(deconv_desc);

            descs.emplace_back(deconv_desc,
                               std::shared_ptr<convolution_forward::primitive_desc>(
                                       new convolution_forward::primitive_desc(*conv_desc, getEngine())));
        } catch(...) {}
    }
}

MKLDNNMemoryDesc MKLDNNDeconvolutionNode::getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    InferenceEngine::TensorDesc desc = idx > 0 ? MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(idx - 1).desc())
                                               : MKLDNNMemoryDesc(primitive_desc_it.diff_dst_primitive_desc(idx).desc());

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
    InferenceEngine::TensorDesc desc = MKLDNNMemoryDesc(primitive_desc_it.diff_src_primitive_desc(idx).desc());
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

REG_MKLDNN_PRIM_FOR(MKLDNNDeconvolutionNode, Deconvolution);
