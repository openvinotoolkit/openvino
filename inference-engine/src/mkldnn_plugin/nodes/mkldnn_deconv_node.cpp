// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_deconv_node.h"
#include "desc_iterator.hpp"
#include <ie_layers.h>
#include <mkldnn.hpp>
#include <string>
#include <vector>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNDeconvolutionNode::MKLDNNDeconvolutionNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng) : MKLDNNNode(layer, eng) {
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

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges.";
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges.";

    auto * deconvLayer = dynamic_cast<DeconvolutionLayer*>(getCnnLayer().get());
    if (deconvLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert deconvolution layer.";
    if (deconvLayer->_weights == nullptr) {
        THROW_IE_EXCEPTION << "Weights are empty for layer: " << deconvLayer->name
                           << " used in MKLDNN node: " << getName() << "\n"
                           << "Use ReadWeights and SetWeights methods of InferenceEngine::CNNNetReader"
                           << " to load them from .bin part of the IR";
    }
    withGroups = (deconvLayer->_group > 1);
    isDW = withGroups && deconvLayer->_group == deconvLayer->_out_depth &&
            deconvLayer->_group == deconvLayer->input()->getDims()[1];
    withBiases = (deconvLayer->_biases != nullptr && deconvLayer->_biases->size() != 0);
    if (withBiases)
        biases = deconvLayer->_biases;

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
                deconvLayer->_kernel_y,
                deconvLayer->_kernel_x
        };
    } else {
        weightDims = {
                deconvLayer->input()->getTensorDesc().getDims()[1],
                deconvLayer->_out_depth,
                deconvLayer->_kernel_y,
                deconvLayer->_kernel_x
        };
    }

    internalBlobs.push_back(createInternalBlob(weightDims, true));

    stride = {static_cast<int>(deconvLayer->_stride_y), static_cast<int>(deconvLayer->_stride_x)};
    paddingL = {static_cast<int>(deconvLayer->_padding_y), static_cast<int>(deconvLayer->_padding_x)};
    dilation = {static_cast<int>(deconvLayer->_dilation_y) - 1, static_cast<int>(deconvLayer->_dilation_x) - 1};
    paddingR = {0, 0};

    weightsDims = MKLDNNDims(weightDims);

    for (int i = 0; i < 2; i++) {
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

void MKLDNNDeconvolutionNode::execute(mkldnn::stream strm) {
    if (prim) {
        strm.submit({*prim});
    }
    if (withBiases) {
        const auto *bias = biases->buffer().as<const float*>();

        auto& dst = getChildEdgeAt(0)->getMemory();

        float *output = reinterpret_cast<float*>(dst.GetData()) +
                        dst.GetDescriptor().data.layout_desc.blocking.offset_padding;

        const auto &dims = dst.GetDims();

        const int N = dims[0];
        const int C = dims[1];
        const int H = dims[2];
        const int W = dims[3];

        memory::format fmt = MKLDNNMemoryDesc(getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc).getFormat();

        const int blksize = fmt == memory::nChw16c ? 16 :
                            fmt == memory::nChw8c ? 8 : 1;

#   pragma omp parallel for collapse(4) schedule(static)
        for (int n = 0; n < N; ++n)
        for (int c = 0; c < C / blksize; ++c)
        for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w) {
            const int off =
                    n * C * H * W + c * H * W * blksize +
                    h * W * blksize + w * blksize;
            auto o = &output[off];
            for (int bc = 0; bc < blksize; ++bc) {
                o[bc] += bias[c*blksize + bc];
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

    auto prim_desc = createPrimitiveDescriptor<convolution_backward_data::primitive_desc,
            convolution_backward_data::desc, convolution_forward::primitive_desc>();

    prim.reset(new convolution_backward_data(prim_desc,
            getParentEdgeAt(0)->getMemory().GetPrimitive(),
            internalBlobMemory[0]->GetPrimitive(),
            getChildEdgeAt(0)->getMemory().GetPrimitive()));
}

void MKLDNNDeconvolutionNode::createDescriptor(const std::vector<InferenceEngine::TensorDesc> &inputDesc,
                                               const std::vector<InferenceEngine::TensorDesc> &outputDesc) {
    MKLDNNMemoryDesc in_candidate(inputDesc[0]);
    MKLDNNMemoryDesc out_candidate(outputDesc[0]);
    auto in_fmt = in_candidate.getFormat();
    auto out_fmt = out_candidate.getFormat();
    int O_IND = withGroups ? 1 : 0;
    int I_IND = withGroups ? 2 : 1;

    // grouping and autoblicking is not compatible
    if ((withGroups && !isDW) && (in_candidate.blocksExtended() || out_candidate.blocksExtended()))
        return;

    MKLDNNDims blocked_weightDims(weightsDims);

    if (!withGroups) {
        if (in_fmt == memory::nChw16c) {
            blocked_weightDims[I_IND] = rnd_up(blocked_weightDims[I_IND], 16);
        } else if (in_fmt == memory::nChw8c) {
            blocked_weightDims[I_IND] = rnd_up(blocked_weightDims[I_IND], 8);
        }

        if (out_fmt == memory::nChw16c) {
            blocked_weightDims[O_IND] = rnd_up(blocked_weightDims[O_IND], 16);
        } else if (out_fmt == memory::nChw8c) {
            blocked_weightDims[O_IND] = rnd_up(blocked_weightDims[O_IND], 8);
        }
    } else if (isDW) {
        if (out_fmt != in_fmt)
            return;

        if (in_fmt == memory::nChw16c) {
            blocked_weightDims[0] = rnd_up(blocked_weightDims[0], 16);
        } else if (in_fmt == memory::nChw8c) {
            blocked_weightDims[0] = rnd_up(blocked_weightDims[0], 8);
        }
    }

    MKLDNNMemoryDesc wgh_candidate{blocked_weightDims, in_candidate.getDataType(), memory::any};

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
    InferenceEngine::TensorDesc desc = MKLDNNMemoryDesc(primitive_desc_it.diff_dst_primitive_desc(idx).desc());
    if (desc.getLayout() == InferenceEngine::Layout::ANY)
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getParentEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getLayout()));
    else
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getParentEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getBlockingDesc()));
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
