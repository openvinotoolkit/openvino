// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_fullyconnected_node.h"
#include "desc_iterator.hpp"
#include <ie_layers.h>
#include <string>
#include <vector>
#include <mkldnn_extension_utils.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNFullyConnectedNode::MKLDNNFullyConnectedNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng) : MKLDNNNode(layer, eng) {
    internalBlobDesc.emplace_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(0).desc());
    });
    internalBlobDesc.emplace_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        if (internalBlobs.size() <= 1)
            return MKLDNNMemoryDesc();
        return MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(1).desc());
    });
}

void MKLDNNFullyConnectedNode::getSupportedDescriptors() {
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

    auto * fcLayer = dynamic_cast<FullyConnectedLayer*>(getCnnLayer().get());
    if (fcLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert fully connected layer.";
    if (fcLayer->_weights == nullptr) {
        THROW_IE_EXCEPTION << "Weights are empty for layer: " << fcLayer->name
                           << " used in MKLDNN node: " << getName() << "\n"
                           << "Use ReadWeights and SetWeights methods of InferenceEngine::CNNNetReader"
                           << " to load them from .bin part of the IR";
    }

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges.";
    if (getParentEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges.";

    MKLDNNDims inDims(fcLayer->input()->getDims());

    if (inDims.ndims() == 2) {
        weightsDims = {fcLayer->_out_num, static_cast<size_t>(inDims.size(1))};
    } else if (inDims.ndims() == 4) {
        weightsDims = {fcLayer->_out_num, static_cast<size_t>(inDims[1]), static_cast<size_t>(inDims[2]),
                       static_cast<size_t>(inDims[3])};
    } else {
        THROW_IE_EXCEPTION << "Unsupported source format for FC layer. Expected 4 or 2, got: "
                           << inDims.ndims() << " dims.";
    }

    internalBlobs.push_back(createInternalBlob(weightsDims, true));

    bool withBiases = (fcLayer->_biases != nullptr && fcLayer->_biases->size() != 0);
    if (withBiases) {
        biasesDims.push_back(static_cast<int>(fcLayer->_out_num));
        internalBlobs.push_back(createInternalBlob(biasesDims, false));
    }

    for (auto format : getAvailableFormatsForDims(getParentEdgeAt(0)->getDims())) {
        MKLDNNMemoryDesc in_candidate(inDims, inputDataType, format);
        MKLDNNMemoryDesc out_candidate(getChildEdgeAt(0)->getDims(), outputDataType, memory::any);

        createDescriptor({in_candidate}, {out_candidate});
    }
}

void MKLDNNFullyConnectedNode::createPrimitive() {
    if (prim)
        return;

    auto prim_desc = createPrimitiveDescriptor<inner_product_forward::primitive_desc, inner_product_forward::desc>();

    if (internalBlobs.size() > 1) {
        prim.reset(new inner_product_forward(prim_desc,
                                             getParentEdgeAt(0)->getMemory().GetPrimitive(),
                                             internalBlobMemory[0]->GetPrimitive(),
                                             internalBlobMemory[1]->GetPrimitive(),
                                             getChildEdgeAt(0)->getMemory().GetPrimitive()));
    } else {
        prim.reset(new inner_product_forward(prim_desc,
                                             getParentEdgeAt(0)->getMemory().GetPrimitive(),
                                             internalBlobMemory[0]->GetPrimitive(),
                                             getChildEdgeAt(0)->getMemory().GetPrimitive()));
    }
}

bool MKLDNNFullyConnectedNode::created() const {
    return getType() == FullyConnected;
}

memory::format MKLDNNFullyConnectedNode::weightsFormatForSrcFormat(memory::format sourceFormat) {
    switch (sourceFormat) {
        case memory::format::x:
            return memory::format::x;
        case memory::format::nc:
            return memory::format::oi;
        case memory::format::nchw:
            return memory::format::oihw;
        case memory::format::nChw8c:
            return memory::format::oIhw8i;
        case memory::format::nChw16c:
            return memory::format::oIhw16i;
        default:
            THROW_IE_EXCEPTION << "Unsupported source format for node " << getName();
    }
}

const std::vector<impl_desc_type>& MKLDNNFullyConnectedNode::getPrimitivesPriority() {
    std::vector<impl_desc_type> priorities = {
            impl_desc_type::unknown,
            impl_desc_type::gemm_blas,
            impl_desc_type::gemm_avx512,
            impl_desc_type::gemm_avx2,
            impl_desc_type::gemm_sse42,
            impl_desc_type::gemm_any,
            impl_desc_type::gemm,
            impl_desc_type::jit_uni_dw,
            impl_desc_type::jit_uni_1x1,
            impl_desc_type::jit_uni,
            impl_desc_type::jit_avx512_dw,
            impl_desc_type::jit_avx512_1x1,
            impl_desc_type::jit_avx512,
            impl_desc_type::jit_avx2_dw,
            impl_desc_type::jit_avx2_1x1,
            impl_desc_type::jit_avx2,
            impl_desc_type::jit_sse42_dw,
            impl_desc_type::jit_sse42_1x1,
            impl_desc_type::jit_sse42,
            impl_desc_type::ref,
    };
    for (const auto& impl : priorities) {
        if (std::find(implPriorities.begin(), implPriorities.end(), impl) == implPriorities.end())
            implPriorities.push_back(impl);
    }
    return implPriorities;
}

void MKLDNNFullyConnectedNode::createDescriptor(const std::vector<InferenceEngine::TensorDesc> &inputDesc,
                                                const std::vector<InferenceEngine::TensorDesc> &outputDesc) {
    MKLDNNMemoryDesc in_candidate(inputDesc[0]);
    MKLDNNMemoryDesc out_candidate(outputDesc[0]);
    memory::format weights_fmt = weightsFormatForSrcFormat(in_candidate.getFormat());

    MKLDNNMemoryDesc wgh_candidate(MKLDNNDims(weightsDims), in_candidate.getDataType(), weights_fmt);
    MKLDNNMemoryDesc bias_candidate(MKLDNNDims(biasesDims), in_candidate.getDataType(), memory::any);

    if (internalBlobs.size() > 1) {
        MKLDNNDescriptor desc(std::shared_ptr<inner_product_forward::desc>(
                new inner_product_forward::desc(prop_kind::forward_scoring, in_candidate, wgh_candidate,
                                                bias_candidate, out_candidate)));
        descs.push_back(desc);
    } else {
        MKLDNNDescriptor desc(std::shared_ptr<inner_product_forward::desc>(
                new inner_product_forward::desc(prop_kind::forward_scoring, in_candidate, wgh_candidate,
                                                out_candidate)));
        descs.push_back(desc);
    }
}
