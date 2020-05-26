// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_reshape_node.h"
#include <ie_layers.h>
#include <string>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNReshapeNode::MKLDNNReshapeNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(layer, eng, cache) {}

void MKLDNNReshapeNode::getSupportedDescriptors() {
    if (getParentEdges().size() != 1 && getParentEdges().size() != 2)
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();
}

void MKLDNNReshapeNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    // The reshape implementation do nothing, just report correct layout config.
    // Additional auto inserted reorders will do all required routine if following
    // nodes doesn't natively support specified layout.
    //
    // The default and mandatory config is:
    //    plain_input -> plain_output[inplace]
    //
    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
    precision = getCnnLayer()->outData[0]->getPrecision();
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    auto& inDims = getParentEdgeAt(0)->getDims();
    auto& outDims = getChildEdgeAt(0)->getDims();

    // Current reshape implementation is simple memory reinterpret,
    // same precision on input and output is required
    if (inputDataType != outputDataType)
        inputDataType = outputDataType;

    auto get_plain_layout = [] (const MKLDNNDims &dims) {
        return MKLDNNMemory::GetPlainFormat(dims);
    };

    auto get_tail_c_layout = [] (const MKLDNNDims &dims) {
        switch (dims.ndims()) {
            case 2: return memory::nc;
            case 3: return memory::nwc;
            case 4: return memory::nhwc;
            case 5: return memory::ndhwc;
            default: return memory::format_undef;
        }
    };

    auto get_blocked_4c_layout = [] (const MKLDNNDims &dims) {
        switch (dims.ndims()) {
            case 2: return memory::nc; // blocked 2D is the same as plain 2D
            case 3: return memory::nCw4c;
            case 4: return memory::nChw4c;
            case 5: return memory::nCdhw4c;
            default: return memory::format_undef;
        }
    };

    auto get_blocked_8c_layout = [] (const MKLDNNDims &dims) {
        switch (dims.ndims()) {
            case 2: return memory::nc; // blocked 2D is the same as plain 2D
            case 3: return memory::nCw8c;
            case 4: return memory::nChw8c;
            case 5: return memory::nCdhw8c;
            default: return memory::format_undef;
        }
    };

    auto get_blocked_16c_layout = [] (const MKLDNNDims &dims) {
        switch (dims.ndims()) {
            case 2: return memory::nc; // blocked 2D is the same as plain 2D
            case 3: return memory::nCw16c;
            case 4: return memory::nChw16c;
            case 5: return memory::nCdhw16c;
            default: return memory::format_undef;
        }
    };

    auto add_reshape_config = [&] (memory::format in_format, memory::format out_format) {
        InferenceEngine::LayerConfig config;
        config.dynBatchSupport = true;
        config.inConfs.resize(1);
        config.inConfs[0].inPlace = -1;
        config.inConfs[0].constant = false;
        config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, in_format);

        config.outConfs.resize(1);
        config.outConfs[0].inPlace = 0;
        config.outConfs[0].constant = false;
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, out_format);
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown, out_format);
    };

    // Default plain layout config [d0,d1,d2,...,dn] like NCHW
    add_reshape_config(get_plain_layout(inDims), get_plain_layout(outDims));

    // Not trivial layouts which permute second dimension (channels) NC...
    if (inDims.ndims() > 1 && inDims.ndims() < 6 &&
        outDims.ndims() > 1 && outDims.ndims() < 6) {
        if (inDims[0] == inDims[0] && inDims[1] == inDims[1]) {
            // Permuted layout config [d0,d2,...,dn,d1] like NHWC
            add_reshape_config(get_tail_c_layout(inDims), get_tail_c_layout(outDims));

            // Blocked layout config [d0,d1/B,...,dn,B] like NCHW4c, NCHW8c or NCHW16c
            add_reshape_config(get_blocked_16c_layout(inDims), get_blocked_16c_layout(outDims));
            add_reshape_config(get_blocked_8c_layout(inDims), get_blocked_8c_layout(outDims));
            add_reshape_config(get_blocked_4c_layout(inDims), get_blocked_4c_layout(outDims));
        }
    }
}

void MKLDNNReshapeNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";
}

bool MKLDNNReshapeNode::created() const {
    return getType() == Reshape || getType() == Flatten;
}
REG_MKLDNN_PRIM_FOR(MKLDNNReshapeNode, Reshape);
