// Copyright (C) 2018 Intel Corporation
//
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

MKLDNNReshapeNode::MKLDNNReshapeNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng) : MKLDNNNode(layer, eng) {}

void MKLDNNReshapeNode::getSupportedDescriptors() {
    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges.";
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges.";
}

void MKLDNNReshapeNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
    precision = getCnnLayer()->outData[0]->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    auto& inDims = getParentEdgeAt(0)->getDims();
    auto& outDims = getChildEdgeAt(0)->getDims();
    memory::format outFormat = MKLDNNMemory::GetPlainFormat(outDims);
    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(1);
    config.outConfs.resize(1);
    config.inConfs[0].inPlace = -1;
    config.inConfs[0].constant = false;
    config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, MKLDNNMemory::GetPlainFormat(getParentEdgeAt(0)->getDims()));
    config.outConfs[0].inPlace = 0;
    config.outConfs[0].constant = false;
    config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, outFormat);
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
    if (inDims.ndims() == 4 && inDims[1] % 8 == 0 && outDims.ndims() == 4 &&outDims[1] % 8 == 0) {
        outFormat = memory::format::any;
    }
    config.inConfs[0].inPlace = -1;
    config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::format::any);
    config.outConfs[0].inPlace = -1;
    config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, outFormat);

    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
}

void MKLDNNReshapeNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor does not set.";

    if (srcMemPtr->GetData() != dstMemPtr->GetData()) {
        InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
        if (precision != InferenceEngine::Precision::FP32)
            precision = InferenceEngine::Precision::FP32;
        auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
        precision = getCnnLayer()->outData[0]->getPrecision();
        if (precision != InferenceEngine::Precision::FP32)
            precision = InferenceEngine::Precision::FP32;
        auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

        auto dims = getParentEdgeAt(0)->getDims();

        srcMem.reset(new MKLDNNMemory(getEngine()));
        srcMem->Create(dims, inputDataType, MKLDNNMemory::GetPlainFormat(dims));

        dstMem.reset(new MKLDNNMemory(getEngine()));
        dstMem->Create(getChildEdgeAt(0)->getDims(), outputDataType,
                       MKLDNNMemory::GetPlainFormat(getChildEdgeAt(0)->getDims()), srcMem->GetData());

        if (srcMemPtr->GetSize() == srcMem->GetSize()) {
            srcPrim.reset(new mkldnn::reorder(srcMemPtr->GetPrimitive(), srcMem->GetPrimitive()));
        } else {
            // Autoblocking mode
            memory::dims dims = srcMem->GetDims();  // contains logical dims

            memory::desc src_d = srcMemPtr->GetDescriptor();
            void *src_data_hdl = srcMemPtr->GetPrimitive().get_data_handle();

            for (int i = 0; i < dims.size(); i++)
                src_d.data.dims[i] =  dims[i];

            memory::primitive_desc tmp_src_pd(src_d, getEngine());
            src_blocked = std::make_shared<MKLDNNMemory>(getEngine());
            src_blocked->Create(src_d, src_data_hdl);

            srcPrim.reset(new mkldnn::reorder(src_blocked->GetPrimitive(), srcMem->GetPrimitive()));
        }

        if (dstMemPtr->GetSize() == dstMem->GetSize()) {
            dstPrim.reset(new mkldnn::reorder(dstMem->GetPrimitive(), dstMemPtr->GetPrimitive()));
        } else {
            // Autoblocking mode
            memory::dims dims = srcMem->GetDims();

            memory::desc dst_d = dstMemPtr->GetDescriptor();
            void *dst_data_hdl = dstMemPtr->GetPrimitive().get_data_handle();

            for (int i = 0; i < dims.size(); i++)
                dst_d.data.dims[i] =  dims[i];

            dst_blocked = std::make_shared<MKLDNNMemory>(getEngine());
            dst_blocked->Create(dst_d, dst_data_hdl);

            dstPrim.reset(new mkldnn::reorder(dst_blocked->GetPrimitive(), dstMemPtr->GetPrimitive()));
        }
    }
}

void MKLDNNReshapeNode::setDynamicBatchLim(int lim) {
    dynBatchLim = lim;
    if (srcPrim && dstPrim) {
        auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
        auto &srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
        memory::desc src_d = srcMemPtr->GetDescriptor();
        memory::desc dst_d = dstMemPtr->GetDescriptor();
        void *src_data_hdl = srcMemPtr->GetPrimitive().get_data_handle();
        void *dst_data_hdl = dstMemPtr->GetPrimitive().get_data_handle();
        srcMem = std::make_shared<MKLDNNMemory>(getEngine());
        src_d.data.dims[0] = batchToProcess();
        srcMem->Create(src_d, src_data_hdl);
        dstMemPtr = std::make_shared<MKLDNNMemory>(getEngine());
        src_d.data.dims[0] = batchToProcess();
        dstMemPtr->Create(src_d, src_data_hdl);

        if (src_blocked && dst_blocked) {
            src_d = src_blocked->GetDescriptor();
            dst_d = dst_blocked->GetDescriptor();
            src_data_hdl = src_blocked->GetPrimitive().get_data_handle();
            dst_data_hdl = dst_blocked->GetPrimitive().get_data_handle();
        }
        src_blocked = std::make_shared<MKLDNNMemory>(getEngine());
        src_d.data.dims[0] = batchToProcess();
        src_blocked->Create(src_d, src_data_hdl);

        dst_blocked = std::make_shared<MKLDNNMemory>(getEngine());
        dst_d.data.dims[0] = batchToProcess();
        dst_blocked->Create(dst_d, dst_data_hdl);
        srcPrim = std::make_shared<mkldnn::reorder>(src_blocked->GetPrimitive(), srcMem->GetPrimitive());
        dstPrim = std::make_shared<mkldnn::reorder>(dst_blocked->GetPrimitive(), dstMemPtr->GetPrimitive());
    }
}

void MKLDNNReshapeNode::execute(mkldnn::stream strm) {
    if (srcPrim && dstPrim) {
        if (src_blocked)
            src_blocked->GetPrimitive().set_data_handle(getParentEdgeAt(0)->getMemory().GetPrimitive().get_data_handle());
        if (dst_blocked)
            dst_blocked->GetPrimitive().set_data_handle(getChildEdgeAt(0)->getMemory().GetPrimitive().get_data_handle());
        strm.submit({*srcPrim, *dstPrim});
    }
}

bool MKLDNNReshapeNode::created() const {
    return getType() == Reshape || getType() == Flatten;
}
