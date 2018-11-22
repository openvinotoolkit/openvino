// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_reorder_node.h"
#include <memory>
#include <string>
#include <algorithm>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include "ie_parallel.hpp"

using namespace mkldnn;
using namespace MKLDNNPlugin;

MKLDNNReorderNode::MKLDNNReorderNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng) : MKLDNNNode(layer, eng) {
}

void MKLDNNReorderNode::getSupportedDescriptors() {
    if (outDims.empty() && output.getLayout() != InferenceEngine::Layout::ANY)
        outDims.push_back(MKLDNNDims(output.getDims()));
    if (inDims.empty() && input.getLayout() != InferenceEngine::Layout::ANY)
        inDims.push_back(MKLDNNDims(input.getDims()));
    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();
}

void MKLDNNReorderNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto inputDataType = MKLDNNMemoryDesc(input).getDataType();
    auto outputDataType = MKLDNNMemoryDesc(output).getDataType();

    auto parent = getParentEdgeAt(0)->getParent();
    auto child = getChildEdgeAt(0)->getChild();

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(1);
    config.outConfs.resize(1);
    config.inConfs[0].inPlace = -1;
    config.inConfs[0].constant = false;
    config.outConfs[0].inPlace = -1;
    config.outConfs[0].constant = false;
    if (input.getLayout() != InferenceEngine::Layout::ANY && output.getLayout() != InferenceEngine::Layout::ANY) {
        config.inConfs[0].desc = input;
        config.outConfs[0].desc = output;
    } else if (parent->getSelectedPrimitiveDescriptor() != nullptr &&
               child->getSelectedPrimitiveDescriptor() != nullptr) {
        config.inConfs[0].desc = parent->getSelectedPrimitiveDescriptor()->getConfig().outConfs[0].desc;
        config.outConfs[0].desc = child->getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc;
    } else {
        config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::format::any);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::format::any);
    }

    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::reorder);
}

void MKLDNNReorderNode::createPrimitive() {
    auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto &srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor does not set.";

    mkldnn::primitive_attr attr;

    if (_scales) {
        std::vector<float> scales;

        float* scaleData = static_cast<float*>(_scales->buffer());

        for (size_t i = 0; i < _scales->size(); i++) {
            scales.push_back(scaleData[i]);
        }

        int mask = 0;
        int oc_dim_id = 1;
        mask = 1 << oc_dim_id;

        attr.set_output_scales(mask, scales);
        attr.set_int_output_round_mode(round_nearest);
    }

    if (srcMemPtr->GetSize() == dstMemPtr->GetSize()) {
        InferenceEngine::Precision dstPrec = getChildEdgeAt(0)->getDesc().getPrecision();
        InferenceEngine::Precision srcPrec = getParentEdgeAt(0)->getDesc().getPrecision();

        if ((srcPrec == InferenceEngine::Precision::I8 && dstPrec == InferenceEngine::Precision::U8)) {
            // This reorder actually does nothing so we declare it in-place.
            dstMemPtr->GetPrimitive().set_data_handle(srcMemPtr->GetPrimitive().get_data_handle());
        } else {
            try {
                // No autoblocking. Reorder can be applied as is

                reorder::primitive_desc pd = reorder::primitive_desc(srcMemPtr->GetPrimitiveDescriptor(), dstMemPtr->GetPrimitiveDescriptor(), attr);
                prim.reset(new mkldnn::reorder(srcMemPtr->GetPrimitive(), dstMemPtr->GetPrimitive()));
            } catch (...) {}
        }
    } else {
        // Autoblocking case. nchw<=>nChw8c are only supported, but memory descriptor
        // should be with strides. Prepare it from enlarged blob
        memory::dims dims = srcMemPtr->GetDims();
        memory::dims dims_dst = dstMemPtr->GetDims();

        for (int i = 0; i < dims.size(); i++)  // min dims is a logical dims
            dims[i] = std::min(dims[i], dims_dst[i]);

        memory::desc src_d = srcMemPtr->GetDescriptor();
        void *src_data_hdl = srcMemPtr->GetPrimitive().get_data_handle();

        memory::desc dst_d = dstMemPtr->GetDescriptor();
        void *dst_data_hdl = dstMemPtr->GetPrimitive().get_data_handle();

        for (int i = 0; i < dims.size(); i++)
            src_d.data.dims[i] = dst_d.data.dims[i] = dims[i];

        src_blocked = std::make_shared<MKLDNNMemory>(getEngine());
        src_blocked->Create(src_d, src_data_hdl);

        dst_blocked = std::make_shared<MKLDNNMemory>(getEngine());
        dst_blocked->Create(dst_d, dst_data_hdl);

        // output blob should be zeroed. NaN value can occur in untouched place.
        dstMemPtr->FillZero();

        reorder::primitive_desc pd = reorder::primitive_desc(src_blocked->GetPrimitiveDescriptor(), dst_blocked->GetPrimitiveDescriptor(), attr);

        prim.reset(new mkldnn::reorder(pd, src_blocked->GetPrimitive(), dst_blocked->GetPrimitive()));
    }
}

const std::vector<impl_desc_type>& MKLDNNReorderNode::getPrimitivesPriority() {
    implPriorities = {impl_desc_type::reorder};
    return implPriorities;
}

bool MKLDNNReorderNode::created() const {
    return getType() == Reorder;
}

void MKLDNNReorderNode::execute(mkldnn::stream strm) {
    if (prim) {
        if (src_blocked)
            src_blocked->GetPrimitivePtr()->set_data_handle(getParentEdgeAt(0)->getMemory().GetPrimitive().get_data_handle());
        if (dst_blocked)
            dst_blocked->GetPrimitivePtr()->set_data_handle(getChildEdgeAt(0)->getMemory().GetPrimitive().get_data_handle());
        MKLDNNNode::execute(strm);
    } else {
        InferenceEngine::Precision dstPrec = getChildEdgeAt(0)->getDesc().getPrecision();
        InferenceEngine::Precision srcPrec = getParentEdgeAt(0)->getDesc().getPrecision();
        if ((srcPrec == InferenceEngine::Precision::I8 && dstPrec == InferenceEngine::Precision::U8)) {
            // Do nothing here
        } else {
            auto srcBlbPtr = getParentEdgeAt(0)->getBlob();
            auto dstBlbPtr = getChildEdgeAt(0)->getBlob();

            assert(srcBlbPtr->size() == dstBlbPtr->size());
            int data_size = srcBlbPtr->size();

            const auto* src_data = srcBlbPtr->cbuffer().as<const float *>();
            auto* dst_data = dstBlbPtr->buffer().as<float *>();

            InferenceEngine::parallel_for(data_size, [&](int i) {
                dst_data[dstBlbPtr->getTensorDesc().offset(i)] = src_data[srcBlbPtr->getTensorDesc().offset(i)];
            });
        }
    }
}

void MKLDNNReorderNode::setDynamicBatchLim(int lim) {
    dynBatchLim = lim;
    if (prim) {
        auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
        auto &srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
        memory::desc src_d = srcMemPtr->GetDescriptor();
        memory::desc dst_d = dstMemPtr->GetDescriptor();
        void *src_data_hdl = srcMemPtr->GetPrimitive().get_data_handle();
        void *dst_data_hdl = dstMemPtr->GetPrimitive().get_data_handle();

        if (src_blocked && dst_blocked) {
            src_d = src_blocked->GetDescriptor();
            dst_d = dst_blocked->GetDescriptor();
            src_data_hdl = src_blocked->GetPrimitive().get_data_handle();
            dst_data_hdl = dst_blocked->GetPrimitive().get_data_handle();
        }
        src_blocked = std::make_shared<MKLDNNMemory>(getEngine());
        src_d.data.dims[0] = batchToProcess();
        src_d.data.layout_desc.blocking.padding_dims[0] = batchToProcess();
        src_blocked->Create(src_d, src_data_hdl);

        dst_blocked = std::make_shared<MKLDNNMemory>(getEngine());
        dst_d.data.dims[0] = batchToProcess();
        dst_d.data.layout_desc.blocking.padding_dims[0] = batchToProcess();
        dst_blocked->Create(dst_d, dst_data_hdl);
        prim.reset(new mkldnn::reorder(src_blocked->GetPrimitive(), dst_blocked->GetPrimitive()));
    }
}
