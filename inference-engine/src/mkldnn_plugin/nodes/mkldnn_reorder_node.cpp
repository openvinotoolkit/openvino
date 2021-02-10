// Copyright (C) 2018-2020 Intel Corporation
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

MKLDNNReorderNode::MKLDNNReorderNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &w_cache) :
        MKLDNNNode(layer, eng, w_cache) {
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
    if (isOptimized) {
        config.inConfs[0].inPlace = 0;
        config.outConfs[0].inPlace = 0;
    }
    if (input.getLayout() != InferenceEngine::Layout::ANY && output.getLayout() != InferenceEngine::Layout::ANY) {
        config.inConfs[0].desc = input;
        config.outConfs[0].desc = output;
    } else if (parent->getSelectedPrimitiveDescriptor() != nullptr &&
               child->getSelectedPrimitiveDescriptor() != nullptr) {
        config.inConfs[0].desc = parent->getSelectedPrimitiveDescriptor()->getConfig().outConfs[0].desc;
        config.outConfs[0].desc = child->getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc;
    } else {
        config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::format_tag::any);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::format_tag::any);
    }

    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::reorder, MKLDNNMemory::Convert(config.outConfs[0].desc.getLayout()));
}

void MKLDNNReorderNode::createPrimitive() {
    auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto &srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";

    if (!isOptimized)
    createReorderPrimitive(srcMemPtr->GetDescriptor(), srcMemPtr->GetPrimitive().get_data_handle(),
            dstMemPtr->GetDescriptor(), dstMemPtr->GetPrimitive().get_data_handle());
}

void MKLDNNReorderNode::createReorderPrimitive(const mkldnn::memory::desc &srcDesc, void* srcPtr, const mkldnn::memory::desc &dstDesc, void* dstPtr) {
    src_blocked = std::make_shared<MKLDNNMemory>(getEngine());
    src_blocked->Create(srcDesc, srcPtr, false);

    dst_blocked = std::make_shared<MKLDNNMemory>(getEngine());
    dst_blocked->Create(dstDesc, dstPtr, false);

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
    }

    auto createReorder = [&]() -> bool {
        // No autoblocking. Reorder can be applied as is
        reorder::primitive_desc pd = mkldnn::reorder::primitive_desc(src_blocked->GetPrimitive(), dst_blocked->GetPrimitive(), attr, true);

        if (!pd)
            return false;

        auto info = pd.impl_info_str();
        supportedPrimitiveDescriptors[0].setImplementationType(parse_impl_name(info));

        prim.reset(new mkldnn::reorder(pd));
        return true;
    };

    auto success = createReorder();
    if (!success) {
        // TODO: We should keep shape consistency for const and expected shape for node.
        //       If it requires reshape operation it should explicitly injected into graph.
        //
        // There is a limitation for IE representing of weights for grouped convolutions. IE doesn't
        // split group dimension in separate shape dimension. IE use OIHW, but mkldnn expect GOIHW.
        // So we will perform implicit reshape to dst shape.
        //
        // MKLDNN doesn't support direct reorders from planar data formats to grouped weights formats.
        // Code block below tries to detect such cases and reinterpret data planar formats (e.g. nchw)
        // as grouped weights planar formats (e.g. goihw) since they have same physical memory layout.
        if (src_blocked->GetDesc().isPlainFormat() &&
            src_blocked->GetDims().size() + 1 == dst_blocked->GetDims().size()) {
            const auto newDims = dst_blocked->GetDims();
            const auto newFormat = MKLDNNMemory::GetPlainFormat(newDims);

            auto newDesc = mkldnn::memory::desc(newDims, src_blocked->GetDataType(), newFormat);
            src_blocked->Create(newDesc, srcPtr, false);

            success = createReorder();
        }
    }

    if (!success) {
        THROW_IE_EXCEPTION << "Cannot create reorder primitive: unsupported reorder case";
    }

    auto src = getParentEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
    auto dst = getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
    primArgs = {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst}};
}

const std::vector<impl_desc_type>& MKLDNNReorderNode::getPrimitivesPriority() {
    implPriorities = {impl_desc_type::reorder};
    return implPriorities;
}

bool MKLDNNReorderNode::created() const {
    return getType() == Reorder;
}

void MKLDNNReorderNode::execute(mkldnn::stream strm) {
    if (isOptimized)
        return;

    src_blocked->GetPrimitivePtr()->set_data_handle(getParentEdgeAt(0)->getMemory().GetPrimitive().get_data_handle());
    dst_blocked->GetPrimitivePtr()->set_data_handle(getChildEdgeAt(0)->getMemory().GetPrimitive().get_data_handle());

    MKLDNNNode::execute(strm);
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

        src_d.data.dims[0] = batchToProcess();
        src_d.data.padded_dims[0] = batchToProcess();

        dst_d.data.dims[0] = batchToProcess();
        dst_d.data.padded_dims[0] = batchToProcess();

        createReorderPrimitive(src_d, src_data_hdl, dst_d, dst_data_hdl);
    }
}
REG_MKLDNN_PRIM_FOR(MKLDNNReorderNode, Reorder);
