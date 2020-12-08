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
        config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::format::any);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::format::any);
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
        attr.set_int_output_round_mode(round_nearest);
    }

    auto createReorder = [&]() {
        // No autoblocking. Reorder can be applied as is
        reorder::primitive_desc pd = reorder::primitive_desc(src_blocked->GetPrimitiveDescriptor(), dst_blocked->GetPrimitiveDescriptor(), attr);

        const char *info;
        mkldnn_primitive_desc_query(pd.get(), mkldnn::convert_to_c(impl_info_str), 0, &info);
        supportedPrimitiveDescriptors[0].setImplementationType(parse_impl_name(std::string(info)));
        supportedPrimitiveDescriptors[0].setOutputLayouts(static_cast<memory::format>(dstDesc.data.format));

        prim.reset(new mkldnn::reorder(pd, src_blocked->GetPrimitive(), dst_blocked->GetPrimitive()));
    };

    try {
        createReorder();
    } catch (...) {
        // MKLDNN doesn't support direct reorders from planar data formats to grouped weights formats.
        // Code block below tries to detect such cases and reinterpret data planar formats (e.g. nchw)
        // as grouped weights planar formats (e.g. goihw) since they have same physical memory layout.
        if (MKLDNNMemory::GetPlainFormat(src_blocked->GetDims()) == src_blocked->GetFormat() &&
            src_blocked->GetDims().size() + 1 == dst_blocked->GetDims().size()) {
            try {
                mkldnn::memory::dims newDims = dst_blocked->GetDims();
                mkldnn::memory::format newFormat;
                if (MKLDNNMemory::IsGroupedFormat(dst_blocked->GetFormat())) {
                    newFormat = src_blocked->GetDims().size() == 4 ? memory::goihw :
                                src_blocked->GetDims().size() == 5 ? memory::goidhw :
                                src_blocked->GetFormat();
                } else {
                    newFormat = src_blocked->GetDims().size() == 4 ? memory::ncdhw :
                                src_blocked->GetFormat();
                }

                auto newDesc = mkldnn::memory::desc(newDims, src_blocked->GetDataType(), newFormat);
                src_blocked->Create(newDesc, srcPtr, false);

                createReorder();
            } catch (...) {
                THROW_IE_EXCEPTION << "Cannot create reorder primitive: unsupported reorder case";
            }
        // MKLDNN doesn't support direct reorders between planar data formats in case they have different rank but the same number of elements.
        // Code block below detects these cases and substitute src dims with dst ones.
        } else if (MKLDNNMemory::GetPlainFormat(src_blocked->GetDims()) == src_blocked->GetFormat() &&
                   MKLDNNMemory::GetPlainFormat(dst_blocked->GetDims()) == dst_blocked->GetFormat() &&
                   src_blocked->GetElementsCount() == dst_blocked->GetElementsCount()) {
            try {
                auto newDesc = mkldnn::memory::desc(dst_blocked->GetDims(), src_blocked->GetDataType(), dst_blocked->GetFormat());
                src_blocked->Create(newDesc, srcPtr, false);

                createReorder();
            } catch (...) {
                THROW_IE_EXCEPTION << "Cannot create reorder primitive: unsupported reorder case";
            }
        } else {
            THROW_IE_EXCEPTION << "Cannot create reorder primitive: unsupported reorder case";
        }
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
        src_d.data.layout_desc.blocking.padding_dims[0] = batchToProcess();

        dst_d.data.dims[0] = batchToProcess();
        dst_d.data.layout_desc.blocking.padding_dims[0] = batchToProcess();

        createReorderPrimitive(src_d, src_data_hdl, dst_d, dst_data_hdl);
    }
}
REG_MKLDNN_PRIM_FOR(MKLDNNReorderNode, Reorder);
