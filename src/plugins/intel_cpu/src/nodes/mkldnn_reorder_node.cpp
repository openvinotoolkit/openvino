// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_reorder_node.h"
#include <memory>
#include <string>
#include <algorithm>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include "ie_parallel.hpp"
#include "utils/general_utils.h"
#include <cpu/x64/cpu_isa_traits.hpp>
#include "nodes/common/cpu_memcpy.h"
#include "nodes/common/cpu_convert.h"
#include "mkldnn_convert_node.h"
#include <common/primitive_hashing_utils.hpp>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

namespace {
struct ReorderKey {
    mkldnn::memory::desc src;
    mkldnn::memory::desc dest;
    size_t hash() const;
    bool operator==(const ReorderKey& rhs) const;
};

size_t ReorderKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;
    seed = hash_combine(seed, get_md_hash(src.data));
    seed = hash_combine(seed, get_md_hash(dest.data));

    return seed;
}

bool ReorderKey::operator==(const ReorderKey& rhs) const {
    bool retVal = true;
    retVal = src == rhs.src && dest == rhs.dest;
    return retVal;
}

}  // namespace

bool MKLDNNReorderNode::isExecutable() const {
    return MKLDNNNode::isExecutable() && !isOptimized;
}

MKLDNNReorderNode::MKLDNNReorderNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &w_cache) :
        MKLDNNNode(op, eng, w_cache) {
    IE_THROW() << "Can't create reorder node from ngraph node";
}

MKLDNNReorderNode::MKLDNNReorderNode(const std::string& name, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &w_cache) :
        MKLDNNNode("Reorder", name, eng, w_cache) {}

void MKLDNNReorderNode::getSupportedDescriptors() {
    if (getParentEdges().size() != 1)
        IE_THROW() << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        IE_THROW() << "Incorrect number of output edges for layer " << getName();
}

void MKLDNNReorderNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto parent = getParentEdgeAt(0)->getParent();
    auto child = getChildEdgeAt(0)->getChild();

    NodeConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(1);
    config.outConfs.resize(1);
    config.inConfs[0].inPlace(-1);
    config.inConfs[0].constant(false);
    config.outConfs[0].inPlace(-1);
    config.outConfs[0].constant(false);
    if (isOptimized) {
        config.inConfs[0].inPlace(0);
        config.outConfs[0].inPlace(0);
    }
    if (input && output) {
        config.inConfs[0].setMemDesc(input);
        config.outConfs[0].setMemDesc(output);
    } else if (parent->getSelectedPrimitiveDescriptor() != nullptr &&
               child->getSelectedPrimitiveDescriptor() != nullptr) {
        config.inConfs[0].setMemDesc(parent->getSelectedPrimitiveDescriptor()->getConfig().outConfs[0].getMemDesc());
        config.outConfs[0].setMemDesc(child->getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].getMemDesc());
    } else {
        IE_THROW() << "Cannot initialize supported PDs for Reorder node with name `" << getName() << "`";
    }

    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::reorder);

    // must be to initialize here since shapes are unknown at the time of Reorder node creation
    isDynamic = !(config.inConfs[0].getMemDesc()->isDefined() && config.outConfs[0].getMemDesc()->isDefined());

    if (isDynamic && (config.inConfs[0].getMemDesc()->getShape().getRank() != config.outConfs[0].getMemDesc()->getShape().getRank()))
        IE_THROW() << "Reorder node doesn't support case when input and output shapes have different rank and dynamic";
    if (!isOptimized) {
        const auto &inShape = getInputShapeAtPort(0);
        if (MKLDNNPlugin::one_of(inShape.getRank(), 4, 5) &&
                config.inConfs[0].getMemDesc()->hasLayoutType(LayoutType::nspc) &&
                config.outConfs[0].getMemDesc()->hasLayoutType(LayoutType::ncsp) &&
                config.inConfs[0].getMemDesc()->getPrecision() == Precision::FP32 &&
                config.outConfs[0].getMemDesc()->getPrecision() == Precision::FP32) {
            // oneDNN JIT reorder shows bad perf for nspc to ncsp reorder case so we fallback on simple c++ implementation
            isNspc2NcspCase = true;
        } else if (!impl::cpu::x64::mayiuse(impl::cpu::x64::avx2) &&
                   MKLDNNPlugin::one_of(inShape.getRank(), 4, 5) &&
                   config.inConfs[0].getMemDesc()->hasLayoutType(LayoutType::ncsp) &&
                   config.outConfs[0].getMemDesc()->hasLayoutType(LayoutType::nspc) &&
                   config.inConfs[0].getMemDesc()->getPrecision() == config.outConfs[0].getMemDesc()->getPrecision() &&
                   config.inConfs[0].getMemDesc()->getPrecision().size() == 1) {
            // oneDNN doesn't provide JIT reorder impl for non-avx2 targets so we fallback on simple c++ implementation which shows better perf
            isNcsp2NspcCase = true;
        }
    }
}

void MKLDNNReorderNode::createPrimitive() {
    if (inputShapesDefined()) {
        if (needPrepareParams())
            prepareParams();
        updateLastInputDims();
    }
}

void MKLDNNReorderNode::executeDynamicImpl(mkldnn::stream strm) {
    execute(strm);
}

void MKLDNNReorderNode::prepareParams() {
    if (!isOptimized) {
        auto &srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
        auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
        if (!dstMemPtr || !dstMemPtr->isAllocated())
            IE_THROW() << "Destination memory didn't allocate.";
        if (!srcMemPtr || !srcMemPtr->isAllocated())
            IE_THROW() << "Input memory didn't allocate.";
        if (getSelectedPrimitiveDescriptor() == nullptr)
            IE_THROW() << "Preferable primitive descriptor is not set.";

        auto isSupportedDesc = [](const MemoryDesc& desc) {
            if (!desc.isDefined()) {
                return false;
            }
            if (!(desc.getType() & MemoryDescType::Blocked)) {
                return false;
            }
            if ((desc.getType() & MemoryDescType::Mkldnn) && !desc.as<const DnnlMemoryDesc>()->hasEmptyExtraData()) {
                return false;
            }
            return true;
        };

        const auto&  parentDesc = srcMemPtr->getDesc();
        const auto&  childDesc = dstMemPtr->getDesc();
        if ((isNspc2NcspCase || isNcsp2NspcCase) && isSupportedDesc(childDesc) && isSupportedDesc(parentDesc)) {
            const auto &inDims = srcMemPtr->getStaticDims();
            // Check that child strides are consistent with parent dims if the child is inplace.
            // The strides must be dense except for the channel one (since the child num channels might differ)
            const auto childSubBlocksAreDense = [&]() {
                const auto& dstStrides = childDesc.as<BlockedMemoryDesc>()->getStrides();
                const auto& dstOrder = childDesc.as<BlockedMemoryDesc>()->getOrder();
                const size_t channelDim = 1;
                if (dstStrides.back() != 1)
                    return false;
                for (int i = inDims.size() - 1; i > 0; i--) {
                    if (dstStrides[i-1] != dstStrides[i] * inDims[dstOrder[i]] && dstOrder[i] != channelDim)
                        return false;
                }
                return true;
            };
            if (isNspc2NcspCase) {
                canUseNspc2Ncsp = inDims[1] <= 64 && inDims[1] >= 16 &&
                                  (parentDesc.as<BlockedMemoryDesc>()->getPaddedElementsCount() / inDims[1]) >= 128 &&
                                  childSubBlocksAreDense();
            } else if (isNcsp2NspcCase) {
                canUseNcsp2Nspc = childSubBlocksAreDense();
            }
        }
        if (!canUseNcsp2Nspc && !canUseNspc2Ncsp) {
            if (!dstMemPtr || !dstMemPtr->isAllocated())
                IE_THROW() << "Destination memory didn't allocate.";
            if (!srcMemPtr || !srcMemPtr->isAllocated())
                IE_THROW() << "Input memory didn't allocate.";
            if (getSelectedPrimitiveDescriptor() == nullptr)
                IE_THROW() << "Preferable primitive descriptor is not set.";

            createReorderPrimitive(srcMemPtr->GetDescWithType<DnnlMemoryDesc>()->getDnnlDesc(), srcMemPtr->GetData(),
                                   dstMemPtr->GetDescWithType<DnnlMemoryDesc>()->getDnnlDesc(), dstMemPtr->GetData());
        }
    }
}

void MKLDNNReorderNode::createReorderPrimitive(const mkldnn::memory::desc& srcDesc,
                                               void* srcPtr,
                                               const mkldnn::memory::desc& dstDesc,
                                               void* dstPtr) {
    const auto engine = getEngine();
    src_blocked = std::make_shared<MKLDNNMemory>(engine);
    src_blocked->Create(MKLDNNExtensionUtils::makeDescriptor(srcDesc), srcPtr, false);

    dst_blocked = std::make_shared<MKLDNNMemory>(engine);
    dst_blocked->Create(MKLDNNExtensionUtils::makeDescriptor(dstDesc), dstPtr, false);

    impl_desc_type impl_type;
    ReorderKey key = {src_blocked->GetPrimitive().get_desc(), dst_blocked->GetPrimitive().get_desc()};

    auto builder = [&engine, &impl_type](const ReorderKey& key) -> std::shared_ptr<mkldnn::primitive> {
        mkldnn::primitive_attr attr;
        reorder::primitive_desc pd = mkldnn::reorder::primitive_desc(engine, key.src, engine, key.dest, attr, true);

        if (!pd)
            return nullptr;
        auto info = pd.impl_info_str();
        impl_type = parse_impl_name(info);
        return std::make_shared<mkldnn::reorder>(pd);
    };

    auto cache = getRuntimeCache();
    std::pair<std::shared_ptr<mkldnn::primitive>, CacheEntryBase::LookUpStatus> result{
        nullptr,
        CacheEntryBase::LookUpStatus::Miss};
    // TODO: We should keep shape consistency for const and expected shape for node.
    //       If it requires reshape operation it should explicitly injected into graph.
    //
    // There is a limitation for IE representing of weights for grouped convolutions. IE doesn't
    // split group dimension in separate shape dimension. IE use OIHW, but mkldnn expect GOIHW.
    // So we will perform implicit reshape to dst shape.
    //
    // MKLDNN doesn't support direct reorders for tensors of different rank. The code below tries to
    // perform such conversion if the source tensor can be reshaped to the destination rank. This is
    // useful in situations when rank in IR does not much rank that is required by the oneDNN primitive,
    // but the input tensor can be reshaped (e.g. weights for grouped convolutions, biases etc.)
    if (src_blocked->getDesc().hasLayoutType(LayoutType::ncsp) &&
        src_blocked->GetShape().getRank() != dst_blocked->GetShape().getRank()) {
        const auto newDims = dst_blocked->getStaticDims();
        const auto newFormat = MKLDNNExtensionUtils::GetPlainFormatByRank(newDims.size());

        auto newDesc = mkldnn::memory::desc(MKLDNNExtensionUtils::convertToDnnlDims(newDims),
                                            src_blocked->GetDataType(),
                                            newFormat);
        src_blocked->Create(MKLDNNExtensionUtils::makeDescriptor(newDesc), srcPtr, false);

        key.src = src_blocked->GetPrimitive().get_desc();
        result = cache->getOrCreate(key, builder);
    } else {
        result = cache->getOrCreate(key, builder);
    }

    if (!result.first) {
        IE_THROW() << "Cannot create reorder primitive: unsupported reorder case";
    }
    prim = result.first;
    supportedPrimitiveDescriptors[0].setImplementationType(impl_type);
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

void MKLDNNReorderNode::optimizedNcsp2Nspc() {
    auto parentEdge = getParentEdgeAt(0);
    auto childEdge = getChildEdgeAt(0);

    auto inDims = parentEdge->getMemory().GetShape().getStaticDims();
    const auto dstStrides = childEdge->getMemoryPtr()->GetDescWithType<BlockedMemoryDesc>()->getStrides();
    const size_t ndims = inDims.size();
    const size_t DIM0 = inDims[0];
    const size_t DIM1 = inDims[1];
    const size_t DIM2 = ndims == 5 ? inDims[ndims - 3] : 1;
    const size_t DIM3 = inDims[ndims - 2];
    const size_t DIM4 = inDims[ndims - 1];

    auto src_data = reinterpret_cast<const uint8_t *>(parentEdge->getMemoryPtr()->GetPtr());
    auto dst_data = reinterpret_cast<uint8_t *>(childEdge->getMemoryPtr()->GetPtr());

    const size_t src_batch_stride = DIM1 * DIM2 * DIM3 * DIM4;
    const size_t dst_batch_stride = dstStrides[0];
    const size_t dst_channel_stride = dstStrides[ndims-2];
    const size_t stride1 = DIM2 * DIM3 * DIM4;
    const size_t stride2 = DIM2 * DIM3;

    parallel_for3d(DIM0, DIM1, stride2, [&](size_t dim0, size_t dim1, size_t j) {
        size_t src_off = dim0 * src_batch_stride + j * DIM4 + dim1 * stride1;
        size_t dst_off = dim0 * dst_batch_stride + j * DIM4 * dst_channel_stride + dim1;

        for (size_t dim4 = 0; dim4 < DIM4; ++dim4) {
            dst_data[dst_off] = src_data[src_off];
            src_off++;
            dst_off += dst_channel_stride;
        }
    });
}

void MKLDNNReorderNode::optimizedNspc2Ncsp() {
    auto parentEdge = getParentEdgeAt(0);
    auto childEdge = getChildEdgeAt(0);

    auto inDims = parentEdge->getMemory().GetShape().getStaticDims();
    const size_t ndims = inDims.size();
    const size_t DIM0 = inDims[0];
    const size_t DIM1 = inDims[1];
    const size_t DIM2 = ndims == 5 ? inDims[ndims - 3] : 1;
    const size_t DIM3 = inDims[ndims - 2];
    const size_t DIM4 = inDims[ndims - 1];

    auto src_data = reinterpret_cast<const float *>(parentEdge->getMemoryPtr()->GetPtr());
    auto dst_data = reinterpret_cast<float *>(childEdge->getMemoryPtr()->GetPtr());

    const auto dstStrides = childEdge->getMemoryPtr()->GetDescWithType<BlockedMemoryDesc>()->getStrides();
    const size_t block_size = DIM2 * DIM3 * DIM4;
    const size_t src_batch_stride = block_size * DIM1;
    const size_t dst_batch_stride = dstStrides[0];
    parallel_for2d(DIM0, block_size, [&](size_t b, size_t j) {
        auto src_off = b * src_batch_stride + j * DIM1;
        auto dst_off = b * dst_batch_stride + j;
        for (size_t dim1 = 0; dim1 < DIM1; ++dim1) {
            dst_data[dst_off] = src_data[src_off];
            src_off++;
            dst_off += block_size;
        }
    });
}

void MKLDNNReorderNode::execute(mkldnn::stream strm) {
    if (isOptimized)
        return;

    if (canUseNspc2Ncsp) {
        optimizedNspc2Ncsp();
    } else if (canUseNcsp2Nspc) {
        optimizedNcsp2Nspc();
    } else {
        src_blocked->setDataHandle(getParentEdgeAt(0)->getMemory().GetData());
        dst_blocked->setDataHandle(getChildEdgeAt(0)->getMemory().GetData());

        MKLDNNNode::execute(strm);
    }
}

void MKLDNNReorderNode::setDynamicBatchLim(int lim) {
    dynBatchLim = lim;
    if (prim) {
        auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
        auto &srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
        memory::desc src_d = srcMemPtr->GetDescWithType<DnnlMemoryDesc>()->getDnnlDesc();
        memory::desc dst_d = dstMemPtr->GetDescWithType<DnnlMemoryDesc>()->getDnnlDesc();
        void *src_data_hdl = srcMemPtr->GetData();
        void *dst_data_hdl = dstMemPtr->GetData();

        src_d.data.dims[0] = batchToProcess();
        src_d.data.padded_dims[0] = batchToProcess();

        dst_d.data.dims[0] = batchToProcess();
        dst_d.data.padded_dims[0] = batchToProcess();

        createReorderPrimitive(src_d, src_data_hdl, dst_d, dst_data_hdl);
    }
}

std::string MKLDNNReorderNode::getReorderArgs(const MemoryDesc &parentDesc, const MemoryDesc &childDesc) {
    std::string inArgs, outArgs;
    if (parentDesc.getPrecision() != childDesc.getPrecision()) {
        inArgs += (inArgs.empty() ? "" : "_") + std::string(parentDesc.getPrecision().name());
        outArgs += (outArgs.empty() ? "" : "_") + std::string(childDesc.getPrecision().name());
    }
    auto formatSrc = parentDesc.serializeFormat();
    auto formatDst = childDesc.serializeFormat();
    if (formatSrc != formatDst || one_of(std::string("undef"), formatSrc, formatDst)) {
        inArgs += (inArgs.empty() ? "" : "_") + formatSrc;
        outArgs += (outArgs.empty() ? "" : "_") + formatDst;
    }
    return inArgs + "_" + outArgs;
}

void MKLDNNReorderNode::reorderData(const MKLDNNMemory &input, const MKLDNNMemory &output) {
    if (!input.getDesc().isDefined() || !output.getDesc().isDefined())
        IE_THROW() << "Can't reorder data with dynamic shapes";

    if (input.GetShape().hasZeroDims() || output.GetShape().hasZeroDims()) {
        return;
    }

    if (input.getDesc().isCompatible(output.getDesc())) {
        auto srcPtr = static_cast<uint8_t*>(input.GetPtr());
        auto dstPtr = static_cast<uint8_t*>(output.GetPtr());

        auto copySize = output.GetSize();
        cpu_memcpy(dstPtr, srcPtr, copySize);
    } else {
        std::unique_ptr<mkldnn::reorder> pReorder;
        mkldnn::memory srcMemory;
        std::vector<uint8_t> tmpBuff;

        try {
            pReorder = std::unique_ptr<mkldnn::reorder>(new mkldnn::reorder(input.GetPrimitive(), output.GetPrimitive()));
            srcMemory = input.GetPrimitive();
        }
        catch (const mkldnn::error& err) {
            if (mkldnn_unimplemented == err.status && output.GetDataType() != input.GetDataType() && MKLDNNConvertNode::isSupportedDesc(input.getDesc()) &&
                    MKLDNNConvertNode::isSupportedDesc(output.getDesc())) {
                //we probably could not make the reorder because there is no one supporting this precision conversion
                //lets try to convert data first using cpu_convert
                auto data = static_cast<const uint8_t *>(input.GetPtr());
                tmpBuff.resize(input.GetSize());

                const auto outPrc = MKLDNNExtensionUtils::DataTypeToIEPrecision(output.GetDataType());
                cpu_convert(data, tmpBuff.data(), MKLDNNExtensionUtils::DataTypeToIEPrecision(input.GetDataType()),
                            outPrc, input.GetSize() / input.getDesc().getPrecision().size());

                MKLDNNMemory tmpMem(output.getEngine());
                auto tmpDesc = input.getDesc().cloneWithNewPrecision(outPrc);
                tmpMem.Create(std::move(tmpDesc), tmpBuff.data());

                pReorder = std::unique_ptr<mkldnn::reorder>(new mkldnn::reorder(tmpMem.GetPrimitive(), output.GetPrimitive()));
                srcMemory = tmpMem.GetPrimitive();
            } else {
                throw;
            }
        }
        if (pReorder) {
            mkldnn::stream loc_stream(output.getEngine(), mkldnn::stream::flags::in_order);
            auto dstMemory = output.GetPrimitive();
            pReorder->execute(loc_stream, srcMemory, dstMemory);
        } else {
            IE_THROW() << "Could not make mkldnn reorder.";
        }
    }
}

std::vector<VectorDims> MKLDNNReorderNode::shapeInfer() const {
    return {getParentEdgesAtPort(0)[0]->getMemory().getStaticDims()};
}

REG_MKLDNN_PRIM_FOR(MKLDNNReorderNode, Reorder);
