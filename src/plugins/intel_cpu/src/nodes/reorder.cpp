// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder.h"
#include <memory>
#include <string>
#include <algorithm>
#include <dnnl_types.h>
#include <dnnl_extension_utils.h>
#include "ie_parallel.hpp"
#include "utils/general_utils.h"
#include <cpu/x64/cpu_isa_traits.hpp>
#include "nodes/common/cpu_memcpy.h"
#include "nodes/common/cpu_convert.h"
#include "nodes/common/reorder_prim.h"
#include "convert.h"
#include <common/primitive_hashing_utils.hpp>
#include <shape_inference/shape_inference_pass_through.hpp>
#include "executors/transpose_list.hpp"

using namespace dnnl;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

bool Reorder::isExecutable() const {
    return Node::isExecutable() && !isOptimized;
}

Reorder::Reorder(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context) :
        Node(op, context, PassThroughShapeInferFactory()) {
    IE_THROW() << "Can't create reorder node from ngraph node";
}

Reorder::Reorder(const std::string& name, const GraphContext::CPtr context) :
        Node("Reorder", name, context) {}

void Reorder::getSupportedDescriptors() {
    if (getParentEdges().size() != 1)
        IE_THROW() << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        IE_THROW() << "Incorrect number of output edges for layer " << getName();
}

void Reorder::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto parent = getParentEdgeAt(0)->getParent();
    auto child = getChildEdgeAt(0)->getChild();

    NodeConfig config;
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
    if (isDynamicNode() && !shapeInference) {
        shapeInference = std::make_shared<ShapeInferPassThrough>();
    }

    if (isDynamic && (config.inConfs[0].getMemDesc()->getShape().getRank() != config.outConfs[0].getMemDesc()->getShape().getRank()))
        IE_THROW() << "Reorder node with name: " << getName() << " doesn't support case when input and output shapes have different rank and dynamic";
    if (!isOptimized) {
        const auto &inShape = getInputShapeAtPort(0);
        if (one_of(inShape.getRank(), 4u, 5u) &&
                config.inConfs[0].getMemDesc()->hasLayoutType(LayoutType::nspc) &&
                config.outConfs[0].getMemDesc()->hasLayoutType(LayoutType::ncsp) &&
                config.inConfs[0].getMemDesc()->getPrecision() == Precision::FP32 &&
                config.outConfs[0].getMemDesc()->getPrecision() == Precision::FP32) {
            // oneDNN JIT reorder shows bad perf for nspc to ncsp reorder case so we fallback on simple c++ implementation
            isNspc2NcspCase = true;
        } else if (!impl::cpu::x64::mayiuse(impl::cpu::x64::avx2) &&
                   one_of(inShape.getRank(), 4u, 5u) &&
                   config.inConfs[0].getMemDesc()->hasLayoutType(LayoutType::ncsp) &&
                   config.outConfs[0].getMemDesc()->hasLayoutType(LayoutType::nspc) &&
                   config.inConfs[0].getMemDesc()->getPrecision() == config.outConfs[0].getMemDesc()->getPrecision() &&
                   config.inConfs[0].getMemDesc()->getPrecision().size() == 1) {
            // oneDNN doesn't provide JIT reorder impl for non-avx2 targets so we fallback on simple c++ implementation which shows better perf
            isNcsp2NspcCase = true;
        }
    }
}

void Reorder::createPrimitive() {
    if (shapesDefined()) {
        if (needPrepareParams())
            prepareParams();
        updateLastInputDims();
    }
}

void Reorder::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

#if defined(OV_CPU_ARM_ENABLE_FP16)
void Reorder::prepareReorderAsTranspose(MemoryDescPtr parentDesc, MemoryDescPtr childDesc) {
    auto getOrderAndBlockedDims = [](const MemoryDesc& lhs, const MemoryDesc& rhs) -> std::pair<std::vector<size_t>, std::vector<size_t>> {
        const auto& in = lhs.as<BlockedMemoryDesc>()->getBlockDims();
        const auto rank = lhs.getShape().getRank();

        if (lhs.hasLayoutType(LayoutType::ncsp) && rhs.hasLayoutType(LayoutType::nspc)) {
            if (rank == 4)
                return {{0, 2, 3, 1}, {in[0], in[2], in[3], in[1]}};
            else
                return {{0, 2, 1}, {in[0], in[2], in[1]}};

        } else if (lhs.hasLayoutType(LayoutType::nspc) && rhs.hasLayoutType(LayoutType::ncsp)) {
            if (rank == 4)
                return {{0, 3, 1, 2}, {in[0], in[3], in[1], in[2]}};
            else
                return {{0, 2, 1}, {in[0], in[2], in[1]}};
        } else {
            if (rank == 4)
                return {{0, 1, 2, 3}, in};
            else
                return {{0, 1, 2}, in};
        }
    };

    auto order = getOrderAndBlockedDims(*parentDesc, *childDesc);
    const auto& transposeOrder = order.first;
    const auto& transposedBlockDims = order.second;

    auto transposedDesc = std::make_shared<CpuBlockedMemoryDesc>(parentDesc->getPrecision(), Shape{transposedBlockDims});

    TransposeParams transposeParams;
    transposeParams.permuteParams.src_block_dims = parentDesc->as<BlockedMemoryDesc>()->getBlockDims();
    transposeParams.permuteParams.src_block_order = parentDesc->as<BlockedMemoryDesc>()->getOrder();
    transposeParams.permuteParams.dst_block_dims = transposedBlockDims;
    transposeParams.permuteParams.dst_block_order = transposeParams.permuteParams.src_block_order;
    transposeParams.permuteParams.order = transposeOrder;
    transposeParams.permuteParams.data_size = parentDesc->getPrecision().size();

    auto transpose_context = std::make_shared<ExecutorContext>(context, getImplPriority());
    auto factory = std::make_shared<TransposeExecutorFactory>(transposeParams,
                                                              std::vector<MemoryDescPtr>{parentDesc},
                                                              std::vector<MemoryDescPtr>{transposedDesc},
                                                              transpose_context);
    dnnl::primitive_attr attr;
    transposeExecutor = factory->makeExecutor(transposeParams,
                                              {parentDesc},
                                              {transposedDesc},
                                              attr);
    getSelectedPrimitiveDescriptor()->setImplementationType(transposeExecutor->getImplType());
    return;
}
#endif // OV_CPU_ARM_ENABLE_FP16

void Reorder::prepareParams() {
    if (isOptimized)
        return;

    auto srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    auto dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
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
        if ((desc.getType() & MemoryDescType::Dnnl) && !desc.as<const DnnlMemoryDesc>()->hasEmptyExtraData()) {
            return false;
        }
        return true;
    };

    const auto&  parentDesc = srcMemPtr->getDescPtr();
    const auto&  childDesc = dstMemPtr->getDescPtr();

#if defined(OV_CPU_ARM_ENABLE_FP16)
    // @todo current oneDNN v3.2 lacks optimized jit implementation for fp16 reorders.
    // Use transpose executor as a temporary WA.
    if (everyone_is(Precision::FP16, parentDesc->getPrecision(), childDesc->getPrecision()) &&
        ((parentDesc->hasLayoutType(LayoutType::ncsp) && childDesc->hasLayoutType(LayoutType::nspc)) ||
         (parentDesc->hasLayoutType(LayoutType::nspc) && childDesc->hasLayoutType(LayoutType::ncsp))) &&
        one_of(parentDesc->getShape().getRank(), 3, 4)) {
        return prepareReorderAsTranspose(parentDesc, childDesc);
    }
#endif

    if ((isNspc2NcspCase || isNcsp2NspcCase) && isSupportedDesc(*childDesc) && isSupportedDesc(*parentDesc)) {
        const auto &inDims = srcMemPtr->getStaticDims();
        // Check that child strides are consistent with parent dims if the child is inplace.
        // The strides must be dense except for the channel one (since the child num channels might differ)
        const auto childSubBlocksAreDense = [&]() {
            const auto& dstStrides = childDesc->as<BlockedMemoryDesc>()->getStrides();
            const auto& dstOrder = childDesc->as<BlockedMemoryDesc>()->getOrder();
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
                (parentDesc->as<BlockedMemoryDesc>()->getPaddedElementsCount() / inDims[1]) >= 128 &&
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

        createReorderPrimitive(srcMemPtr->getDescWithType<DnnlMemoryDesc>()->getDnnlDesc(), srcMemPtr->getData(),
                               dstMemPtr->getDescWithType<DnnlMemoryDesc>()->getDnnlDesc(), dstMemPtr->getData());
    }
}

void Reorder::createReorderPrimitive(const dnnl::memory::desc& srcDesc,
                                     void* srcPtr,
                                     const dnnl::memory::desc& dstDesc,
                                     void* dstPtr) {
    auto selectedPD = getSelectedPrimitiveDescriptor();
    if (!selectedPD)
        IE_THROW() << "Preferable primitive descriptor is not set.";

    const auto engine = getEngine();
    src_blocked = std::make_shared<Memory>(engine, DnnlExtensionUtils::makeDescriptor(srcDesc), srcPtr, false);

    dst_blocked = std::make_shared<Memory>(engine, DnnlExtensionUtils::makeDescriptor(dstDesc), dstPtr, false);

    auto src_desc = src_blocked->getPrimitive().get_desc();
    if (!src_permutation.empty()) {
        // reorder requires exact matching of logical dimensions between src & dst
        // sometime we have to permute source's logical dimensions to satisfy
        // this requirement, this dosn't affect plugin's node input memory desc.
        /// for (i = 0; i < ndims(); i++)
        ///     new_desc.dims()[permutation[i]] = dims()[i];
        src_desc = src_desc.permute_axes(src_permutation);
    }

    auto dst_desc = dst_blocked->getPrimitive().get_desc();

    // TODO: We should keep shape consistency for const and expected shape for node.
    //       If it requires reshape operation it should explicitly injected into graph.
    //
    // There is a limitation for IE representing of weights for grouped convolutions. IE doesn't
    // split group dimension in separate shape dimension. IE use OIHW, but onednn expect GOIHW.
    // So we will perform implicit reshape to dst shape.
    //
    // oneDNN doesn't support direct reorders for tensors of different rank. The code below tries to
    // perform such conversion if the source tensor can be reshaped to the destination rank. This is
    // useful in situations when rank in IR does not much rank that is required by the oneDNN primitive,
    // but the input tensor can be reshaped (e.g. weights for grouped convolutions, biases etc.)
    if (src_blocked->getDesc().hasLayoutType(LayoutType::ncsp) &&
        src_blocked->getShape().getRank() != dst_blocked->getShape().getRank()) {
        const auto newDims = dst_blocked->getStaticDims();
        const auto newFormat = DnnlExtensionUtils::GetPlainFormatByRank(newDims.size());

        auto newDesc = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(newDims),
                                            src_blocked->getDataType(),
                                            newFormat);
        src_blocked = std::make_shared<Memory>(getEngine(), DnnlExtensionUtils::makeDescriptor(newDesc), srcPtr, false);

        src_desc = src_blocked->getPrimitive().get_desc();
    }

    auto result = getReorderPrim(context->getParamsCache(), getEngine(), src_desc, dst_desc);
    if (!result) {
        IE_THROW() << "Cannot create reorder primitive: unsupported reorder case";
    }
    prim = result;

    selectedPD->setImplementationType(
        parse_impl_name(DnnlExtensionUtils::query_impl_info_str(prim.get_primitive_desc())));

    auto src = getParentEdgesAtPort(0)[0]->getMemoryPtr()->getPrimitive();
    auto dst = getChildEdgesAtPort(0)[0]->getMemoryPtr()->getPrimitive();
    primArgs = {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst}};

#ifdef CPU_DEBUG_CAPS
    if (prim) {
        auto pd = prim.get_primitive_desc();
        DEBUG_LOG("verbose##", getName(), "##", DnnlExtensionUtils::query_pd_info(pd), "\n");
    }
#endif
}

const std::vector<impl_desc_type>& Reorder::getDefaultImplPriority() {
    static const std::vector<impl_desc_type> priorities = {impl_desc_type::reorder};

    return priorities;
}

bool Reorder::created() const {
    return getType() == Type::Reorder;
}

void Reorder::optimizedNcsp2Nspc() {
    auto parentEdge = getParentEdgeAt(0);
    auto childEdge = getChildEdgeAt(0);

    auto inDims = parentEdge->getMemory().getShape().getStaticDims();
    const auto dstStrides = childEdge->getMemoryPtr()->getDescWithType<BlockedMemoryDesc>()->getStrides();
    const size_t ndims = inDims.size();
    const size_t DIM0 = inDims[0];
    const size_t DIM1 = inDims[1];
    const size_t DIM2 = ndims == 5 ? inDims[ndims - 3] : 1;
    const size_t DIM3 = inDims[ndims - 2];
    const size_t DIM4 = inDims[ndims - 1];

    auto src_data = reinterpret_cast<const uint8_t *>(parentEdge->getMemoryPtr()->getData());
    auto dst_data = reinterpret_cast<uint8_t *>(childEdge->getMemoryPtr()->getData());

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

void Reorder::optimizedNspc2Ncsp() {
    auto parentEdge = getParentEdgeAt(0);
    auto childEdge = getChildEdgeAt(0);

    auto inDims = parentEdge->getMemory().getShape().getStaticDims();
    const size_t ndims = inDims.size();
    const size_t DIM0 = inDims[0];
    const size_t DIM1 = inDims[1];
    const size_t DIM2 = ndims == 5 ? inDims[ndims - 3] : 1;
    const size_t DIM3 = inDims[ndims - 2];
    const size_t DIM4 = inDims[ndims - 1];

    auto src_data = reinterpret_cast<const float *>(parentEdge->getMemoryPtr()->getData());
    auto dst_data = reinterpret_cast<float *>(childEdge->getMemoryPtr()->getData());

    const auto dstStrides = childEdge->getMemoryPtr()->getDescWithType<BlockedMemoryDesc>()->getStrides();
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

void Reorder::execute(dnnl::stream strm) {
#if defined(OV_CPU_ARM_ENABLE_FP16)
    if (transposeExecutor) {
        auto dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
        auto srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
        int MB = srcMemPtr->getStaticDims()[0];
        return transposeExecutor->exec({srcMemPtr}, {dstMemPtr}, MB);
    }
#endif

    if (isOptimized) {
        DEBUG_LOG("#", getExecIndex(), " Reorder ", getName(), "  is Optimized.",
                   " input @", getParentEdgeAt(0)->getMemory().getData(),
                   " output @", getChildEdgeAt(0)->getMemory().getData());
        return;
    }

    if (canUseNspc2Ncsp) {
        optimizedNspc2Ncsp();
    } else if (canUseNcsp2Nspc) {
        optimizedNcsp2Nspc();
    } else {
        if (prim) {
            prim.execute(strm, primArgs);
        } else {
            IE_THROW() << "Reorder node with name " << getName() << " doesn't have an initialized primitive";
        }
    }
}

std::string Reorder::getReorderArgs(const MemoryDesc &parentDesc, const MemoryDesc &childDesc) {
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

void Reorder::reorderData(const IMemory &input, const IMemory &output, MultiCachePtr cache) {
    if (!input.getDesc().isDefined() || !output.getDesc().isDefined())
        IE_THROW() << "Can't reorder data with dynamic shapes";

    if (input.getShape().hasZeroDims() || output.getShape().hasZeroDims()) {
        return;
    }

    if (input.getDesc().isCompatible(output.getDesc())) {
        auto srcPtr = static_cast<uint8_t*>(input.getData());
        auto dstPtr = static_cast<uint8_t*>(output.getData());

        auto copySize = output.getSize();
        cpu_memcpy(dstPtr, srcPtr, copySize);
    } else {
        dnnl::reorder reorder;
        std::vector<uint8_t> tmpBuff;

        auto srcMemory = input.getPrimitive();
        auto dstMemory = output.getPrimitive();
        auto engine = dstMemory.get_engine();
        // try directly reorder
        reorder = getReorderPrim(cache, dstMemory.get_engine(), srcMemory.get_desc(), dstMemory.get_desc());
        if (!reorder) {
            // try precision conversion then do the reorder
            if (output.getDataType() != input.getDataType() && Convert::isSupportedDesc(input.getDesc()) &&
                Convert::isSupportedDesc(output.getDesc())) {
                //we probably could not make the reorder because there is no one supporting this precision conversion
                //lets try to convert data first using cpu_convert
                auto data = static_cast<const uint8_t *>(input.getData());
                tmpBuff.resize(input.getSize());

                const auto outPrc = DnnlExtensionUtils::DataTypeToIEPrecision(output.getDataType());
                cpu_convert(data, tmpBuff.data(), DnnlExtensionUtils::DataTypeToIEPrecision(input.getDataType()),
                            outPrc, input.getSize() / input.getDesc().getPrecision().size());

                auto tmpDesc = input.getDesc().cloneWithNewPrecision(outPrc);
                Memory tmpMem(engine, std::move(tmpDesc), tmpBuff.data());

                srcMemory = tmpMem.getPrimitive();
                reorder = getReorderPrim(cache, dstMemory.get_engine(), srcMemory.get_desc(), dstMemory.get_desc());
            }
            if (!reorder) {
                IE_THROW() << "No reorder available for the following tensor descriptors: "
                    << input.getDesc().serializeFormat() << " and " << output.getDesc().serializeFormat();
            }
        }
        if (reorder) {
            dnnl::stream loc_stream(engine, dnnl::stream::flags::in_order);
            reorder.execute(loc_stream, {{DNNL_ARG_FROM, srcMemory}, {DNNL_ARG_TO, dstMemory}});
        } else {
            IE_THROW() << "Could not make onednn reorder.";
        }
    }
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
