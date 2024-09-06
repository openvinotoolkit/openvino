// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder.h"
#include <memory>
#include <string>
#include <dnnl_types.h>
#include <dnnl_extension_utils.h>
#include "openvino/core/parallel.hpp"
#include "utils/general_utils.h"
#include <cpu/x64/cpu_isa_traits.hpp>
#include "nodes/common/cpu_memcpy.h"
#include "nodes/common/cpu_convert.h"
#include "nodes/common/reorder_prim.h"
#include "convert.h"
#include <common/primitive_hashing_utils.hpp>
#include <shape_inference/shape_inference_pass_through.hpp>

#include "convert.h"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "nodes/common/cpu_convert.h"
#include "nodes/common/cpu_memcpy.h"
#include "nodes/common/reorder_prim.h"
#include "openvino/core/parallel.hpp"
#include "shape_inference/shape_inference_pass_through.hpp"
#include "utils/precision_support.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/transpose_list.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

bool Reorder::isExecutable() const {
    return Node::isExecutable() && !isOptimized;
}

Reorder::Reorder(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context) :
        Node(op, context, PassThroughShapeInferFactory()) {
    THROW_CPU_NODE_ERR("could not create CPU node from Core node.");
}

Reorder::Reorder(const MemoryDesc& input, const MemoryDesc& output, const std::string& name, const GraphContext::CPtr context) :
    Node("Reorder", {input.getShape()}, {output.getShape()}, {input.getPrecision()}, {output.getPrecision()}, name, context) {
    this->input = input.clone();
    this->output = output.clone();
}

void Reorder::getSupportedDescriptors() {
    if (getParentEdges().size() != 1)
        THROW_CPU_NODE_ERR("has incorrect number of input edges.");
    if (getChildEdges().empty())
        THROW_CPU_NODE_ERR("has incorrect number of output edges.");
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
        THROW_CPU_NODE_ERR("could not initialize supported PDs.");
    }

    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::reorder);

    // must be to initialize here since shapes are unknown at the time of Reorder node creation
    isDynamic = !(config.inConfs[0].getMemDesc()->isDefined() && config.outConfs[0].getMemDesc()->isDefined());
    if (isDynamicNode() && !shapeInference) {
        shapeInference = std::make_shared<ShapeInferPassThrough>();
    }

    if (isDynamic && (config.inConfs[0].getMemDesc()->getShape().getRank() != config.outConfs[0].getMemDesc()->getShape().getRank()))
        THROW_CPU_NODE_ERR("doesn't support case when input and output shapes have different rank and dynamic.");
    if (!isOptimized) {
        const auto &inShape = getInputShapeAtPort(0);
        if (one_of(inShape.getRank(), 4u, 5u) &&
                config.inConfs[0].getMemDesc()->hasLayoutType(LayoutType::nspc) &&
                config.outConfs[0].getMemDesc()->hasLayoutType(LayoutType::ncsp) &&
                config.inConfs[0].getMemDesc()->getPrecision() == ov::element::f32 &&
                config.outConfs[0].getMemDesc()->getPrecision() == ov::element::f32) {
            // oneDNN JIT reorder shows bad perf for nspc to ncsp reorder case so we fallback on simple c++ implementation
            isNspc2NcspCase = true;
        } else if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2) &&
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
    getSelectedPrimitiveDescriptor()->setImplementationType(transposeExecutor->implType());
    return;
}

void Reorder::prepareParams() {
    if (isOptimized)
        return;

    auto srcMemPtr = getSrcMemoryAtPort(0);
    auto dstMemPtr = getDstMemoryAtPort(0);
    if (!dstMemPtr || !dstMemPtr->isDefined())
        THROW_CPU_NODE_ERR("has undefined destination memory object.");
    if (!srcMemPtr || !srcMemPtr->isDefined())
        THROW_CPU_NODE_ERR("has undefined input memory object.");
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_CPU_NODE_ERR("does not have preferable primitive descriptor.");

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

#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
    // @todo current oneDNN v3.2 lacks optimized jit implementation for fp16 reorders.
    // Use transpose executor as a temporary WA.
    if (everyone_is(ov::element::f16, parentDesc->getPrecision(), childDesc->getPrecision()) &&
        ((parentDesc->hasLayoutType(LayoutType::ncsp) && childDesc->hasLayoutType(LayoutType::nspc)) ||
         (parentDesc->hasLayoutType(LayoutType::nspc) && childDesc->hasLayoutType(LayoutType::ncsp))) &&
        one_of(parentDesc->getShape().getRank(), 3u, 4u)) {
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
        if (!dstMemPtr || !dstMemPtr->isDefined())
            THROW_CPU_NODE_ERR("has undefined destination memory object.");
        if (!srcMemPtr || !srcMemPtr->isDefined())
            THROW_CPU_NODE_ERR("has undefined input memory object.");
        if (getSelectedPrimitiveDescriptor() == nullptr)
            THROW_CPU_NODE_ERR("does not have preferable primitive descriptor.");

        createReorderPrimitive(srcMemPtr->getDescWithType<DnnlMemoryDesc>(),
                               dstMemPtr->getDescWithType<DnnlMemoryDesc>());
    }
}

void Reorder::createReorderPrimitive(const DnnlMemoryDescPtr& srcDesc, const DnnlMemoryDescPtr& dstDesc) {
    auto selectedPD = getSelectedPrimitiveDescriptor();
    if (!selectedPD)
        THROW_CPU_NODE_ERR("does not have preferable primitive descriptor.");

    const auto engine = getEngine();
    auto src_desc = srcDesc->getDnnlDesc();
    if (!src_permutation.empty()) {
        CPU_NODE_ASSERT(src_permutation.size() == static_cast<size_t>(src_desc.get_ndims()),
                        "src_permutation size (",
                        src_permutation.size(),
                        ") doesn't match with src_desc ndims(",
                        src_desc.get_ndims(),
                        ")");
        // reorder requires exact matching of logical dimensions between src & dst
        // sometime we have to permute source's logical dimensions to satisfy
        // this requirement, this dosn't affect plugin's node input memory desc.
        /// for (i = 0; i < ndims(); i++)
        ///     new_desc.dims()[permutation[i]] = dims()[i];
        src_desc = src_desc.permute_axes(src_permutation);
    }

    auto dst_desc = dstDesc->getDnnlDesc();

    // TODO: We should keep shape consistency for const and expected shape for node.
    //       If it requires reshape operation it should explicitly injected into graph.
    //
    // There is a limitation for OV representing of weights for grouped convolutions. OV doesn't
    // split group dimension in separate shape dimension. OV use OIHW, but onednn expect GOIHW.
    // So we will perform implicit reshape to dst shape.
    //
    // oneDNN doesn't support direct reorders for tensors of different rank. The code below tries to
    // perform such conversion if the source tensor can be reshaped to the destination rank. This is
    // useful in situations when rank in IR does not much rank that is required by the oneDNN primitive,
    // but the input tensor can be reshaped (e.g. weights for grouped convolutions, biases etc.)
    if (srcDesc->hasLayoutType(LayoutType::ncsp) && srcDesc->getShape().getRank() != dstDesc->getShape().getRank()) {
        const auto newDims = dstDesc->getShape().getStaticDims();
        const auto newFormat = DnnlExtensionUtils::GetPlainFormatByRank(newDims.size());

        src_desc = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(newDims),
                                      DnnlExtensionUtils::ElementTypeToDataType(srcDesc->getPrecision()),
                                      newFormat);
    }

    DEBUG_LOG("CreateReorderPrimitive is called for node", getName(), " src desc: ", src_desc, " dst_desc: ", dst_desc);
    CPU_NODE_ASSERT(src_desc.get_ndims() == dst_desc.get_ndims(), "OneDNN doesn't support reorder with different ranks.");
    auto result = getReorderPrim(context->getParamsCache(), getEngine(), src_desc, dst_desc);
    CPU_NODE_ASSERT(result, "could not create reorder primitive: unsupported reorder case.");
    prim = result;

    selectedPD->setImplementationType(
        parse_impl_name(DnnlExtensionUtils::query_impl_info_str(prim.get_primitive_desc())));

    auto src = getSrcMemoryAtPort(0)->getPrimitive();
    auto dst = getDstMemoryAtPort(0)->getPrimitive();
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

    auto src_data = parentEdge->getMemoryPtr()->getDataAs<const uint8_t>();
    auto dst_data = childEdge->getMemoryPtr()->getDataAs<uint8_t>();

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

    auto src_data = parentEdge->getMemoryPtr()->getDataAs<const float>();
    auto dst_data = childEdge->getMemoryPtr()->getDataAs<float>();

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
#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
    if (transposeExecutor) {
        auto dstMemPtr = getDstMemoryAtPort(0);
        auto srcMemPtr = getSrcMemoryAtPort(0);
        return transposeExecutor->exec({srcMemPtr}, {dstMemPtr});
    }
#endif

    if (isOptimized) {
        DEBUG_LOG("#", getExecIndex(), " Reorder ", getName(), "  is Optimized.",
                   " input @", getSrcDataAtPort(0),
                   " output @", getDstDataAtPort(0));
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
            THROW_CPU_NODE_ERR("doesn't have an initialized primitive.");
        }
    }
}

std::string Reorder::getReorderArgs(const MemoryDesc &parentDesc, const MemoryDesc &childDesc) {
    std::string inArgs, outArgs;
    if (parentDesc.getPrecision() != childDesc.getPrecision()) {
        inArgs += (inArgs.empty() ? "" : "_") + std::string(parentDesc.getPrecision().get_type_name());
        outArgs += (outArgs.empty() ? "" : "_") + std::string(childDesc.getPrecision().get_type_name());
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
        OPENVINO_THROW("Can't reorder data with dynamic shapes");

    if (input.getShape().hasZeroDims() || output.getShape().hasZeroDims()) {
        return;
    }

    if (input.getDesc().isCompatible(output.getDesc())) {
        if (input.getDesc().getPrecision() == element::string) {
            auto srcPtr = input.getDataAs<StringMemory::OvString>();
            auto dstPtr = output.getDataAs<StringMemory::OvString>();
            std::copy(srcPtr, srcPtr + output.getShape().getElementsCount(), dstPtr);
        } else {
            auto srcPtr = static_cast<uint8_t*>(input.getData());
            auto dstPtr = static_cast<uint8_t*>(output.getData());

            auto copySize = output.getSize();
            cpu_memcpy(dstPtr, srcPtr, copySize);
        }
    } else {
        dnnl::reorder reorder;
        std::vector<uint8_t> tmpBuff;

        auto srcMemory = input.getPrimitive();
        auto dstMemory = output.getPrimitive();

        auto srcMemoryDesc = srcMemory.get_desc();
        auto dstMemoryDesc = dstMemory.get_desc();

        auto engine = dstMemory.get_engine();

        if (srcMemoryDesc.get_ndims() != dstMemoryDesc.get_ndims()) {
            //rank mismatch, try to reshape source mem descriptor
            constexpr bool allowEmpty = true;
            auto reshapedSrcMemDesc = srcMemoryDesc.reshape(dstMemoryDesc.get_dims(), allowEmpty);
            if (reshapedSrcMemDesc) {
                srcMemoryDesc = reshapedSrcMemDesc;
                srcMemory = dnnl::memory(srcMemoryDesc, engine, srcMemory.get_data_handle());
            }
        }

        // try directly reorder
        reorder = getReorderPrim(cache, engine, srcMemoryDesc, dstMemoryDesc);
        if (!reorder) {
            // try precision conversion then do the reorder
            if (output.getDataType() != input.getDataType() && Convert::isSupportedDesc(input.getDesc()) &&
                Convert::isSupportedDesc(output.getDesc())) {
                //we probably could not make the reorder because there is no one supporting this precision conversion
                //lets try to convert data first using cpu_convert
                auto data = static_cast<const uint8_t *>(input.getData());
                tmpBuff.resize(output.getSize());

                const auto outPrc = DnnlExtensionUtils::DataTypeToElementType(output.getDataType());
                cpu_convert(data, tmpBuff.data(), DnnlExtensionUtils::DataTypeToElementType(input.getDataType()),
                            outPrc, input.getSize() / input.getDesc().getPrecision().size());

                auto tmpDesc = input.getDesc().cloneWithNewPrecision(outPrc);
                Memory tmpMem(engine, std::move(tmpDesc), tmpBuff.data());

                srcMemory = tmpMem.getPrimitive();
                reorder = getReorderPrim(cache, dstMemory.get_engine(), srcMemory.get_desc(), dstMemory.get_desc());
            }
            if (!reorder) {
                OPENVINO_THROW("No reorder available for the following tensor descriptors: ",
                               input.getDesc().serializeFormat(),
                               " and ",
                               output.getDesc().serializeFormat());
            }
        }
        if (reorder) {
            dnnl::stream loc_stream(engine, dnnl::stream::flags::in_order);
            reorder.execute(loc_stream, {{DNNL_ARG_FROM, srcMemory}, {DNNL_ARG_TO, dstMemory}});
        } else {
            OPENVINO_THROW("Could not make onednn reorder.");
        }
    }
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
