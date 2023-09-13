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

using namespace dnnl;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

bool Reorder::isExecutable() const {
    return Node::isExecutable() && !isOptimized;
}

Reorder::Reorder(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context) :
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
        createReorderExecutor(srcMemPtr, dstMemPtr);
    }
}

void Reorder::createReorderExecutor(const ov::intel_cpu::MemoryCPtr& src, const ov::intel_cpu::MemoryCPtr& dst) {
    auto selectedPD = getSelectedPrimitiveDescriptor();
    if (!selectedPD)
        IE_THROW() << "Preferable primitive descriptor is not set.";

    auto engine = getEngine();
    auto cache = context->getParamsCache();

    executor_ptr = std::make_shared<ReorderExecutor>(engine, cache, src, dst, src_permutation);
    executor_ptr->prepareParams(engine, cache, src, dst);
    if (executor_ptr->getPrimitive()) {
        selectedPD->setImplementationType(parse_impl_name(
            DnnlExtensionUtils::query_impl_info_str(executor_ptr->getPrimitive().get_primitive_desc())));
    }

#ifdef CPU_DEBUG_CAPS
    if (executor_ptr->getPrimitive()) {
        auto pd = executor_ptr->getPrimitive().get_primitive_desc();
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
        if (executor_ptr) {
            executor_ptr->setDescs(input, output);
            executor_ptr->updateMem(getParentEdgeAt(0)->getMemoryPtr(), getChildEdgeAt(0)->getMemoryPtr());
            if (!executor_ptr->exec(strm)) {
                OPENVINO_THROW("Reorder node with name ", getName(), " doesn't have an initialized primitive");
            }
        } else {
            OPENVINO_THROW(getName(), "Reorder node didn't create ReorderExecutor!");
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

void Reorder::reorderData2(const IMemory& input, const IMemory& output, MultiCachePtr cache) {
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
        auto engine = output.getPrimitive().get_engine();
        std::vector<int> src_permutation = {};
        auto executor = std::make_shared<ReorderExecutor>(engine,
                                                          cache,
                                                          static_cast<MemoryCPtr>(&input),
                                                          static_cast<MemoryCPtr>(&output),
                                                          src_permutation);
        executor->prepareParams(engine, cache, static_cast<MemoryCPtr>(&input), static_cast<MemoryCPtr>(&output));
        dnnl::stream loc_stream(engine, dnnl::stream::flags::in_order);
        executor->exec(loc_stream);
    }
}

//
// ReorderExecutor implement
//
Reorder::ReorderExecutor::ReorderExecutor(const dnnl::engine& engine,
                                          MultiCachePtr& cache,
                                          const ov::intel_cpu::MemoryCPtr& src,
                                          const ov::intel_cpu::MemoryCPtr& dst,
                                          const std::vector<int> src_permutation) {
    //  Dnnl data types is a limited precision range comparing with MemoryDesc precision
    //        MemoryDesc precision:  BIN, BOOL, I4, U4, I8, U8, I16, U16, BF16, FP16, I32, U32, FP32, I64, U64, FP64
    //        DNNL dataType: bin, s8, u8, f16, bf16, s32, f32, f64
    //
    //  To choose the supported dnnl data type from src/dst MemoryDesc, strict same precison or intermediate precision
    //
    //  [case 1]: Input/output precision has the same precision, means PrecisionX -> precisionX reorder:
    //  1. If MemoryDesc input/output precision has corresponding dnnl data type:
    //     dnnl::reoder should support all dnnl data type
    //     Note: it seems dnnl::reoder reoder doesn's support fp64->fp64, bin->bin, in this case will throw exception.
    //  2. If MemoryDesc input/output precision doesn't have corresponding dnnl data type, throw exception
    //      for example:
    //            I64->I64, U64->U64, I4->I4, U4->U4, I64->I64, etc
    //
    // [case 2]: Input/output precision are the different precision with different layout, supposed PrecisionX -> precisionY reorder:
    // 1. Convert PrecisionX--> data_type::x, PrecisionY--> data_type::y
    //        Check whether dnnl::reorder support x->y reorder, if support there will no other additional operation
    // 2. If dnnl::reorder doesn't support x->y reorder or there is corresponding dnnl data type
    //    1). Choose smaller data size, set PrecisionZ as intermediate precision
    //       For example:
    //              X=FP64, Y=FP32, will choose Z=FP32
    //              X=U8, Y=FP32, will choose Z=U8
    //       Note: If data size is same, will choose src precision as intermediate precision:
    //              X=U32, Y=FP32, will choose Z=U32
    //    2). If PrecisionZ is not supported by dnnl::reoder, throw exception:
    //        For example:
    //              X=FP64, Y=I64 --> Z=I64, not supported, throw exception
    //              X=FP64, Y=U32 --> Z=U32, not supported, throw exception
    //    3). Allocate memory for intermediate PrecisionZ, and do data conversion before or after dnnl::reorder
    //            If PrecisionZ == PrecisionX, do data conversion after dnnl::reorder
    //                input(PrecX) --> reorder(PrecX) --> conversion(PrecX->PrecY) --> output(PrecY)
    //            If PrecisionZ == PrecisionY, do data conversion before dnnl::reorder
    //                input(PreX) --> conversion(PrecX->PrecY) -> reorder(PrecY) --> output(PrecY)
    // [case 3]: Input/output precision are the different precision with same layout, in this case only precision is needed
    //    Don't need do reorder, only do data conversion
    //

    auto src_prc = src->getDesc().getPrecision();
    auto dst_prc = dst->getDesc().getPrecision();
    OPENVINO_ASSERT(src_prc != InferenceEngine::Precision::UNSPECIFIED,
                    "ReorderExecutor input precision is unspecified!");
    OPENVINO_ASSERT(dst_prc != InferenceEngine::Precision::UNSPECIFIED,
                    "ReorderExecutor output precision is unspecified!");

    prim = dnnl::reorder();
    // if (src_prc != dst_prc) {
    //    if (src->getDescWithType<BlockedMemoryDesc>()->getOrder() ==
    //        dst->getDescWithType<BlockedMemoryDesc>()->getOrder()) {
    //        need_reorder = false;
    //        src_blocked = std::make_shared<Memory>(engine, src->getDescPtr(), src->getData(), false);
    //        dst_blocked = std::make_shared<Memory>(engine, dst->getDescPtr(), dst->getData(), false);
    //        pre_converter = std::make_shared<IntermConverter>(src_blocked, src_prc, dst_blocked, dst_prc);
    //        return;
    //    }
    // }

    need_reorder = true;
    auto src_data_type = DnnlExtensionUtils::IEPrecisionToDataType(src_prc);
    auto dst_data_type = DnnlExtensionUtils::IEPrecisionToDataType(dst_prc);

    // dnnl::memory::data_type doesn't support this precision
    if (src_data_type == dnnl::memory::data_type::undef && dst_data_type == dnnl::memory::data_type::undef) {
        OPENVINO_THROW("Reorder doesn't support: ", src_prc, " -> ", dst_prc);
    }

    src_blocked = std::make_shared<Memory>(engine, src->getDescPtr(), src->getData(), false);
    dst_blocked = std::make_shared<Memory>(engine, dst->getDescPtr(), dst->getData(), false);

    dnnl::memory::desc src_desc, dst_desc;
    if (src_data_type != dnnl::memory::data_type::undef && dst_data_type != dnnl::memory::data_type::undef) {
        // Check whether dnnl::reorder directly support input->output
        src_desc = updateSrcDesc(engine, src_permutation);
        dst_desc = dst_blocked->getPrimitive().get_desc();
        auto result = getReorderPrim(cache, engine, src_desc, dst_desc);

        if (result) {
            prim = result;
            DEBUG_LOG("** (",
                      DnnlExtensionUtils::DataTypeToIEPrecision(src_desc.get_data_type()),
                      ", ",
                      DnnlExtensionUtils::DataTypeToIEPrecision(dst_desc.get_data_type()),
                      ")",
                      " is supported reorder");
            return;
        }
        // dnnl::reorder doesn't directly support input->output
        if (src_data_type == dst_data_type) {
            // dnnl::reorder doesn't support fp64->fp64 and bin->bin.
            OPENVINO_THROW("dnnl::reorder doesn't support: ", src_prc, " -> ", dst_prc);
        }
    }

    auto intermediate_data_type = src_data_type;
    if (src_data_type != dnnl::memory::data_type::undef && dst_data_type != dnnl::memory::data_type::undef) {
        // input and output are different data types, choose the data type with smaller data size as intermediate type
        if (dnnl::memory::data_type_size(src_data_type) > dnnl::memory::data_type_size(dst_data_type)) {
            intermediate_data_type = dst_data_type;
        }
    } else if (src_data_type == dnnl::memory::data_type::undef) {
        intermediate_data_type = dst_data_type;
    } else if (dst_data_type == dnnl::memory::data_type::undef) {
        intermediate_data_type = src_data_type;
    }

    scratch_ptr = std::make_shared<DnnlScratchPad>(engine);
    if (intermediate_data_type == dst_data_type) {
        // Need preConvert
        InferenceEngine::Precision out_prec = DnnlExtensionUtils::DataTypeToIEPrecision(intermediate_data_type);
        MemoryDescPtr out_desc = src->getDesc().cloneWithNewPrecision(out_prec);
        auto out_mem = scratch_ptr->createScratchPadMem(out_desc);
        pre_converter = std::make_shared<IntermConverter>(src_blocked, src_prc, out_mem, out_prec);
        src_blocked = out_mem;
        post_converter = nullptr;
    } else {
        // Need postConvert
        InferenceEngine::Precision in_prec = DnnlExtensionUtils::DataTypeToIEPrecision(intermediate_data_type);
        MemoryDescPtr in_desc = dst->getDesc().cloneWithNewPrecision(in_prec);
        auto in_mem = scratch_ptr->createScratchPadMem(in_desc);
        post_converter = std::make_shared<IntermConverter>(in_mem, in_prec, dst_blocked, dst_prc);
        dst_blocked = in_mem;
        pre_converter = nullptr;
    }

    src_desc = updateSrcDesc(engine, src_permutation);
    dst_desc = dst_blocked->getPrimitive().get_desc();
    auto result = getReorderPrim(cache, engine, src_desc, dst_desc);

    if (result) {
        prim = result;
        DEBUG_LOG("** (",
                  DnnlExtensionUtils::DataTypeToIEPrecision(src_desc.get_data_type()),
                  ", ",
                  DnnlExtensionUtils::DataTypeToIEPrecision(dst_desc.get_data_type()),
                  ")",
                  " is supported reorder");
        return;
    }

    OPENVINO_THROW("dnnl::reorder doesn't support: ",
                   DnnlExtensionUtils::DataTypeToIEPrecision(src_desc.get_data_type()),
                   " -> ",
                   DnnlExtensionUtils::DataTypeToIEPrecision(dst_desc.get_data_type()));
}

dnnl::memory::desc Reorder::ReorderExecutor::updateSrcDesc(const dnnl::engine& engine,
                                                           const std::vector<int> src_permutation) {
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

        auto newDesc =
            dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(newDims), src_blocked->getDataType(), newFormat);
        auto _srcPtr = src_blocked->getData();
        src_blocked = std::make_shared<Memory>(engine, DnnlExtensionUtils::makeDescriptor(newDesc), _srcPtr, false);

        src_desc = src_blocked->getPrimitive().get_desc();
    }
    return src_desc;
}

void Reorder::ReorderExecutor::prepareParams(const dnnl::engine& engine,
                                             MultiCachePtr& cache,
                                             const ov::intel_cpu::MemoryCPtr& src,
                                             const ov::intel_cpu::MemoryCPtr& dst) {
    if (!need_reorder)
        return;

    auto dnnl_mem_src = src_blocked->getPrimitive();
    auto dnnl_mem_dst = dst_blocked->getPrimitive();
    if (!pre_converter)
        dnnl_mem_src = src->getPrimitive();
    if (!post_converter)
        dnnl_mem_dst = dst->getPrimitive();
    primArgs = {{DNNL_ARG_SRC, dnnl_mem_src}, {DNNL_ARG_DST, dnnl_mem_dst}};
}

void Reorder::ReorderExecutor::updateMem(const ov::intel_cpu::MemoryPtr& src, const ov::intel_cpu::MemoryPtr& dst) {
    // Update due to changeDefaultPtr maybe update input/output ptr
    if (need_reorder) {
        if (pre_converter)
            pre_converter->setInputMem(src);
        if (post_converter)
            post_converter->setOutputMem(dst);
    } else if (pre_converter) {
        pre_converter->setInputMem(src);
        pre_converter->setOutputMem(dst);
    }
}

void Reorder::ReorderExecutor::IntermConverter::convert() {
    OPENVINO_ASSERT(src_mem && dst_mem, "ReorderExecutor::IntermConverter has no input/output!");
    OPENVINO_ASSERT(src_mem->isAllocated());
    OPENVINO_ASSERT(dst_mem->isAllocated());

    auto src = static_cast<const uint8_t*>(src_mem->getData());
    auto dst = dst_mem->getData();
    size_t size = src_mem->getSize() / src_mem->getDesc().getPrecision().size();

    cpu_convert(src, dst, src_prec, dst_prec, size);
}

void Reorder::ReorderExecutor::preConvert() {
    if (!pre_converter)
        return;
    if (input)
        pre_converter->setInputPrec(input->getPrecision());
    pre_converter->convert();
}

void Reorder::ReorderExecutor::postConvert() {
    if (!post_converter)
        return;
    if (output)
        post_converter->setOutputPrec(output->getPrecision());
    post_converter->convert();
}

bool Reorder::ReorderExecutor::exec(dnnl::stream strm) {
    preConvert();
    if (need_reorder) {
        if (prim) {
            prim.execute(strm, primArgs);
        } else {
            return false;
        }
    }
    postConvert();
    return true;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
