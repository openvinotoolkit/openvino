// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_fullyconnected_primitive.hpp"

#include <oneapi/dnnl/dnnl_types.h>

#include <common/primitive_attr.hpp>
#include <common/primitive_desc_iface.hpp>
#include <common/primitive_iface.hpp>
#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu_memory.h"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "dnnl_postops_composer.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/executors/dnnl/dnnl_shape_agnostic_data.hpp"
#include "nodes/executors/dnnl/dnnl_utils.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "utils/cpu_utils.hpp"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {

using namespace dnnl;
using namespace ov::element;
using namespace executor;

// @todo rewrite using hash_builder
size_t DnnlFCPrimitive::Key::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;

    for (const auto& ptr : {src, wei, bias, dst}) {
        if (ptr) {
            seed = hash_combine(seed, get_md_hash(*ptr->getDnnlDesc().get()));
        }
    }

    seed = hash_combine(seed, get_attr_hash(*attr.get()));
    seed = hash_combine(seed, sparseWeights);
    seed = hash_combine(seed, modelType);

    return seed;
}

bool DnnlFCPrimitive::Key::operator==(const Key& rhs) const {
    bool result = true;

    if (src != rhs.src) {
        result = result && src && rhs.src && src->getDnnlDesc() == rhs.src->getDnnlDesc();
    }
    if (wei != rhs.wei) {
        result = result && wei && rhs.wei && wei->getDnnlDesc() == rhs.wei->getDnnlDesc();
    }
    if (bias != rhs.bias) {
        result = result && bias && rhs.bias && bias->getDnnlDesc() == rhs.bias->getDnnlDesc();
    }
    if (dst != rhs.dst) {
        result = result && dst && rhs.dst && dst->getDnnlDesc() == rhs.dst->getDnnlDesc();
    }

    result =
        result && *attr.get() == *rhs.attr.get() && sparseWeights == rhs.sparseWeights && modelType == rhs.modelType;

    return result;
}

std::shared_ptr<DnnlFCPrimitive> DnnlFCPrimitive::create(const MemoryArgs& memory,
                                                         const FCAttrs& attrs,
                                                         const ExecutorContext::CPtr context,
                                                         const DnnlShapeAgnosticDataPtr& shapeAgnosticData) {
    const auto& srcDesc = MemoryDescUtils::convertToDnnlMemoryDesc(memory.at(ARG_SRC)->getDescPtr());
    const auto& weiDesc = MemoryDescUtils::convertToDnnlMemoryDesc(memory.at(ARG_WEI)->getDescPtr());
    const auto& biaDesc = MemoryDescUtils::convertToDnnlMemoryDesc(memory.at(ARG_BIAS)->getDescPtr());
    const auto& dstDesc = MemoryDescUtils::convertToDnnlMemoryDesc(memory.at(ARG_DST)->getDescPtr());

    Key dnnlFCKey{srcDesc,
                  weiDesc,
                  biaDesc,
                  dstDesc,
                  shapeAgnosticData->primAttrs.attr,
                  attrs.sparseWeights,
                  attrs.modelType};

    auto builder = [&context](const Key& dnnlKey) {
        return std::make_shared<DnnlFCPrimitive>(dnnlKey, context->getEngine(), context->getImplPriorities());
    };

    auto runtimeCache = context->getRuntimeCache();
    const auto result = runtimeCache->getOrCreate(dnnlFCKey, builder);
    const auto& primitive = result.first;
    assert(primitive);

    return primitive;
}

DnnlMemoryDescPtr DnnlFCPrimitive::makeTransposedWeightDescriptor(const DnnlMemoryDescPtr& srcDesc,
                                                                  const DnnlMemoryDescPtr& dstDesc,
                                                                  bool weightsNonTransposed) {
    if (!weightsNonTransposed) {
        return srcDesc;
    }

    const auto& weiDesc = srcDesc->getDnnlDesc();
    auto wDims = weiDesc.get_dims();
    dnnl::memory::dims wDims2D = reshapeDownToRank<2>(wDims);

    const auto transposedWeiDesc = dnnl::memory::desc{wDims2D, weiDesc.get_data_type(), dnnl::memory::format_tag::ba};

    return DnnlExtensionUtils::makeDescriptor(transposedWeiDesc);
}

bool DnnlFCPrimitive::useWeightsDecompressionImpl(const ov::element::Type inputType,
                                                  const ov::element::Type weightsType,
                                                  const ov::intel_cpu::Config::ModelType modelType) {
    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
        if (one_of(inputType, f32, bf16) && one_of(weightsType, u8, i8, nf4, u4, i4, f4e2m1)) {
            return true;
        }

        if (modelType == ov::intel_cpu::Config::ModelType::LLM) {
            // f16c kernel saves memory footprint with additional decompression computational overhead
            // which is only meaningful on LLM with small batch-size.
            // TODO: fall-back to use f32 weights on large batch-size
            if (inputType == f32 && one_of(weightsType, f16, bf16)) {
                return true;
            }
        }
    }
    return false;
}

static bool useDynamicQuantizationImpl(size_t dqGroupSize,
                                       const MemoryDescPtr& srcDesc,
                                       const MemoryDescPtr& weightsDesc,
                                       const MemoryArgs& memory,
                                       bool needTranspose) {
    if (dqGroupSize == 0) {
        return false;
    }

    if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2_vnni) &&
        !dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_vnni)) {
        return false;
    }

    if (srcDesc->getPrecision() != ov::element::f32) {
        return false;
    }

    MemoryCPtr zpPtr =
        memory.count(ARG_WEI | ARG_ATTR_ZERO_POINTS) ? memory.at(ARG_WEI | ARG_ATTR_ZERO_POINTS) : nullptr;
    // For dynamic quantization, VNNI accumulation requires weight to be unsigned.
    // To support dynamic quantization with weights symmetrically quantized as i8/i4
    // w/o zero-point, we will transform weight to u8/u4 weight with zp 128/8.
    if (!one_of(weightsDesc->getPrecision(), ov::element::u8, ov::element::u4) &&
        !((one_of(weightsDesc->getPrecision(), ov::element::i8, ov::element::i4) && !zpPtr))) {
        return false;
    }
    if (zpPtr && !one_of(zpPtr->getDesc().getPrecision(), ov::element::u8, ov::element::u4, ov::element::dynamic)) {
        return false;
    }

    const size_t simdWidth = 16;
    if (dqGroupSize % simdWidth) {
        return false;
    }

    MemoryCPtr scalesPtr = memory.count(ARG_WEI | ARG_ATTR_SCALES) ? memory.at(ARG_WEI | ARG_ATTR_SCALES) : nullptr;
    int ic = weightsDesc->getShape().getStaticDims()[1];
    if (scalesPtr && scalesPtr->getShape().getRank() != 1) {
        auto scalesDims = scalesPtr->getShape().getStaticDims();
        auto groupsNum = scalesDims[1];
        size_t groupSize = ic / groupsNum;
        if (groupsNum != 1 && groupSize % std::min(dqGroupSize, groupSize)) {
            return false;
        }
    }

    if (zpPtr && zpPtr->getShape().getRank() != 1) {
        auto zpDims = zpPtr->getShape().getStaticDims();
        int groupsNum = zpDims[1];
        size_t groupSize = ic / groupsNum;
        if (groupsNum != 1 && groupSize % std::min(dqGroupSize, groupSize)) {
            return false;
        }
    }

    return true;
}

static DnnlPrimitiveAttrs createPrimitiveAttrs(const FCAttrs& attrs,
                                               const PostOps& postOps,
                                               const MemoryArgs& memory,
                                               const ExecutorContext::CPtr& context,
                                               bool useDynamicQuantization) {
    const auto& srcDesc = memory.at(ARG_SRC)->getDescPtr();
    const auto& weiDesc = memory.at(ARG_WEI)->getDescPtr();
    const auto& dstDesc = memory.at(ARG_DST)->getDescPtr();

    const auto& originalDims = dstDesc->getShape().getMinDims();
    const auto& dims = reshapeDownToRank<2>(originalDims);

    auto isINT8 =
        one_of(srcDesc->getPrecision(), ov::element::u8, ov::element::i8) && weiDesc->getPrecision() == ov::element::i8;
    auto outputDataType = DnnlExtensionUtils::ElementTypeToDataType(dstDesc->getPrecision());

    DnnlPostOpsComposer
        dnnlpoc(postOps, context->getEngine(), dims, dims.size() - 1, isINT8, 1 << 0, memory, outputDataType);

    if (memory.count(ARG_WEI | ARG_ATTR_SCALES)) {
        auto dstPrc = memory.at(ARG_WEI | ARG_ATTR_SCALES)->getPrecision();
        if (dstPrc != f8e8m0 || useDynamicQuantization) {
            dstPrc = ov::element::f32;
        }

        dnnlpoc.appendDecompressionScalesLegacy(memory.at(ARG_WEI | ARG_ATTR_SCALES),
                                                !attrs.weightsNonTransposed,
                                                dstPrc);
    }

    if (memory.count(ARG_WEI | ARG_ATTR_ZERO_POINTS)) {
        auto dstPrc = useDynamicQuantization ? ov::element::u8 : ov::element::f32;
        dnnlpoc.appendDecompressionZeroPointsLegacy(memory.at(ARG_WEI | ARG_ATTR_ZERO_POINTS),
                                                    !attrs.weightsNonTransposed,
                                                    dstPrc);
    }

    if (useDynamicQuantization) {
        auto wei_precision = weiDesc->getPrecision();
        bool is_symmetric_weights = (wei_precision == ov::element::i8) || (wei_precision == ov::element::i4);
        if (is_symmetric_weights) {
            // dynamic Quantization needs unsigned quantized weights, conversion from i8/i4 to u8/u4 by adding 128/8
            // introduces 128/8 as zero-points.
            uint8_t zp_value = (wei_precision == ov::element::i8) ? 128 : 8;
            DnnlBlockedMemoryDesc zpMemoryDesc(ov::element::u8, Shape({1}));
            auto decompressionSubtractPtr = std::make_shared<Memory>(context->getEngine(), zpMemoryDesc, &zp_value);
            dnnlpoc.appendDecompressionZeroPointsLegacy(decompressionSubtractPtr,
                                                        !attrs.weightsNonTransposed,
                                                        ov::element::u8);
        }
        dnnlpoc.setDynamicQuantizationParams(attrs.dynamicQuantizationGroupSize);
    }

    return dnnlpoc.compose();
}

static dnnl::memory::desc normalizeDescriptor(const dnnl::memory::desc& desc) {
    const auto& dims = desc.get_dims();

    if (dims.size() > 2) {
        return desc.reshape(reshapeDownToRank<2>(dims));
    }

    return desc;
}

static dnnl::inner_product_forward::primitive_desc createDescriptorInternal(const dnnl::memory::desc& inputDesc,
                                                                            const dnnl::memory::desc& weightDesc,
                                                                            const dnnl::memory::desc& biasDesc,
                                                                            const dnnl::memory::desc& outputDesc,
                                                                            const dnnl::primitive_attr& attr,
                                                                            const dnnl::engine& engine,
                                                                            const bool useSparseWeights,
                                                                            const bool useWeightsDecompression) {
    const auto normalizedInputDesc = normalizeDescriptor(inputDesc);
    const auto normalizedOutputDesc = normalizeDescriptor(outputDesc);
    const auto normalizedWeightDesc = normalizeDescriptor(weightDesc);

    const auto indt = normalizedInputDesc.get_data_type();
    auto wdt = indt;

    if (useWeightsDecompression) {
        wdt = normalizedWeightDesc.get_data_type();

        // dynamic quantization with symmetric quantized weights needs unsigned weights
        uint64_t dynQuantGroupSize = 0;
        attr.get_src_dyn_quant_params(dynQuantGroupSize);
        if (dynQuantGroupSize > 0) {
            if (wdt == dnnl::memory::data_type::s8) {
                wdt = memory::data_type::u8;
            }
            if (wdt == dnnl::memory::data_type::s4) {
                wdt = memory::data_type::u4;
            }
        }
    } else if (indt == dnnl::memory::data_type::u8 || indt == dnnl::memory::data_type::s8) {
        wdt = memory::data_type::s8;
    }

    const dnnl::memory::desc weightsDesc =
        useSparseWeights ? dnnl::memory::desc().sparse_desc(normalizedWeightDesc.get_dims(), wdt)
                         : dnnl::memory::desc(normalizedWeightDesc.get_dims(), wdt, memory::format_tag::any);

    return {engine,
            dnnl::prop_kind::forward_inference,
            normalizedInputDesc,
            weightsDesc,
            biasDesc,
            normalizedOutputDesc,
            attr};
}

static primitive_desc createPrimitiveDesc(const dnnl::memory::desc& inputDesc,
                                          const dnnl::memory::desc& weightDesc,
                                          const dnnl::memory::desc& biasDesc,
                                          const dnnl::memory::desc& outputDesc,
                                          const dnnl::primitive_attr& attr,
                                          const dnnl::engine& engine,
                                          const std::vector<impl_desc_type>& implPriorities,
                                          const bool useSparseWeights,
                                          const bool useWeightsDecompression) {
    auto prim_desc = createDescriptorInternal(inputDesc,
                                              weightDesc,
                                              biasDesc,
                                              outputDesc,
                                              attr,
                                              engine,
                                              useSparseWeights,
                                              useWeightsDecompression);
    OPENVINO_ASSERT(prim_desc, "Failed to create inner_product primitive descriptor");
    auto first_desc = dnnl::inner_product_forward::primitive_desc(prim_desc.get());

    const bool found = DnnlExtensionUtils::find_implementation(prim_desc, [&](impl_desc_type implType) {
        return contains(implPriorities, implType);
    });

    if (found) {
        return std::move(prim_desc);
    }

    return std::move(first_desc);
}

static VectorDims makeDummyInputDims(const Shape& inShape, const Shape& wShape) {
    const auto& weightDims = wShape.getStaticDims();

    auto inMinDims = inShape.getMinDims();
    auto inMaxDims = inShape.getMaxDims();
    inMinDims.back() = weightDims.back();
    inMaxDims.back() = weightDims.back();

    return MemoryDescUtils::makeDummyShape(Shape(inMinDims, inMaxDims)).getStaticDims();
}

static VectorDims makeDummyOutputDims(const VectorDims& inShape, const VectorDims& wShape, const size_t out_rank) {
    size_t activationRank = inShape.size();
    size_t channelRank = wShape.size() - 1;
    // activation   weight    output_shape
    // NCHW         CoCHW     NCo
    // TNC          CoC       TNCo
    // NC           CoC       NCo
    VectorDims outputShape(out_rank, 1);
    // set Co
    outputShape.back() = wShape[0];
    // set batch dims
    size_t batchRank = activationRank - channelRank;
    size_t startIdx = out_rank - batchRank - 1;
    for (size_t i = 0; i < batchRank; i++) {
        outputShape[i + startIdx] = inShape[i];
    }

    return outputShape;
}

DnnlShapeAgnosticDataPtr DnnlFCPrimitive::createShapeAgnosticData(const FCAttrs& attrs,
                                                                  const PostOps& postOps,
                                                                  const MemoryArgs& memory,
                                                                  const ExecutorContext::CPtr& context,
                                                                  const bool cacheWeights) {
    DEBUG_LOG("Creating shape agnostic data");
    auto srcDesc = memory.at(ARG_SRC)->getDescPtr();
    const auto& weiDesc = memory.at(ARG_WEI)->getDescPtr();
    const auto& biasDesc = memory.at(ARG_BIAS)->getDescPtr();
    auto dstDesc = memory.at(ARG_DST)->getDescPtr();

    const auto useWeightsDecompression =
        useWeightsDecompressionImpl(srcDesc->getPrecision(), weiDesc->getPrecision(), attrs.modelType);
    const auto useDynamicQuantization =
        useWeightsDecompression && useDynamicQuantizationImpl(attrs.dynamicQuantizationGroupSize,
                                                              srcDesc,
                                                              weiDesc,
                                                              memory,
                                                              !attrs.weightsNonTransposed);

    const auto postOpData = createPrimitiveAttrs(attrs, postOps, memory, context, useDynamicQuantization);

    if (!cacheWeights) {
        return std::make_shared<DnnlShapeAgnosticData>(postOpData);
    }

    if (srcDesc->getShape().isDynamic()) {
        const auto& inShape = srcDesc->getShape();
        const auto& wShape = weiDesc->getShape();
        const auto& inDymmyDims = makeDummyInputDims(inShape, wShape);
        srcDesc = srcDesc->cloneWithNewDims(inDymmyDims);
        const auto& outDymmyDims =
            makeDummyOutputDims(inDymmyDims, wShape.getStaticDims(), dstDesc->getShape().getRank());
        dstDesc = dstDesc->cloneWithNewDims(outDymmyDims);
    }

    const dnnl::memory::desc srcDnnlDesc = MemoryDescUtils::convertToDnnlMemoryDesc(srcDesc)->getDnnlDesc();
    const dnnl::memory::desc weiDnnlDesc = MemoryDescUtils::convertToDnnlMemoryDesc(weiDesc)->getDnnlDesc();
    const dnnl::memory::desc dstDnnlDesc = MemoryDescUtils::convertToDnnlMemoryDesc(dstDesc)->getDnnlDesc();
    const dnnl::memory::desc biaDnnlDesc = MemoryDescUtils::convertToDnnlMemoryDesc(biasDesc)->getDnnlDesc();

    const auto useSparseWeights = attrs.sparseWeights;
    const auto primDesc = createPrimitiveDesc(srcDnnlDesc,
                                              weiDnnlDesc,
                                              biaDnnlDesc,
                                              dstDnnlDesc,
                                              postOpData.attr,
                                              context->getEngine(),
                                              context->getImplPriorities(),
                                              useSparseWeights,
                                              useWeightsDecompression);

    const auto weightsDesc = DnnlExtensionUtils::makeDescriptor(primDesc.weights_desc());
    auto originalWeightsDesc = MemoryDescUtils::convertToDnnlMemoryDesc(weiDesc);

    originalWeightsDesc = makeTransposedWeightDescriptor(originalWeightsDesc, weightsDesc, attrs.weightsNonTransposed);

    // ignore the result since we just need to put the packed weights into the cache
    (void)utils::prepareWeightsMemory(originalWeightsDesc,
                                      weightsDesc,
                                      memory.at(ARG_WEI),
                                      context,
                                      useDynamicQuantization);

    return std::make_shared<DnnlShapeAgnosticData>(postOpData);
}

static impl_desc_type implTypeFromPrimDesc(const dnnl::primitive_desc& primDesc) {
    const auto implType = parse_impl_name(primDesc.impl_info_str());
    if (implType == ov::intel_cpu::brgemm_avx512_amx &&
        primDesc.weights_desc().get_format_kind() == memory::format_kind::sparsed) {
        return ov::intel_cpu::brgemm_sparse_avx512_amx;
    }

    return implType;
}

DnnlFCPrimitive::DnnlFCPrimitive(const Key& key,
                                 const dnnl::engine& engine,
                                 const std::vector<impl_desc_type>& implPriorities)
    : m_stream(dnnl::stream(engine)),
      m_primDesc(createPrimitiveDesc(
          key.src->getDnnlDesc(),
          key.wei->getDnnlDesc(),
          key.bias->getDnnlDesc(),
          key.dst->getDnnlDesc(),
          key.attr,
          engine,
          implPriorities,
          key.sparseWeights,
          useWeightsDecompressionImpl(key.src->getPrecision(), key.wei->getPrecision(), key.modelType))),
      m_implType(implTypeFromPrimDesc(m_primDesc)),
      m_srcDesc(DnnlExtensionUtils::makeDescriptor(m_primDesc.src_desc())),
      m_weiDesc(DnnlExtensionUtils::makeDescriptor(m_primDesc.weights_desc())),
      m_dstDesc(DnnlExtensionUtils::makeDescriptor(m_primDesc.dst_desc())),
      m_scratchPadDesc(DnnlExtensionUtils::makeDescriptor(m_primDesc.scratchpad_desc())),
      m_prim(primitive(m_primDesc)) {}

void DnnlFCPrimitive::execute(const dnnl_primitive_args& primArgs) const {
    m_prim.execute(m_stream, primArgs);
}

}  // namespace ov::intel_cpu
