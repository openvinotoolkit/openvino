// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/executors/dnnl/dnnl_matmul_primitive.hpp"

#include <oneapi/dnnl/dnnl_types.h>

#include <common/primitive_attr.hpp>
#include <common/primitive_desc_iface.hpp>
#include <common/primitive_iface.hpp>
#include <cstdint>
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
#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/executors/dnnl/dnnl_shape_agnostic_data.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/matmul_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "utils/debug_capabilities.h"
#include "dnnl_utils.hpp"

namespace ov {
namespace intel_cpu {

using namespace dnnl;
using namespace ov::element;
using namespace executor;

// @todo rewrite using hash_builder
size_t DnnlMatMulPrimitive::Key::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;

    for (const auto& ptr : {src, wei, bias, dst}) {
        if (ptr) {
            seed = hash_combine(seed, get_md_hash(*ptr->getDnnlDesc().get()));
        }
    }

    seed = hash_combine(seed, get_attr_hash(*attr.get()));

    return seed;
}

bool DnnlMatMulPrimitive::Key::operator==(const Key& rhs) const {
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

    return result;
}

std::shared_ptr<DnnlMatMulPrimitive> DnnlMatMulPrimitive::create(const MemoryArgs& memory,
                                                                 const MatMulAttrs& attrs,
                                                                 const ExecutorContext::CPtr context,
                                                                 const DnnlShapeAgnosticDataPtr& shapeAgnosticData) {
    const auto& srcDesc = MemoryDescUtils::convertToDnnlMemoryDesc(memory.at(ARG_SRC)->getDescPtr());
    const auto& weiDesc = MemoryDescUtils::convertToDnnlMemoryDesc(memory.at(ARG_WEI)->getDescPtr());
    const auto& biaDesc = MemoryDescUtils::convertToDnnlMemoryDesc(memory.at(ARG_BIAS)->getDescPtr());
    const auto& dstDesc = MemoryDescUtils::convertToDnnlMemoryDesc(memory.at(ARG_DST)->getDescPtr());

    Key dnnlFCKey{
        srcDesc,
        weiDesc,
        biaDesc,
        dstDesc,
        shapeAgnosticData->primAttrs.attr,
        attrs.activation_k_dim,
        attrs.activation_offset
    };

    auto builder = [&context](const Key& dnnlKey) {
        return std::make_shared<DnnlMatMulPrimitive>(dnnlKey, context->getEngine(), context->getImplPriorities());
    };

    auto runtimeCache = context->getRuntimeCache();
    const auto result = runtimeCache->getOrCreate(dnnlFCKey, builder);
    const auto& primitive = result.first;
    assert(primitive);

    return primitive;
}

bool DnnlMatMulPrimitive::useWeightsDecompressionImpl(const ov::element::Type inputType,
                                                      const ov::element::Type weightsType) {
    return dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2) && one_of(inputType, f32, bf16) &&
           one_of(weightsType, u8, nf4, u4, i4);
}

bool DnnlMatMulPrimitive::useDynamicQuantizationImpl(size_t dqGroupSize,
                                                     const MemoryDescPtr srcDesc,
                                                     const MemoryDescPtr weightsDesc,
                                                     MemoryCPtr scalesPtr,
                                                     MemoryCPtr zpPtr,
                                                     bool needTranspose) {
    if (dqGroupSize == 0)
        return false;

    if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2_vnni) &&
        !dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_vnni))
        return false;

    if (srcDesc->getPrecision() != ov::element::f32)
        return false;

    if (!one_of(weightsDesc->getPrecision(), ov::element::u8, ov::element::u4))
        return false;

    if (zpPtr && !one_of(zpPtr->getDesc().getPrecision(), ov::element::u8, ov::element::u4, ov::element::undefined))
        return false;

    const size_t simdWidth = 16;
    if (dqGroupSize % simdWidth)
        return false;

    if (weightsDesc->getPrecision() == ov::element::u4) {
        int ic = weightsDesc->getShape().getStaticDims()[1];
        int minGroupSize = INT_MAX;
        if (scalesPtr && scalesPtr->getShape().getRank() == 3) {
            auto scalesDims = scalesPtr->getShape().getStaticDims();
            auto groupsNum = needTranspose ? scalesDims[1] : scalesDims[0];
            minGroupSize = ic / groupsNum;
        }
        if (zpPtr && zpPtr->getShape().getRank() == 3) {
            auto zpDims = zpPtr->getShape().getStaticDims();
            int groupsNum = needTranspose ? zpDims[1] : zpDims[0];
            minGroupSize = std::min(minGroupSize, ic / groupsNum);
        }

        const size_t minLoopSize = 8;
        if (minGroupSize != INT_MAX && minGroupSize % minLoopSize)
            return false;
    }

    return true;
}

template <typename T>
static std::vector<T> normalizeDimsTo2D(const std::vector<T>& dims) {
    return {std::accumulate(dims.begin(), dims.end() - 1, (T)1, std::multiplies<T>()), dims[dims.size() - 1]};
}

static DnnlPrimitiveAttrs createPrimitiveAttrs(const MatMulAttrs& attrs,
                                               const PostOps& postOps,
                                               const MemoryArgs& memory,
                                               ExecutorContext::CPtr context,
                                               bool useDynamicQuantization) {
    const auto& srcDesc = memory.at(ARG_SRC)->getDescPtr();
    const auto& weiDesc = memory.at(ARG_WEI)->getDescPtr();
    const auto& dstDesc = memory.at(ARG_DST)->getDescPtr();

    const auto& originalDims = dstDesc->getShape().getMinDims();
    const auto& dims = originalDims;

    auto isINT8 =
        one_of(srcDesc->getPrecision(), ov::element::u8, ov::element::i8) && weiDesc->getPrecision() == ov::element::i8;
    auto outputDataType = DnnlExtensionUtils::ElementTypeToDataType(dstDesc->getPrecision());

    DnnlPostOpsComposer dnnlpoc(postOps,
                                context->getEngine(),
                                dims,
                                dims.size() - 1,
                                isINT8,
                                1 << 0,
                                attrs.dequantizationScales,
                                !memory.at(ARG_BIAS)->getDesc().empty(),
                                outputDataType);

    return dnnlpoc.compose();
}

static dnnl::memory::desc normalizeDescriptor(const dnnl::memory::desc& desc) {
    const auto& dims = desc.get_dims();

    if (dims.size() > 2)
        return desc.reshape(normalizeDimsTo2D(dims));

    return desc;
}

static dnnl::matmul::primitive_desc createDescriptorInternal(const dnnl::memory::desc& inputDesc,
                                                             const dnnl::memory::desc& weightDesc,
                                                             const dnnl::memory::desc& biasDesc,
                                                             const dnnl::memory::desc& outputDesc,
                                                             const dnnl::primitive_attr& attr,
                                                             const dnnl::engine& engine,
                                                             const bool useSparseWeights,
                                                             const bool useWeightsDecompression,
                                                             const bool weightsNonTransposed,
                                                             int64_t K,
                                                             int64_t offset) {
    auto wDims = weightDesc.get_dims();
    if (!weightsNonTransposed) {
        std::swap(wDims[wDims.size() - 1], wDims[wDims.size() - 2]);
    }

    while (inputDesc.get_ndims() > wDims.size()) {
        wDims.insert(wDims.begin(), 1);
    }

    auto bDims = biasDesc.get_dims();
    while (!biasDesc.is_zero() && outputDesc.get_ndims() > bDims.size()) {
        bDims.insert(bDims.begin(), 1);
    }
    auto newBiasDesc = !biasDesc.is_zero() && biasDesc.get_ndims() != bDims.size() ? biasDesc.reshape(bDims) : biasDesc;

    const auto normalizedOutputDesc = outputDesc;
    auto inputSumDims = inputDesc.get_dims();
    inputSumDims.back() = K;

    dnnl::memory::dims offsets(inputSumDims.size(), 0);
    offsets.back() = offset;
    // std::cout << "initial intput desc: " << inputDesc << "\n";
    // std::cout << "inputSumDims" << PrintableVector<dnnl::memory::dim>(inputSumDims) << "\n";
    // std::cout << "offsets" << PrintableVector<dnnl::memory::dim>(offsets) << "\n";

    auto inputSubDesc = inputDesc.submemory_desc(inputSumDims, offsets);

    if (K == 0) {
        inputSubDesc = inputDesc;
    }

    const auto indt = inputSubDesc.get_data_type();
    auto wdt = indt;

    // if (useWeightsDecompression) {
    //     wdt = weightDesc.get_data_type();
    // } else if (indt == dnnl::memory::data_type::u8 || indt == dnnl::memory::data_type::s8) {
    //     wdt = memory::data_type::s8;
    // }

    const dnnl::memory::desc weightsDesc =
        useSparseWeights ? dnnl::memory::desc().sparse_desc(wDims, wdt)
                         : dnnl::memory::desc(wDims, wdt, memory::format_tag::any);

    // std::cout << "initial intput desc: " << inputDesc << "\n";
    // std::cout << "intput desc: " << inputSubDesc << "\n";
    // std::cout << "initial wei desc: " << weightDesc << "\n";
    // std::cout << "wei desc: " << weightsDesc << "\n";
    // std::cout << "initial bias desc: " << biasDesc << "\n";
    // std::cout << "bias desc: " << newBiasDesc << "\n";
    // std::cout << "output desc: " << normalizedOutputDesc << "\n";

    return dnnl::matmul::primitive_desc(engine,
                                        inputSubDesc,
                                        weightsDesc,
                                        newBiasDesc,
                                        normalizedOutputDesc,
                                        attr);
}

static primitive_desc createPrimitiveDesc(const dnnl::memory::desc& inputDesc,
                                          const dnnl::memory::desc& weightDesc,
                                          const dnnl::memory::desc& biasDesc,
                                          const dnnl::memory::desc& outputDesc,
                                          const dnnl::primitive_attr& attr,
                                          const dnnl::engine& engine,
                                          const std::vector<impl_desc_type>& implPriorities,
                                          const bool useSparseWeights,
                                          const bool useWeightsDecompression,
                                          const bool weightsNonTransposed,
                                          int64_t K,
                                          int64_t offset) {
    auto prim_desc = createDescriptorInternal(inputDesc,
                                              weightDesc,
                                              biasDesc,
                                              outputDesc,
                                              attr,
                                              engine,
                                              useSparseWeights,
                                              useWeightsDecompression,
                                              weightsNonTransposed,
                                              K,
                                              offset);
    OPENVINO_ASSERT(prim_desc, "Failed to create matmul primitive descriptor");
    auto first_desc = dnnl::matmul::primitive_desc(prim_desc.get());

    const bool found = DnnlExtensionUtils::find_implementation(prim_desc, [&](impl_desc_type implType) {
        return contains(implPriorities, implType);
    });

    if (found)
        return std::move(prim_desc);

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

DnnlShapeAgnosticDataPtr DnnlMatMulPrimitive::createShapeAgnosticData(const FCAttrs& attrs,
                                                                      const PostOps& postOps,
                                                                      const MemoryArgs& memory,
                                                                      const ExecutorContext::CPtr context,
                                                                      const bool cacheWeights) {
    DEBUG_LOG("Creating shape agnostic data");
    auto srcDesc = memory.at(ARG_SRC)->getDescPtr();
    const auto& weiDesc = memory.at(ARG_WEI)->getDescPtr();
    const auto& biasDesc = memory.at(ARG_BIAS)->getDescPtr();
    auto dstDesc = memory.at(ARG_DST)->getDescPtr();
    MatMulAttrs mmAttrs{false, false, attrs.dequantizationScales, attrs.activation_k_dim, attrs.activation_offset};
    // std::cout << "### Weights transposed? " << attrs.weightsNonTransposed  << "\n";

    const auto postOpData = createPrimitiveAttrs(mmAttrs, postOps, memory, context, false);

    if (!cacheWeights)
        return std::make_shared<DnnlShapeAgnosticData>(postOpData);

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

    const auto primDesc = createPrimitiveDesc(srcDnnlDesc,
                                              weiDnnlDesc,
                                              biaDnnlDesc,
                                              dstDnnlDesc,
                                              postOpData.attr,
                                              context->getEngine(),
                                              context->getImplPriorities(),
                                              false,
                                              false,
                                              attrs.weightsNonTransposed,
                                              mmAttrs.activation_k_dim,
                                              mmAttrs.activation_offset);

    const auto weightsDesc = DnnlExtensionUtils::makeDescriptor(primDesc.weights_desc());
    auto originalWeightsDesc = MemoryDescUtils::convertToDnnlMemoryDesc(weiDesc);
    // if (attrs.weightsNonTransposed)
    //     originalWeightsDesc = utils::makeTransposedWeightDescriptor(originalWeightsDesc, weightsDesc);

    // ignore the result since we just need to put the packed weights into the cache
    (void)utils::prepareWeightsMemory(originalWeightsDesc, weightsDesc, memory.at(ARG_WEI), context);

    return std::make_shared<DnnlShapeAgnosticData>(postOpData);
}

static impl_desc_type implTypeFromPrimDesc(const dnnl::primitive_desc primDesc) {
    const auto implType = parse_impl_name(primDesc.impl_info_str());
    if (implType == ov::intel_cpu::brgemm_avx512_amx &&
        primDesc.weights_desc().get_format_kind() == memory::format_kind::sparsed) {
        return ov::intel_cpu::brgemm_sparse_avx512_amx;
    }

    return implType;
}

DnnlMatMulPrimitive::DnnlMatMulPrimitive(const Key& key,
                                         const dnnl::engine& engine,
                                         const std::vector<impl_desc_type>& implPriorities)
    : m_stream(dnnl::stream(engine)),
      m_primDesc(createPrimitiveDesc(key.src->getDnnlDesc(),
                                     key.wei->getDnnlDesc(),
                                     key.bias->getDnnlDesc(),
                                     key.dst->getDnnlDesc(),
                                     key.attr,
                                     engine,
                                     implPriorities,
                                     false,
                                     false,
                                     false,
                                     key.K,
                                     key.offset)),
      m_implType(implTypeFromPrimDesc(m_primDesc)),
      m_srcDesc(DnnlExtensionUtils::makeDescriptor(m_primDesc.src_desc())),
      m_weiDesc(DnnlExtensionUtils::makeDescriptor(m_primDesc.weights_desc())),
      m_dstDesc(DnnlExtensionUtils::makeDescriptor(m_primDesc.dst_desc())),
      m_scratchPadDesc(DnnlExtensionUtils::makeDescriptor(m_primDesc.scratchpad_desc())),
      m_prim(primitive(m_primDesc)) {}

void DnnlMatMulPrimitive::execute(const dnnl_primitive_args& primArgs) const {
    m_prim.execute(m_stream, primArgs);
}

}  // namespace intel_cpu
}  // namespace ov
