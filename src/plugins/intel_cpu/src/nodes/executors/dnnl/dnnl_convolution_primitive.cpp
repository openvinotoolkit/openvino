// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/executors/dnnl/dnnl_convolution_primitive.hpp"

#include <common/primitive_desc_iface.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "dnnl_extension_utils.h"
#include "dnnl_postops_composer.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/executors/convolution_config.hpp"
#include "nodes/executors/dnnl/dnnl_aliases.hpp"
#include "nodes/executors/dnnl/dnnl_fullyconnected_primitive.hpp"
#include "nodes/executors/dnnl/dnnl_shape_agnostic_data.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"

namespace ov {
namespace intel_cpu {

using namespace dnnl;
using namespace executor;

// @todo rewrite using hash_builder
size_t DnnlConvolutionPrimitive::Key::hash() const {
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

bool DnnlConvolutionPrimitive::Key::operator==(const Key& rhs) const {
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

    result = result && *attr.get() == *rhs.attr.get();

    return result;
}

// make a fake shape: N, C, W
template <typename T>
static std::vector<T> normalizeDims(const std::vector<T>& dims) {
    assert(one_of(static_cast<int>(dims.size()), 2, 3));

    if (dims.size() == 3) {
        return {dims[0], dims[2], dims[1]};
    }

    return {dnnl::memory::dim{1}, dims[1], dims[0]};
}

static dnnl::convolution_forward::primitive_desc createDescriptorInternal(const dnnl::memory::desc& inputDesc,
                                                                          const dnnl::memory::desc& weightDesc,
                                                                          const dnnl::memory::desc& biasDesc,
                                                                          const dnnl::memory::desc& outputDesc,
                                                                          const dnnl::primitive_attr& attr,
                                                                          const dnnl::engine& engine) {
    const auto normalizedInDims = normalizeDims(inputDesc.get_dims());
    const auto convInDesc = dnnl::memory::desc(normalizedInDims, inputDesc.get_data_type(), memory::format_tag::nwc);
    const auto normalizedOutDims = normalizeDims(outputDesc.get_dims());
    const auto convOutDesc = dnnl::memory::desc(normalizedOutDims, outputDesc.get_data_type(), memory::format_tag::nwc);

    // @todo create general mapping from node configuration to backend configuration
    static const std::map<memory::data_type, memory::data_type> weightsTypeByInputType{
        // input data type        weights data type
        {memory::data_type::f32, memory::data_type::f32},
        {memory::data_type::f16, memory::data_type::f16},
        {memory::data_type::bf16, memory::data_type::bf16},
        {memory::data_type::u8, memory::data_type::s8},
        {memory::data_type::s8, memory::data_type::s8},
    };

    // make a fake shape: OC, IC, 1
    const auto& weightDims = weightDesc.get_dims();
    const dnnl::memory::dims normalizedWeightDims{static_cast<dnnl::memory::dim>(weightDims[0]),
                                                  static_cast<dnnl::memory::dim>(weightDims[1]),
                                                  dnnl::memory::dim{1}};
    const auto weightDataType = weightsTypeByInputType.at(inputDesc.get_data_type());
    const auto convWeightDescAny =
        dnnl::memory::desc(normalizedWeightDims, weightDataType, dnnl::memory::format_tag::any);

    return dnnl::convolution_forward::primitive_desc(engine,
                                                     prop_kind::forward_inference,
                                                     dnnl::algorithm::convolution_direct,
                                                     convInDesc,
                                                     convWeightDescAny,
                                                     biasDesc,
                                                     convOutDesc,
                                                     dnnl::memory::dims{1},  // stride
                                                     dnnl::memory::dims{0},  // dilation
                                                     dnnl::memory::dims{0},  // paddingL
                                                     dnnl::memory::dims{0},  // paddingR
                                                     attr);
}

static primitive_desc createPrimitiveDesc(const dnnl::engine& engine,
                                          const dnnl::memory::desc& inputDesc,
                                          const dnnl::memory::desc& weightDesc,
                                          const dnnl::memory::desc& biasDesc,
                                          const dnnl::memory::desc& outputDesc,
                                          const dnnl::primitive_attr& attr,
                                          const std::vector<impl_desc_type>& implPriorities) {
    auto prim_desc = createDescriptorInternal(inputDesc, weightDesc, biasDesc, outputDesc, attr, engine);
    auto first_desc = dnnl::convolution_forward::primitive_desc(prim_desc.get());

    for (auto preferredImplType : implPriorities) {
        const bool found = DnnlExtensionUtils::find_implementation(prim_desc, preferredImplType);

        if (found)
            return std::move(prim_desc);
    }

    return std::move(first_desc);
}

static DnnlPrimitiveAttrs createPrimitiveAttrs(const ConvAttrs& attrs,
                                               const PostOps& postOps,
                                               const MemoryArgs& memory,
                                               ExecutorContext::CPtr context) {
    const auto& srcDesc = memory.at(ARG_SRC)->getDescPtr();
    const auto& weiDesc = memory.at(ARG_WEI)->getDescPtr();
    const auto& dstDesc = memory.at(ARG_DST)->getDescPtr();

    const auto& originalDims = dstDesc->getShape().getMinDims();
    const auto& dims = normalizeDims(originalDims);

    auto isINT8 =
        one_of(srcDesc->getPrecision(), ov::element::u8, ov::element::i8) && weiDesc->getPrecision() == ov::element::i8;
    auto outputDataType = DnnlExtensionUtils::ElementTypeToDataType(dstDesc->getPrecision());

    DnnlPostOpsComposer
        dnnlpoc(postOps, context->getEngine(), dims, 1, isINT8, 1 << 0, {}, attrs.withBias, outputDataType);

    return dnnlpoc.compose();
}

DnnlShapeAgnosticDataPtr DnnlConvolutionPrimitive::createShapeAgnosticData(const FCAttrs& attrs,
                                                                           const PostOps& postOps,
                                                                           const MemoryArgs& memory,
                                                                           const ExecutorContext::CPtr context,
                                                                           const bool cacheWeights) {
    DEBUG_LOG("Creating shape agnostic data");
    ConvAttrs convAttrs{attrs.withBias};

    const auto postOpData = createPrimitiveAttrs(convAttrs, postOps, memory, context);

    return std::make_shared<DnnlShapeAgnosticData>(postOpData);
}

void DnnlConvolutionPrimitive::execute(const dnnl_primitive_args& primArgs) const {
    m_prim.execute(m_stream, primArgs);
}

std::shared_ptr<DnnlConvolutionPrimitive> DnnlConvolutionPrimitive::create(
    const MemoryArgs& memory,
    const ConvAttrs& attrs,
    const ExecutorContext::CPtr context,
    const DnnlShapeAgnosticDataPtr& shapeAgnosticData) {
    const auto& srcDesc = MemoryDescUtils::convertToDnnlMemoryDesc(memory.at(ARG_SRC)->getDescPtr());
    const auto& weiDesc = MemoryDescUtils::convertToDnnlMemoryDesc(memory.at(ARG_WEI)->getDescPtr());
    const auto& biaDesc = MemoryDescUtils::convertToDnnlMemoryDesc(memory.at(ARG_BIAS)->getDescPtr());
    const auto& dstDesc = MemoryDescUtils::convertToDnnlMemoryDesc(memory.at(ARG_DST)->getDescPtr());

    const Key dnnlConvKey{srcDesc, weiDesc, biaDesc, dstDesc, shapeAgnosticData->primAttrs.attr};

    auto builder = [&context](const Key& dnnlKey) {
        return std::make_shared<DnnlConvolutionPrimitive>(dnnlKey, context->getEngine(), context->getImplPriorities());
    };

    auto runtimeCache = context->getRuntimeCache();
    const auto result = runtimeCache->getOrCreate(dnnlConvKey, builder);
    const auto& primitive = result.first;
    assert(primitive);

    return primitive;
}

DnnlMemoryDescPtr DnnlConvolutionPrimitive::makeTransposedWeightDescriptor(const DnnlMemoryDescPtr srcDesc,
                                                                           const DnnlMemoryDescPtr dstDesc,
                                                                           bool weightsNonTransposed) {
    return DnnlFCPrimitive::makeTransposedWeightDescriptor(srcDesc, dstDesc, weightsNonTransposed);
}

DnnlConvolutionPrimitive::DnnlConvolutionPrimitive(const Key& key,
                                                   const dnnl::engine& engine,
                                                   const std::vector<impl_desc_type>& implPriorities)
    : m_stream(dnnl::stream(engine)),
      m_primDesc(createPrimitiveDesc(engine,
                                     key.src->getDnnlDesc(),
                                     key.wei->getDnnlDesc(),
                                     key.bias->getDnnlDesc(),
                                     key.dst->getDnnlDesc(),
                                     key.attr,
                                     implPriorities)),
      m_implType(parse_impl_name(m_primDesc.impl_info_str())),
      m_srcDesc(DnnlExtensionUtils::makeDescriptor(m_primDesc.src_desc())),
      m_weiDesc(DnnlExtensionUtils::makeDescriptor(m_primDesc.weights_desc())),
      m_dstDesc(DnnlExtensionUtils::makeDescriptor(m_primDesc.dst_desc())),
      m_scratchPadDesc(DnnlExtensionUtils::makeDescriptor(m_primDesc.scratchpad_desc())),
      m_prim(primitive(m_primDesc)) {}

}  // namespace intel_cpu
}  // namespace ov
