// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/executors/dnnl/dnnl_convolution_primitive.hpp"

#include <oneapi/dnnl/dnnl_types.h>

#include <algorithm>
#include <cassert>
#include <common/c_types_map.hpp>
#include <common/primitive_attr.hpp>
#include <common/primitive_hashing_utils.hpp>
#include <common/utils.hpp>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "dnnl_postops_composer.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/executors/convolution_config.hpp"
#include "nodes/executors/dnnl/dnnl_aliases.hpp"
#include "nodes/executors/dnnl/dnnl_fullyconnected_primitive.hpp"
#include "nodes/executors/dnnl/dnnl_post_op_data.hpp"
#include "nodes/executors/dnnl/dnnl_shape_agnostic_data.hpp"
#include "nodes/executors/dnnl/dnnl_utils.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/graph_emitter.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "post_ops.hpp"
#include "shape_inference/custom/convolution.hpp"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

using namespace dnnl;
using namespace executor;

DnnlConvolutionPrimitive::IntermediateReorders::IntermediateReorders(const Key& key,
                                                                     const dnnl::primitive_desc& primDesc,
                                                                     const dnnl::engine& engine) {
    if (key.fcSemantic) {
        return;  // currently 'any' format is never used for src / dst memory for FullyConnected
    }

    enum class AllocateMemoryFor : uint8_t { Src, Dst };

    const auto& postOps = primDesc.get_primitive_attr().get_post_ops();

    const bool withSum = [&postOps] {
        for (int i = 0; i < postOps.len(); ++i) {
            if (postOps.kind(i) == dnnl::primitive::kind::sum) {
                return true;
            }
        }
        return false;
    }();

    auto createIfNotEqual = [](const dnnl::memory::desc& src,
                               const dnnl::memory::desc& dst,
                               AllocateMemoryFor allocate,
                               const dnnl::engine& engine) -> IntermediateReorder {
        auto reorderDesc = dnnl::reorder::primitive_desc(engine, src, engine, dst);
        auto reorder = dnnl::reorder(reorderDesc);
        const auto& memDescToAllocate = allocate == AllocateMemoryFor::Dst ? dst : src;

        return {reorder, memDescToAllocate};
    };

    if (key.src->getDnnlDesc() != primDesc.src_desc()) {
        m_inputReorders[DNNL_ARG_SRC] =
            createIfNotEqual(key.src->getDnnlDesc(), primDesc.src_desc(), AllocateMemoryFor::Dst, engine);
    }

    // In the case of fusing sum, we have to reorder the output data before executing the primitive,
    // since the output data are used as an accumulator for the covolution computations.
    if (withSum && key.dst->getDnnlDesc() != primDesc.dst_desc()) {
        m_inputReorders[DNNL_ARG_DST] =
            createIfNotEqual(key.dst->getDnnlDesc(), primDesc.dst_desc(), AllocateMemoryFor::Dst, engine);
    }

    if (key.nonConstantWeights && key.wei->getDnnlDesc() != primDesc.weights_desc()) {
        m_inputReorders[DNNL_ARG_WEIGHTS] =
            createIfNotEqual(key.wei->getDnnlDesc(), primDesc.weights_desc(), AllocateMemoryFor::Dst, engine);
    }

    if (key.dst->getDnnlDesc() != primDesc.dst_desc()) {
        m_outputReorders[DNNL_ARG_DST] =
            createIfNotEqual(primDesc.dst_desc(), key.dst->getDnnlDesc(), AllocateMemoryFor::Src, engine);
    }
}

bool DnnlConvolutionPrimitive::IntermediateReorders::empty() const {
    return m_inputReorders.empty() && m_outputReorders.empty();
}

static const std::map<memory::data_type, memory::data_type> weightsTypeByInputType{
    // input data type       weights data type
    {memory::data_type::f32, memory::data_type::f32},
    {memory::data_type::f16, memory::data_type::f16},
    {memory::data_type::bf16, memory::data_type::bf16},
    {memory::data_type::u8, memory::data_type::s8},
    {memory::data_type::s8, memory::data_type::s8},
};

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

    seed = get_vector_hash(seed, stride);
    seed = get_vector_hash(seed, dilation);
    seed = get_vector_hash(seed, paddingL);
    seed = get_vector_hash(seed, paddingR);

    seed = hash_combine(seed, get_attr_hash(*attr.get()));
    seed = hash_combine(seed, fcSemantic);
    seed = hash_combine(seed, nonConstantWeights);

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

    result = result && stride == rhs.stride;
    result = result && dilation == rhs.dilation;

    result = result && *attr.get() == *rhs.attr.get();
    result = result && fcSemantic == rhs.fcSemantic;
    result = result && nonConstantWeights == rhs.nonConstantWeights;

    return result;
}

// make a fake shape: N, C, W
template <typename T>
static std::vector<T> normalizeDims(const std::vector<T>& dims) {
    assert(any_of(static_cast<int>(dims.size()), 2, 3));

    if (dims.size() == 3) {
        return {dims[0], dims[2], dims[1]};
    }

    return {dnnl::memory::dim{1}, dims[1], dims[0]};
}

static dnnl::convolution_forward::primitive_desc createInnerProductDescriptor(const dnnl::memory::desc& inputDesc,
                                                                              const dnnl::memory::desc& weightDesc,
                                                                              const dnnl::memory::desc& biasDesc,
                                                                              const dnnl::memory::desc& outputDesc,
                                                                              const std::vector<size_t>& stride,
                                                                              const std::vector<size_t>& dilation,
                                                                              const std::vector<ptrdiff_t>& paddingL,
                                                                              const std::vector<ptrdiff_t>& paddingR,
                                                                              const dnnl::primitive_attr& attr,
                                                                              const dnnl::engine& engine) {
    const auto normalizedInDims = normalizeDims(inputDesc.get_dims());
    const auto convInDesc = dnnl::memory::desc(normalizedInDims, inputDesc.get_data_type(), memory::format_tag::nwc);
    const auto normalizedOutDims = normalizeDims(outputDesc.get_dims());
    const auto convOutDesc = dnnl::memory::desc(normalizedOutDims, outputDesc.get_data_type(), memory::format_tag::nwc);

    // make a fake shape: OC, IC, 1
    const auto& weightDims = weightDesc.get_dims();
    const dnnl::memory::dims normalizedWeightDims{static_cast<dnnl::memory::dim>(weightDims[0]),
                                                  static_cast<dnnl::memory::dim>(weightDims[1]),
                                                  dnnl::memory::dim{1}};
    const auto weightDataType = weightsTypeByInputType.at(inputDesc.get_data_type());
    const auto convWeightDescAny =
        dnnl::memory::desc(normalizedWeightDims, weightDataType, dnnl::memory::format_tag::any);

    // TODO: migrate on convolution_auto algorithm for x64
#if defined(OPENVINO_ARCH_X86_64)
    const dnnl::algorithm algorithm = dnnl::algorithm::convolution_direct;
#else
    const dnnl::algorithm algorithm = dnnl::algorithm::convolution_auto;
#endif

    return {engine,
            prop_kind::forward_inference,
            algorithm,
            convInDesc,
            convWeightDescAny,
            biasDesc,
            convOutDesc,
            dnnl::memory::dims(stride.begin(), stride.end()),
            dnnl::memory::dims(dilation.begin(), dilation.end()),
            dnnl::memory::dims(paddingL.begin(), paddingL.end()),
            dnnl::memory::dims(paddingR.begin(), paddingR.end()),
            attr};
}

static dnnl::convolution_forward::primitive_desc createConvolutionDescriptor(const dnnl::memory::desc& inputDesc,
                                                                             const dnnl::memory::desc& weightDesc,
                                                                             const dnnl::memory::desc& biasDesc,
                                                                             const dnnl::memory::desc& outputDesc,
                                                                             const std::vector<size_t>& stride,
                                                                             const std::vector<size_t>& dilation,
                                                                             const std::vector<ptrdiff_t>& paddingL,
                                                                             const std::vector<ptrdiff_t>& paddingR,
                                                                             const dnnl::primitive_attr& attr,
                                                                             const dnnl::engine& engine) {
    const auto weightDataType = weightsTypeByInputType.at(inputDesc.get_data_type());
    const auto weightDescAny = dnnl::memory::desc(weightDesc.get_dims(), weightDataType, dnnl::memory::format_tag::any);
    // TODO: migrate on convolution_auto algorithm for x64
#if defined(OPENVINO_ARCH_X86_64)
    const dnnl::algorithm algorithm = dnnl::algorithm::convolution_direct;
#else
    const dnnl::algorithm algorithm = dnnl::algorithm::convolution_auto;
#endif
    return {engine,
            prop_kind::forward_inference,
            algorithm,
            inputDesc,
            weightDescAny,
            biasDesc,
            outputDesc,
            dnnl::memory::dims(stride.begin(), stride.end()),
            dnnl::memory::dims(dilation.begin(), dilation.end()),
            dnnl::memory::dims(paddingL.begin(), paddingL.end()),
            dnnl::memory::dims(paddingR.begin(), paddingR.end()),
            attr,
            true};
}

static std::tuple<primitive_desc, size_t> selectPrimitiveDescWithMultipleAttributes(
    const dnnl::engine& engine,
    const dnnl::memory::desc& inputDesc,
    const dnnl::memory::desc& weightDesc,
    const dnnl::memory::desc& biasDesc,
    const dnnl::memory::desc& outputDesc,
    const std::vector<size_t>& stride,
    const std::vector<size_t>& dilation,
    const std::vector<ptrdiff_t>& paddingL,
    const std::vector<ptrdiff_t>& paddingR,
    const std::vector<dnnl::primitive_attr>& attrs,
    const std::vector<impl_desc_type>& implPriorities) {
    auto createPrimitiveDescriptor = [&](const dnnl::primitive_attr& attr) {
        return createConvolutionDescriptor(inputDesc,
                                           weightDesc,
                                           biasDesc,
                                           outputDesc,
                                           stride,
                                           dilation,
                                           paddingL,
                                           paddingR,
                                           attr,
                                           engine);
    };

    struct PrimitiveDescWithPriority {
        dnnl::primitive_desc prim_desc;
        size_t attrId = 0UL;
        size_t priority = 0UL;
    };

    PrimitiveDescWithPriority prim_desc_w_priority{dnnl::primitive_desc(), 0, implPriorities.size()};
    const bool first_match = implPriorities.front() == impl_desc_type::unknown;

    // try all the provided attributes and select the one which results in a primitive desc with the highest priority
    for (size_t attrId = 0; attrId < attrs.size(); attrId++) {
        const auto& attr = attrs[attrId];

        auto cur_desc = createPrimitiveDescriptor(attr);

        DnnlExtensionUtils::for_each_implementation(
            cur_desc,
            first_match,
            [&](impl_desc_type implType) {  // is acceptable implementation
                return contains(implPriorities, implType);
            },
            [&](dnnl::primitive_desc& desc) {  // is implementation with highest priority
                const impl_desc_type descImplType = parse_impl_name(desc.impl_info_str());
                const auto it = std::find(implPriorities.begin(), implPriorities.end(), descImplType);
                const size_t priorityId = std::distance(implPriorities.begin(), it);
                const size_t highestPriority = prim_desc_w_priority.priority;
                if (priorityId < highestPriority) {
                    auto desc_copy = dnnl::primitive_desc(DnnlExtensionUtils::clone_primitive_desc(desc.get(true)));
                    prim_desc_w_priority = {desc_copy, attrId, priorityId};
                }
            });
    }

    return {prim_desc_w_priority.prim_desc, prim_desc_w_priority.attrId};
}

static primitive_desc createPrimitiveDesc(const dnnl::memory::desc& inputDesc,
                                          const dnnl::memory::desc& weightDesc,
                                          const dnnl::memory::desc& biasDesc,
                                          const dnnl::memory::desc& outputDesc,
                                          const std::vector<size_t>& stride,
                                          const std::vector<size_t>& dilation,
                                          const std::vector<ptrdiff_t>& paddingL,
                                          const std::vector<ptrdiff_t>& paddingR,
                                          const dnnl::primitive_attr& attr,
                                          const dnnl::engine& engine,
                                          bool fcSemantic,
                                          const std::vector<impl_desc_type>& implPriorities,
                                          const impl_desc_type defaultImplType) {
    auto createPrimitiveDescriptor = [&](const dnnl::primitive_attr& attr) {
        return fcSemantic ? createInnerProductDescriptor(inputDesc,
                                                         weightDesc,
                                                         biasDesc,
                                                         outputDesc,
                                                         stride,
                                                         dilation,
                                                         paddingL,
                                                         paddingR,
                                                         attr,
                                                         engine)
                          : createConvolutionDescriptor(inputDesc,
                                                        weightDesc,
                                                        biasDesc,
                                                        outputDesc,
                                                        stride,
                                                        dilation,
                                                        paddingL,
                                                        paddingR,
                                                        attr,
                                                        engine);
    };

    auto createConvolutionDescriptorAny = [](const dnnl::memory::desc& inputDesc,
                                             const dnnl::memory::desc& weightDesc,
                                             const dnnl::memory::desc& biasDesc,
                                             const dnnl::memory::desc& outputDesc,
                                             const std::vector<size_t>& stride,
                                             const std::vector<size_t>& dilation,
                                             const std::vector<ptrdiff_t>& paddingL,
                                             const std::vector<ptrdiff_t>& paddingR,
                                             const dnnl::primitive_attr& attr,
                                             const dnnl::engine& engine) {
        auto inputDescAny =
            dnnl::memory::desc(inputDesc.get_dims(), inputDesc.get_data_type(), memory::format_tag::any);
        auto outputDescAny =
            dnnl::memory::desc(outputDesc.get_dims(), outputDesc.get_data_type(), memory::format_tag::any);
        return createConvolutionDescriptor(inputDescAny,
                                           weightDesc,
                                           biasDesc,
                                           outputDescAny,
                                           stride,
                                           dilation,
                                           paddingL,
                                           paddingR,
                                           attr,
                                           engine);
    };

    auto prim_desc = createPrimitiveDescriptor(attr);
    // keep first implementation descriptor to fallback to it if no other implementation is found
    auto first_desc = prim_desc;
    // if default implementation type is not specified, try to find the best implementation
    if (defaultImplType == impl_desc_type::undef) {
        if (!prim_desc) {
            // fallback to 'any' implementation
            return createConvolutionDescriptorAny(inputDesc,
                                                  weightDesc,
                                                  biasDesc,
                                                  outputDesc,
                                                  stride,
                                                  dilation,
                                                  paddingL,
                                                  paddingR,
                                                  attr,
                                                  engine);
            return std::move(prim_desc);
        }

        for (auto preferredImplType : implPriorities) {
            // the only way to fully reset primitive_desc after iterating over the implementations is to re-create it
            const bool found = DnnlExtensionUtils::find_implementation(prim_desc, preferredImplType);

            if (found) {
                return std::move(prim_desc);
            }

            prim_desc = createPrimitiveDescriptor(attr);
        }

        return std::move(first_desc);
    }
    // try to use a default implementations type (created using dummy shapes) if specified
    const bool found = DnnlExtensionUtils::find_implementation(prim_desc, defaultImplType);

    if (found) {
        return std::move(prim_desc);
    }

    if (fcSemantic) {  // fallback to the first implementation if used as FC executor
        return std::move(first_desc);
    }

    // fallback to 'any' implementation
    return createConvolutionDescriptorAny(inputDesc,
                                          weightDesc,
                                          biasDesc,
                                          outputDesc,
                                          stride,
                                          dilation,
                                          paddingL,
                                          paddingR,
                                          attr,
                                          engine);
}

static std::vector<DnnlPrimitiveAttrs> createPrimitiveAttrs(const ConvAttrs& attrs,
                                                            const MemoryArgs& memory,
                                                            const ExecutorContext::CPtr& context) {
    const auto& srcDesc = memory.at(ARG_SRC)->getDescPtr();
    const auto& weiDesc = memory.at(ARG_WEI)->getDescPtr();
    const auto& dstDesc = memory.at(ARG_DST)->getDescPtr();

    const auto& originalOutputDims = dstDesc->getShape().getMinDims();
    const auto& outputDims = attrs.fcSemantic ? normalizeDims(originalOutputDims) : originalOutputDims;

    auto isINT8 =
        any_of(srcDesc->getPrecision(), ov::element::u8, ov::element::i8) && weiDesc->getPrecision() == ov::element::i8;
    auto outputDataType = DnnlExtensionUtils::ElementTypeToDataType(dstDesc->getPrecision());

    const auto weightScaleMask = attrs.isGrouped ? 3 : 1 << 0;
    constexpr int channelDimIdx = 1;

    if (attrs.fcSemantic) {
        // use original post ops and zero points in case if used as FC executor
        return {DnnlPostOpsComposer(attrs.postOps,
                                    context->getEngine(),
                                    outputDims,
                                    channelDimIdx,
                                    isINT8,
                                    weightScaleMask,
                                    memory,
                                    outputDataType,
                                    attrs.dqScales,
                                    PostOpsMode::Original,
                                    false)
                    .compose()};
    }

    DnnlPostOpsComposer legacyPostOpsLegacyZeroPoints(attrs.postOps,
                                                      context->getEngine(),
                                                      outputDims,
                                                      channelDimIdx,
                                                      isINT8,
                                                      weightScaleMask,
                                                      memory,
                                                      outputDataType,
                                                      attrs.dqScales,
                                                      PostOpsMode::Legacy,
                                                      true);
    // first try to compose using legacy post ops
    auto legacyCompose = legacyPostOpsLegacyZeroPoints.compose();

    // check if legacy compose is enough
    auto attrContainsPostOp = [](const dnnl::primitive_attr& attr, const dnnl::impl::primitive_kind_t kind) -> bool {
        const auto ops = attr.get_post_ops();
        return ops.get()->find(kind) != -1;
    };

    // dw-conv would be fused into conv only on AVX2 platform.
    if (attrContainsPostOp(legacyCompose.attr, dnnl::impl::primitive_kind::convolution)) {
        DEBUG_LOG("Attribute contains conv post op. Use legacy post ops");
        return {legacyCompose};
    }

    if (attrs.inputZeroPointsType == ZeroPointsType::None &&
        !attrContainsPostOp(legacyCompose.attr, dnnl::impl::primitive_kind::depthwise) &&
        !attrContainsPostOp(legacyCompose.attr, dnnl::impl::primitive_kind::quantization)) {
        DEBUG_LOG("Attribute already contains no legacy post ops");
        return {legacyCompose};
    }

    if (attrs.inputZeroPointsType == ZeroPointsType::PerChannel) {
        DEBUG_LOG("Per channel zero point can only supported with legacy post ops");
        return {legacyCompose};
    }

    // @todo avoid extra step of creating config to get the brgconv availability
    auto config = createConfig(memory, attrs);
    if (!DnnlConvolutionPrimitive::isBrgConvAvailable(config)) {
        DEBUG_LOG("Brgconv is not available. Skip extra attribute");
        return {legacyCompose};
    }

    if (!dstDesc->hasLayoutType(LayoutType::nspc)) {
        DEBUG_LOG("Brgemm convolution supports only nspc layout. Use legacy post ops");
        return {legacyCompose};
    }

    std::vector<DnnlPrimitiveAttrs> attributeVariants{legacyCompose};

    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx) &&
        attrs.inputZeroPointsType == ZeroPointsType::PerTensor) {
        DnnlPostOpsComposer legacyPostOpsOriginalZeroPoints(attrs.postOps,
                                                            context->getEngine(),
                                                            outputDims,
                                                            channelDimIdx,
                                                            isINT8,
                                                            weightScaleMask,
                                                            memory,
                                                            outputDataType,
                                                            attrs.dqScales,
                                                            PostOpsMode::Legacy,
                                                            false);
        attributeVariants.emplace_back(legacyPostOpsOriginalZeroPoints.compose());

        return attributeVariants;
    }

    DnnlPostOpsComposer originalPostOpsOriginalZeroPoints(attrs.postOps,
                                                          context->getEngine(),
                                                          outputDims,
                                                          channelDimIdx,
                                                          isINT8,
                                                          weightScaleMask,
                                                          memory,
                                                          outputDataType,
                                                          attrs.dqScales,
                                                          PostOpsMode::Original,
                                                          false);
    attributeVariants.emplace_back(originalPostOpsOriginalZeroPoints.compose());

    return attributeVariants;
}

constexpr auto dilated(const int64_t dim, const int64_t dilation) {
    constexpr int64_t inf_bound = -1;  //!< Infinite bound value for dimension.
    return (dim < 1) ? inf_bound : (dilation + 1) * (dim - 1) + 1;
}

static std::pair<int64_t, int64_t> padding(const int64_t dim,
                                           const int64_t kernel_size,
                                           const int64_t dilation,
                                           const int64_t stride) {
    const auto dilated_kernel = dilated(kernel_size, dilation);
    const int64_t tmp = (dim + stride - 1) / stride;

    const auto padding = std::max<int64_t>(0, (tmp - 1) * stride + dilated_kernel - dim);
    const auto left_padding = padding / 2;

    return {left_padding, padding - left_padding};
}

static std::tuple<std::vector<ptrdiff_t>, std::vector<ptrdiff_t>> apply_auto_pad(const VectorDims& data_shape,
                                                                                 const VectorDims& weights_shape,
                                                                                 const std::vector<size_t>& strides,
                                                                                 const std::vector<size_t>& dilations,
                                                                                 AutoPaddingType type) {
    const auto num_spatial = strides.size();
    std::vector<ptrdiff_t> padB(num_spatial);
    std::vector<ptrdiff_t> padE(num_spatial);

    auto data_dim = data_shape.size() - num_spatial;
    auto kernel_dim = weights_shape.size() - num_spatial;

    const auto padding_swap = type == AutoPaddingType::SAME_UPPER;
    auto& pad_b = padding_swap ? padB : padE;
    auto& pad_e = padding_swap ? padE : padB;

    for (size_t i = 0; i < num_spatial; ++i, ++data_dim, ++kernel_dim) {
        std::tie(pad_b[i], pad_e[i]) =
            padding(data_shape[data_dim], weights_shape[kernel_dim], dilations[i], strides[i]);
    }

    return {padB, padE};
}

VectorDims static makeInputDummyShape(const Shape& inputShape,
                                      const Shape& weightShape,
                                      const std::vector<size_t>& strides,
                                      const std::vector<size_t>& dilation,
                                      const std::vector<ptrdiff_t>& paddingL,
                                      const std::vector<ptrdiff_t>& paddingR,
                                      bool isGrouped) {
    // There are a bunch of heuristics mostly aimed to guess the most appropriate oneDNN implementation, to reduce
    // thie amount of the implementation mismatch and the internal reordering as a consequence.
    constexpr Dim dummyInputDim = 64;

    const size_t spatialRank = strides.size();
    const auto& weightDims = weightShape.getStaticDims();
    const size_t filterStartIndx = weightDims.size() - spatialRank;

    VectorDims dummyInputShapeVals(inputShape.getRank(), dummyInputDim);

    const auto G = isGrouped ? weightDims[0] : 1;
    const auto IC = isGrouped ? weightDims[2] : weightDims[1];
    dummyInputShapeVals[1] = G * IC;

    for (size_t i = 0; i < spatialRank; i++) {
        if (weightDims[filterStartIndx + i] > dummyInputShapeVals[2 + i]) {
            constexpr Dim dummyOutputDim = 16;
            dummyInputShapeVals[2 + i] = (dummyOutputDim - 1) * strides[i] - (paddingL[i] + paddingR[i]) +
                                         weightDims[filterStartIndx + i] +
                                         (weightDims[filterStartIndx + i] - 1) * (dilation[i]);
        }
    }

    return MemoryDescUtils::makeDummyShape(inputShape, dummyInputShapeVals).getStaticDims();
}

static std::tuple<MemoryDescPtr, MemoryDescPtr> createDummySrcDstDescs(const ConvAttrs& attrs,
                                                                       const MemoryArgs& memory) {
    const auto& srcShape = memory.at(ARG_SRC)->getShape();
    const auto& weiShape = memory.at(ARG_WEI)->getShape();
    const auto& strides = attrs.stride;
    const auto& dilation = attrs.dilation;
    auto paddingL = attrs.paddingL;
    if (paddingL.size() < strides.size()) {
        paddingL.resize(strides.size(), 0);
    }

    auto paddingR = attrs.paddingR;
    if (paddingR.size() < strides.size()) {
        paddingR.resize(strides.size(), 0);
    }

    auto dummyInputDims =
        makeInputDummyShape(srcShape, weiShape, strides, dilation, paddingL, paddingR, attrs.isGrouped);

    auto ovDilation = dilation;
    std::transform(ovDilation.begin(), ovDilation.end(), ovDilation.begin(), [](size_t d) {
        return d + 1;
    });

    auto dummyOutputDims = node::convolution_shape_infer(dummyInputDims,
                                                         weiShape.getStaticDims(),
                                                         strides,
                                                         ovDilation,
                                                         paddingL,
                                                         paddingR,
                                                         attrs.autoPadding != AutoPaddingType::None,
                                                         attrs.isGrouped);

    auto srcDesc = memory.at(ARG_SRC)->getDescPtr();
    const auto& weiDesc = memory.at(ARG_WEI)->getDescPtr();
    auto dstDesc = memory.at(ARG_DST)->getDescPtr();

    srcDesc = srcDesc->cloneWithNewDims(dummyInputDims);
    dstDesc = dstDesc->cloneWithNewDims(dummyOutputDims);

    if (attrs.autoPadding != AutoPaddingType::None) {
        std::tie(paddingL, paddingR) = apply_auto_pad(srcDesc->getShape().getDims(),
                                                      weiDesc->getShape().getDims(),
                                                      attrs.stride,
                                                      attrs.dilation,
                                                      attrs.autoPadding);
    }

    return {srcDesc, dstDesc};
}

DnnlShapeAgnosticDataPtr DnnlConvolutionPrimitive::createShapeAgnosticData(const ConvAttrs& attrs,
                                                                           const MemoryArgs& memory,
                                                                           const ExecutorContext::CPtr& context,
                                                                           const bool cacheWeights) {
    // @todo we might want to enable weights prepacking for dynamic shapes as well
    const bool undefinedInputShapes = !memory.at(ARG_SRC)->isDefined();
    const bool cacheWeightsWithUndefData = undefinedInputShapes && cacheWeights;
    OPENVINO_ASSERT(!cacheWeightsWithUndefData,
                    "dnnl convolution weights caching for dynamic shapes is not implemented");

    auto [srcDesc, dstDesc] = undefinedInputShapes
                                  ? createDummySrcDstDescs(attrs, memory)
                                  : std::tuple{memory.at(ARG_SRC)->getDescPtr(), memory.at(ARG_DST)->getDescPtr()};

    const auto& weiDesc = memory.at(ARG_WEI)->getDescPtr();
    const auto& biasDesc = memory.at(ARG_BIAS)->getDescPtr();

    const bool autoPadding = attrs.autoPadding != AutoPaddingType::None;
    const bool calculatePadding = undefinedInputShapes && autoPadding;
    auto [paddingL, paddingR] = calculatePadding ? apply_auto_pad(srcDesc->getShape().getDims(),
                                                                  weiDesc->getShape().getDims(),
                                                                  attrs.stride,
                                                                  attrs.dilation,
                                                                  attrs.autoPadding)
                                                 : std::tuple(attrs.paddingL, attrs.paddingR);

    const dnnl::memory::desc srcDnnlDesc = MemoryDescUtils::convertToDnnlMemoryDesc(srcDesc)->getDnnlDesc();
    const dnnl::memory::desc weiDnnlDesc = MemoryDescUtils::convertToDnnlMemoryDesc(weiDesc)->getDnnlDesc();
    const dnnl::memory::desc dstDnnlDesc = MemoryDescUtils::convertToDnnlMemoryDesc(dstDesc)->getDnnlDesc();
    const dnnl::memory::desc biaDnnlDesc = MemoryDescUtils::convertToDnnlMemoryDesc(biasDesc)->getDnnlDesc();

    const auto primitiveAttributes = createPrimitiveAttrs(attrs, memory, context);

    std::vector<dnnl::primitive_attr> dnnlAttrVariants;
    dnnlAttrVariants.reserve(primitiveAttributes.size());

    for (const auto& postOp : primitiveAttributes) {
        dnnlAttrVariants.push_back(postOp.attr);
    }

    auto [primitiveDesc, attrId] = selectPrimitiveDescWithMultipleAttributes(context->getEngine(),
                                                                             srcDnnlDesc,
                                                                             weiDnnlDesc,
                                                                             biaDnnlDesc,
                                                                             dstDnnlDesc,
                                                                             attrs.stride,
                                                                             attrs.dilation,
                                                                             paddingL,
                                                                             paddingR,
                                                                             dnnlAttrVariants,
                                                                             context->getImplPriorities());

    // with all the current hacks it is not guaranteed that a primitive descriptor will be created
    if (cacheWeights && primitiveDesc) {
        auto originalWeightsDesc = MemoryDescUtils::convertToDnnlMemoryDesc(weiDesc);
        const auto weightsDesc = DnnlExtensionUtils::makeDescriptor(primitiveDesc.weights_desc());
        (void)utils::prepareWeightsMemory(originalWeightsDesc, weightsDesc, memory.at(ARG_WEI), context);
    }

    const auto defaultImpType = primitiveDesc ? parse_impl_name(primitiveDesc.impl_info_str()) : impl_desc_type::undef;

    return std::make_shared<DnnlShapeAgnosticData>(primitiveAttributes.at(attrId), defaultImpType);
}

DnnlShapeAgnosticDataPtr DnnlConvolutionPrimitive::createShapeAgnosticData(const FCAttrs& fcAttrs,
                                                                           const MemoryArgs& memory,
                                                                           const ExecutorContext::CPtr& context,
                                                                           const bool cacheWeights) {
    const bool cacheWeightsWithUndefData = !memory.at(ARG_SRC)->isDefined() && cacheWeights;
    OPENVINO_ASSERT(!cacheWeightsWithUndefData,
                    "dnnl convolution weights caching for dynamic shapes is not implemented");

    ConvAttrs attrs{{1},
                    {0},
                    {0},
                    {0},
                    AutoPaddingType::None,
                    fcAttrs.withBias,
                    fcAttrs.weightsNonTransposed,
                    false,
                    false,
                    true,
                    false,
                    {},
                    {},
                    fcAttrs.postOps};

    const auto postOpData = createPrimitiveAttrs(attrs, memory, context);
    OPENVINO_ASSERT(postOpData.size() == 1, "Single attribute variant is expected when used as FC executor");

    return std::make_shared<DnnlShapeAgnosticData>(postOpData.front());
}

void DnnlConvolutionPrimitive::execute(dnnl_primitive_args& primArgs) {
    if (m_intermediateReorders.empty()) {  // fast path
        m_prim.execute(m_stream, primArgs);
        return;
    }

    // keep original memory to restore it after the execution
    std::unordered_map<int, dnnl::memory> originalMemory;
    // execute intermediate src reorders
    for (auto& [id, reorder] : m_intermediateReorders.m_inputReorders) {
        auto& [primitive, dstMemoryDesc] = reorder;
        if (auto primArg = primArgs.find(id); primArg != primArgs.end()) {
            auto& [id, srcMemory] = *primArg;
            dnnl::memory dstMemory(dstMemoryDesc, m_stream.get_engine());
            primitive.execute(m_stream, srcMemory, dstMemory);
            originalMemory[id] = primArgs[id];
            primArgs[id] = dstMemory;
        }
    }
    // prepare intermediate dst memory
    auto& outputReorders = m_intermediateReorders.m_outputReorders;
    if (auto outputReorder = outputReorders.find(DNNL_ARG_DST); outputReorder != outputReorders.end()) {
        auto& [id, reorder] = *outputReorder;
        auto& [primitive, srcMemoryDesc] = reorder;
        dnnl::memory srcMemory(srcMemoryDesc, m_stream.get_engine());
        originalMemory[id] = primArgs[id];
        primArgs[id] = srcMemory;
    }

    m_prim.execute(m_stream, primArgs);
    // execute intermediate dst reorders
    if (const auto& outputReorder = outputReorders.find(DNNL_ARG_DST); outputReorder != outputReorders.end()) {
        const auto& [id, reorder] = *outputReorder;
        const auto& primitive = reorder.m_reorder;
        primitive.execute(m_stream, primArgs[id], originalMemory[id]);
    }
    // restore original memory
    for (const auto& [id, mem] : originalMemory) {
        primArgs[id] = mem;
    }
}

std::shared_ptr<DnnlConvolutionPrimitive> DnnlConvolutionPrimitive::create(
    const MemoryArgs& memory,
    const ConvAttrs& attrs,
    const ExecutorContext::CPtr& context,
    const DnnlShapeAgnosticDataPtr& shapeAgnosticData) {
    const auto& srcDesc = MemoryDescUtils::convertToDnnlMemoryDesc(memory.at(ARG_SRC)->getDescPtr());
    const auto& weiDesc = MemoryDescUtils::convertToDnnlMemoryDesc(memory.at(ARG_WEI)->getDescPtr());
    const auto& biaDesc = MemoryDescUtils::convertToDnnlMemoryDesc(memory.at(ARG_BIAS)->getDescPtr());
    const auto& dstDesc = MemoryDescUtils::convertToDnnlMemoryDesc(memory.at(ARG_DST)->getDescPtr());

    auto getPaddings = [&attrs](const VectorDims& dataShape, const VectorDims& weightsShape) {
        const bool fusedDWconv = std::any_of(attrs.postOps.begin(), attrs.postOps.end(), [](const auto& p) {
            return p.type() == typeid(DepthwiseConvolutionPostOp);
        });

        if (attrs.autoPadding == AutoPaddingType::None ||  // auto padding disabled
            fusedDWconv) {  // auto padding enabled, but paddingR is calculated manually for fused convolution
            return std::make_tuple(attrs.paddingL, attrs.paddingR);
        }

        return apply_auto_pad(dataShape, weightsShape, attrs.stride, attrs.dilation, attrs.autoPadding);
    };

    auto [paddingL, paddingR] = getPaddings(srcDesc->getShape().getDims(), weiDesc->getShape().getDims());

    const Key dnnlConvKey{srcDesc,
                          weiDesc,
                          biaDesc,
                          dstDesc,
                          attrs.stride,
                          attrs.dilation,
                          paddingL,
                          paddingR,
                          shapeAgnosticData->m_primAttrs.attr,
                          attrs.fcSemantic,
                          attrs.nonConstantWeights};

    const auto defaultImplType = shapeAgnosticData->m_implType;

    auto builder = [&context, defaultImplType](const Key& dnnlKey) {
        return std::make_shared<DnnlConvolutionPrimitive>(dnnlKey,
                                                          context->getEngine(),
                                                          context->getImplPriorities(),
                                                          defaultImplType);
    };

    auto runtimeCache = context->getRuntimeCache();
    const auto result = runtimeCache->getOrCreate(dnnlConvKey, builder);
    const auto& primitive = result.first;
    assert(primitive);

    return primitive;
}

DnnlMemoryDescPtr DnnlConvolutionPrimitive::makeTransposedWeightDescriptor(const DnnlMemoryDescPtr& srcDesc,
                                                                           const DnnlMemoryDescPtr& dstDesc,
                                                                           const ConvAttrs& attrs) {
    FCAttrs fcAttrs{};
    fcAttrs.withBias = attrs.withBias;
    fcAttrs.weightsNonTransposed = attrs.weightsNonTransposed;

    return DnnlFCPrimitive::makeTransposedWeightDescriptor(srcDesc, dstDesc, fcAttrs);
}

DnnlMemoryDescPtr DnnlConvolutionPrimitive::makeTransposedWeightDescriptor(const DnnlMemoryDescPtr& srcDesc,
                                                                           const DnnlMemoryDescPtr& dstDesc,
                                                                           const FCAttrs& attrs) {
    return DnnlFCPrimitive::makeTransposedWeightDescriptor(srcDesc, dstDesc, attrs);
}

std::tuple<size_t, size_t, size_t, size_t> DnnlConvolutionPrimitive::getChannelParams(const ConvConfig& config) {
    const auto& attrs = config.attrs;
    const auto& weightDesc = config.descs.at(ARG_WEI);
    const auto& weightDims = weightDesc->getShape().getStaticDims();

    const auto groupNum = attrs.isGrouped ? weightDims[0] : 1;
    const auto groupIC = attrs.isGrouped ? weightDims[2] : weightDims[1];
    const auto IC = attrs.isGrouped ? groupNum * groupIC : groupIC;
    const auto groupOC = attrs.isGrouped ? weightDims[1] : weightDims[0];

    return std::make_tuple(groupNum, groupIC, IC, groupOC);
}

bool DnnlConvolutionPrimitive::isJitPlanarAvailable(const ConvConfig& config) {
    // Only apply this heuristic logic on FP32 IR. IC=1, OC=1 would disable brgconv on avx2.
    const bool isAvx2FP32 = !dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) &&
                            dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2) && !config.attrs.isGraphQuantized;

    const auto [groupNum, groupIC, IC, groupOC] = getChannelParams(config);

    return all_of(1U, IC, groupOC * groupNum) && isAvx2FP32;
}

bool DnnlConvolutionPrimitive::isBrgConvAvailable(const ConvConfig& config) {
    // When avx2 brgconv heuristic case,  disable brgconv to WA the regression.
    const bool isBrgConvAvailable =
        dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2) && !isJitPlanarAvailable(config);

    return isBrgConvAvailable;
}

bool DnnlConvolutionPrimitive::isNspcAvailable(const ConvConfig& config) {
    using impl::cpu::x64::mayiuse;

    // do not use in non-quantized networks until it is enforced externally
    if (!config.attrs.isGraphQuantized) {
        return false;
        // @todo master implementation had the following logic as well:
        //     auto predicate = [](memory::format_tag tag) {
        //         return any_of(tag, memory::format_tag::nwc, memory::format_tag::nhwc, memory::format_tag::ndhwc);
        //     };
        //     if (std::none_of(inputMemoryFormatsFilter.begin(), inputMemoryFormatsFilter.end(), predicate)) {
        //         return false;
        //     }
        // }
    }
    // AVX2 heuristic
    if (isJitPlanarAvailable(config)) {
        return false;
    }

    // A bunch of heuristics are designed to cut off not optimal nspc convolution applications
    auto inpDims = config.descs.at(ARG_SRC)->getShape().getDims();
    auto outDims = config.descs.at(ARG_DST)->getShape().getDims();
    auto ndims = inpDims.size();

    const auto [groupNum, groupIC, IC, groupOC] = getChannelParams(config);

    bool isDepthWise = config.attrs.isGrouped && 1 == groupOC && 1 == groupIC;

    if (isDepthWise) {
        // 1d equivalent cases are painfully slow
        return inpDims.size() != 3 && 1 != inpDims[inpDims.size() - 2];
    }

    // it was empirically observed that the nspc convolutions perform much slower than the blocked ones if the
    // channels number more than the specific value
    size_t spatialRank = ndims - 2;  // two means batch dim plus channels dim

    bool is1x1 = false;

    if (!config.attrs.isGrouped) {
        const auto& weightDims = config.descs.at(ARG_WEI)->getShape().getDims();
        auto weightDimsReversItr = weightDims.crbegin();
        auto strideReversItr = config.attrs.stride.crbegin();
        auto paddingLreversItr = config.attrs.paddingL.crbegin();
        auto paddingRreversItr = config.attrs.paddingR.crbegin();

        for (size_t i = 0; i < spatialRank; ++i) {
            is1x1 = *(weightDimsReversItr++) == 1 && *(strideReversItr++) == 1 && *(paddingLreversItr++) == 0 &&
                    *(paddingRreversItr++) == 0;
        }
    }

    // if the activation field size is 1x1 the avx512 1x1 nspc convolution pollutes caches so that the layer after
    // the convolution performs slow
    if (mayiuse(impl::cpu::x64::avx512_core) && is1x1) {
        auto end = inpDims.rbegin();
        std::advance(end, spatialRank);
        if (std::all_of(inpDims.rbegin(), end, [](size_t x) {
                return dimsEqualStrong(1, x);
            })) {
            return false;
        }
    }

    unsigned thresholdNumChannels = 128U;  // for avx and below
    if (is1x1) {
        thresholdNumChannels = 2048U;
    } else if (mayiuse(impl::cpu::x64::avx512_core)) {
        thresholdNumChannels = 512U;
    }

    size_t OC = outDims[1];
    if (std::max(IC, OC) >= thresholdNumChannels) {
        return false;
    }

    if (!mayiuse(impl::cpu::x64::avx)) {
        // SSE41 nspc convolutions do not support ic and oc tails yet
        // the blocked implementation is faster than gemm
        if ((IC % 8) || (OC % 8)) {
            return false;
        }
    }

    return true;
}

DnnlConvolutionPrimitive::DnnlConvolutionPrimitive(const Key& key,
                                                   const dnnl::engine& engine,
                                                   const std::vector<impl_desc_type>& implPriorities,
                                                   const impl_desc_type defaultImplType)
    : m_stream(dnnl::stream(engine)),
      m_primDesc(createPrimitiveDesc(key.src->getDnnlDesc(),
                                     key.wei->getDnnlDesc(),
                                     key.bias->getDnnlDesc(),
                                     key.dst->getDnnlDesc(),
                                     key.stride,
                                     key.dilation,
                                     key.paddingL,
                                     key.paddingR,
                                     {key.attr},
                                     engine,
                                     key.fcSemantic,
                                     implPriorities,
                                     defaultImplType)),
      m_implType(parse_impl_name(m_primDesc.impl_info_str())),
      m_srcDesc(DnnlExtensionUtils::makeDescriptor(m_primDesc.src_desc())),
      m_weiDesc(DnnlExtensionUtils::makeDescriptor(m_primDesc.weights_desc())),
      m_dstDesc(DnnlExtensionUtils::makeDescriptor(m_primDesc.dst_desc())),
      m_scratchPadDesc(DnnlExtensionUtils::makeDescriptor(m_primDesc.scratchpad_desc())),
      m_prim(primitive(m_primDesc)),
      m_intermediateReorders(key, m_primDesc, engine) {}

}  // namespace ov::intel_cpu
