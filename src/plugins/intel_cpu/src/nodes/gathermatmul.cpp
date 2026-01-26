// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gathermatmul.h"

#include <oneapi/dnnl/dnnl_common_types.h>
#include <oneapi/dnnl/dnnl_types.h>

#include <bitset>
#include <common/primitive_hashing_utils.hpp>
#include <common/utils.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/blocked_desc_creator.h"
#include "config.h"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu_memory.h"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "graph_context.h"
#include "memory_desc/blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "node.h"
#include "node_config.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "shape_inference/custom/gathermatmul.hpp"
#include "transformations/cpu_opset/common/op/batch_gather_matmul.hpp"
#include "transformations/cpu_opset/common/op/batch_gather_matmul_compressed.hpp"
#include "transformations/utils/utils.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {

struct onednn_matmul_key {
    dnnl::memory::desc src_md;
    dnnl::memory::desc weights_md;
    VectorDims scale_shape;
    VectorDims zp_shape;
    bool has_bias;

    [[nodiscard]] size_t hash() const {
        using namespace dnnl::impl;
        using namespace dnnl::impl::primitive_hashing;

        size_t seed = 0;
        seed = hash_combine(seed, get_md_hash(*src_md.get()));
        seed = hash_combine(seed, get_md_hash(*weights_md.get()));
        seed = get_vector_hash(seed, scale_shape);
        seed = get_vector_hash(seed, zp_shape);
        seed = hash_combine(seed, has_bias);
        return seed;
    }

    bool operator==(const onednn_matmul_key& rhs) const {
        return src_md == rhs.src_md && weights_md == rhs.weights_md && scale_shape == rhs.scale_shape &&
               zp_shape == rhs.zp_shape && has_bias == rhs.has_bias;
    }
};

class GatherMatmul::onednn_matmul {
public:
    onednn_matmul() = delete;
    onednn_matmul(const onednn_matmul&) = delete;
    onednn_matmul(onednn_matmul&&) = delete;
    onednn_matmul& operator=(const onednn_matmul&) = delete;
    onednn_matmul& operator=(onednn_matmul&&) = delete;

    onednn_matmul(const dnnl::engine& eng, const onednn_matmul_key& key) : m_has_bias(key.has_bias) {
        const auto& src_md = key.src_md;
        const auto& weights_md = key.weights_md;
        auto scale_shape = key.scale_shape;
        auto zp_shape = key.zp_shape;

        const auto K = weights_md.get_dims()[1];
        const auto N = weights_md.get_dims()[0];
        const auto M = src_md.get_dims()[0];

        if (!scale_shape.empty()) {
            if (all_of(1U, scale_shape.size(), scale_shape[0])) {
                scale_shape.push_back(1);
            }
            OPENVINO_ASSERT(scale_shape.size() == 2, "Unsupported scale shape ", vec2str(scale_shape));
            const auto K_groups = scale_shape.back();
            OPENVINO_ASSERT((K % K_groups) == 0, "Incompatible number of groups ", K_groups, " for K ", K);
            init_w_scales(scale_shape);
            if (!zp_shape.empty()) {
                if (all_of(1U, zp_shape.size(), zp_shape[0])) {
                    zp_shape.push_back(1);
                }
                OPENVINO_ASSERT(zp_shape.size() == 2, "Unsupported zero points shape ", vec2str(zp_shape));
                init_w_zp(zp_shape);
            }
        }

        m_input_md = src_md;
        m_output_md =
            dnnl::memory::desc(dnnl::memory::dims({M, N}), src_md.get_data_type(), dnnl::memory::format_tag::ab);

        dnnl::memory::desc bias_md;
        if (m_has_bias) {
            bias_md =
                dnnl::memory::desc(dnnl::memory::dims({N}), dnnl::memory::data_type::f32, dnnl::memory::format_tag::a);
        }

        auto inner_product_primitive_desc =
            dnnl::inner_product_forward::primitive_desc(eng,
                                                        dnnl::prop_kind::forward_inference,
                                                        m_input_md,
                                                        weights_md,
                                                        bias_md,
                                                        m_output_md,
                                                        attr);

        m_impl_type = parse_impl_name(inner_product_primitive_desc.impl_info_str());

        m_wei_md = inner_product_primitive_desc.weights_desc();
        m_prim = dnnl::inner_product_forward(inner_product_primitive_desc);

        dnnl::memory inp_memory(m_input_md, eng, DNNL_MEMORY_NONE);
        dnnl::memory out_memory(m_output_md, eng, DNNL_MEMORY_NONE);
        dnnl::memory wei_memory(m_wei_md, eng, DNNL_MEMORY_NONE);
        dnnl::memory bias_memory;
        if (m_has_bias) {
            bias_memory = dnnl::memory(bias_md, eng, DNNL_MEMORY_NONE);
        }
        dnnl::memory scale_memory;
        if (!scale_shape.empty()) {
            scale_memory = dnnl::memory(m_scale_md, eng, DNNL_MEMORY_NONE);
        }
        dnnl::memory zp_memory;
        if (!zp_shape.empty()) {
            zp_memory = dnnl::memory(m_zp_md, eng, DNNL_MEMORY_NONE);
        }
        args = make_args(inp_memory, out_memory, wei_memory, bias_memory, scale_memory, zp_memory);
    }

    void exec(const dnnl::stream& astream,
              void* src,
              void* dst,
              void* weight,
              void* bias = nullptr,
              void* scale = nullptr,
              void* zp = nullptr) {
        args[DNNL_ARG_SRC].set_data_handle(src);
        args[DNNL_ARG_DST].set_data_handle(dst);
        args[DNNL_ARG_WEIGHTS].set_data_handle(weight);
        if (bias) {
            args[DNNL_ARG_BIAS].set_data_handle(bias);
        }
        if (scale) {
            args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS].set_data_handle(scale);
        }
        if (zp) {
            args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS].set_data_handle(zp);
        }
        m_prim.execute(astream, args);
    }

    [[nodiscard]] dnnl::memory::desc get_weights_md() const {
        return m_wei_md;
    }

    [[nodiscard]] dnnl::memory::desc get_scale_md() const {
        return m_scale_md;
    }

    [[nodiscard]] dnnl::memory::desc get_zp_md() const {
        return m_zp_md;
    }

    [[nodiscard]] ov::intel_cpu::impl_desc_type get_impl_type() const {
        return m_impl_type;
    }

private:
    void init_w_scales(const VectorDims& scale_shape) {
        constexpr auto data_type = dnnl::memory::data_type::f32;
        const auto scale_dims = DnnlExtensionUtils::convertToDnnlDims(scale_shape);
        attr.set_scales_dims(DNNL_ARG_WEIGHTS, scale_dims, data_type);
        m_scale_md = dnnl::memory::desc(scale_dims, data_type, dnnl::memory::format_tag::ba);
    }
    void init_w_zp(const VectorDims& zp_shape) {
        constexpr auto data_type = dnnl::memory::data_type::f32;
        const auto zp_dims = DnnlExtensionUtils::convertToDnnlDims(zp_shape);
        attr.set_zero_points_dims(DNNL_ARG_WEIGHTS, zp_dims, data_type);
        m_zp_md = dnnl::memory::desc(zp_dims, data_type, dnnl::memory::format_tag::ba);
    }
    static std::unordered_map<int, dnnl::memory> make_args(dnnl::memory& src,
                                                           dnnl::memory& dst,
                                                           dnnl::memory& weight,
                                                           dnnl::memory& bias,
                                                           dnnl::memory& scale,
                                                           dnnl::memory& zp) {
        std::unordered_map<int, dnnl::memory> args;
        args.insert({DNNL_ARG_SRC, src});
        args.insert({DNNL_ARG_WEIGHTS, weight});
        args.insert({DNNL_ARG_DST, dst});
        if (bias) {
            args.insert({DNNL_ARG_BIAS, bias});
        }
        if (scale) {
            args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scale});
        }
        if (zp) {
            args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, zp});
        }
        return args;
    }

    dnnl::primitive m_prim;
    dnnl::memory::desc m_input_md;
    dnnl::memory::desc m_output_md;
    dnnl::memory::desc m_wei_md;
    dnnl::memory::desc m_scale_md;
    dnnl::memory::desc m_zp_md;
    dnnl::primitive_attr attr;

    ov::intel_cpu::impl_desc_type m_impl_type = ov::intel_cpu::impl_desc_type::unknown;

    std::unordered_map<int, dnnl::memory> args;

    bool m_has_bias = false;
};

bool GatherMatmul::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        // Check if the operation is BatchGatherMatmul or BatchGatherMatmulCompressed
        const bool isBatchGatherMatmul = ov::is_type<ov::intel_cpu::BatchGatherMatmul>(op);
        const bool isBatchGatherMatmulCompressed = ov::is_type<ov::intel_cpu::BatchGatherMatmulCompressed>(op);

        if (!isBatchGatherMatmul && !isBatchGatherMatmulCompressed) {
            errorMessage = "Only BatchGatherMatmul and BatchGatherMatmulCompressed operations are supported. Got: " +
                           std::string(op->get_type_info().name);
            return false;
        }

        // Check that weights input (port 1) is constant
        if (!ov::op::util::is_on_path<ov::op::v0::Constant>(op->input_value(WEIGHTS))) {
            errorMessage = "Only constant weights are supported for GatherMatmul operation";
            return false;
        }

        // For compressed variant, check that scales and zero points are constant
        if (isBatchGatherMatmulCompressed) {
            if (op->get_input_size() > WEIGHT_SCALES) {
                if (!ov::op::util::is_on_path<ov::op::v0::Constant>(op->input_value(WEIGHT_SCALES))) {
                    errorMessage = "Only constant weight scales are supported for GatherMatmul operation";
                    return false;
                }
            }

            if (op->get_input_size() > WEIGHT_ZERO_POINTS) {
                if (!ov::op::util::is_on_path<ov::op::v0::Constant>(op->input_value(WEIGHT_ZERO_POINTS))) {
                    errorMessage = "Only constant weight zero points are supported for GatherMatmul operation";
                    return false;
                }
            }
        }

        // Check that bias (if present) is constant
        if (op->get_input_size() > BIAS) {
            const auto& biasInput = op->input_value(BIAS);
            // Skip validation if bias is dynamic (empty constant)
            if (biasInput.get_element_type() != ov::element::dynamic) {
                if (!ov::op::util::is_on_path<ov::op::v0::Constant>(biasInput)) {
                    errorMessage = "Only constant bias is supported for GatherMatmul operation";
                    return false;
                }
            }
        }

    } catch (...) {
        return false;
    }

    return true;
}

bool GatherMatmul::isSupportedCompressedOperation([[maybe_unused]] const std::shared_ptr<ov::Node>& op,
                                                  [[maybe_unused]] size_t IC,
                                                  [[maybe_unused]] size_t OC,
                                                  [[maybe_unused]] size_t G,
                                                  [[maybe_unused]] const Config& config) noexcept {
#ifdef OPENVINO_ARCH_X86_64
    // copy paste from FullyConnected
    try {
        std::string errorMessage;
        if (!isSupportedOperation(op, errorMessage)) {
            return false;
        }

        if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
            return false;
        }

        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx) &&
            config.inferencePrecision == ov::element::bf16) {
            // OneDNN AMX IP implementation has limited shapes support due to performance considerations. As a
            // current solution conditions below are copied from OneDNN to make sure correct IP impl will be
            // used since fallback one doesn't support weights decompression feature.
            constexpr size_t simdWidth = 16;
            constexpr size_t vnniFactor = 2;
            constexpr size_t maxSize = 512;
            constexpr size_t amxRow = vnniFactor * simdWidth;

            if ((IC <= amxRow && OC <= amxRow) || (IC <= maxSize && OC <= maxSize && IC % amxRow != 0)) {
                return false;
            }
        }

        if (IC % G != 0) {
            return false;  // sanity check IC must be evenly divided by the group size
        }

        if (IC / G < 4) {
            return false;  // minimal group size should be 4
        }

        if (OC == 1) {
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
#else
    return false;
#endif
}

ov::element::TypeVector GatherMatmul::getSupportedCompressedWeightsTypes([[maybe_unused]] bool apply_fp8) {
    using ov::element::Type_t;

#ifdef OPENVINO_ARCH_X86_64
    return {Type_t::u8, Type_t::i8, Type_t::u4, Type_t::i4};
#else
    return {};
#endif
}

ov::element::TypeVector GatherMatmul::getSupportedCompressedActivationsTypes() {
    using ov::element::Type_t;
    // @todo enable for bf16 as well
    // after EnforceInferencePrecision is replaced with ConvertPrecision
    return {Type_t::f32};
}

GatherMatmul::GatherMatmul(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, GatherMatmulShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    // Determine the algorithm type
    if (ov::is_type<ov::intel_cpu::BatchGatherMatmulCompressed>(op)) {
        algorithm = Algorithm::GatherMatmulCompressed;
    } else {
        algorithm = Algorithm::GatherMatmulDefault;
    }
}

void GatherMatmul::initSupportedPrimitiveDescriptors() {
    const auto& srcTypes = getOriginalInputPrecisions();
    auto dstTypes = getOriginalOutputPrecisions();

    if (!fusedWith.empty()) {
        dstTypes = fusedWith.back()->getOriginalOutputPrecisions();
    }

    NodeConfig nodeConfig;

    if (srcTypes.front() == ov::element::bf16 && dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx)) {
        // enable bf16 amx optimizations
        bf16_amx_mode = true;
    }

    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    for (size_t i = 0; i < srcTypes.size(); i++) {
        if (srcTypes[i] == element::dynamic) {
            nodeConfig.inConfs.emplace_back(MemoryDescUtils::makeEmptyDesc());
            continue;
        }
        const auto srcDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(srcTypes[i], getInputShapeAtPort(i));
        nodeConfig.inConfs.emplace_back(srcDesc);
    }

    for (size_t i = 0; i < dstTypes.size(); i++) {
        const auto dstDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(dstTypes[i], getOutputShapeAtPort(i));
        nodeConfig.outConfs.emplace_back(dstDesc);
    }

    supportedPrimitiveDescriptors.emplace_back(nodeConfig, impl_desc_type::undef);
}

void GatherMatmul::createPrimitive() {
    // we use gemv here and the shapes of this op in fact are determined as the weigths are const
    auto weightsMemoryDesc = getBaseMemDescAtInputPort(WEIGHTS);
    CPU_NODE_ASSERT(weightsMemoryDesc->isDefined(), "Weights memory descriptor is not defined");

    auto srcDesc = getBaseMemDescAtInputPort(DATA);
    auto biasDesc = getBaseMemDescAtInputPort(BIAS);
    auto dstDesc = getBaseMemDescAtOutputPort(0);

    // Now we construct memory descriptors for input as [1, K]
    //  weights memory descriptor as [N, K] since the weights are expected to be transposed

    auto src_precision = srcDesc->getPrecision();
    auto weights_precision = weightsMemoryDesc->getPrecision();

    // onednn doesn't not support bf16+f16
    if (ov::element::bf16 == src_precision && any_of(weights_precision, ov::element::f16, ov::element::f32)) {
        weights_precision = ov::element::bf16;
    }

    const auto& weiDims = weightsMemoryDesc->getShape().getStaticDims();

    int N = weiDims[weiDims.size() - 2];
    int K = weiDims[weiDims.size() - 1];

    VectorDims scale_shape{};
    VectorDims zp_shape{};
    if (algorithm == Algorithm::GatherMatmulCompressed) {
        auto scaleDesc = getBaseMemDescAtInputPort(WEIGHT_SCALES);
        if (scaleDesc && !scaleDesc->empty()) {
            const auto& fullScalesShape = scaleDesc->getShape().getStaticDims();
            if (1 == fullScalesShape.size()) {
                CPU_NODE_ASSERT(fullScalesShape[0] == 1, "Expect broadcastable scales shape.");
                scale_shape.push_back(fullScalesShape[0]);
            } else {
                scale_shape.assign(fullScalesShape.begin() + 1, fullScalesShape.end());
            }
        }
        auto zpDesc = getBaseMemDescAtInputPort(WEIGHT_ZERO_POINTS);
        if (zpDesc && !zpDesc->empty()) {
            const auto& fullZeroPointsShape = zpDesc->getShape().getStaticDims();
            if (1 == fullZeroPointsShape.size()) {
                CPU_NODE_ASSERT(fullZeroPointsShape[0] == 1, "Expect broadcastable zero points shape.");
                zp_shape.push_back(fullZeroPointsShape[0]);
            } else {
                zp_shape.assign(fullZeroPointsShape.begin() + 1, fullZeroPointsShape.end());
            }
        }
    }

    dnnl::memory::desc src_md({1, K},
                              DnnlExtensionUtils::ElementTypeToDataType(src_precision),
                              dnnl::memory::format_tag::ab);

    dnnl::memory::desc weights_md({N, K},
                                  DnnlExtensionUtils::ElementTypeToDataType(weights_precision),
                                  dnnl::memory::format_tag::any);

    onednn_matmul_key key{src_md, weights_md, scale_shape, zp_shape, biasDesc && !biasDesc->empty()};

    auto cache = context->getParamsCache();
    const auto& eng = getEngine();
    std::tie(gemv_impl, std::ignore) = cache->getOrCreate(key, [&eng](const onednn_matmul_key& k) {
        return std::make_shared<onednn_matmul>(eng, k);
    });

    // repack weights
    // we build gemv impl, but in fact there is B weights to gather, so we have to process 3D weights, scales and zp
    // tensors
    auto gemvWeightsDesc =
        MemoryDescUtils::convertToBlockedMemoryDesc(DnnlExtensionUtils::makeDescriptor(gemv_impl->get_weights_md()));

    // now let's build 3d descriptors
    auto addBatchDim = [](const BlockedMemoryDescPtr& desc, size_t batchDim) -> DnnlMemoryDescPtr {
        const auto& weightsDims = desc->getShape().getStaticDims();
        const auto& weightsBlockDims = desc->getBlockDims();
        const auto& weightsOrder = desc->getOrder();
        // at this point we assume that the tensors are dense and have no padded dims
        VectorDims newDims = {batchDim};
        newDims.insert(newDims.end(), weightsDims.begin(), weightsDims.end());
        VectorDims newBlockDims = {batchDim};
        newBlockDims.insert(newBlockDims.end(), weightsBlockDims.begin(), weightsBlockDims.end());
        VectorDims newOrder(weightsOrder.size() + 1);
        newOrder[0] = 0;
        for (size_t i = 0; i < weightsOrder.size(); i++) {
            newOrder[i + 1] = weightsOrder[i] + 1;
        }
        auto targetDesc =
            std::make_shared<CpuBlockedMemoryDesc>(desc->getPrecision(), Shape(newDims), newBlockDims, newOrder);
        return MemoryDescUtils::convertToDnnlMemoryDesc(targetDesc);
    };

    auto targetWeightsDesc = addBatchDim(gemvWeightsDesc, weiDims[0]);

    m_weightsMemory =
        prepareWeightMemory(targetWeightsDesc, MemoryDescUtils::convertToDnnlMemoryDesc(weightsMemoryDesc));

    if (!scale_shape.empty()) {
        auto expectedScaleMemDesc =
            MemoryDescUtils::convertToDnnlMemoryDesc(DnnlExtensionUtils::makeDescriptor(gemv_impl->get_scale_md()));
        auto scales = getSrcMemoryAtPort(WEIGHT_SCALES);
        CPU_NODE_ASSERT(scales && scales->isDefined(), "Weight scales memory is not defined");
        const auto& scDims = scales->getShape().getStaticDims();
        expectedScaleMemDesc =
            addBatchDim(MemoryDescUtils::convertToBlockedMemoryDesc(expectedScaleMemDesc), scDims[0]);
        if (expectedScaleMemDesc->isCompatible(scales->getDesc())) {
            m_scalesMemory = scales;
        } else {
            m_scalesMemory = std::make_shared<Memory>(getEngine(), expectedScaleMemDesc);
            m_scalesMemory->load(*scales, false, false);
        }
    }

    if (!zp_shape.empty()) {
        auto expectedZpMemDesc =
            MemoryDescUtils::convertToDnnlMemoryDesc(DnnlExtensionUtils::makeDescriptor(gemv_impl->get_zp_md()));
        auto zps = getSrcMemoryAtPort(WEIGHT_ZERO_POINTS);
        CPU_NODE_ASSERT(zps && zps->isDefined(), "Weight zero points memory is not defined");
        const auto& zpDims = zps->getShape().getStaticDims();
        expectedZpMemDesc = addBatchDim(MemoryDescUtils::convertToBlockedMemoryDesc(expectedZpMemDesc), zpDims[0]);
        if (expectedZpMemDesc->isCompatible(zps->getDesc())) {
            m_zpMemory = zps;
        } else {
            m_zpMemory = std::make_shared<Memory>(getEngine(), expectedZpMemDesc);
            m_zpMemory->load(*zps, false, false);
        }
    }

    Node::createPrimitive();

    // set the actual implementation type
    getSelectedPrimitiveDescriptor()->setImplementationType(gemv_impl->get_impl_type());
}

bool GatherMatmul::needPrepareParams() const {
    if (bf16_amx_mode && Node::needPrepareParams()) {
        auto srcMem = getSrcMemoryAtPort(DATA);
        const auto& srcShape = srcMem->getStaticDims();
        const auto M = srcShape[1];
        if (M != 1) {
            return true;
        }
    }
    return false;
}

// AMX tile has 16 rows, so to avoid partial tiles it's better to pad M dimension to 16 multiple
static Dim normalizeM(Dim M) {
    if (M < 512) {
        M = rnd_up(M, 16);
    } else if (M < 1024) {
        M = rnd_up(M, 32);  // 2 tile blocking - better tile register utilization
    } else {
        M = rnd_up(M, 256);  // better L2 cache blocking
    }
    return M;
}

void GatherMatmul::prepareParams() {
    auto srcMem = getSrcMemoryAtPort(DATA);
    const auto& srcShape = srcMem->getStaticDims();
    const Dim M = normalizeM(srcShape[1]);
    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();

    const auto srcPrc = srcMem->getDesc().getPrecision();

    auto dstMem = getDstMemoryAtPort(0);
    const auto& dstShape = dstMem->getStaticDims();

    m_tmpInputDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(srcPrc, Shape({M, srcShape[2]}));
    m_tmpOutputDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(srcPrc, Shape({M, dstShape[2]}));

    const size_t srcSize = rnd_up(m_tmpInputDesc->getCurrentMemSize(), 64);  // 64 bytes is the cache line size
    const size_t totalSize = srcSize + m_tmpOutputDesc->getCurrentMemSize();
    auto scratchPadDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(ov::element::u8, Shape({totalSize}));
    m_tmpInpBuffer = getScratchPadMem(scratchPadDesc);

    CPU_NODE_ASSERT(gemv_impl, "GEMV implementation is not created");

    dnnl::memory::desc src_md({static_cast<dnnl::memory::dim>(M), static_cast<dnnl::memory::dim>(srcShape[2])},
                              DnnlExtensionUtils::ElementTypeToDataType(srcPrc),
                              dnnl::memory::format_tag::ab);
    auto weights_md = gemv_impl->get_weights_md();

    VectorDims scale_shape{};
    VectorDims zp_shape{};

    if (m_scalesMemory) {
        const auto& fullScaleDims = m_scalesMemory->getStaticDims();
        if (1 == fullScaleDims.size()) {
            scale_shape.push_back(fullScaleDims[0]);
        } else {
            scale_shape.assign(fullScaleDims.begin() + 1, fullScaleDims.end());
        }
    }

    if (m_zpMemory) {
        const auto& fullZpDims = m_zpMemory->getStaticDims();
        if (1 == fullZpDims.size()) {
            zp_shape.push_back(fullZpDims[0]);
        } else {
            zp_shape.assign(fullZpDims.begin() + 1, fullZpDims.end());
        }
    }

    auto biasMemory = getSrcMemoryAtPort(BIAS);
    const auto& biasDesc = biasMemory->getDesc();
    onednn_matmul_key key{src_md, weights_md, scale_shape, zp_shape, !biasDesc.empty()};

    auto cache = context->getParamsCache();
    const auto& eng = getEngine();
    std::tie(gemm_impl, std::ignore) = cache->getOrCreate(key, [&eng](const onednn_matmul_key& k) {
        return std::make_shared<onednn_matmul>(eng, k);
    });
}

bool GatherMatmul::isExecutable() const {
    return !isInputTensorAtPortEmpty(0);  // only data shape matters
}

namespace {
class OffsetHelper {
public:
    static OffsetHelper createOffsetHelper(const MemoryPtr& mem) {
        static const VectorDims empty_dims;
        std::bitset<2> broadcast_mask;
        if (nullptr == mem || mem->getDesc().empty()) {
            return {nullptr, empty_dims, broadcast_mask, 0};
        }
        return createOffsetHelper(*mem);
    }

    static OffsetHelper createOffsetHelper(const IMemory& mem) {
        std::bitset<2> broadcast_mask;
        auto* base_ptr = static_cast<uint8_t*>(mem.getData());
        auto desc = mem.getDescWithType<BlockedMemoryDesc>();
        const auto& strides = desc->getStrides();
        const auto prc = desc->getPrecision();
        const auto& shape = desc->getShape().getStaticDims();
        for (size_t i = 0; i < shape.size() && i < 2; i++) {
            if (shape[i] == 1) {
                broadcast_mask.set(i);
            }
        }
        return {base_ptr, strides, broadcast_mask, prc.bitwidth()};
    }

    void* operator()(size_t i0) const {
        if (!m_base_ptr) {
            return nullptr;
        }
        if (m_broadcast_mask.test(0)) {
            i0 = 0;
        }
        const size_t offset_bits = i0 * m_strides[0] * m_num_bits;
        const size_t offset = div_up(offset_bits, 8);  // 8 bits in byte
        return m_base_ptr + offset;
    }

    void* operator()(size_t i0, size_t i1) const {
        if (!m_base_ptr) {
            return nullptr;
        }
        if (m_broadcast_mask.test(0)) {
            i0 = 0;
        }
        if (m_broadcast_mask.test(1)) {
            i1 = 0;
        }
        const size_t offset_bits = i0 * m_strides[0] * m_num_bits + i1 * m_strides[1] * m_num_bits;
        const size_t offset = div_up(offset_bits, 8);  // 8 bits in byte
        return m_base_ptr + offset;
    }

    [[nodiscard]] void* get_base() const {
        return m_base_ptr;
    }

private:
    OffsetHelper(uint8_t* base_ptr, const VectorDims& strides, std::bitset<2> broadcast_mask, size_t num_bits)
        : m_base_ptr(base_ptr),
          m_strides(strides),
          m_num_bits(num_bits),
          m_broadcast_mask(broadcast_mask) {}

    uint8_t* m_base_ptr = nullptr;
    const VectorDims& m_strides;
    size_t m_num_bits;
    std::bitset<2> m_broadcast_mask;
};
}  // namespace

void GatherMatmul::execute(const dnnl::stream& strm) {
    const auto& srcMem = getParentEdgeAt(DATA)->getMemoryPtr();
    const auto& biasMem = getParentEdgeAt(BIAS)->getMemoryPtr();
    const auto& indexMem = getParentEdgeAt(INDICES)->getMemoryPtr();

    const auto& dstMem = getChildEdgeAt(0)->getMemoryPtr();

    const auto& indexShape = indexMem->getStaticDims();
    size_t M = indexShape[0];
    size_t indices_size = indexShape[1];  // number of elements to be gathered per each m index

    auto src_offset = OffsetHelper::createOffsetHelper(srcMem);
    auto dst_offset = OffsetHelper::createOffsetHelper(dstMem);
    auto wei_offset = OffsetHelper::createOffsetHelper(m_weightsMemory);
    auto bias_offset = OffsetHelper::createOffsetHelper(biasMem);
    auto scale_offset = OffsetHelper::createOffsetHelper(m_scalesMemory);
    auto zp_offset = OffsetHelper::createOffsetHelper(m_zpMemory);
    auto index_offset = OffsetHelper::createOffsetHelper(indexMem);

    // input 1 is a tensor A[B, M, K]
    // input 2 is a tensor B[G, K, N] (transposed) - G is the gather axis
    // input 3 is the gather indices I [M, B]
    // for each b in B and m in M:
    //    gathered_weights = B[I[m,b], :, :] (has shape [K, N]^T)
    //    output[b,m,:] = MatMul(A[b,m,:], gathered_weights)

    if (M > 1) {
        const size_t gather_axis_size = m_weightsMemory->getStaticDims()[0];

        // all the gather idx for corresponding m index
        std::vector<std::pair<int32_t, int32_t>> gather_idx_map(gather_axis_size * M);
        std::vector<int32_t> elements_per_gather_indx(gather_axis_size, 0);
        for (size_t m = 0; m < M; m++) {
            const auto* gather_ids = static_cast<const int32_t*>(index_offset(m));
            for (size_t i = 0; i < indices_size; i++) {
                int32_t gather_axis_index = gather_ids[i];
                CPU_NODE_ASSERT(gather_axis_index >= 0 && static_cast<size_t>(gather_axis_index) < gather_axis_size,
                                "Invalid gather_id ",
                                gather_axis_index,
                                " for m ",
                                m);
                auto& index = elements_per_gather_indx[gather_axis_index];
                gather_idx_map[gather_axis_index * M + index] = {m, i};
                index++;
            }
        }

        if (bf16_amx_mode) {
            // When AMX is available, we use GEMM for better performance
            // first we pack all the tokens corresponding to a specific expert into a temporary buffer
            // then we call GEMM for that expert on that temporary buffer
            // and finally scatter the results to result memory
            CPU_NODE_ASSERT(m_tmpInpBuffer, "Temporary input/output memory is not created");
            CPU_NODE_ASSERT(m_tmpInputDesc, "Temporary input memory desc is not created");
            CPU_NODE_ASSERT(m_tmpOutputDesc, "Temporary output memory desc is not created");

            const auto element_size = m_tmpInputDesc->getPrecision().size();
            const auto K_size = m_tmpInputDesc->getShape().getStaticDims()[1];
            const auto M_size = m_tmpInputDesc->getShape().getStaticDims()[0];
            const auto N_size = dstMem->getStaticDims()[2];

            auto* input_ptr = m_tmpInpBuffer->getDataAs<uint8_t>();
            auto* output_ptr =
                input_ptr + rnd_up(m_tmpInputDesc->getCurrentMemSize(), 64);  // 64 bytes is the cache line size

            Memory tmpInput(getEngine(), m_tmpInputDesc, input_ptr);
            Memory tmpOutput(getEngine(), m_tmpOutputDesc, output_ptr);

            auto tmp_input_offset = OffsetHelper::createOffsetHelper(tmpInput);
            auto tmp_dst_offset = OffsetHelper::createOffsetHelper(tmpOutput);

            CPU_NODE_ASSERT(gemm_impl, "GEMM implementation is not created");
            for (size_t gather_axis_index = 0; gather_axis_index < gather_axis_size; gather_axis_index++) {
                const size_t num_valid_rows = elements_per_gather_indx[gather_axis_index];
                if (0 == num_valid_rows) {
                    continue;
                }

                parallel_for(M_size, [&](size_t m) {
                    auto* dst_row = tmp_input_offset(m);

                    if (m < num_valid_rows) {
                        const auto row_id = gather_idx_map[gather_axis_index * M + m].first;
                        const auto batch_index = gather_idx_map[gather_axis_index * M + m].second;
                        const auto* src_data = src_offset(batch_index, row_id);
                        std::memcpy(dst_row, src_data, K_size * element_size);
                    } else {
                        // Zero padding for rows beyond num_valid_tokens
                        std::memset(dst_row, 0, K_size * element_size);
                    }
                });

                auto* src = tmp_input_offset.get_base();
                auto* dst = tmp_dst_offset.get_base();
                auto* wei = wei_offset(gather_axis_index);
                auto* bias = bias_offset(gather_axis_index);
                auto* scale = scale_offset(gather_axis_index);
                auto* zp = zp_offset(gather_axis_index);
                gemm_impl->exec(strm, src, dst, wei, bias, scale, zp);

                // Immediately scatter results while they're hot in cache
                parallel_for(num_valid_rows, [&](size_t m) {
                    const auto* src_row = tmp_dst_offset(m);
                    const auto row_id = gather_idx_map[gather_axis_index * M + m].first;
                    const auto batch_index = gather_idx_map[gather_axis_index * M + m].second;
                    auto* dst_row = dst_offset(batch_index, row_id);
                    std::memcpy(dst_row, src_row, N_size * element_size);
                });
            }
        } else {
            // For the default SIMD it's better to simply call GEMV
            CPU_NODE_ASSERT(gemv_impl, "GEMM implementation is not created");
            for (size_t gather_axis_index = 0; gather_axis_index < gather_axis_size; gather_axis_index++) {
                if (0 == elements_per_gather_indx[gather_axis_index]) {
                    continue;
                }
                auto* wei = wei_offset(gather_axis_index);
                auto* bias = bias_offset(gather_axis_index);
                auto* scale = scale_offset(gather_axis_index);
                auto* zp = zp_offset(gather_axis_index);
                for (int32_t m = 0; m < elements_per_gather_indx[gather_axis_index]; ++m) {
                    const auto row_id = gather_idx_map[gather_axis_index * M + m].first;
                    const auto batch_index = gather_idx_map[gather_axis_index * M + m].second;
                    auto* src = src_offset(batch_index, row_id);
                    auto* dst = dst_offset(batch_index, row_id);
                    gemv_impl->exec(strm, src, dst, wei, bias, scale, zp);
                }
            }
        }
    } else {
        CPU_NODE_ASSERT(gemv_impl, "GEMM implementation is not created");

        constexpr size_t m = 0;
        auto* gather_ids = static_cast<int32_t*>(index_offset(m));
        for (size_t i = 0; i < indices_size; i++) {
            int32_t gather_axis_index = gather_ids[i];
            auto* src = src_offset(i, m);
            auto* dst = dst_offset(i, m);
            auto* wei = wei_offset(gather_axis_index);
            auto* bias = bias_offset(gather_axis_index);
            auto* scale = scale_offset(gather_axis_index);
            auto* zp = zp_offset(gather_axis_index);
            gemv_impl->exec(strm, src, dst, wei, bias, scale, zp);
        }
    }
}

void GatherMatmul::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

bool GatherMatmul::created() const {
    return getType() == Type::GatherMatmul;
}

}  // namespace ov::intel_cpu::node
