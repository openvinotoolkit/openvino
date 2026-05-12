// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_gathermatmul_executor.hpp"

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
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "nodes/common/blocked_desc_creator.h"
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#    include <cpu/x64/cpu_isa_traits.hpp>
#endif
#include "cpu_memory.h"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "memory_desc/blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/executors/dnnl/dnnl_utils.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/gathermatmul_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "thread_pool_imp.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

// ---- InnerProductKey ---------------------------------------------------------

struct InnerProductKey {
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

    bool operator==(const InnerProductKey& rhs) const {
        return src_md == rhs.src_md && weights_md == rhs.weights_md && scale_shape == rhs.scale_shape &&
               zp_shape == rhs.zp_shape && has_bias == rhs.has_bias;
    }
};

// ---- InnerProduct (oneDNN inner_product wrapper) ----------------------------

class GatherMatmulDnnlExecutor::InnerProduct {
public:
    InnerProduct() = delete;
    InnerProduct(const InnerProduct&) = delete;
    InnerProduct(InnerProduct&&) = delete;
    InnerProduct& operator=(const InnerProduct&) = delete;
    InnerProduct& operator=(InnerProduct&&) = delete;

    InnerProduct(const dnnl::engine& eng, const std::shared_ptr<ThreadPool>& threadPool, const InnerProductKey& key)
        : m_stream(make_stream(eng, threadPool)),
          m_has_bias(key.has_bias) {
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

        auto ip_prim_desc = dnnl::inner_product_forward::primitive_desc(eng,
                                                                        dnnl::prop_kind::forward_inference,
                                                                        m_input_md,
                                                                        weights_md,
                                                                        bias_md,
                                                                        m_output_md,
                                                                        m_attr);

        m_impl_type = parse_impl_name(ip_prim_desc.impl_info_str());
        m_wei_md = ip_prim_desc.weights_desc();
        m_prim = dnnl::inner_product_forward(ip_prim_desc);

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
        m_args = make_args(inp_memory, out_memory, wei_memory, bias_memory, scale_memory, zp_memory);
    }

    void exec(void* src, void* dst, void* weight, void* bias = nullptr, void* scale = nullptr, void* zp = nullptr) {
        m_args[DNNL_ARG_SRC].set_data_handle(src);
        m_args[DNNL_ARG_DST].set_data_handle(dst);
        m_args[DNNL_ARG_WEIGHTS].set_data_handle(weight);
        if (bias) {
            m_args[DNNL_ARG_BIAS].set_data_handle(bias);
        }
        if (scale) {
            m_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS].set_data_handle(scale);
        }
        if (zp) {
            m_args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS].set_data_handle(zp);
        }
        m_prim.execute(m_stream, m_args);
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
    [[nodiscard]] impl_desc_type get_impl_type() const {
        return m_impl_type;
    }

private:
    void init_w_scales(const VectorDims& scale_shape) {
        constexpr auto data_type = dnnl::memory::data_type::f32;
        const auto scale_dims = DnnlExtensionUtils::convertToDnnlDims(scale_shape);
        m_attr.set_scales_dims(DNNL_ARG_WEIGHTS, scale_dims, data_type);
        m_scale_md = dnnl::memory::desc(scale_dims, data_type, dnnl::memory::format_tag::ba);
    }

    void init_w_zp(const VectorDims& zp_shape) {
        constexpr auto data_type = dnnl::memory::data_type::f32;
        const auto zp_dims = DnnlExtensionUtils::convertToDnnlDims(zp_shape);
        m_attr.set_zero_points_dims(DNNL_ARG_WEIGHTS, zp_dims, data_type);
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

    dnnl::stream m_stream;
    dnnl::primitive m_prim;
    dnnl::memory::desc m_input_md;
    dnnl::memory::desc m_output_md;
    dnnl::memory::desc m_wei_md;
    dnnl::memory::desc m_scale_md;
    dnnl::memory::desc m_zp_md;
    dnnl::primitive_attr m_attr;
    std::unordered_map<int, dnnl::memory> m_args;
    impl_desc_type m_impl_type = impl_desc_type::unknown;
    bool m_has_bias = false;
};

// ---- OffsetHelper ------------------------------------------------------------

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
        const size_t offset = div_up(offset_bits, 8);
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
        const size_t offset = div_up(offset_bits, 8);
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

// ---- normalizeM helper -------------------------------------------------------

static Dim normalizeM(Dim M) {
    if (M < 512) {
        M = rnd_up(M, 16);
    } else if (M < 1024) {
        M = rnd_up(M, 32);
    } else {
        M = rnd_up(M, 256);
    }
    return M;
}

// ---- GatherMatmulDnnlExecutor -----------------------------------------------

bool GatherMatmulDnnlExecutor::supports(const GatherMatmulConfig& config) {
#ifdef OPENVINO_ARCH_X86_64
    // Allow empty (dynamic) src descriptor — actual type is resolved at createPrimitive time
    if ((config.descs.count(ARG_SRC) != 0U) && !config.descs.at(ARG_SRC)->empty()) {
        const auto src_prc = config.descs.at(ARG_SRC)->getPrecision();
        if (!any_of(src_prc, ov::element::f32, ov::element::bf16)) {
            return false;
        }
    }
    // For compressed (int) weights, require AVX2
    if ((config.descs.count(ARG_WEI) != 0U) && !config.descs.at(ARG_WEI)->empty()) {
        const auto wei_prc = config.descs.at(ARG_WEI)->getPrecision();
        if (any_of(wei_prc, ov::element::u8, ov::element::i8, ov::element::u4, ov::element::i4)) {
            if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
                return false;
            }
        }
    }
    return true;
#else
    return false;
#endif
}

GatherMatmulDnnlExecutor::GatherMatmulDnnlExecutor([[maybe_unused]] const GatherMatmulAttrs& attrs,
                                                   const MemoryArgs& memory,
                                                   const ExecutorContext::CPtr& context)
    : m_context(context) {
    m_withBias = !memory.at(ARG_BIAS)->getDesc().empty();

    const auto& weightsMemory = memory.at(ARG_WEI);
    const auto& srcMemory = memory.at(ARG_SRC);

    auto src_precision = srcMemory->getDesc().getPrecision();
    auto weights_precision = weightsMemory->getDesc().getPrecision();
#ifdef OPENVINO_ARCH_X86_64
    m_bf16AmxMode =
        (src_precision == ov::element::bf16 && dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx));
#endif

    const auto& weiDims = weightsMemory->getShape().getStaticDims();
    const dnnl::memory::dim N = weiDims[weiDims.size() - 2];
    const dnnl::memory::dim K = weiDims[weiDims.size() - 1];

    // Determine scale/zp shapes (per-group, removing the leading batch/gather dim)
    VectorDims scale_shape{};
    VectorDims zp_shape{};

    const auto& scalesMem = memory.at(ARG_SRC_3);
    if (scalesMem && !scalesMem->getDesc().empty()) {
        const auto& fullScalesShape = scalesMem->getShape().getStaticDims();
        if (1 == fullScalesShape.size()) {
            OPENVINO_ASSERT(fullScalesShape[0] == 1, "Expect broadcastable scales shape.");
            scale_shape.push_back(fullScalesShape[0]);
        } else {
            scale_shape.assign(fullScalesShape.begin() + 1, fullScalesShape.end());
        }
    }

    const auto& zpMem = memory.at(ARG_SRC_4);
    if (zpMem && !zpMem->getDesc().empty()) {
        const auto& fullZpShape = zpMem->getShape().getStaticDims();
        if (1 == fullZpShape.size()) {
            OPENVINO_ASSERT(fullZpShape[0] == 1, "Expect broadcastable zero points shape.");
            zp_shape.push_back(fullZpShape[0]);
        } else {
            zp_shape.assign(fullZpShape.begin() + 1, fullZpShape.end());
        }
    }

    dnnl::memory::desc src_md({1, K},
                              DnnlExtensionUtils::ElementTypeToDataType(src_precision),
                              dnnl::memory::format_tag::ab);
    dnnl::memory::desc weights_md({N, K},
                                  DnnlExtensionUtils::ElementTypeToDataType(weights_precision),
                                  dnnl::memory::format_tag::any);

    const bool has_bias = m_withBias;
    InnerProductKey key{src_md, weights_md, scale_shape, zp_shape, has_bias};

    const auto& eng = context->getEngine();
    const auto threadPool = context->getThreadPool();
    auto cache = context->getRuntimeCache();
    std::tie(m_gemvImpl, std::ignore) = cache->getOrCreate(key, [&eng, &threadPool](const InnerProductKey& k) {
        return std::make_shared<InnerProduct>(eng, threadPool, k);
    });

    // Repack weights: convert from [G, K, N] to [G, (packed_N, K)] format expected by oneDNN
    auto gemvWeightsDesc =
        MemoryDescUtils::convertToBlockedMemoryDesc(DnnlExtensionUtils::makeDescriptor(m_gemvImpl->get_weights_md()));

    auto addBatchDim = [](const BlockedMemoryDescPtr& desc, size_t batchDim) -> DnnlMemoryDescPtr {
        const auto& weightsDims = desc->getShape().getStaticDims();
        const auto& weightsBlockDims = desc->getBlockDims();
        const auto& weightsOrder = desc->getOrder();
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
    auto srcWeightsDesc = MemoryDescUtils::convertToDnnlMemoryDesc(weightsMemory->getDescPtr());

    m_weightsMemory = utils::prepareWeightsMemory(srcWeightsDesc,
                                                  targetWeightsDesc,
                                                  weightsMemory,
                                                  eng,
                                                  cache,
                                                  context->getWeightsCache(),
                                                  context->getPrivateWeightCache(),
                                                  threadPool);

    if (!scale_shape.empty()) {
        auto expectedScaleMemDesc =
            MemoryDescUtils::convertToDnnlMemoryDesc(DnnlExtensionUtils::makeDescriptor(m_gemvImpl->get_scale_md()));
        const auto& scDims = scalesMem->getShape().getStaticDims();
        expectedScaleMemDesc =
            addBatchDim(MemoryDescUtils::convertToBlockedMemoryDesc(expectedScaleMemDesc), scDims[0]);
        if (expectedScaleMemDesc->isCompatible(scalesMem->getDesc())) {
            m_scalesMemory = std::const_pointer_cast<IMemory>(scalesMem);
        } else {
            m_scalesMemory = std::make_shared<Memory>(eng, expectedScaleMemDesc);
            m_scalesMemory->load(*scalesMem, false, false);
        }
    }

    if (!zp_shape.empty()) {
        auto expectedZpMemDesc =
            MemoryDescUtils::convertToDnnlMemoryDesc(DnnlExtensionUtils::makeDescriptor(m_gemvImpl->get_zp_md()));
        const auto& zpDims = zpMem->getShape().getStaticDims();
        expectedZpMemDesc = addBatchDim(MemoryDescUtils::convertToBlockedMemoryDesc(expectedZpMemDesc), zpDims[0]);
        if (expectedZpMemDesc->isCompatible(zpMem->getDesc())) {
            m_zpMemory = std::const_pointer_cast<IMemory>(zpMem);
        } else {
            m_zpMemory = std::make_shared<Memory>(eng, expectedZpMemDesc);
            m_zpMemory->load(*zpMem, false, false);
        }
    }

    m_implType = m_gemvImpl->get_impl_type();
}

bool GatherMatmulDnnlExecutor::update(const MemoryArgs& memory) {
    if (!m_bf16AmxMode) {
        return true;
    }

    const auto& srcMem = memory.at(ARG_SRC);
    const auto& srcShape = srcMem->getStaticDims();
    // srcShape is [B, M, K]
    const Dim M = normalizeM(srcShape[1]);
    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    const auto srcPrc = srcMem->getDesc().getPrecision();

    const auto& dstMem = memory.at(ARG_DST);
    const auto& dstShape = dstMem->getStaticDims();

    m_tmpInputDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(srcPrc, Shape({M, srcShape[2]}));
    m_tmpOutputDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(srcPrc, Shape({M, dstShape[2]}));

    const size_t srcSize = rnd_up(m_tmpInputDesc->getCurrentMemSize(), 64);
    const size_t totalSize = srcSize + m_tmpOutputDesc->getCurrentMemSize();
    auto scratchPadDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(ov::element::u8, Shape({totalSize}));
    m_tmpInpBuffer = m_context->getScratchPad()->createScratchPadMem(scratchPadDesc);

    OPENVINO_ASSERT(m_gemvImpl, "GEMV implementation is not created");

    dnnl::memory::desc src_md({static_cast<dnnl::memory::dim>(M), static_cast<dnnl::memory::dim>(srcShape[2])},
                              DnnlExtensionUtils::ElementTypeToDataType(srcPrc),
                              dnnl::memory::format_tag::ab);
    auto weights_md = m_gemvImpl->get_weights_md();

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

    InnerProductKey key{src_md, weights_md, scale_shape, zp_shape, m_withBias};
    const auto& eng = m_context->getEngine();
    const auto threadPool = m_context->getThreadPool();
    auto cache = m_context->getRuntimeCache();
    std::tie(m_gemmImpl, std::ignore) = cache->getOrCreate(key, [&eng, &threadPool](const InnerProductKey& k) {
        return std::make_shared<InnerProduct>(eng, threadPool, k);
    });
    return true;
}

void GatherMatmulDnnlExecutor::execute(const MemoryArgs& memory) {
    const auto& cpu_parallel = m_context->getCpuParallel();
    const auto& srcMem = memory.at(ARG_SRC);
    const auto& biasMem = memory.at(ARG_BIAS);
    const auto& indexMem = memory.at(ARG_SRC_1);
    const auto& dstMem = memory.at(ARG_DST);

    const auto& indexShape = indexMem->getStaticDims();
    size_t M = indexShape[0];
    size_t indices_size = indexShape[1];

    auto src_offset = OffsetHelper::createOffsetHelper(srcMem);
    auto dst_offset = OffsetHelper::createOffsetHelper(dstMem);
    auto wei_offset = OffsetHelper::createOffsetHelper(m_weightsMemory);
    auto bias_offset = OffsetHelper::createOffsetHelper(biasMem);
    auto scale_offset = OffsetHelper::createOffsetHelper(m_scalesMemory);
    auto zp_offset = OffsetHelper::createOffsetHelper(m_zpMemory);
    auto index_offset = OffsetHelper::createOffsetHelper(indexMem);

    if (M > 1) {
        const size_t gather_axis_size = m_weightsMemory->getStaticDims()[0];

        std::vector<std::pair<int32_t, int32_t>> gather_idx_map(gather_axis_size * M);
        std::vector<int32_t> elements_per_gather_indx(gather_axis_size, 0);
        for (size_t m = 0; m < M; m++) {
            const auto* gather_ids = static_cast<const int32_t*>(index_offset(m));
            for (size_t i = 0; i < indices_size; i++) {
                int32_t gather_axis_index = gather_ids[i];
                OPENVINO_ASSERT(gather_axis_index >= 0 && static_cast<size_t>(gather_axis_index) < gather_axis_size,
                                "Invalid gather_id ",
                                gather_axis_index,
                                " for m ",
                                m);
                auto& index = elements_per_gather_indx[gather_axis_index];
                gather_idx_map[gather_axis_index * M + index] = {m, i};
                index++;
            }
        }

        if (m_bf16AmxMode) {
            OPENVINO_ASSERT(m_tmpInpBuffer, "Temporary input/output memory is not created");
            OPENVINO_ASSERT(m_tmpInputDesc, "Temporary input memory desc is not created");
            OPENVINO_ASSERT(m_tmpOutputDesc, "Temporary output memory desc is not created");

            const auto element_size = m_tmpInputDesc->getPrecision().size();
            const auto K_size = m_tmpInputDesc->getShape().getStaticDims()[1];
            const auto M_size = m_tmpInputDesc->getShape().getStaticDims()[0];
            const auto N_size = dstMem->getStaticDims()[2];

            auto* input_ptr = m_tmpInpBuffer->getDataAs<uint8_t>();
            auto* output_ptr = input_ptr + rnd_up(m_tmpInputDesc->getCurrentMemSize(), 64);

            Memory tmpInput(m_context->getEngine(), m_tmpInputDesc, input_ptr);
            Memory tmpOutput(m_context->getEngine(), m_tmpOutputDesc, output_ptr);

            auto tmp_input_offset = OffsetHelper::createOffsetHelper(tmpInput);
            auto tmp_dst_offset = OffsetHelper::createOffsetHelper(tmpOutput);

            OPENVINO_ASSERT(m_gemmImpl, "GEMM implementation is not created");
            for (size_t gather_axis_index = 0; gather_axis_index < gather_axis_size; gather_axis_index++) {
                const size_t num_valid_rows = elements_per_gather_indx[gather_axis_index];
                if (0 == num_valid_rows) {
                    continue;
                }

                cpu_parallel->parallel_for(M_size, [&](size_t m) {
                    auto* dst_row = tmp_input_offset(m);
                    if (m < num_valid_rows) {
                        const auto row_id = gather_idx_map[gather_axis_index * M + m].first;
                        const auto batch_index = gather_idx_map[gather_axis_index * M + m].second;
                        const auto* src_data = src_offset(batch_index, row_id);
                        std::memcpy(dst_row, src_data, K_size * element_size);
                    } else {
                        std::memset(dst_row, 0, K_size * element_size);
                    }
                });

                auto* src = tmp_input_offset.get_base();
                auto* dst = tmp_dst_offset.get_base();
                auto* wei = wei_offset(gather_axis_index);
                auto* bias = bias_offset(gather_axis_index);
                auto* scale = scale_offset(gather_axis_index);
                auto* zp = zp_offset(gather_axis_index);
                m_gemmImpl->exec(src, dst, wei, bias, scale, zp);

                cpu_parallel->parallel_for(num_valid_rows, [&](size_t m) {
                    const auto* src_row = tmp_dst_offset(m);
                    const auto row_id = gather_idx_map[gather_axis_index * M + m].first;
                    const auto batch_index = gather_idx_map[gather_axis_index * M + m].second;
                    auto* dst_row = dst_offset(batch_index, row_id);
                    std::memcpy(dst_row, src_row, N_size * element_size);
                });
            }
        } else {
            OPENVINO_ASSERT(m_gemvImpl, "GEMV implementation is not created");
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
                    m_gemvImpl->exec(src, dst, wei, bias, scale, zp);
                }
            }
        }
    } else {
        OPENVINO_ASSERT(m_gemvImpl, "GEMV implementation is not created");

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
            m_gemvImpl->exec(src, dst, wei, bias, scale, zp);
        }
    }
}

impl_desc_type GatherMatmulDnnlExecutor::implType() const {
    return m_implType;
}

}  // namespace ov::intel_cpu
