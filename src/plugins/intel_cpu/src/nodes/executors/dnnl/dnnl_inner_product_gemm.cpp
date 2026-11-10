// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_inner_product_gemm.hpp"

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#    include <cpu/x64/cpu_isa_traits.hpp>
#endif
#include <oneapi/dnnl/dnnl_common_types.h>
#include <oneapi/dnnl/dnnl_types.h>

#include <common/primitive_hashing.hpp>
#include <common/primitive_hashing_utils.hpp>
#include <common/utils.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "memory_desc/blocked_memory_desc.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "thread_pool_imp.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::utils {

size_t InnerProductKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;
    seed = hash_combine(seed, get_md_hash(*src_md.get()));
    seed = hash_combine(seed, get_md_hash(*weights_md.get()));
    seed = hash_combine(seed, get_md_hash(*bias_md.get()));
    seed = hash_combine(seed, static_cast<size_t>(dst_data_type));
    seed = get_vector_hash(seed, scale_shape);
    seed = get_vector_hash(seed, zp_shape);
    seed = hash_combine(seed, static_cast<size_t>(zp_dt));
    seed = hash_combine(seed, src_dyn_quant_group_size);

    return seed;
}

bool InnerProductKey::operator==(const InnerProductKey& rhs) const {
    return src_md == rhs.src_md && weights_md == rhs.weights_md && bias_md == rhs.bias_md &&
           dst_data_type == rhs.dst_data_type && scale_shape == rhs.scale_shape && zp_shape == rhs.zp_shape &&
           zp_dt == rhs.zp_dt && src_dyn_quant_group_size == rhs.src_dyn_quant_group_size;
}

InnerProduct::InnerProduct(const dnnl::engine& eng,
                           const std::shared_ptr<ThreadPool>& thread_pool,
                           const InnerProductKey& key)
    : m_stream(make_stream(eng, thread_pool)) {
    const auto& src_md = key.src_md;
    const auto& weights_md = key.weights_md;
    auto scale_shape = key.scale_shape;
    auto zp_shape = key.zp_shape;

    const auto weiDims = weights_md.get_dims();
    const auto K = weiDims[1];
    const auto N = weiDims[0];
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
            init_w_zp(zp_shape, key.src_dyn_quant_group_size > 0 ? key.zp_dt : dnnl::memory::data_type::f32);
        }
    }

    if (key.src_dyn_quant_group_size > 0) {
        m_attr.set_src_dyn_quant_params(key.src_dyn_quant_group_size);
    }

    m_input_md = src_md;
    m_output_md = dnnl::memory::desc(dnnl::memory::dims({M, N}), key.dst_data_type, dnnl::memory::format_tag::ab);

    const auto& bias_md = key.bias_md;

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
    if (!bias_md.is_zero()) {
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

void InnerProduct::exec(void* src, void* dst, void* weight, void* bias, void* scale, void* zp) {
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

void InnerProduct::init_w_scales(const VectorDims& scale_shape) {
    constexpr auto data_type = dnnl::memory::data_type::f32;
    const auto scale_dims = DnnlExtensionUtils::convertToDnnlDims(scale_shape);
    m_attr.set_scales_dims(DNNL_ARG_WEIGHTS, scale_dims, data_type);
    m_scale_md = dnnl::memory::desc(scale_dims, data_type, dnnl::memory::format_tag::ba);
}

void InnerProduct::init_w_zp(const VectorDims& zp_shape, dnnl::memory::data_type data_type) {
    const auto zp_dims = DnnlExtensionUtils::convertToDnnlDims(zp_shape);
    m_attr.set_zero_points_dims(DNNL_ARG_WEIGHTS, zp_dims, data_type);
    m_zp_md = dnnl::memory::desc(zp_dims, data_type, dnnl::memory::format_tag::ba);
}

std::unordered_map<int, dnnl::memory> InnerProduct::make_args(dnnl::memory& src,
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

Dim normalize_m(Dim M) {
    if (M < 512) {
        M = rnd_up(M, 16);
    } else if (M < 1024) {
        M = rnd_up(M, 32);
    } else {
        M = rnd_up(M, 256);
    }
    return M;
}

dnnl::memory::desc make_bias_md(dnnl::memory::dim N, const MemoryPtr& bias_mem) {
    if (!bias_mem || bias_mem->getDesc().empty()) {
        return {};
    }
    const auto bias_precision = bias_mem->getDesc().getPrecision();
    return dnnl::memory::desc(dnnl::memory::dims({N}),
                              DnnlExtensionUtils::ElementTypeToDataType(bias_precision),
                              dnnl::memory::format_tag::a);
}

DnnlMemoryDescPtr add_batch_dim(const BlockedMemoryDescPtr& desc, size_t batch_dim) {
    const auto& weights_dims = desc->getShape().getStaticDims();
    const auto& weights_block_dims = desc->getBlockDims();
    const auto& weights_order = desc->getOrder();
    VectorDims new_dims = {batch_dim};
    new_dims.insert(new_dims.end(), weights_dims.begin(), weights_dims.end());
    VectorDims new_block_dims = {batch_dim};
    new_block_dims.insert(new_block_dims.end(), weights_block_dims.begin(), weights_block_dims.end());
    VectorDims new_order(weights_order.size() + 1);
    new_order[0] = 0;
    for (size_t i = 0; i < weights_order.size(); i++) {
        new_order[i + 1] = weights_order[i] + 1;
    }
    auto target_desc =
        std::make_shared<CpuBlockedMemoryDesc>(desc->getPrecision(), Shape(new_dims), new_block_dims, new_order);
    return MemoryDescUtils::convertToDnnlMemoryDesc(target_desc);
}

bool is_supported_compressed_inner_product([[maybe_unused]] size_t IC,
                                           [[maybe_unused]] size_t OC,
                                           [[maybe_unused]] size_t G,
                                           [[maybe_unused]] const ov::element::Type& inference_precision) {
#ifdef OPENVINO_ARCH_X86_64
    if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
        return false;
    }

    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx) &&
        inference_precision == ov::element::bf16) {
        // OneDNN AMX IP implementation has limited shapes support due to performance considerations. As a
        // current solution conditions below are copied from OneDNN to make sure correct IP impl will be
        // used since fallback one doesn't support weights decompression feature.
        constexpr size_t simd_width = 16;
        constexpr size_t vnni_factor = 2;
        constexpr size_t max_size = 512;
        constexpr size_t amx_row = vnni_factor * simd_width;

        if ((IC <= amx_row && OC <= amx_row) || (IC <= max_size && OC <= max_size && IC % amx_row != 0)) {
            return false;
        }
    }

    if (G == 0 || IC % G != 0) {
        return false;  // sanity check IC must be evenly divided by the group size
    }

    if (IC / G < 4) {
        return false;  // minimal group size should be 4
    }

    return OC != 1;
#else
    return false;
#endif
}

bool supports_inner_product([[maybe_unused]] const MemoryDescArgs& descs) {
#ifdef OPENVINO_ARCH_X86_64
    // Allow empty (dynamic) src descriptor — actual type is resolved at createPrimitive time
    if ((descs.count(ARG_SRC) != 0U) && !descs.at(ARG_SRC)->empty()) {
        const auto src_prc = descs.at(ARG_SRC)->getPrecision();
        if (!any_of(src_prc, ov::element::f32, ov::element::bf16)) {
            return false;
        }
    }
    // For compressed (int) weights, require AVX2
    if ((descs.count(ARG_WEI) != 0U) && !descs.at(ARG_WEI)->empty()) {
        const auto wei_prc = descs.at(ARG_WEI)->getPrecision();
        if (any_of(wei_prc, ov::element::u8, ov::element::i8, ov::element::u4, ov::element::i4, ov::element::u3)) {
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

VectorDims per_group_shape(const MemoryPtr& mem) {
    VectorDims shape{};
    if (!mem || mem->getDesc().empty()) {
        return shape;
    }
    const auto& full_shape = mem->getShape().getStaticDims();
    if (full_shape.size() == 1LU) {
        OPENVINO_ASSERT(full_shape[0] == 1, "Expect broadcastable per-group shape, got ", vec2str(full_shape));
        shape.push_back(full_shape[0]);
    } else {
        shape.assign(full_shape.begin() + 1, full_shape.end());
    }
    return shape;
}

MemoryPtr prepare_decompression_memory(const MemoryPtr& mem,
                                       const dnnl::memory::desc& prim_md,
                                       const dnnl::engine& eng) {
    const auto group_count = mem->getShape().getStaticDims()[0];
    auto expected_desc =
        add_batch_dim(MemoryDescUtils::convertToBlockedMemoryDesc(
                          MemoryDescUtils::convertToDnnlMemoryDesc(DnnlExtensionUtils::makeDescriptor(prim_md))),
                      group_count);

    MemoryPtr src = mem;
    if (mem->getShape().getRank() == 1LU) {
        auto desc = std::make_shared<CpuBlockedMemoryDesc>(mem->getDesc().getPrecision(),
                                                           Shape(VectorDims(expected_desc->getShape().getRank(), 1LU)));
        src = std::make_shared<Memory>(eng, desc);
        std::memcpy(src->getData(), mem->getData(), mem->getSize());
    }

    if (expected_desc->isCompatible(src->getDesc())) {
        return src;
    }
    auto result = std::make_shared<Memory>(eng, expected_desc);
    result->load(*src, false, false);
    return result;
}

std::pair<size_t, dnnl::memory::data_type> dyn_quant_params(uint64_t dq_group_size,
                                                            ov::element::Type wei_precision,
                                                            ov::element::Type src_precision,
                                                            ov::element::Type zp_precision,
                                                            bool zp_non_negative_i8,
                                                            size_t K,
                                                            const VectorDims& scale_shape,
                                                            const VectorDims& zp_shape,
                                                            bool allow_u4) {
    const auto disabled = std::make_pair(size_t{0}, dnnl::memory::data_type::f32);
    if (dq_group_size == 0 || dq_group_size == std::numeric_limits<uint64_t>::max() || dq_group_size % 32 != 0) {
        return disabled;
    }
    // u3 weights come with i8 zero points. Dynamic quantization converts the zero points to u8, which is
    // lossless only when none of them is negative.
    const bool zp_ok = any_of(zp_precision,
                              ov::element::u8,
                              ov::element::u2,
                              ov::element::u3,
                              ov::element::u4,
                              ov::element::dynamic) ||
                       (zp_precision == ov::element::i8 && zp_non_negative_i8);
    const bool is_supported_weights =
        wei_precision == ov::element::u3 || (allow_u4 && wei_precision == ov::element::u4);
    if (!is_supported_weights || !zp_ok) {
        return disabled;
    }
    if (K < dq_group_size) {
        return disabled;
    }
    for (const auto* shape : {&scale_shape, &zp_shape}) {
        if (shape->size() >= 2 && (*shape)[1] > 1 && (K / (*shape)[1]) % dq_group_size) {
            return disabled;
        }
    }
    if (src_precision == ov::element::bf16) {
        if (!ov::with_cpu_x86_bfloat16() || ov::with_cpu_x86_avx512_core_amx()) {
            return disabled;
        }
    } else if (!ov::with_cpu_x86_avx2_vnni() && !ov::with_cpu_x86_avx512_core_vnni()) {
        return disabled;
    }
    const auto zp_dt =
        zp_precision == ov::element::dynamic ? dnnl::memory::data_type::f32 : dnnl::memory::data_type::u8;
    return {static_cast<size_t>(dq_group_size), zp_dt};
}

}  // namespace ov::intel_cpu::utils
