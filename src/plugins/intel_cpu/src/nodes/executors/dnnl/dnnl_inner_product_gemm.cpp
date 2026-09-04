// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_inner_product_gemm.hpp"

#include <oneapi/dnnl/dnnl_common_types.h>
#include <oneapi/dnnl/dnnl_types.h>

#include <common/primitive_hashing.hpp>
#include <common/primitive_hashing_utils.hpp>
#include <common/utils.hpp>
#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <unordered_map>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "memory_desc/blocked_memory_desc.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "thread_pool_imp.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::dnnl_utils {

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
    return seed;
}

bool InnerProductKey::operator==(const InnerProductKey& rhs) const {
    return src_md == rhs.src_md && weights_md == rhs.weights_md && bias_md == rhs.bias_md &&
           dst_data_type == rhs.dst_data_type && scale_shape == rhs.scale_shape && zp_shape == rhs.zp_shape;
}

InnerProduct::InnerProduct(const dnnl::engine& eng,
                           const std::shared_ptr<ThreadPool>& threadPool,
                           const InnerProductKey& key)
    : m_stream(make_stream(eng, threadPool)) {
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

void InnerProduct::init_w_zp(const VectorDims& zp_shape) {
    constexpr auto data_type = dnnl::memory::data_type::f32;
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

Dim normalizeM(Dim M) {
    if (M < 512) {
        M = rnd_up(M, 16);
    } else if (M < 1024) {
        M = rnd_up(M, 32);
    } else {
        M = rnd_up(M, 256);
    }
    return M;
}

dnnl::memory::desc makeBiasMd(dnnl::memory::dim N, const MemoryPtr& biasMem) {
    if (!biasMem || biasMem->getDesc().empty()) {
        return {};
    }
    const auto bias_precision = biasMem->getDesc().getPrecision();
    return dnnl::memory::desc(dnnl::memory::dims({N}),
                              DnnlExtensionUtils::ElementTypeToDataType(bias_precision),
                              dnnl::memory::format_tag::a);
}

DnnlMemoryDescPtr addBatchDim(const BlockedMemoryDescPtr& desc, size_t batchDim) {
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
}

}  // namespace ov::intel_cpu::dnnl_utils
