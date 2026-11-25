// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_grouped_matmul_executor.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <tuple>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "nodes/executors/common/offset_helper.hpp"
#include "nodes/executors/dnnl/dnnl_inner_product_gemm.hpp"
#include "nodes/executors/dnnl/dnnl_utils.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/grouped_matmul_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

using utils::InnerProduct;
using utils::InnerProductKey;

bool GroupedMatMulDnnlExecutor::supports(const GroupedMatMulConfig& config) {
    return utils::supports_inner_product(config.descs);
}

GroupedMatMulDnnlExecutor::GroupedMatMulDnnlExecutor(const GroupedMatMulAttrs& attrs,
                                                     const MemoryArgs& memory,
                                                     const ExecutorContext::CPtr& context)
    : m_context(context) {
    const auto& src_memory = memory.at(ARG_GMM_DATA);
    const auto src_precision = src_memory->getDesc().getPrecision();
    m_is_batched = src_memory->getShape().getRank() == 3;

    const auto& wei_dims = memory.at(ARG_GMM_WEIGHTS)->getShape().getStaticDims();
    const dnnl::memory::dim K = wei_dims[wei_dims.size() - 1];

    auto gemv = utils::prepare_gemv_inner_product(memory, context, attrs.dynamic_quantization_group_size, {ov::element::u3, ov::element::u4, ov::element::u8});
    m_weights_memory = std::move(gemv.weights);
    m_scales_memory = std::move(gemv.scales);
    m_zp_memory = std::move(gemv.zp);
    const auto gemv_impl = gemv.impl;

    // Only one value can be reported; the per-group primitives are all inner_product over the same
    // weights, so the implementation family does not vary.
    m_impl_type = gemv_impl->get_impl_type();

    m_src_data_type = DnnlExtensionUtils::ElementTypeToDataType(src_precision);
    m_K = K;
    m_key_template = std::move(gemv.key);
    m_key_template.src_md = {};
    m_key_template.weights_md = gemv_impl->get_weights_md();
}

// inner_product bakes M into the primitive and a group's row count is offsets *data*, which is not valid in
// prepareParams() (it also doesn't run when only the routing changes): so the primitives are taken from
// the runtime cache in execute().
GroupedMatMulDnnlExecutor::InnerProductPtr GroupedMatMulDnnlExecutor::impl_for(Dim rows) {
    auto key = m_key_template;
    key.src_md =
        dnnl::memory::desc({static_cast<dnnl::memory::dim>(rows), m_K}, m_src_data_type, dnnl::memory::format_tag::ab);

    const auto& eng = m_context->getEngine();
    const auto thread_pool = m_context->getThreadPool();
    auto [impl, _] = m_context->getRuntimeCache()->getOrCreate(key, [&eng, &thread_pool](const InnerProductKey& k) {
        return std::make_shared<InnerProduct>(eng, thread_pool, k);
    });
    return impl;
}

bool GroupedMatMulDnnlExecutor::update([[maybe_unused]] const MemoryArgs& memory) {
    // The per-group primitives depend on offsets contents, which are not valid here. See impl_for.
    return true;
}

void GroupedMatMulDnnlExecutor::execute(const MemoryArgs& memory) {
    const auto& src_mem = memory.at(ARG_GMM_DATA);
    const auto& dst_mem = memory.at(ARG_DST);
    const auto& offsets_mem = memory.at(ARG_GMM_OFFSETS);

    auto src_offset = OffsetHelper::createOffsetHelper(src_mem);
    auto dst_offset = OffsetHelper::createOffsetHelper(dst_mem);
    auto wei_offset = OffsetHelper::createOffsetHelper(m_weights_memory);
    auto scale_offset = OffsetHelper::createOffsetHelper(m_scales_memory);
    auto zp_offset = OffsetHelper::createOffsetHelper(m_zp_memory);

    const auto& src_shape = src_mem->getStaticDims();
    const size_t G = m_weights_memory->getStaticDims()[0];
    const size_t N = dst_mem->getStaticDims().back();
    const size_t total_rows = m_is_batched ? src_shape[0] * src_shape[1] : src_shape[0];

    // 3D x 3D indexes src / dst as [G, M, ...], 2D x 3D as a flat [T, ...]
    auto src_at = [&](size_t g, size_t row) {
        return m_is_batched ? src_offset(g, row) : src_offset(row);
    };
    auto dst_at = [&](size_t g, size_t row) {
        return m_is_batched ? dst_offset(g, row) : dst_offset(row);
    };

    const int32_t* offsets = nullptr;
    if (!m_is_batched) {
        OPENVINO_ASSERT(offsets_mem && !offsets_mem->getDesc().empty(),
                        "GroupedMatMul: the 2D x 3D form requires the offsets input");
        OPENVINO_ASSERT(offsets_mem->getPrecision() == ov::element::i32,
                        "GroupedMatMul: offsets must be i32, got ",
                        offsets_mem->getPrecision());
        OPENVINO_ASSERT(offsets_mem->getStaticDims()[0] == G,
                        "GroupedMatMul: offsets size ",
                        offsets_mem->getStaticDims()[0],
                        " does not match the number of groups ",
                        G);
        offsets = offsets_mem->getDataAs<int32_t>();
    }

    // Row range owned by group g. The 2D x 3D offsets are cumulative exclusive end boundaries; the
    // 3D x 3D form gives every group its own M rows, indexed within the group.
    size_t prev_end = 0;
    for (size_t g = 0; g < G; g++) {
        size_t start = 0;
        size_t end = m_is_batched ? src_shape[1] : static_cast<size_t>(offsets[g]);
        if (!m_is_batched) {
            start = prev_end;
            OPENVINO_ASSERT(end >= start && end <= total_rows,
                            "GroupedMatMul: offsets must be non-decreasing and not exceed the row count, got ",
                            end,
                            " after ",
                            start,
                            " for a total of ",
                            total_rows,
                            " rows");
            prev_end = end;
        }
        const size_t rows = end - start;
        if (0 == rows) {
            continue;
        }

        auto* wei = wei_offset(g);
        auto* scale = scale_offset(g);
        auto* zp = zp_offset(g);

        // Rows are contiguous, so the primitives run in place. Padding the tail is not possible (it would read
        // and write past the group), so the rows are split into chunks of the normalize_m granularity plus a
        // short tail: this keeps the set of distinct row counts small.
        size_t chunk_start = start;
        size_t remaining = rows;
        while (remaining > 0) {
            const size_t granularity = utils::normalize_m_granularity(remaining);
            const size_t chunk = remaining < granularity ? remaining : remaining - remaining % granularity;
            impl_for(chunk)->exec(src_at(g, chunk_start), dst_at(g, chunk_start), wei, nullptr, scale, zp);
            chunk_start += chunk;
            remaining -= chunk;
        }
    }

    // offsets[G-1] is allowed to be smaller than the token count; the rows past the last group are
    // left untouched by every GEMM, and the reference implementation defines them as zero.
    if (!m_is_batched && (prev_end < total_rows)) {
        const auto dst_element_size = dst_mem->getDesc().getPrecision().size();
        std::memset(dst_at(0, prev_end), 0, (total_rows - prev_end) * N * dst_element_size);
    }
}

impl_desc_type GroupedMatMulDnnlExecutor::implType() const {
    return m_impl_type;
}

}  // namespace ov::intel_cpu
