// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ref_transpose.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "nodes/common/cpu_memcpy.h"
#include "nodes/common/permute_kernel.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/transpose.hpp"
#include "nodes/executors/transpose_config.hpp"
#include "openvino/core/parallel.hpp"

namespace ov::intel_cpu {

RefTransposeExecutor::RefTransposeExecutor(const TransposeAttrs& attrs, ExecutorContext::CPtr context)
    : TransposeExecutor(attrs, std::move(context)) {}

static inline size_t parallel_init(size_t start, size_t nDims, const VectorDims& dims, VectorDims& indexes) {
    for (int j = nDims - 1; j >= 0; j--) {
        indexes[j] = start % dims[j];
        start = start / dims[j];
    }
    return start;
}

static inline void parallel_step(size_t nDims, const VectorDims& dims, VectorDims& indexes) {
    for (int j = nDims - 1; j >= 0; --j) {
        ++indexes[j];
        if (indexes[j] < dims[j]) {
            break;
        }
        indexes[j] = 0;
    }
}

void RefTransposeExecutor::referenceExecute(const uint8_t* src_data,
                                            uint8_t* dst_data,
                                            const jit_permute_config_params& jcp,
                                            const int mb) {
    VectorDims dst_dims = jcp.dst_block_dims;
    const VectorDims dst_strides = jcp.dst_strides;
    const VectorDims src_strides = jcp.src_strides;
    const size_t data_size = jcp.data_size;
    const size_t ndims = dst_dims.size();

    if (static_cast<int>(dst_dims[0]) != mb) {
        dst_dims[0] = mb;
    }

    size_t work_amount = std::accumulate(dst_dims.begin(), dst_dims.end(), 1, std::multiplies<>());

    auto get_idx = [ndims, data_size](const VectorDims& indexes, const VectorDims& strides) {
        size_t idx = 0;
        for (size_t i = 0; i < ndims; ++i) {
            idx += indexes[i] * strides[i];
        }
        return idx * data_size;
    };

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0;
        size_t end = 0;
        VectorDims indexes(ndims, 0);
        splitter(work_amount, nthr, ithr, start, end);

        parallel_init(start, ndims, dst_dims, indexes);

        for (size_t iwork = start; iwork < end; ++iwork) {
            const size_t dst_idx = get_idx(indexes, dst_strides);
            const size_t src_idx = get_idx(indexes, src_strides);
            cpu_memcpy(&dst_data[dst_idx], &src_data[src_idx], data_size);

            parallel_step(ndims, dst_dims, indexes);
        }
    });
}

void RefTransposeExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) {
    const auto* src_data = src[0]->getDataAs<const uint8_t>();
    auto* dst_data = dst[0]->getDataAs<uint8_t>();
    const int MB = src[0]->getStaticDims()[0];
    referenceExecute(src_data, dst_data, jcp, MB);
}

bool RefTransposeExecutor::init([[maybe_unused]] const MemoryArgs& memory) {
    jcp = TransposeExecutor::prepareParams(permuteParams);
    return true;
}

ExecutorPtr RefTransposeExecutor::create(const TransposeAttrs& attrs,
                                         [[maybe_unused]] const MemoryArgs& memory,
                                         const ExecutorContext::CPtr& context) {
    return std::make_shared<RefTransposeExecutor>(attrs, context);
}

}  // namespace ov::intel_cpu
