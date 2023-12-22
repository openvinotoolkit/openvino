// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ref_transpose.hpp"
#include "openvino/core/parallel.hpp"
#include "nodes/common/cpu_memcpy.h"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {

static inline size_t parallel_init(size_t start, size_t nDims, const SizeVector& dims, SizeVector& indexes) {
    for (int j = nDims - 1; j >= 0; j--) {
        indexes[j] = start % dims[j];
        start = start / dims[j];
    }
    return start;
}

static inline void parallel_step(size_t nDims, const SizeVector& dims, SizeVector& indexes) {
    for (int j = nDims - 1; j >= 0; --j) {
        ++indexes[j];
        if (indexes[j] < dims[j])
            break;
        else
            indexes[j] = 0;
    }
}

void RefTransposeExecutor::referenceExecute(const uint8_t* src_data, uint8_t* dst_data, jit_permute_config_params jcp, const int mb) {
    SizeVector dst_dims = jcp.dst_block_dims;
    const SizeVector dst_strides = jcp.dst_strides;
    const SizeVector src_strides = jcp.src_strides;
    const size_t data_size = jcp.data_size;
    const size_t ndims = dst_dims.size();

    if (static_cast<int>(dst_dims[0]) != mb)
        dst_dims[0] = mb;

    size_t work_amount = std::accumulate(dst_dims.begin(), dst_dims.end(), 1, std::multiplies<size_t>());

    auto get_idx = [ndims, data_size](const SizeVector& indexes, const SizeVector& strides) {
        size_t idx = 0;
        for (size_t i = 0; i < ndims; ++i)
            idx += indexes[i] * strides[i];
        return idx * data_size;
    };

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        SizeVector indexes(ndims, 0);
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

void RefTransposeExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const int MB) {
    const uint8_t* src_data = reinterpret_cast<const uint8_t*>(src[0]->getData());
    uint8_t* dst_data = reinterpret_cast<uint8_t*>(dst[0]->getData());
    referenceExecute(src_data, dst_data, jcp, MB);
}

bool RefTransposeExecutor::init(const TransposeParams &transposeParams,
                                const std::vector<MemoryDescPtr> &srcDescs,
                                const std::vector<MemoryDescPtr> &dstDescs,
                                const dnnl::primitive_attr &attr) {
    jcp = TransposeExecutor::prepareParams(transposeParams.permuteParams);
    return true;
}

}   // namespace intel_cpu
}   // namespace ov