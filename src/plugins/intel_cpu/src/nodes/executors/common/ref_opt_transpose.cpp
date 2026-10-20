// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ref_opt_transpose.hpp"

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "cpu_memory.h"
#include "cpu_parallel.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/common/ref_opt_transpose.hpp"
#include "nodes/executors/transpose.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "selective_build.h"

namespace ov::intel_cpu {
namespace {

struct TransposeContext {
    MemoryCPtr srcMemPtr;
    MemoryPtr dstMemPtr;
    size_t MB;
    std::shared_ptr<CpuParallel> cpuParallel;
};

template <typename T>
void transpose_to_0312(const size_t MB,
                       const MemoryCPtr& srcMemPtr,
                       MemoryPtr& dstMemPtr,
                       const std::shared_ptr<CpuParallel>& cpuParallel) {
    const auto* const src_data = srcMemPtr->getDataAs<const T>();
    auto dst_data = dstMemPtr->getDataAs<T>();

    const size_t DIM1 = srcMemPtr->getStaticDims()[1];
    const size_t DIM2 = srcMemPtr->getStaticDims()[2];
    const size_t DIM3 = srcMemPtr->getStaticDims()[3];

    cpuParallel->parallel_for3d(MB, DIM1, DIM2, [&](const size_t n, const size_t dim1, const size_t dim2) {
        for (size_t dim3 = 0; dim3 < DIM3; ++dim3) {
            const size_t src_off = n * DIM1 * DIM2 * DIM3 + dim1 * DIM2 * DIM3 + dim2 * DIM3 + dim3;
            const size_t dst_off = n * DIM1 * DIM2 * DIM3 + dim3 * DIM1 * DIM2 + dim1 * DIM2 + dim2;

            dst_data[dst_off] = src_data[src_off];
        }
    });
}

template <typename T>
void transpose_to_04123(const size_t MB,
                        const MemoryCPtr& srcMemPtr,
                        MemoryPtr& dstMemPtr,
                        const std::shared_ptr<CpuParallel>& cpuParallel) {
    const auto* const src_data = srcMemPtr->getDataAs<const T>();
    auto dst_data = dstMemPtr->getDataAs<T>();

    const size_t DIM1 = srcMemPtr->getStaticDims()[1];
    const size_t DIM2 = srcMemPtr->getStaticDims()[2];
    const size_t DIM3 = srcMemPtr->getStaticDims()[3];
    const size_t DIM4 = srcMemPtr->getStaticDims()[4];

    cpuParallel->parallel_for4d(
        MB,
        DIM1,
        DIM2,
        DIM3,
        [&](const size_t n, const size_t dim1, const size_t dim2, const size_t dim3) {
            for (size_t dim4 = 0; dim4 < DIM4; ++dim4) {
                const size_t src_off =
                    n * DIM1 * DIM2 * DIM3 * DIM4 + dim1 * DIM2 * DIM3 * DIM4 + dim2 * DIM3 * DIM4 + dim3 * DIM4 + dim4;
                const size_t dst_off =
                    n * DIM1 * DIM2 * DIM3 * DIM4 + dim4 * DIM1 * DIM2 * DIM3 + dim1 * DIM2 * DIM3 + dim2 * DIM3 + dim3;

                dst_data[dst_off] = src_data[src_off];
            }
        });
}

template <typename T>
void transpose_to_051234(const size_t MB,
                         const MemoryCPtr& srcMemPtr,
                         MemoryPtr& dstMemPtr,
                         const std::shared_ptr<CpuParallel>& cpuParallel) {
    const auto* const src_data = srcMemPtr->getDataAs<const T>();
    auto dst_data = dstMemPtr->getDataAs<T>();

    const size_t DIM1 = srcMemPtr->getStaticDims()[1];
    const size_t DIM2 = srcMemPtr->getStaticDims()[2];
    const size_t DIM3 = srcMemPtr->getStaticDims()[3];
    const size_t DIM4 = srcMemPtr->getStaticDims()[4];
    const size_t DIM5 = srcMemPtr->getStaticDims()[5];

    cpuParallel->parallel_for5d(
        MB,
        DIM1,
        DIM2,
        DIM3,
        DIM4,
        [&](const size_t n, const size_t dim1, const size_t dim2, const size_t dim3, const size_t dim4) {
            for (size_t dim5 = 0; dim5 < DIM5; ++dim5) {
                const size_t src_off = n * DIM1 * DIM2 * DIM3 * DIM4 * DIM5 + dim1 * DIM2 * DIM3 * DIM4 * DIM5 +
                                       dim2 * DIM3 * DIM4 * DIM5 + dim3 * DIM4 * DIM5 + dim4 * DIM5 + dim5;
                const size_t dst_off = n * DIM5 * DIM1 * DIM2 * DIM3 * DIM4 + dim5 * DIM1 * DIM2 * DIM3 * DIM4 +
                                       dim1 * DIM2 * DIM3 * DIM4 + dim2 * DIM3 * DIM4 + dim3 * DIM4 + dim4;

                dst_data[dst_off] = src_data[src_off];
            }
        });
}

template <typename T>
struct TransposeOptimizedEmitter {
    void operator()(TransposeContext& ctx) {
        switch (ctx.srcMemPtr->getStaticDims().size()) {
        case 4:
            transpose_to_0312<T>(ctx.MB, ctx.srcMemPtr, ctx.dstMemPtr, ctx.cpuParallel);
            break;
        case 5:
            transpose_to_04123<T>(ctx.MB, ctx.srcMemPtr, ctx.dstMemPtr, ctx.cpuParallel);
            break;
        case 6:
            transpose_to_051234<T>(ctx.MB, ctx.srcMemPtr, ctx.dstMemPtr, ctx.cpuParallel);
            break;
        default:
            OPENVINO_THROW("Transpose supports optimized execution with only 4D, 5D and 6D shapes");
        }
    }
};
}  // namespace
void RefOptimizedTransposeExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) {
    const size_t dataSize = src[0]->getDesc().getPrecision().size();
    const auto MB = src[0]->getStaticDims()[0];
    TransposeContext ctx = {src[0], dst[0], MB, context->getCpuParallel()};
    OV_SWITCH(intel_cpu,
              TransposeOptimizedEmitter,
              ctx,
              dataSize,
              OV_CASE(1U, element_type_traits<ov::element::u8>::value_type),
              OV_CASE(2U, element_type_traits<ov::element::u16>::value_type),
              OV_CASE(4U, element_type_traits<ov::element::i32>::value_type));
}

bool RefOptimizedTransposeExecutor::init([[maybe_unused]] const TransposeParams& transposeParams,
                                         [[maybe_unused]] const std::vector<MemoryDescPtr>& srcDescs,
                                         [[maybe_unused]] const std::vector<MemoryDescPtr>& dstDescs,
                                         [[maybe_unused]] const dnnl::primitive_attr& attr) {
    return true;
}

}  // namespace ov::intel_cpu
