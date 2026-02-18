// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ref_opt_transpose.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

#include "cpu_memory.h"
#include "cpu_parallel.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/common/ref_opt_transpose.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/transpose_config.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "selective_build.h"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {
namespace {

struct TransposeContext {
    MemoryCPtr srcMemPtr;
    MemoryPtr dstMemPtr;
    int MB;
    std::shared_ptr<CpuParallel> cpuParallel;
};

template <typename T>
void transpose_to_0312(const int MB,
                       const MemoryCPtr& srcMemPtr,
                       MemoryPtr& dstMemPtr,
                       const std::shared_ptr<CpuParallel>& cpuParallel) {
    const auto* const src_data = srcMemPtr->getDataAs<const T>();
    auto dst_data = dstMemPtr->getDataAs<T>();

    const int DIM1 = srcMemPtr->getStaticDims()[1];
    const int DIM2 = srcMemPtr->getStaticDims()[2];
    const int DIM3 = srcMemPtr->getStaticDims()[3];

    cpuParallel->parallel_for3d(MB, DIM1, DIM2, [&](const int n, const int dim1, const int dim2) {
        for (int dim3 = 0; dim3 < DIM3; ++dim3) {
            const int src_off = n * DIM1 * DIM2 * DIM3 + dim1 * DIM2 * DIM3 + dim2 * DIM3 + dim3;
            const int dst_off = n * DIM1 * DIM2 * DIM3 + dim3 * DIM1 * DIM2 + dim1 * DIM2 + dim2;

            dst_data[dst_off] = src_data[src_off];
        }
    });
}

template <typename T>
void transpose_to_04123(const int MB,
                        const MemoryCPtr& srcMemPtr,
                        MemoryPtr& dstMemPtr,
                        const std::shared_ptr<CpuParallel>& cpuParallel) {
    const auto* const src_data = srcMemPtr->getDataAs<const T>();
    auto dst_data = dstMemPtr->getDataAs<T>();

    const int DIM1 = srcMemPtr->getStaticDims()[1];
    const int DIM2 = srcMemPtr->getStaticDims()[2];
    const int DIM3 = srcMemPtr->getStaticDims()[3];
    const int DIM4 = srcMemPtr->getStaticDims()[4];

    cpuParallel->parallel_for4d(MB, DIM1, DIM2, DIM3, [&](const int n, const int dim1, const int dim2, const int dim3) {
        for (int dim4 = 0; dim4 < DIM4; ++dim4) {
            const int src_off =
                n * DIM1 * DIM2 * DIM3 * DIM4 + dim1 * DIM2 * DIM3 * DIM4 + dim2 * DIM3 * DIM4 + dim3 * DIM4 + dim4;
            const int dst_off =
                n * DIM1 * DIM2 * DIM3 * DIM4 + dim4 * DIM1 * DIM2 * DIM3 + dim1 * DIM2 * DIM3 + dim2 * DIM3 + dim3;

            dst_data[dst_off] = src_data[src_off];
        }
    });
}

template <typename T>
void transpose_to_051234(const int MB,
                         const MemoryCPtr& srcMemPtr,
                         MemoryPtr& dstMemPtr,
                         const std::shared_ptr<CpuParallel>& cpuParallel) {
    const auto* const src_data = srcMemPtr->getDataAs<const T>();
    auto dst_data = dstMemPtr->getDataAs<T>();

    const int DIM1 = srcMemPtr->getStaticDims()[1];
    const int DIM2 = srcMemPtr->getStaticDims()[2];
    const int DIM3 = srcMemPtr->getStaticDims()[3];
    const int DIM4 = srcMemPtr->getStaticDims()[4];
    const int DIM5 = srcMemPtr->getStaticDims()[5];

    cpuParallel->parallel_for5d(
        MB,
        DIM1,
        DIM2,
        DIM3,
        DIM4,
        [&](const int n, const int dim1, const int dim2, const int dim3, const int dim4) {
            for (int dim5 = 0; dim5 < DIM5; ++dim5) {
                const int src_off = n * DIM1 * DIM2 * DIM3 * DIM4 * DIM5 + dim1 * DIM2 * DIM3 * DIM4 * DIM5 +
                                    dim2 * DIM3 * DIM4 * DIM5 + dim3 * DIM4 * DIM5 + dim4 * DIM5 + dim5;
                const int dst_off = n * DIM5 * DIM1 * DIM2 * DIM3 * DIM4 + dim5 * DIM1 * DIM2 * DIM3 * DIM4 +
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

bool RefOptimizedTransposeExecutor::supports(const TransposeConfig& config) {
    static const std::vector<std::vector<size_t>> optimizedOrders = {
        std::vector<size_t>{0, 3, 1, 2},
        std::vector<size_t>{0, 4, 1, 2, 3},
        std::vector<size_t>{0, 5, 1, 2, 3, 4},
    };

    const auto& srcDesc = config.descs.at(ARG_SRC);
    if (srcDesc->hasLayoutType(LayoutType::ncsp) &&
        std::find(optimizedOrders.begin(), optimizedOrders.end(), config.attrs.permuteParams.order) !=
            optimizedOrders.end()) {
        return true;
    }

    DEBUG_LOG("RefOptimizedTransposeExecutor is not supported, because passed order is not optimized");
    return false;
}

ExecutorPtr RefOptimizedTransposeExecutor::create(const TransposeAttrs& attrs,
                                                  [[maybe_unused]] const MemoryArgs& memory,
                                                  const ExecutorContext::CPtr& context) {
    return std::make_shared<RefOptimizedTransposeExecutor>(attrs, context);
}

bool RefOptimizedTransposeExecutor::init([[maybe_unused]] const MemoryArgs& memory) {
    return true;
}

void RefOptimizedTransposeExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) {
    const size_t dataSize = src[0]->getDesc().getPrecision().size();
    const int MB = src[0]->getStaticDims()[0];
    TransposeContext ctx = {src[0], dst[0], MB, context->getCpuParallel()};
    OV_SWITCH(intel_cpu,
              TransposeOptimizedEmitter,
              ctx,
              dataSize,
              OV_CASE(1U, element_type_traits<ov::element::u8>::value_type),
              OV_CASE(2U, element_type_traits<ov::element::u16>::value_type),
              OV_CASE(4U, element_type_traits<ov::element::i32>::value_type));
}

}  // namespace ov::intel_cpu
