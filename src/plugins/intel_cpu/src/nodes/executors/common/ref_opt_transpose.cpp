// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ref_opt_transpose.hpp"

#include "openvino/core/parallel.hpp"

namespace ov::intel_cpu {
namespace {

struct TransposeContext {
    MemoryCPtr srcMemPtr;
    MemoryPtr dstMemPtr;
    int MB;
};

template <typename T>
void transpose_to_0312(const int MB, const MemoryCPtr& srcMemPtr, MemoryPtr& dstMemPtr) {
    const auto src_data = srcMemPtr->getDataAs<const T>();
    auto dst_data = dstMemPtr->getDataAs<T>();

    const int DIM1 = srcMemPtr->getStaticDims()[1];
    const int DIM2 = srcMemPtr->getStaticDims()[2];
    const int DIM3 = srcMemPtr->getStaticDims()[3];

    parallel_for3d(MB, DIM1, DIM2, [&](const int n, const int dim1, const int dim2) {
        for (int dim3 = 0; dim3 < DIM3; ++dim3) {
            const int src_off = n * DIM1 * DIM2 * DIM3 + dim1 * DIM2 * DIM3 + dim2 * DIM3 + dim3;
            const int dst_off = n * DIM1 * DIM2 * DIM3 + dim3 * DIM1 * DIM2 + dim1 * DIM2 + dim2;

            dst_data[dst_off] = src_data[src_off];
        }
    });
}

template <typename T>
void transpose_to_04123(const int MB, const MemoryCPtr& srcMemPtr, MemoryPtr& dstMemPtr) {
    const auto src_data = srcMemPtr->getDataAs<const T>();
    auto dst_data = dstMemPtr->getDataAs<T>();

    const int DIM1 = srcMemPtr->getStaticDims()[1];
    const int DIM2 = srcMemPtr->getStaticDims()[2];
    const int DIM3 = srcMemPtr->getStaticDims()[3];
    const int DIM4 = srcMemPtr->getStaticDims()[4];

    parallel_for4d(MB, DIM1, DIM2, DIM3, [&](const int n, const int dim1, const int dim2, const int dim3) {
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
void transpose_to_051234(const int MB, const MemoryCPtr& srcMemPtr, MemoryPtr& dstMemPtr) {
    const auto src_data = srcMemPtr->getDataAs<const T>();
    auto dst_data = dstMemPtr->getDataAs<T>();

    const int DIM1 = srcMemPtr->getStaticDims()[1];
    const int DIM2 = srcMemPtr->getStaticDims()[2];
    const int DIM3 = srcMemPtr->getStaticDims()[3];
    const int DIM4 = srcMemPtr->getStaticDims()[4];
    const int DIM5 = srcMemPtr->getStaticDims()[5];

    parallel_for5d(MB,
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
            transpose_to_0312<T>(ctx.MB, ctx.srcMemPtr, ctx.dstMemPtr);
            break;
        case 5:
            transpose_to_04123<T>(ctx.MB, ctx.srcMemPtr, ctx.dstMemPtr);
            break;
        case 6:
            transpose_to_051234<T>(ctx.MB, ctx.srcMemPtr, ctx.dstMemPtr);
            break;
        default:
            OPENVINO_THROW("Transpose supports optimized execution with only 4D, 5D and 6D shapes");
        }
    }
};
}  // namespace
void RefOptimizedTransposeExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) {
    const size_t dataSize = src[0]->getDesc().getPrecision().size();
    const int MB = src[0]->getStaticDims()[0];
    TransposeContext ctx = {src[0], dst[0], MB};
    OV_SWITCH(intel_cpu,
              TransposeOptimizedEmitter,
              ctx,
              dataSize,
              OV_CASE(1u, element_type_traits<ov::element::u8>::value_type),
              OV_CASE(2u, element_type_traits<ov::element::u16>::value_type),
              OV_CASE(4u, element_type_traits<ov::element::i32>::value_type));
}

bool RefOptimizedTransposeExecutor::init(const TransposeParams& transposeParams,
                                         const std::vector<MemoryDescPtr>& srcDescs,
                                         const std::vector<MemoryDescPtr>& dstDescs,
                                         const dnnl::primitive_attr& attr) {
    return true;
}

}  // namespace ov::intel_cpu
