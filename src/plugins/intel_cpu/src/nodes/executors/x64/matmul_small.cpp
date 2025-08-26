// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matmul_small.hpp"

#include <oneapi/dnnl/dnnl_types.h>

#include <common/primitive_attr.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>
#include <oneapi/dnnl/dnnl.hpp>
#include <unordered_map>

#include "dnnl_extension_utils.h"
#include "nodes/executors/debug_messages.hpp"
#include "nodes/executors/dnnl/dnnl_matmul_primitive.hpp"
#include "nodes/executors/dnnl/dnnl_post_op_data.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/matmul_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/kernels/x64/jit_matmul_small.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"

using namespace dnnl;
using namespace dnnl::impl;

namespace ov::intel_cpu {

MatMulSmallExecutor::MatMulSmallExecutor(const MatMulAttrs& attrs,
                                         const MemoryArgs& memory,
                                         const ExecutorContext::CPtr& context)
    : shapeAgnosticData(DnnlMatMulPrimitive::createShapeAgnosticData(attrs, memory, context, false)) {}

// Extract post ops args from primArgs
// comply Eltwise::appendPostOps and depthwise injector prepare_binary_args() on how to map
void MatMulSmallExecutor::prepare_binary_args(const DnnlPrimitiveAttrs& primAttrs) {
    std::unordered_map<int, dnnl::memory> c_args = primAttrs.dnnlArgs;
    m_post_ops_args.clear();
    auto* dnnlPrimitiveAttr = shapeAgnosticData->m_primAttrs.attr.get();
    const auto& attr = *dnnlPrimitiveAttr;
    const auto& post_ops = attr.post_ops_;
    m_post_ops_args.reserve(post_ops.entry_.size());
    unsigned idx = 0;
    for (const auto& post_op : post_ops.entry_) {
        if (post_op.is_binary() || post_op.is_depthwise() || post_op.is_quantization()) {
            m_post_ops_args.emplace_back(
                c_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_1].get_data_handle());
        }
        ++idx;
    }
}


[[maybe_unused]] static inline bool noWeightsDecompression(const MatMulConfig& config) {
    return !DnnlMatMulPrimitive::useWeightsDecompressionImpl(srcType(config), weiType(config));
}

[[maybe_unused]] static inline bool noSparseDecompression(const MatMulConfig& config) {
    return !(config.attrs.sparseWeights);
}

bool MatMulSmallExecutor::supports([[maybe_unused]] const MatMulConfig& config) {
    VERIFY(noSparseDecompression(config), UNSUPPORTED_SPARSE_WEIGHTS);
    VERIFY(noWeightsDecompression(config), UNSUPPORTED_WEIGHTS_DECOMPRESSION);

    VERIFY(!config.attrs.transposeA && !config.attrs.transposeB, "unsupported strides");
    VERIFY(all_of(ov::element::f32, srcType(config), weiType(config), dstType(config)), UNSUPPORTED_SRC_PRECISIONS);

    const auto& biasDesc = config.descs.at(ARG_BIAS);
    VERIFY(biasDesc->empty(), "bias is not supported");

    const auto& srcDesc0 = config.descs.at(ARG_SRC);
    const auto& srcDesc1 = config.descs.at(ARG_WEI);
    const auto srcRank0 = srcDesc0->getShape().getRank();
    const auto srcRank1 = srcDesc1->getShape().getRank();

    VERIFY(srcRank0 >= 2, UNSUPPORTED_SRC_RANK);
    VERIFY(srcRank0 == srcRank1, UNSUPPORTED_SRC_RANK);

    return true;
}

bool MatMulSmallExecutor::update(const MemoryArgs& memory) {
    const auto& srcDesc = memory.at(ARG_SRC)->getDescPtr();
    const auto& weiDesc = memory.at(ARG_WEI)->getDescPtr();

    const auto& shape_in0 = srcDesc->getShape().getStaticDims();
    const auto& shape_in1 = weiDesc->getShape().getStaticDims();

    m_matmul_attrs.M = *++shape_in0.rbegin();
    m_matmul_attrs.K = *shape_in0.rbegin();
    m_matmul_attrs.N = *shape_in1.rbegin();

    jit_matmul_small_config_params jcp{m_matmul_attrs.M, m_matmul_attrs.K, m_matmul_attrs.N};

    m_matmul_attrs.WA =
        std::accumulate(shape_in0.begin(), shape_in0.end() - 2, static_cast<size_t>(1), std::multiplies<>());

    auto* dnnlPrimitiveAttr = shapeAgnosticData->m_primAttrs.attr.get();
    if (mayiuse(cpu::x64::avx512_core)) {
        m_matmul_kernel =
            std::make_shared<jit_uni_matmul_small_kernel_f32<cpu::x64::avx512_core>>(jcp, *dnnlPrimitiveAttr);
    } else if (mayiuse(cpu::x64::avx2)) {
        m_matmul_kernel = std::make_shared<jit_uni_matmul_small_kernel_f32<cpu::x64::avx2>>(jcp, *dnnlPrimitiveAttr);
    } else if (mayiuse(cpu::x64::sse41)) {
        m_matmul_kernel = std::make_shared<jit_uni_matmul_small_kernel_f32<cpu::x64::sse41>>(jcp, *dnnlPrimitiveAttr);
    } else {
        OPENVINO_THROW("Can't create jit jit_uni_matmul_small_kernel_f32 kernel");
    }

    m_matmul_kernel->create_ker();

    // prepare_binary_args
    if (!shapeAgnosticData->m_primAttrs.attr.get()->post_ops_.entry_.empty()) {
        prepare_binary_args(shapeAgnosticData->m_primAttrs);
    }

    return true;
}

void MatMulSmallExecutor::execute(const MemoryArgs& memory) {
    const auto& in1 = memory.at(ARG_SRC);
    const auto& in2 = memory.at(ARG_WEI);
    const auto& out = memory.at(ARG_DST);

    const auto* src_data = in1->getDataAs<const float>();
    const auto* wei_data = in2->getDataAs<const float>();
    auto* dst_data = out->getDataAs<float>();

    const auto M = m_matmul_attrs.M;
    const auto K = m_matmul_attrs.K;
    const auto N = m_matmul_attrs.N;

    const auto& out_shape = out->getDesc().getShape().getStaticDims();
    const auto& OC = out_shape.size() >= 3 ? out_shape[out_shape.size() - 3] : 1;

    const auto src_spatial_size = M * K;
    const auto wei_spatial_size = K * N;
    const auto dst_spatial_size = M * N;
    const size_t wa = m_matmul_attrs.WA;
    const size_t threads_num = parallel_get_max_threads();

    parallel_nt(threads_num, [&](const int ithr, [[maybe_unused]] const int nthr) {
        size_t start = 0, end = 0;
        splitter(wa, nthr, ithr, start, end);
        jit_matmul_small_call_args args{};
        args.input1 = src_data + start * src_spatial_size;
        args.input2 = wei_data + start * wei_spatial_size;
        args.output = dst_data + start * dst_spatial_size;
        args.B = end - start;
        args.oc_off = start % static_cast<size_t>(OC);
        args.oc = OC;
        args.post_op_data = reinterpret_cast<void*>(m_post_ops_args.data());
        (*m_matmul_kernel)(&args);
    });
}

}  // namespace ov::intel_cpu
