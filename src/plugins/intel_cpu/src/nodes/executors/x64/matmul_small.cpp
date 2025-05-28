// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matmul_small.hpp"

#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>
#include <unordered_map>

#include "jit_matmul_small.hpp"
#include "nodes/common/dnnl_executor.h"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_common.hpp"
#include "oneapi/dnnl/dnnl_types.h"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"

using namespace dnnl;

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
using namespace Xbyak;

namespace ov::intel_cpu {

MatMulSmallExecutor::MatMulSmallExecutor(const MatMulSmallAttrs& matmulAttrs, const dnnl::primitive_desc& pd)
    : DnnlExecutorLegacy(pd) {
    auto jcp = jit_matmul_small_config_params();
    jcp.M = matmulAttrs.M;
    jcp.N = matmulAttrs.N;
    jcp.K = matmulAttrs.K;
    if (mayiuse(cpu::x64::avx512_core)) {
        matmul_kernel = std::make_shared<jit_uni_matmul_small_kernel_f32<cpu::x64::avx512_core>>(jcp);
    } else if (mayiuse(cpu::x64::avx2)) {
        matmul_kernel = std::make_shared<jit_uni_matmul_small_kernel_f32<cpu::x64::avx2>>(jcp);
    } else if (mayiuse(cpu::x64::sse41)) {
        matmul_kernel = std::make_shared<jit_uni_matmul_small_kernel_f32<cpu::x64::sse41>>(jcp);
    } else {
        OPENVINO_THROW("Can't create jit jit_uni_matmul_small_kernel_f32 kernel");
    }

    if (matmul_kernel) {
        matmul_kernel->create_ker();
    }
}

void MatMulSmallExecutor::exec(const std::unordered_map<int, dnnl::memory>& primArgs,
                               [[maybe_unused]] const dnnl::stream& strm) {
    std::unordered_map<int, dnnl::memory> c_args = primArgs;
    auto in1 = c_args[DNNL_ARG_SRC_0];
    auto in2 = c_args[DNNL_ARG_WEIGHTS_0];
    auto out = c_args[DNNL_ARG_DST];
    const auto src_shape = in1.get_desc().get_dims();
    const auto wei_shape = in2.get_desc().get_dims();
    const auto* src_data = static_cast<float*>(in1.get_data_handle());
    const auto* wei_data = static_cast<float*>(in2.get_data_handle());
    auto* dst_data = static_cast<float*>(out.get_data_handle());
    const auto& M = src_shape[src_shape.size() - 2];
    const auto& K = src_shape[src_shape.size() - 1];
    const auto& N = wei_shape[wei_shape.size() - 1];
    const auto& src_spatial_size = M * K;
    const auto& wei_spatial_size = K * N;
    const auto& dst_spatial_size = M * N;
    const size_t threads_num = parallel_get_max_threads();
    const size_t wa =
        std::accumulate(src_shape.begin(), src_shape.end() - 2, static_cast<size_t>(1), std::multiplies<>());
    parallel_nt(threads_num, [&](const int ithr, [[maybe_unused]] const int nthr) {
        size_t start = 0, end = 0;
        splitter(wa, nthr, ithr, start, end);
        jit_matmul_small_call_args args;
        args.input1 = src_data + start * src_spatial_size;
        args.input2 = wei_data + start * wei_spatial_size;
        args.output = dst_data + start * dst_spatial_size;
        args.B = end - start;
        (*matmul_kernel)(&args);
    });
}

}  // namespace ov::intel_cpu