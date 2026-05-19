// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "nodes/executors/x64/softmax_fork_executor.hpp"

#include <algorithm>

#include "common/memory_desc_wrapper.hpp"
#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"
#include "nodes/kernels/x64/jit_softmax_fork_kernel_f32.hpp"
#include "openvino/core/except.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu {

template <cpu_isa_t isa>
struct SoftmaxForkExecutor::KernelExecutor : public SoftmaxForkExecutor::IKernelExecutor {
    explicit KernelExecutor(const jit_softmax_conf_t& jpp) : kernel(jpp) {}

    status_t create() override {
        return kernel.create_kernel();
    }

    void run(const jit_softmax_call_s* args) const override {
        kernel(args);
    }

    jit_softmax_fork_kernel_f32<isa> kernel;
};

SoftmaxForkExecutor::SoftmaxForkExecutor(const std::vector<size_t>& dims,
                                         size_t axis,
                                         ov::element::Type precision,
                                         const dnnl::memory::desc& srcDesc)
    : m_dims(dims),
      m_axis(axis),
      m_precision(precision),
      m_srcDesc(srcDesc) {
    if (m_dims.empty() || m_axis >= m_dims.size()) {
        return;
    }

    const auto ndims = m_dims.size();
    if (ndims == 3) {
        return;
    }

    dnnl::impl::memory_desc_wrapper src_d(m_srcDesc.get());
    const auto srcDataType = src_d.data_type();
    if (!utils::one_of(srcDataType, data_type::f32, data_type::bf16)) {
        return;
    }

    auto dat_tag = utils::pick(ndims - 3,
                               dnnl::impl::format_tag::ncw,
                               dnnl::impl::format_tag::nchw,
                               dnnl::impl::format_tag::ncdhw);
    if (!src_d.is_dense(true) || src_d.matches_one_of_tag(dat_tag) != dat_tag) {
        return;
    }

    m_jpp.outer_size = utils::array_product(m_dims.data(), m_axis);
    m_jpp.channels = m_dims[m_axis];
    m_jpp.inner_size = utils::array_product(m_dims.data() + m_axis + 1, ndims - m_axis - 1);

    if (m_jpp.outer_size < 1 || m_jpp.channels < 1 || m_jpp.inner_size < 1) {
        return;
    }

    if (!utils::one_of(m_precision, ov::element::f32, ov::element::bf16)) {
        return;
    }

    m_isSupported = true;
}

bool SoftmaxForkExecutor::isSupported() const {
    return m_isSupported;
}

status_t SoftmaxForkExecutor::init() {
    if (!m_isSupported) {
        return status::unimplemented;
    }

    if (mayiuse(avx512_core)) {
        return initForIsa<avx512_core>();
    } else if (mayiuse(avx2)) {
        return initForIsa<avx2>();
    } else if (mayiuse(sse41)) {
        return initForIsa<sse41>();
    } else {
        return status::unimplemented;
    }
}

template <cpu_isa_t isa>
status_t SoftmaxForkExecutor::initForIsa() {
    softmax_desc_t desc = {};
    desc.src_desc = *m_srcDesc.get();
    desc.dst_desc = *m_srcDesc.get();
    desc.softmax_axis = static_cast<int>(m_axis);

    dnnl::impl::memory_desc_wrapper src_d(m_srcDesc.get());
    dnnl::impl::memory_desc_wrapper dst_d(m_srcDesc.get());

    jit_softmax_conf_t jpp = {};
    const auto confStatus = jit_softmax_fork_kernel_f32<isa>::init_conf(jpp, desc, src_d, dst_d);
    if (confStatus != status::success) {
        return confStatus;
    }

    // Keep oneDNN a3ca91c4 behavior: fork path only for inner_size > 1.
    if (jpp.inner_size <= 1) {
        return status::unimplemented;
    }

    m_jpp = jpp;
    m_kernelExecutor = std::make_unique<KernelExecutor<isa>>(m_jpp);
    return m_kernelExecutor->create();
}

void SoftmaxForkExecutor::execute(const uint8_t* src, uint8_t* dst) const {
    if (!m_kernelExecutor) {
        OPENVINO_THROW("SoftmaxForkExecutor is not initialized");
    }

    const auto outerSize = m_jpp.outer_size;
    const auto dim = m_jpp.channels * m_jpp.inner_size;
    const auto& jpp = m_jpp;
    dnnl::impl::memory_desc_wrapper data_d(m_srcDesc.get());

    if (jpp.inner_size > 1) {
        const size_t workAmount = outerSize;

        auto ker = [&](const int ithr, const int nthr) {
            size_t start = 0;
            size_t end = 0;

            balance211(workAmount, nthr, ithr, start, end);

            for (size_t iwork = start; iwork < end; ++iwork) {
                const size_t ou = iwork;
                auto args = jit_softmax_call_s();
                args.channels = jpp.channels;
                args.work = jpp.inner_size;
                const size_t off = data_d.off_l(ou * dim);
                args.src = src + off * jpp.dt_size;
                args.dst = dst + off * jpp.dt_size;

                m_kernelExecutor->run(&args);
            }
        };

        parallel(0, ker);
    } else {
        const int ouBlocks = dnnl::impl::utils::div_up(outerSize, jpp.outer_block);
        const size_t workAmount = ouBlocks;

        auto ker = [&](const int ithr, const int nthr) {
            size_t start = 0;
            size_t end = 0;

            balance211(workAmount, nthr, ithr, start, end);

            for (size_t iwork = start; iwork < end; ++iwork) {
                const size_t oub = iwork;
                const size_t work = nstl::min(jpp.outer_block, outerSize - oub * jpp.outer_block);

                auto args = jit_softmax_call_s();
                args.channels = jpp.channels;
                args.work = work;
                const size_t off = data_d.off_l(oub * jpp.outer_block * dim);
                args.src = src + off * jpp.dt_size;
                args.dst = dst + off * jpp.dt_size;

                m_kernelExecutor->run(&args);
            }
        };

        parallel(0, ker);
    }
}

}  // namespace ov::intel_cpu
