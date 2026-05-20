// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "common/c_types_map.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "nodes/kernels/x64/jit_softmax_fork_kernel_f32.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

class SoftmaxForkExecutor {
public:
    SoftmaxForkExecutor(const std::vector<size_t>& dims,
                        size_t axis,
                        ov::element::Type precision,
                        const dnnl::memory::desc& srcDesc);

    [[nodiscard]] bool isSupported() const;
    [[nodiscard]] dnnl::impl::status_t init();
    void execute(const uint8_t* src, uint8_t* dst) const;

private:
    struct IKernelExecutor {
        virtual ~IKernelExecutor() = default;
        virtual dnnl::impl::status_t create() = 0;
        virtual void run(const dnnl::impl::cpu::x64::jit_softmax_call_s* args) const = 0;
    };

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    struct KernelExecutor;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    dnnl::impl::status_t initForIsa();

    std::vector<size_t> m_dims;
    size_t m_axis = 0;
    ov::element::Type m_precision;
    dnnl::memory::desc m_srcDesc;
    dnnl::impl::cpu::x64::jit_softmax_conf_t m_jpp{};
    bool m_isSupported = false;
    std::unique_ptr<IKernelExecutor> m_kernelExecutor;
};

}  // namespace ov::intel_cpu
