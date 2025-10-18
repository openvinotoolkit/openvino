// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "nodes/executors/deconv.hpp"
#include "nodes/executors/aarch64/jit_conv3d.hpp"
#include "nodes/executors/aarch64/jit_conv3d_f32.hpp"

namespace ov::intel_cpu {

class JitDeconv3DExecutor : public DeconvExecutor {
public:
    explicit JitDeconv3DExecutor(ExecutorContext::CPtr context) : DeconvExecutor(std::move(context)) {}
    ~JitDeconv3DExecutor() override = default;

    bool init(const DeconvAttrs& deconvAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr& attr) override;

    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              const void* post_ops_data_) override;

    [[nodiscard]] impl_desc_type getImplType() const override { return impl_desc_type::jit_asimd; }

private:
    std::vector<MemoryDescPtr> m_srcDescs;
    std::vector<MemoryDescPtr> m_dstDescs;
    // kernels
    std::unique_ptr<JitConv3DKernelF16> m_ip_kernel_f16;
    std::unique_ptr<JitConv3DKernelF32> m_ip_kernel_f32;
    bool m_is_fp32{false};

    // packed weights
    std::vector<uint16_t> m_wei_packed_f16;
    std::vector<float>    m_wei_packed_f32;
    bool m_wei_packed_ready_f16{false};
    bool m_wei_packed_ready_f32{false};
    size_t m_padded_IC_f16{0};
    size_t m_padded_IC_f32{0};

    void ensure_weights_packed_f16(const std::vector<MemoryCPtr>& src);
    void ensure_weights_packed_f32(const std::vector<MemoryCPtr>& src);
    void exec_fp16(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst);
    void exec_fp32(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst);
};

class AArch64JitDeconvExecutorBuilder : public DeconvExecutorBuilder {
public:
    [[nodiscard]] bool isSupported(const DeconvAttrs& attrs,
                                   const std::vector<MemoryDescPtr>& srcDescs,
                                   const std::vector<MemoryDescPtr>& dstDescs) const override;
    [[nodiscard]] DeconvExecutorPtr makeExecutor(ExecutorContext::CPtr context) const override {
        return std::make_shared<JitDeconv3DExecutor>(context);
    }
};

}  // namespace ov::intel_cpu
