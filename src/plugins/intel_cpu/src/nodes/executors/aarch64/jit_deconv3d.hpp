// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "nodes/executors/deconv.hpp"
#include "nodes/executors/aarch64/jit_conv3d.hpp"

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
    std::unique_ptr<JitConv3DKernelF16> m_ip_kernel;
    std::vector<uint16_t> m_wei_packed;
    bool m_wei_packed_ready{false};
    size_t m_padded_IC{0};
    void ensure_weights_packed(const std::vector<MemoryCPtr>& src);
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
