// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"
#include "transformations/cpu_opset/x64/op/qkv_proj.hpp"

#if defined(OPENVINO_ARCH_X86_64)
#include "kernels/x64/mlp_kernel.hpp"
#endif

namespace ov {
namespace intel_cpu {
namespace node {

class QKVProjection : public Node {
public:
    QKVProjection(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override {}
    bool created() const override {
        return getType() == Type::QKVProjection;
    }
    void prepareParams() override;
    void executeDynamicImpl(dnnl::stream strm) override {
        execute(strm);
    }
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    struct Impl;
    std::shared_ptr<Impl> m_pimpl;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
