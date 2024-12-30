// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"
#include "transformations/cpu_opset/x64/op/act_sparse_fc.hpp"

#if defined(OPENVINO_ARCH_X86_64)
#    include "kernels/x64/act_sparse_fc_kernel.hpp"
#else
namespace ov {
namespace intel_cpu {
class ActSparseFcKernel {
public:
    // compile time parameters
    ActSparseFcKernel(bool is_quantized, bool is_int4, bool with_zero_points, int ic_group_size);

    void operator()(const float* input,
                    float* output,
                    int M,
                    int IC,
                    int OC,
                    float threshold,
                    float zero_point,
                    const void* W,
                    const float* scales,
                    const uint8_t* zp) {
        OPENVINO_THROW("Unsupported platform.");
    }

    void repack_weights_i4(uint8_t* src, uint8_t* dst, int IC, int OC) {
        OPENVINO_THROW("Unsupported platform.");
    }
};
}  // namespace intel_cpu
}  // namespace ov
#endif

namespace ov {
namespace intel_cpu {
namespace node {

class ActSparseFC : public Node {
public:
    ActSparseFC(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override {}
    bool created() const override {
        return getType() == Type::ActSparseFC;
    }
    bool needPrepareParams() const override {
        return false;  // this is a shape-agnostic kernel
    }
    void createPrimitive() override;
    void executeDynamicImpl(dnnl::stream strm) override {
        execute(strm);
    }
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    std::shared_ptr<ov::intel_cpu::ActSparseFcKernel> m_executor;

    MemoryPtr m_weight;
    MemoryPtr m_zp;
    MemoryPtr m_scales;

    ActSparseFCNode::Config m_config;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
