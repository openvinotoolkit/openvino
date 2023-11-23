// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <ie_common.h>
#include <node.h>

#include <memory>
#include <string>
#include <vector>

#include "transformations/cpu_opset/common/op/sdp.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class ScaledDotProductAttention : public Node {
public:
    ScaledDotProductAttention(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override {}
    bool created() const override {
        return getType() == Type::ScaledDotProductAttention;
    }
    // pastkv may have zero dimension
    bool isExecutable() const override {
        return true;
    }
    bool needPrepareParams() const override {
        return false;
    }
    void executeDynamicImpl(dnnl::stream strm) override {
        execute(strm);
    }
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

    enum KernelTypes { KT_REF, KT_ONEDNN, KT_MLAS};

private:
    struct Executor {
        virtual void execute(dnnl::stream strm, const std::vector<MemoryPtr>& inputs, const std::vector<MemoryPtr>& outputs) = 0;
    };

    ScaledDotProductAttentionStub::Config m_config;
    std::shared_ptr<Executor> m_executor;
    template <KernelTypes KType, typename T> struct AttentionExecutor;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
