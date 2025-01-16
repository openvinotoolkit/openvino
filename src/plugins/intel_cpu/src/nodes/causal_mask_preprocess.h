// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"
#include "transformations/cpu_opset/common/op/causal_mask_preprocess.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class CausalMaskPreprocess : public Node {
public:
    CausalMaskPreprocess(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override {}
    bool created() const override {
        return getType() == Type::CausalMaskPreprocess;
    }
    bool needPrepareParams() const override {
        return false;
    };
    void executeDynamicImpl(dnnl::stream strm) override {
        execute(strm);
    }
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    struct Executor {
        virtual void execute(dnnl::stream strm,
                             intel_cpu::Node * pnode,
                             const intel_cpu::CausalMaskPreprocessNode::Config& config) = 0;
        virtual ~Executor() = default;
    };
    template <typename T>
    struct ExecutorCausalMaskPreprocess;
    intel_cpu::CausalMaskPreprocessNode::Config m_config;
    std::shared_ptr<Executor> m_executor;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
