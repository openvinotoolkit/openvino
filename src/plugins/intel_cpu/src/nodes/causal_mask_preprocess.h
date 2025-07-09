// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_types.h"
#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"
#include "transformations/cpu_opset/common/op/causal_mask_preprocess.hpp"

namespace ov::intel_cpu::node {

class CausalMaskPreprocess : public Node {
public:
    CausalMaskPreprocess(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {}
    bool created() const override {
        return getType() == Type::CausalMaskPreprocess;
    }
    bool needPrepareParams() const override {
        return false;
    };
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    struct Executor {
        virtual void execute(const dnnl::stream& strm,
                             intel_cpu::Node* pnode,
                             const intel_cpu::CausalMaskPreprocessNode::Config& config) = 0;
        virtual ~Executor() = default;
    };
    template <typename T>
    struct ExecutorCausalMaskPreprocess;
    intel_cpu::CausalMaskPreprocessNode::Config m_config;
    std::shared_ptr<Executor> m_executor;
};

}  // namespace ov::intel_cpu::node
