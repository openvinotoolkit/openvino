// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include <string>
#include <memory>
#include <vector>
#include <tuple>
#include "nodes/executors/executor_factory.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class MVN : public Node {
public:
    MVN(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    bool created() const override;
    void execute(dnnl::stream strm) override;
    void executeDynamicImpl(dnnl::stream strm) override;
    bool canBeInPlace() const override {
        return false;
    }

    inline bool getAcrossChannels() const {
        return mvnAttrs.initAcrossChannels_;
    }

    inline bool getNormalizeVariance() const {
        return mvnAttrs.normalizeVariance_;
    }

    bool canFuse(const NodePtr& node) const override;
    void prepareParams() override;
    void createPrimitive() override;

private:
    ExecutorPtr createExecutor();
    MVNAttrs mvnAttrs;
    PostOps postOps;
    MemoryArgs memory;
    ExecutorFactoryPtr<MVNAttrs, node::MVN> factory;
    ExecutorPtr executor = nullptr;
    bool onlyUnaryPostOps = true;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
