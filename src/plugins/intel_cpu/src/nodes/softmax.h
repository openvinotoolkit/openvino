// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common/dnnl_executor.h"
#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

class SoftMax : public Node {
public:
    SoftMax(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void initOptimalPrimitiveDescriptor() override;
    void createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                          const std::vector<MemoryDescPtr>& outputDesc) override;
    void getSupportedDescriptors() override;
    bool created() const override;
    AttrPtr initPrimitiveAttr() override;
    void prepareParams() override;
    void execute(const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    using executorPtr = std::shared_ptr<DnnlExecutor>;
    executorPtr execPtr = nullptr;
    size_t axis = 0;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
