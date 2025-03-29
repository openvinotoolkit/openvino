// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

class GatherElements : public Node {
public:
    GatherElements(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override{};
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

protected:
    void executeDynamicImpl(const dnnl::stream& strm) override;
    void prepareParams() override;

private:
    const size_t dataIndex_ = 0;
    const size_t indicesIndex_ = 1;

    size_t axis_;
    size_t dataTypeSize_ = 0;
    int strideAxDst_ = 0;
    int dstAxDim_ = 0;
    int strideAx1Diff_ = 0;

    template <typename dataType>
    void directExecution();
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
