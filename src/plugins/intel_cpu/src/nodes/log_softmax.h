// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

class LogSoftmax : public Node {
public:
    LogSoftmax(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override{};
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;

    void prepareParams() override;
    void executeDynamicImpl(const dnnl::stream& strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    int axis;
    size_t reducedAxisSize = 0;
    size_t reducedAxisStride = 1;
    size_t axisStep = 1;
    bool isLastDim = false;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
