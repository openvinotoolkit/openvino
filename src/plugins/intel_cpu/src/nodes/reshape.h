// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <string>
#include <vector>
#include <memory>
#include "input.h"

namespace ov {
namespace intel_cpu {
namespace node {

class Reshape : public Node {
public:
    Reshape(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    bool created() const override;
    bool isExecutable() const override;

    bool needShapeInfer() const override;
    bool needPrepareParams() const override { return false; }
    void executeDynamicImpl(dnnl::stream strm) override;
    void execute(dnnl::stream strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    mutable std::vector<int> lastSecondInputValues;

    std::string errorPrefix;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
