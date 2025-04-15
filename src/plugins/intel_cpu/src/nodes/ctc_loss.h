// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

class CTCLoss : public Node {
public:
    CTCLoss(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override{};
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    void executeDynamicImpl(const dnnl::stream& strm) override;
    bool needPrepareParams() const override {
        return false;
    };

private:
    bool ctcMergeRepeated;
    bool preprocessCollapseRepeated;
    bool unique;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
